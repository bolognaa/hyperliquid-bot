import time
from datetime import datetime, timezone
from typing import Dict, Optional

import aiohttp
import numpy as np
import pandas as pd

from core.config import KRAKEN_BASE_URL, KRAKEN_PAIRS, OHLC_INTERVAL, OHLC_CANDLES, TUNABLE
from core.logger import get_logger
from core.state import BotState

logger = get_logger()


class MarketDataAgent:
    def __init__(self, state: BotState):
        self.state = state
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ------------------------------------------------------------------
    # Kraken fetchers
    # ------------------------------------------------------------------

    async def _fetch_ohlc(self, pair: str) -> Optional[pd.DataFrame]:
        since = int(time.time()) - OHLC_CANDLES * OHLC_INTERVAL * 60
        params = {"pair": pair, "interval": OHLC_INTERVAL, "since": since}
        session = await self._get_session()
        try:
            async with session.get(f"{KRAKEN_BASE_URL}/OHLC", params=params) as resp:
                data = await resp.json()
            if data.get("error"):
                logger.error(f"Kraken OHLC error for {pair}: {data['error']}")
                return None
            candles = list(data["result"].values())[0]
            df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
            for col in ["open", "high", "low", "close", "vwap", "volume"]:
                df[col] = df[col].astype(float)
            df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Kraken OHLC fetch failed for {pair}: {e}")
            return None

    async def _fetch_ticker(self, pair: str) -> Optional[float]:
        session = await self._get_session()
        try:
            async with session.get(f"{KRAKEN_BASE_URL}/Ticker", params={"pair": pair}) as resp:
                data = await resp.json()
            if data.get("error"):
                return None
            result = list(data["result"].values())[0]
            return float(result["c"][0])  # last trade price
        except Exception as e:
            logger.error(f"Kraken ticker fetch failed for {pair}: {e}")
            return None

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(closes: pd.Series, period: int) -> float:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0

    @staticmethod
    def _ema(closes: pd.Series, period: int) -> pd.Series:
        return closes.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _macd(closes: pd.Series, fast: int, slow: int, signal: int):
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

    @staticmethod
    def _bollinger(closes: pd.Series, period: int, std: int):
        sma = closes.rolling(window=period).mean()
        std_dev = closes.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        val = atr.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    def _compute_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        closes = df["close"]
        p = TUNABLE

        ema_fast_s = self._ema(closes, p.EMA_FAST)
        ema_slow_s = self._ema(closes, p.EMA_SLOW)

        macd_line, signal_line, macd_hist = self._macd(closes, p.MACD_FAST, p.MACD_SLOW, p.MACD_SIGNAL)
        bb_upper, bb_mid, bb_lower = self._bollinger(closes, p.BB_PERIOD, p.BB_STD)

        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else current_price
        price_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0

        return {
            "price": current_price,
            "rsi": self._rsi(closes, p.RSI_PERIOD),
            "ema_fast": float(ema_fast_s.iloc[-1]),
            "ema_slow": float(ema_slow_s.iloc[-1]),
            "ema_fast_prev": float(ema_fast_s.iloc[-2]) if len(ema_fast_s) > 1 else float(ema_fast_s.iloc[-1]),
            "ema_slow_prev": float(ema_slow_s.iloc[-2]) if len(ema_slow_s) > 1 else float(ema_slow_s.iloc[-1]),
            "macd_line": macd_line,
            "signal_line": signal_line,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
            "atr": self._atr(df, p.ATR_PERIOD),
            "price_change_pct": price_change_pct,
            "recent_closes": closes.tail(5).tolist(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    async def run(self):
        logger.info("MarketDataAgent: fetching data...")
        success_count = 0

        for asset, pair in KRAKEN_PAIRS.items():
            df = await self._fetch_ohlc(pair)
            if df is None or len(df) < 30:
                logger.warning(f"MarketDataAgent: insufficient OHLC data for {asset}")
                continue

            price = await self._fetch_ticker(pair)
            if price is None:
                price = float(df["close"].iloc[-1])

            indicators = self._compute_indicators(df, price)
            await self.state.update_indicators(asset, indicators)
            await self.state.update_ohlc(asset, df)
            logger.info(
                f"{asset}: price=${price:,.2f} RSI={indicators['rsi']:.1f} "
                f"EMA({TUNABLE.EMA_FAST}/{TUNABLE.EMA_SLOW})="
                f"{indicators['ema_fast']:.2f}/{indicators['ema_slow']:.2f}"
            )
            success_count += 1

        status = "ok" if success_count == len(KRAKEN_PAIRS) else "partial"
        await self.state.update_agent_status("market_data", status, f"{success_count}/{len(KRAKEN_PAIRS)} assets updated")

    async def run_price_ticker(self):
        """Fast loop: update just the live price every 10s for accurate SL/TP monitoring."""
        import asyncio
        logger.info("MarketDataAgent: fast price ticker started (10s interval).")
        while self.state.running:
            await asyncio.sleep(10)
            for asset, pair in KRAKEN_PAIRS.items():
                price = await self._fetch_ticker(pair)
                if price is None:
                    continue
                # Patch only the price field — leave all indicators intact
                async with self.state._lock:
                    if self.state.indicators.get(asset):
                        self.state.indicators[asset]["price"] = price

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
