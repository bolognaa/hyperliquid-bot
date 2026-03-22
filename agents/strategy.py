from datetime import datetime, timezone
from typing import Dict, Optional

from core.config import ASSETS, TUNABLE
from core.logger import get_logger
from core.state import BotState

logger = get_logger()


class StrategyAgent:
    def __init__(self, state: BotState):
        self.state = state

    def _trend_signal(self, ind: Dict, sentiment: Dict) -> Optional[Dict]:
        """EMA crossover + MACD confirmation + AI sentiment."""
        p = TUNABLE

        ema_fast = ind["ema_fast"]
        ema_slow = ind["ema_slow"]
        ema_fast_prev = ind.get("ema_fast_prev", ema_fast)
        ema_slow_prev = ind.get("ema_slow_prev", ema_slow)

        macd_hist = ind["macd_hist"]
        macd_line = ind["macd_line"]

        ai_dir = sentiment.get("direction", "neutral")
        ai_conf = sentiment.get("confidence", 0.0)

        # Detect EMA crossover
        bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow
        bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow

        # Current trend (no crossover needed, just position)
        bullish_trend = ema_fast > ema_slow
        bearish_trend = ema_fast < ema_slow

        direction = None
        base_strength = 0.0

        # Crossover gives stronger base signal
        if bullish_cross:
            direction = "buy"
            base_strength = 0.55
        elif bearish_cross:
            direction = "sell"
            base_strength = 0.55
        elif bullish_trend and macd_hist > 0 and macd_line > 0:
            direction = "buy"
            base_strength = 0.40
        elif bearish_trend and macd_hist < 0 and macd_line < 0:
            direction = "sell"
            base_strength = 0.40
        else:
            return None

        # MACD confirmation bonus
        macd_confirms = (direction == "buy" and macd_hist > 0) or (direction == "sell" and macd_hist < 0)
        strength = base_strength + (0.15 if macd_confirms else 0.0)

        # AI confirmation
        if ai_conf >= p.AI_CONFIDENCE_THRESHOLD:
            ai_matches = (direction == "buy" and ai_dir == "bullish") or \
                         (direction == "sell" and ai_dir == "bearish")
            ai_contradicts = (direction == "buy" and ai_dir == "bearish") or \
                              (direction == "sell" and ai_dir == "bullish")
            if ai_matches:
                strength += 0.20
            elif ai_contradicts:
                strength -= 0.20

        return {"direction": direction, "strength": strength, "type": "trend"}

    def _mean_reversion_signal(self, ind: Dict, sentiment: Dict) -> Optional[Dict]:
        """RSI extremes + Bollinger Band touches."""
        p = TUNABLE

        rsi = ind["rsi"]
        price = ind["price"]
        bb_upper = ind["bb_upper"]
        bb_lower = ind["bb_lower"]

        ai_dir = sentiment.get("direction", "neutral")
        ai_conf = sentiment.get("confidence", 0.0)

        direction = None
        base_strength = 0.0

        # RSI signals
        rsi_oversold = rsi < p.RSI_OVERSOLD
        rsi_overbought = rsi > p.RSI_OVERBOUGHT

        # Bollinger Band signals
        at_lower_band = price <= bb_lower * 1.005
        at_upper_band = price >= bb_upper * 0.995

        if rsi_oversold or at_lower_band:
            direction = "buy"
            base_strength = 0.45
            if rsi_oversold:
                base_strength += 0.10
            if at_lower_band:
                base_strength += 0.10
        elif rsi_overbought or at_upper_band:
            direction = "sell"
            base_strength = 0.45
            if rsi_overbought:
                base_strength += 0.10
            if at_upper_band:
                base_strength += 0.10
        else:
            return None

        strength = base_strength

        # AI should at minimum not strongly contradict
        if ai_conf >= p.AI_CONFIDENCE_THRESHOLD:
            ai_matches = (direction == "buy" and ai_dir == "bullish") or \
                         (direction == "sell" and ai_dir == "bearish")
            ai_contradicts = (direction == "buy" and ai_dir == "bearish") or \
                              (direction == "sell" and ai_dir == "bullish")
            if ai_matches:
                strength += 0.15
            elif ai_contradicts:
                strength -= 0.25  # mean reversion against strong AI signal = bad idea

        return {"direction": direction, "strength": strength, "type": "mean_reversion"}

    def _combine_signals(self, trend: Optional[Dict], mr: Optional[Dict]) -> Optional[Dict]:
        """Take the stronger signal; if both point same direction, combine."""
        if trend is None and mr is None:
            return None
        if trend is None:
            return mr
        if mr is None:
            return trend

        # Same direction — boost
        if trend["direction"] == mr["direction"]:
            combined_strength = min(1.0, (trend["strength"] + mr["strength"]) * 0.7)
            return {
                "direction": trend["direction"],
                "strength": combined_strength,
                "type": "combined",
            }

        # Opposing — take the stronger one
        if trend["strength"] >= mr["strength"]:
            return trend
        return mr

    def _calc_leverage(self, strength: float) -> int:
        """Map signal strength to leverage 1-20x."""
        p = TUNABLE
        base = 3
        if strength >= 0.85:
            leverage = 15
        elif strength >= 0.75:
            leverage = 10
        elif strength >= 0.65:
            leverage = 7
        else:
            leverage = base
        return min(leverage, 20)

    def _build_reasoning(self, asset: str, ind: Dict, sentiment: Dict, signal: Dict) -> str:
        rsi = ind.get("rsi", 0)
        macd_hist = ind.get("macd_hist", 0)
        ema_f = ind.get("ema_fast", 0)
        ema_s = ind.get("ema_slow", 0)
        ai_dir = sentiment.get("direction", "neutral")
        ai_conf = sentiment.get("confidence", 0)
        ai_reason = sentiment.get("reasoning", "")

        lines = [
            f"Signal type: {signal['type']} | Direction: {signal['direction']} | Strength: {signal['strength']:.2f}",
            f"RSI={rsi:.1f}, MACD hist={macd_hist:.4f}, EMA fast/slow={ema_f:.2f}/{ema_s:.2f}",
            f"AI: {ai_dir} (conf={ai_conf:.2f}) — {ai_reason}",
        ]
        return " | ".join(lines)

    async def run(self):
        logger.info("StrategyAgent: generating signals...")
        signal_count = 0

        for asset in ASSETS:
            ind = self.state.indicators.get(asset, {})
            sentiment = self.state.sentiment.get(asset, {})

            if not ind:
                logger.warning(f"StrategyAgent: no data for {asset}, skipping.")
                await self.state.update_signals(asset, {"direction": "hold", "strength": 0.0, "reasoning": "No data"})
                continue

            trend = self._trend_signal(ind, sentiment)
            mr = self._mean_reversion_signal(ind, sentiment)
            combined = self._combine_signals(trend, mr)

            if combined is None or combined["strength"] < 0.5:
                signal = {
                    "direction": "hold",
                    "strength": combined["strength"] if combined else 0.0,
                    "leverage": 1,
                    "position_pct": 0.0,
                    "reasoning": "Signal strength below threshold.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                leverage = self._calc_leverage(combined["strength"])
                position_pct = min(TUNABLE.MAX_POSITION_PCT, 0.2 + combined["strength"] * 0.3)
                reasoning = self._build_reasoning(asset, ind, sentiment, combined)
                signal = {
                    "direction": combined["direction"],
                    "strength": combined["strength"],
                    "leverage": leverage,
                    "position_pct": position_pct,
                    "signal_type": combined["type"],
                    "reasoning": reasoning,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                signal_count += 1

            await self.state.update_signals(asset, signal)
            logger.info(f"{asset}: signal={signal['direction']} strength={signal['strength']:.2f} leverage={signal['leverage']}x")

        await self.state.update_agent_status("strategy", "ok", f"{signal_count} actionable signals generated")
