import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiofiles

from core.config import STARTING_BALANCE, ASSETS, STATE_FILE, STATE_SAVE_INTERVAL
from core.logger import get_logger

logger = get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BotState:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.running: bool = True

        # Paper trading
        self.balance: float = STARTING_BALANCE
        self.starting_balance: float = STARTING_BALANCE
        self.total_pnl: float = 0.0
        self.balance_alert_sent: bool = False

        # Positions: {asset: position_dict}
        self.positions: Dict[str, Dict] = {}

        # Trade history (all closed trades)
        self.trades: List[Dict] = []

        # Market data per asset
        self.indicators: Dict[str, Dict] = {a: {} for a in ASSETS}
        self.ohlc: Dict[str, Any] = {a: None for a in ASSETS}

        # Sentiment per asset
        self.sentiment: Dict[str, Dict] = {a: {} for a in ASSETS}

        # Signals per asset
        self.signals: Dict[str, Dict] = {a: {} for a in ASSETS}

        # Agent health
        self.agent_status: Dict[str, Dict] = {
            name: {"last_run": None, "status": "init", "message": ""}
            for name in ["market_data", "sentiment", "strategy", "risk_manager", "executor", "optimizer"]
        }

        # Optimizer history
        self.optimizer_history: List[Dict] = []

        # Trade counter for optimizer
        self.trades_since_last_optimization: int = 0

    # ------------------------------------------------------------------
    # Thread-safe accessors
    # ------------------------------------------------------------------

    async def update_balance(self, delta: float):
        async with self._lock:
            self.balance += delta
            self.total_pnl += delta

    async def open_position(self, position: Dict):
        async with self._lock:
            self.positions[position["asset"]] = position

    async def close_position(self, asset: str) -> Optional[Dict]:
        async with self._lock:
            return self.positions.pop(asset, None)

    async def add_trade(self, trade: Dict):
        async with self._lock:
            self.trades.append(trade)
            self.trades_since_last_optimization += 1

    async def update_indicators(self, asset: str, data: Dict):
        async with self._lock:
            self.indicators[asset] = data

    async def update_ohlc(self, asset: str, df):
        async with self._lock:
            self.ohlc[asset] = df

    async def update_sentiment(self, asset: str, data: Dict):
        async with self._lock:
            self.sentiment[asset] = data

    async def update_signals(self, asset: str, data: Dict):
        async with self._lock:
            self.signals[asset] = data

    async def update_agent_status(self, name: str, status: str, message: str = ""):
        async with self._lock:
            self.agent_status[name] = {
                "last_run": _now(),
                "status": status,
                "message": message,
            }

    async def add_optimizer_entry(self, entry: Dict):
        async with self._lock:
            self.optimizer_history.append(entry)

    async def update_optimizer_entry(self, idx: int, updates: Dict):
        async with self._lock:
            if 0 <= idx < len(self.optimizer_history):
                self.optimizer_history[idx].update(updates)

    async def reset_trade_counter(self):
        async with self._lock:
            self.trades_since_last_optimization = 0

    # ------------------------------------------------------------------
    # Snapshot (safe read for dashboard / serialization)
    # ------------------------------------------------------------------

    async def snapshot(self) -> Dict:
        async with self._lock:
            return {
                "balance": self.balance,
                "starting_balance": self.starting_balance,
                "total_pnl": self.total_pnl,
                "positions": dict(self.positions),
                "trades": list(self.trades[-50:]),
                "indicators": dict(self.indicators),
                "sentiment": dict(self.sentiment),
                "signals": dict(self.signals),
                "agent_status": dict(self.agent_status),
                "optimizer_history": list(self.optimizer_history),
                "running": self.running,
                "trades_since_last_optimization": self.trades_since_last_optimization,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save(self):
        try:
            data = await self.snapshot()
            # ohlc dataframes are not JSON-serialisable — skip
            data.pop("ohlc", None)
            async with aiofiles.open(STATE_FILE, "w") as f:
                await f.write(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"State save failed: {e}")

    async def load(self):
        if not os.path.exists(STATE_FILE):
            logger.info("No persisted state found — starting fresh.")
            return
        try:
            async with aiofiles.open(STATE_FILE, "r") as f:
                data = json.loads(await f.read())
            async with self._lock:
                self.balance = data.get("balance", STARTING_BALANCE)
                self.starting_balance = data.get("starting_balance", STARTING_BALANCE)
                self.total_pnl = data.get("total_pnl", 0.0)
                self.positions = data.get("positions", {})
                self.trades = data.get("trades", [])
                self.sentiment = data.get("sentiment", {a: {} for a in ASSETS})
                self.signals = data.get("signals", {a: {} for a in ASSETS})
                self.agent_status = data.get("agent_status", self.agent_status)
                self.optimizer_history = data.get("optimizer_history", [])
                self.trades_since_last_optimization = data.get("trades_since_last_optimization", 0)
            logger.info(f"State restored — balance: ${self.balance:.2f}, trades: {len(self.trades)}")
        except Exception as e:
            logger.error(f"State load failed: {e}")

    async def auto_save(self):
        while self.running:
            await asyncio.sleep(STATE_SAVE_INTERVAL)
            await self.save()
