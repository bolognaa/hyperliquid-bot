import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.config import OPTIMIZER_TRADE_THRESHOLD, TUNABLE
from core.logger import get_logger
from core.notifier import notify_optimizer_change
from core.state import BotState

logger = get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"win_rate": 0.0, "avg_pnl": 0.0, "sharpe": 0.0, "trade_count": 0}

    pnls = [t.get("pnl", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0

    import numpy as np
    if len(pnls) > 1:
        std = float(np.std(pnls))
        sharpe = (avg_pnl / std) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
        "trade_count": len(pnls),
    }


def _composite_score(metrics: Dict) -> float:
    return (metrics["win_rate"] * 0.4) + (metrics["avg_pnl"] * 0.4) + (metrics["sharpe"] * 0.2)


class OptimizerAgent:
    def __init__(self, state: BotState):
        self.state = state
        self._pending_change: Optional[Dict] = None
        self._baseline_score: float = 0.0

    def _choose_parameter(self) -> Tuple[str, float, float]:
        key = random.choice(list(TUNABLE.BOUNDS.keys()))
        lo, hi = TUNABLE.BOUNDS[key]
        current = TUNABLE.get(key)

        step = (hi - lo) * random.uniform(0.10, 0.20)
        direction = random.choice([-1, 1])
        new_value = current + direction * step
        new_value = max(lo, min(hi, new_value))

        if isinstance(current, int):
            new_value = int(round(new_value))

        return key, current, new_value

    async def _run_experiment(self):
        trades = self.state.trades
        baseline_trades = trades[max(0, len(trades) - OPTIMIZER_TRADE_THRESHOLD):]
        baseline_metrics = _compute_metrics(baseline_trades)
        self._baseline_score = _composite_score(baseline_metrics)

        param, old_val, new_val = self._choose_parameter()

        entry = {
            "timestamp": _now(),
            "parameter": param,
            "old_value": old_val,
            "new_value": new_val,
            "baseline_metrics": baseline_metrics,
            "result_metrics": None,
            "status": "pending",
        }
        await self.state.add_optimizer_entry(entry)
        idx = len(self.state.optimizer_history) - 1

        TUNABLE.set(param, new_val)
        self._pending_change = {
            "param": param,
            "old_value": old_val,
            "new_value": new_val,
            "entry_idx": idx,
            "experiment_start_trade_count": len(self.state.trades),
        }

        logger.info(f"Optimizer: experimenting with {param}: {old_val} -> {new_val} (baseline score={self._baseline_score:.4f})")
        await notify_optimizer_change(
            param, old_val, new_val,
            f"Testing new value. Baseline score: {self._baseline_score:.4f} "
            f"(win_rate={baseline_metrics['win_rate']:.1%}, avg_pnl=${baseline_metrics['avg_pnl']:.3f})"
        )
        await self.state.reset_trade_counter()

    async def _evaluate_experiment(self):
        if not self._pending_change:
            return

        change = self._pending_change
        param = change["param"]
        old_val = change["old_value"]
        new_val = change["new_value"]
        idx = change["entry_idx"]

        start_count = change["experiment_start_trade_count"]
        new_trades = self.state.trades[start_count:]
        new_metrics = _compute_metrics(new_trades)
        new_score = _composite_score(new_metrics)

        if new_score >= self._baseline_score:
            status = "kept"
            decision = f"KEPT (new={new_score:.4f} vs baseline={self._baseline_score:.4f})"
            logger.info(f"Optimizer: {param} change KEPT — {decision}")
        else:
            TUNABLE.set(param, old_val)
            status = "reverted"
            decision = f"REVERTED (new={new_score:.4f} vs baseline={self._baseline_score:.4f})"
            logger.info(f"Optimizer: {param} change REVERTED — {decision}")

        await self.state.update_optimizer_entry(idx, {
            "result_metrics": new_metrics,
            "status": status,
            "decision": decision,
            "evaluated_at": _now(),
        })

        await notify_optimizer_change(
            param,
            f"{old_val} -> {new_val}",
            "KEPT" if status == "kept" else f"REVERTED to {old_val}",
            decision,
        )

        self._pending_change = None
        await self.state.reset_trade_counter()

    async def on_trade_closed(self):
        """Called by executor after every trade close. Only acts when threshold is hit."""
        trades_since = self.state.trades_since_last_optimization

        if trades_since < OPTIMIZER_TRADE_THRESHOLD:
            return

        if len(self.state.trades) < OPTIMIZER_TRADE_THRESHOLD:
            return

        if self._pending_change is not None:
            await self._evaluate_experiment()
            await self.state.update_agent_status("optimizer", "ok", "Experiment evaluated.")
        else:
            await self._run_experiment()
            await self.state.update_agent_status("optimizer", "ok", "New experiment started.")
