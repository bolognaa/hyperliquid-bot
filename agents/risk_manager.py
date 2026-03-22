from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from core.config import (
    ASSETS, TUNABLE, MAX_LEVERAGE, MAX_SAME_DIRECTION,
    STARTING_BALANCE, MAX_DRAWDOWN_PCT,
)
from core.logger import get_logger
from core.state import BotState

logger = get_logger()


class RiskManagerAgent:
    def __init__(self, state: BotState):
        self.state = state

    def _check_drawdown(self, balance: float) -> bool:
        """Return True if max drawdown has been hit."""
        min_allowed = STARTING_BALANCE * (1 - MAX_DRAWDOWN_PCT)
        return balance <= min_allowed

    def _count_direction(self, direction: str) -> int:
        """Count open positions in a given direction."""
        positions = self.state.positions
        return sum(1 for p in positions.values() if p.get("direction") == direction)

    def _calc_position_params(
        self,
        balance: float,
        position_pct: float,
        leverage: int,
        entry_price: float,
        is_long: bool,
    ) -> Tuple[float, float, float, float]:
        """
        Returns: (size_in_coins, stop_loss_price, take_profit_price, margin)

        Stop-loss is sized so that max loss = balance * STOP_LOSS_BALANCE_PCT
        """
        p = TUNABLE
        margin = balance * position_pct          # dollars of collateral
        notional = margin * leverage             # total USD exposure
        size = notional / entry_price            # coins

        # max_loss = balance * stop_loss_pct
        max_loss = balance * p.STOP_LOSS_BALANCE_PCT
        # price must move by (max_loss / notional) to trigger SL
        sl_pct = max_loss / notional if notional > 0 else 0.01
        tp_pct = sl_pct * p.TAKE_PROFIT_MULTIPLIER

        if is_long:
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        return size, sl_price, tp_price, margin

    def _approve_signal(
        self,
        asset: str,
        signal: Dict,
        balance: float,
        indicators: Dict,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Returns: (approved, reason, adjusted_signal)
        """
        p = TUNABLE
        direction = signal.get("direction", "hold")

        if direction == "hold":
            return False, "Signal is HOLD", None

        # Already have an open position for this asset?
        if asset in self.state.positions:
            existing = self.state.positions[asset]
            if existing["direction"] == ("long" if direction == "buy" else "short"):
                return False, f"Already have a {existing['direction']} position in {asset}", None

        # Max drawdown
        if self._check_drawdown(balance):
            return False, f"Max drawdown reached (balance=${balance:.2f})", None

        # Balance too low to trade
        if balance < 5.0:
            return False, f"Balance too low to trade (${balance:.2f})", None

        # Same-direction cap
        pos_direction = "long" if direction == "buy" else "short"
        count = self._count_direction(pos_direction)
        if count >= MAX_SAME_DIRECTION:
            return False, f"Already have {count} {pos_direction} positions (max={MAX_SAME_DIRECTION})", None

        # Signal strength
        if signal.get("strength", 0) < 0.5:
            return False, f"Signal strength too low ({signal.get('strength', 0):.2f})", None

        # Validate/clamp leverage
        requested_leverage = signal.get("leverage", 1)
        leverage = max(1, min(requested_leverage, MAX_LEVERAGE))

        # Validate/clamp position_pct
        requested_pct = signal.get("position_pct", 0.2)
        position_pct = max(0.05, min(requested_pct, p.MAX_POSITION_PCT))

        entry_price = indicators.get("price")
        if not entry_price or entry_price <= 0:
            return False, f"Invalid entry price for {asset}", None

        is_long = direction == "buy"
        size, sl_price, tp_price, margin = self._calc_position_params(
            balance, position_pct, leverage, entry_price, is_long
        )

        if size <= 0 or margin <= 0:
            return False, "Calculated size/margin is zero or negative", None

        adjusted = dict(signal)
        adjusted.update({
            "asset": asset,
            "direction": "long" if is_long else "short",
            "size": size,
            "leverage": leverage,
            "position_pct": position_pct,
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "margin": margin,
            "notional": margin * leverage,
            "approved_at": datetime.now(timezone.utc).isoformat(),
        })

        return True, "Approved", adjusted

    async def run(self):
        logger.info("RiskManagerAgent: evaluating signals...")
        approved_count = 0
        rejected_count = 0
        balance = self.state.balance

        for asset in ASSETS:
            signal = self.state.signals.get(asset, {})
            if not signal or signal.get("direction") == "hold":
                continue

            indicators = self.state.indicators.get(asset, {})
            approved, reason, adjusted_signal = self._approve_signal(asset, signal, balance, indicators)

            if approved and adjusted_signal:
                # Write back the approved+adjusted signal
                await self.state.update_signals(asset, adjusted_signal)
                approved_count += 1
                logger.info(f"{asset}: APPROVED — {reason} | leverage={adjusted_signal['leverage']}x margin=${adjusted_signal['margin']:.2f}")
            else:
                # Mark as HOLD with rejection reason
                await self.state.update_signals(asset, {
                    "direction": "hold",
                    "strength": signal.get("strength", 0),
                    "reasoning": f"REJECTED: {reason}",
                    "generated_at": signal.get("generated_at"),
                })
                rejected_count += 1
                logger.info(f"{asset}: REJECTED — {reason}")

        await self.state.update_agent_status(
            "risk_manager", "ok",
            f"approved={approved_count} rejected={rejected_count}"
        )
