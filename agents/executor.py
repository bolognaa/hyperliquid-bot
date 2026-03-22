import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from core.config import ASSETS, HL_PRIVATE_KEY, HL_TESTNET_URL, POSITION_MONITOR_INTERVAL, MIN_BALANCE_ALERT
from core.logger import get_logger
from core.notifier import notify_trade_opened, notify_trade_closed, notify_low_balance, notify_bot_stopped
from core.state import BotState
from agents.optimizer import OptimizerAgent

logger = get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutorAgent:
    def __init__(self, state: BotState, optimizer: OptimizerAgent):
        self.state = state
        self.optimizer = optimizer
        self._hl_exchange = None
        self._hl_info = None
        self._paper_mode = True
        self._init_hyperliquid()

    def _init_hyperliquid(self):
        """Try to initialise the Hyperliquid SDK. Fall back to paper-only mode."""
        if not HL_PRIVATE_KEY:
            logger.info("ExecutorAgent: HL_PRIVATE_KEY not set — running in pure paper mode.")
            return
        try:
            import eth_account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info

            wallet = eth_account.Account.from_key(HL_PRIVATE_KEY)
            self._hl_info = Info(HL_TESTNET_URL, skip_ws=True)
            self._hl_exchange = Exchange(wallet, HL_TESTNET_URL)
            self._paper_mode = False
            logger.info(f"ExecutorAgent: Hyperliquid testnet connected ({wallet.address[:10]}...).")
        except Exception as e:
            logger.error(f"ExecutorAgent: Hyperliquid init failed ({e}) — falling back to paper mode.")
            self._paper_mode = True

    async def _place_hl_order(self, asset: str, is_long: bool, size: float) -> Optional[Dict]:
        """Place a market order on Hyperliquid testnet."""
        if self._paper_mode or self._hl_exchange is None:
            return {"status": "paper", "message": "Paper mode — no real order placed"}
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._hl_exchange.market_open(asset, is_long, round(size, 6))
            )
            logger.info(f"HL order result for {asset}: {result}")
            return result
        except Exception as e:
            logger.error(f"HL order failed for {asset}: {e}")
            return None

    async def _close_hl_order(self, asset: str) -> Optional[Dict]:
        """Close a position on Hyperliquid testnet."""
        if self._paper_mode or self._hl_exchange is None:
            return {"status": "paper", "message": "Paper mode"}
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._hl_exchange.market_close(asset)
            )
            logger.info(f"HL close result for {asset}: {result}")
            return result
        except Exception as e:
            logger.error(f"HL close failed for {asset}: {e}")
            return None

    async def _open_position(self, signal: Dict):
        """Open a new paper trading position."""
        asset = signal["asset"]
        direction = signal["direction"]
        size = signal["size"]
        leverage = signal["leverage"]
        entry_price = signal["entry_price"]
        margin = signal["margin"]
        stop_loss = signal["stop_loss"]
        take_profit = signal["take_profit"]
        reasoning = signal.get("reasoning", "")

        # Log reasoning BEFORE execution
        logger.info(
            f"\n{'='*60}\n"
            f"TRADE OPENING — {asset} {direction.upper()}\n"
            f"Size: {size:.6f} {asset} @ ${entry_price:,.2f}\n"
            f"Leverage: {leverage}x | Margin: ${margin:.2f} | Notional: ${margin*leverage:.2f}\n"
            f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}\n"
            f"Reasoning: {reasoning}\n"
            f"{'='*60}"
        )

        # Place order on HL testnet (if configured)
        is_long = direction == "long"
        await self._place_hl_order(asset, is_long, size)

        # Track position internally
        position = {
            "id": str(uuid.uuid4())[:8],
            "asset": asset,
            "direction": direction,
            "size": size,
            "leverage": leverage,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "margin": margin,
            "notional": margin * leverage,
            "opened_at": _now(),
            "reasoning": reasoning,
        }
        await self.state.open_position(position)

        # Telegram notification
        await notify_trade_opened(asset, direction, size, leverage, entry_price, margin, reasoning)

        logger.info(f"Position opened: {asset} {direction} | balance=${self.state.balance:.2f}")

    async def _close_position(self, asset: str, current_price: float, reason: str):
        """Close an open paper trading position."""
        position = await self.state.close_position(asset)
        if not position:
            return

        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]
        leverage = position["leverage"]
        margin = position["margin"]

        # P&L calculation
        if direction == "long":
            price_pnl = (current_price - entry_price) * size
        else:
            price_pnl = (entry_price - current_price) * size

        # Close on HL testnet
        await self._close_hl_order(asset)

        # Update paper balance
        await self.state.update_balance(price_pnl)
        new_balance = self.state.balance

        # Record trade
        trade = {
            "id": position["id"],
            "asset": asset,
            "direction": direction,
            "size": size,
            "leverage": leverage,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": price_pnl,
            "pnl_pct": (price_pnl / margin) * 100 if margin > 0 else 0,
            "balance_after": new_balance,
            "opened_at": position["opened_at"],
            "closed_at": _now(),
            "reason": reason,
            "reasoning": position.get("reasoning", ""),
        }
        await self.state.add_trade(trade)

        logger.info(
            f"\n{'='*60}\n"
            f"TRADE CLOSED — {asset} {direction.upper()}\n"
            f"Exit: ${current_price:,.2f} | Entry: ${entry_price:,.2f}\n"
            f"P&L: ${price_pnl:+.2f} | Reason: {reason}\n"
            f"Balance: ${new_balance:.2f}\n"
            f"{'='*60}"
        )

        # Telegram notification
        await notify_trade_closed(asset, direction, price_pnl, new_balance, reason)

        # Low balance alert (only send once per threshold crossing)
        if new_balance < MIN_BALANCE_ALERT and not self.state.balance_alert_sent:
            self.state.balance_alert_sent = True
            await notify_low_balance(new_balance)

        # Check if balance is above threshold again (reset alert)
        if new_balance >= MIN_BALANCE_ALERT:
            self.state.balance_alert_sent = False

        # Kill switch
        if new_balance <= 0:
            logger.critical("Balance hit zero — stopping bot!")
            self.state.running = False
            await notify_bot_stopped("Balance reached $0. Bot halted.")

        # Trigger optimizer check (only acts every N trades)
        await self.optimizer.on_trade_closed()

    # ------------------------------------------------------------------
    # Main execution run (called each trading cycle)
    # ------------------------------------------------------------------

    async def run(self):
        logger.info("ExecutorAgent: processing approved signals...")
        executed = 0

        for asset in ASSETS:
            signal = self.state.signals.get(asset, {})
            direction = signal.get("direction", "hold")

            # Skip HOLD or unapproved signals
            if direction == "hold" or "approved_at" not in signal:
                continue

            # If existing position is opposite direction, close it first
            if asset in self.state.positions:
                existing = self.state.positions[asset]
                existing_dir = existing["direction"]
                new_dir = "long" if direction == "buy" else "short"
                if existing_dir != new_dir:
                    indicators = self.state.indicators.get(asset, {})
                    current_price = indicators.get("price")
                    if current_price:
                        logger.info(f"Closing {existing_dir} {asset} to flip to {new_dir}")
                        await self._close_position(asset, current_price, "signal_flip")
                else:
                    continue  # same direction, skip

            await self._open_position(signal)
            executed += 1

            # Clear the signal so we don't re-execute next cycle
            await self.state.update_signals(asset, {
                "direction": "hold",
                "strength": 0.0,
                "reasoning": f"Executed at {_now()}",
            })

        await self.state.update_agent_status(
            "executor", "ok", f"{executed} trades executed | balance=${self.state.balance:.2f}"
        )

    # ------------------------------------------------------------------
    # Position monitoring loop (runs independently)
    # ------------------------------------------------------------------

    async def monitor_positions(self):
        """Check SL/TP for all open positions every N seconds."""
        while self.state.running:
            await asyncio.sleep(POSITION_MONITOR_INTERVAL)
            if not self.state.positions:
                continue

            for asset in list(self.state.positions.keys()):
                position = self.state.positions.get(asset)
                if not position:
                    continue

                indicators = self.state.indicators.get(asset, {})
                current_price = indicators.get("price")
                if not current_price:
                    continue

                direction = position["direction"]
                sl = position["stop_loss"]
                tp = position["take_profit"]

                # Check stop-loss
                sl_hit = (direction == "long" and current_price <= sl) or \
                         (direction == "short" and current_price >= sl)

                # Check take-profit
                tp_hit = (direction == "long" and current_price >= tp) or \
                         (direction == "short" and current_price <= tp)

                if sl_hit:
                    logger.info(f"STOP-LOSS triggered for {asset} at ${current_price:,.2f} (SL=${sl:,.2f})")
                    await self._close_position(asset, current_price, "stop_loss")
                elif tp_hit:
                    logger.info(f"TAKE-PROFIT triggered for {asset} at ${current_price:,.2f} (TP=${tp:,.2f})")
                    await self._close_position(asset, current_price, "take_profit")

        logger.info("ExecutorAgent: position monitor stopped.")
