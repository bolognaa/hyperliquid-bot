import asyncio
import signal
import sys
import time

from core.config import MARKET_DATA_INTERVAL
from core.logger import setup_logger
from core.notifier import discover_chat_id, close_session
from core.state import BotState
from agents.market_data import MarketDataAgent
from agents.sentiment import SentimentAgent
from agents.strategy import StrategyAgent
from agents.risk_manager import RiskManagerAgent
from agents.executor import ExecutorAgent
from agents.optimizer import OptimizerAgent
from dashboard.app import run_dashboard

logger = setup_logger()

SENTIMENT_INTERVAL = 900  # 15 minutes between AI calls


async def trading_loop(
    state: BotState,
    market_data: MarketDataAgent,
    sentiment: SentimentAgent,
    strategy: StrategyAgent,
    risk_manager: RiskManagerAgent,
    executor: ExecutorAgent,
):
    """Main sequential pipeline: data -> sentiment -> strategy -> risk -> execution."""
    logger.info("Trading loop started.")
    last_sentiment_time = 0.0

    while state.running:
        try:
            logger.info("=" * 50 + " CYCLE START " + "=" * 50)

            await market_data.run()

            # Only query AI every 15 minutes
            now = time.time()
            if now - last_sentiment_time >= SENTIMENT_INTERVAL:
                await sentiment.run()
                last_sentiment_time = now
            else:
                remaining = int(SENTIMENT_INTERVAL - (now - last_sentiment_time))
                logger.info(f"SentimentAgent: skipping (next AI call in {remaining}s)")

            await strategy.run()
            await risk_manager.run()
            await executor.run()

            logger.info(
                f"CYCLE COMPLETE — Balance: ${state.balance:.2f} | "
                f"Open positions: {len(state.positions)} | "
                f"Total trades: {len(state.trades)}"
            )

            await asyncio.sleep(MARKET_DATA_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)
            await asyncio.sleep(10)

    logger.info("Trading loop stopped.")


async def main():
    logger.info("=" * 70)
    logger.info("  Hyperliquid Multi-Agent Trading Bot — Starting Up")
    logger.info("=" * 70)

    state = BotState()
    await state.load()

    asyncio.create_task(discover_chat_id())

    # Initialise agents
    market_data = MarketDataAgent(state)
    sentiment = SentimentAgent(state)
    strategy = StrategyAgent(state)
    risk_manager = RiskManagerAgent(state)
    optimizer = OptimizerAgent(state)
    executor = ExecutorAgent(state, optimizer)

    # Graceful shutdown
    loop = asyncio.get_running_loop()

    def _shutdown(sig_name):
        logger.info(f"Received {sig_name} — shutting down...")
        state.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig.name)

    tasks = [
        asyncio.create_task(
            trading_loop(state, market_data, sentiment, strategy, risk_manager, executor),
            name="trading-loop",
        ),
        asyncio.create_task(executor.monitor_positions(), name="position-monitor"),
        asyncio.create_task(state.auto_save(), name="state-autosave"),
        asyncio.create_task(run_dashboard(state), name="dashboard"),
    ]

    logger.info("All agents started. Dashboard: http://0.0.0.0:3456")
    logger.info(f"Sentiment AI calls every {SENTIMENT_INTERVAL}s. Optimizer triggers after every 10 closed trades.")

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await state.save()
        await market_data.close()
        await sentiment.close()
        await close_session()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
