import asyncio
import os
from typing import Optional

import aiohttp

from core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID_FILE
from core.logger import get_logger

logger = get_logger()

_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
_chat_id: Optional[str] = None
_session: Optional[aiohttp.ClientSession] = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def discover_chat_id() -> Optional[str]:
    """Poll getUpdates to find the most recent chat ID and persist it."""
    global _chat_id
    if _chat_id:
        return _chat_id

    # Try loading from file first
    if os.path.exists(TELEGRAM_CHAT_ID_FILE):
        with open(TELEGRAM_CHAT_ID_FILE) as f:
            stored = f.read().strip()
        if stored:
            _chat_id = stored
            logger.info(f"Telegram chat_id loaded from file: {_chat_id}")
            return _chat_id

    # Also check env variable
    env_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if env_id:
        _chat_id = env_id
        with open(TELEGRAM_CHAT_ID_FILE, "w") as f:
            f.write(_chat_id)
        logger.info(f"Telegram chat_id from env: {_chat_id}")
        return _chat_id

    # Poll updates
    try:
        session = await _get_session()
        async with session.get(f"{_BASE}/getUpdates", params={"limit": 10, "timeout": 5}) as resp:
            data = await resp.json()
        updates = data.get("result", [])
        if updates:
            msg = updates[-1].get("message") or updates[-1].get("channel_post", {})
            chat = msg.get("chat", {})
            cid = str(chat.get("id", ""))
            if cid:
                _chat_id = cid
                with open(TELEGRAM_CHAT_ID_FILE, "w") as f:
                    f.write(_chat_id)
                logger.info(f"Telegram chat_id discovered: {_chat_id}")
                return _chat_id
        logger.warning("Telegram: no updates found — send a message to the bot to register your chat_id.")
    except Exception as e:
        logger.error(f"Telegram discover_chat_id error: {e}")
    return None


async def send(text: str):
    """Send a message to the configured Telegram chat."""
    chat_id = _chat_id or await discover_chat_id()
    if not chat_id:
        logger.warning("Telegram: chat_id unknown, message not sent.")
        return
    try:
        session = await _get_session()
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        async with session.post(f"{_BASE}/sendMessage", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Telegram send error {resp.status}: {body}")
    except Exception as e:
        logger.error(f"Telegram send exception: {e}")


# ------------------------------------------------------------------
# Formatted notification helpers
# ------------------------------------------------------------------

async def notify_trade_opened(asset: str, direction: str, size: float, leverage: int,
                               entry_price: float, margin: float, reasoning: str):
    arrow = "🟢" if direction == "long" else "🔴"
    msg = (
        f"{arrow} <b>TRADE OPENED</b>\n"
        f"Asset: <b>{asset}</b> | Direction: <b>{direction.upper()}</b>\n"
        f"Size: {size:.6f} {asset} | Leverage: {leverage}x\n"
        f"Entry: ${entry_price:,.2f} | Margin: ${margin:.2f}\n\n"
        f"<i>Reasoning:</i>\n{reasoning}"
    )
    await send(msg)


async def notify_trade_closed(asset: str, direction: str, pnl: float,
                               balance_after: float, reason: str):
    emoji = "✅" if pnl >= 0 else "❌"
    sign = "+" if pnl >= 0 else ""
    msg = (
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"Asset: <b>{asset}</b> | Direction: <b>{direction.upper()}</b>\n"
        f"P&L: <b>{sign}${pnl:.2f}</b> | Reason: {reason}\n"
        f"Balance after: <b>${balance_after:.2f}</b>"
    )
    await send(msg)


async def notify_low_balance(balance: float):
    msg = (
        f"⚠️ <b>LOW BALANCE ALERT</b>\n"
        f"Balance dropped to <b>${balance:.2f}</b> (below $25 threshold).\n"
        f"Bot continues trading but monitor closely."
    )
    await send(msg)


async def notify_optimizer_change(parameter: str, old_value, new_value, reasoning: str):
    msg = (
        f"🔧 <b>OPTIMIZER CHANGE</b>\n"
        f"Parameter: <b>{parameter}</b>\n"
        f"Old: <b>{old_value}</b> → New: <b>{new_value}</b>\n"
        f"<i>{reasoning}</i>"
    )
    await send(msg)


async def notify_bot_stopped(reason: str):
    msg = f"🛑 <b>BOT STOPPED</b>\n{reason}"
    await send(msg)


async def close_session():
    global _session
    if _session and not _session.closed:
        await _session.close()
