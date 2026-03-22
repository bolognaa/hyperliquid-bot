import json
from datetime import datetime, timezone
from typing import Dict, Optional

import aiohttp

from core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, AI_MODEL, ASSETS
from core.logger import get_logger
from core.state import BotState

logger = get_logger()

_SYSTEM_PROMPT = (
    "You are a professional crypto trading analyst. Analyze market data and provide "
    "a directional bias. Always respond with valid JSON only — no extra text."
)

_USER_TEMPLATE = """Analyze the following market data for {asset}/USD and provide a directional bias.

Current Price: ${price:,.2f}
24h Change: {price_change_pct:+.2f}%

Indicators:
- RSI(14): {rsi:.1f}
- EMA(9): ${ema_fast:,.2f}  EMA(21): ${ema_slow:,.2f}
- MACD Line: {macd_line:.4f}  Signal: {signal_line:.4f}  Histogram: {macd_hist:.4f}
- Bollinger Upper: ${bb_upper:,.2f}  Mid: ${bb_mid:,.2f}  Lower: ${bb_lower:,.2f}
- ATR(14): ${atr:,.2f}

Recent closing prices (oldest→newest): {recent_closes}

Respond ONLY with this exact JSON structure:
{{
  "direction": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentences explaining your analysis"
}}"""


class SentimentAgent:
    def __init__(self, state: BotState):
        self.state = state
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _query_ai(self, asset: str, indicators: Dict) -> Optional[Dict]:
        if not OPENROUTER_API_KEY:
            logger.warning("SentimentAgent: OPENROUTER_API_KEY not set, using neutral sentiment.")
            return {"direction": "neutral", "confidence": 0.5, "reasoning": "AI not configured."}

        prompt = _USER_TEMPLATE.format(
            asset=asset,
            price=indicators.get("price", 0),
            price_change_pct=indicators.get("price_change_pct", 0),
            rsi=indicators.get("rsi", 50),
            ema_fast=indicators.get("ema_fast", 0),
            ema_slow=indicators.get("ema_slow", 0),
            macd_line=indicators.get("macd_line", 0),
            signal_line=indicators.get("signal_line", 0),
            macd_hist=indicators.get("macd_hist", 0),
            bb_upper=indicators.get("bb_upper", 0),
            bb_mid=indicators.get("bb_mid", 0),
            bb_lower=indicators.get("bb_lower", 0),
            atr=indicators.get("atr", 0),
            recent_closes=indicators.get("recent_closes", []),
        )

        payload = {
            "model": AI_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hyperliquid-bot",
            "X-Title": "HyperliquidBot",
        }

        session = await self._get_session()
        try:
            async with session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                http_status = resp.status
                raw_text = await resp.text()

            # Always log the raw response so we can see exactly what came back
            logger.debug(f"OpenRouter raw response for {asset} (HTTP {http_status}): {raw_text}")

            # Parse JSON
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                logger.error(f"OpenRouter returned non-JSON for {asset} (HTTP {http_status}): {raw_text[:300]}")
                return None

            # Top-level API error
            if "error" in data:
                logger.error(f"OpenRouter API error for {asset}: {data['error']} — skipping (no retry)")
                return None

            # Guard against missing/malformed choices
            choices = data.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                logger.error(f"OpenRouter no choices for {asset}. Full response: {raw_text[:500]}")
                return None

            choice = choices[0]
            # Some models put content directly in choice, others nest under message
            message = choice.get("message") or choice.get("delta") or {}
            content = message.get("content") if isinstance(message, dict) else None

            # Fallback: some models put text at choice level
            if content is None:
                content = choice.get("text") or choice.get("content")

            finish_reason = choice.get("finish_reason", "unknown")

            if content is None:
                logger.error(
                    f"OpenRouter null content for {asset} "
                    f"(finish_reason={finish_reason}). Full response: {raw_text[:500]}"
                )
                return None

            content = str(content).strip()
            if not content:
                logger.error(f"OpenRouter empty content for {asset} (finish_reason={finish_reason})")
                return None

            # Strip markdown code fences if present
            if "```" in content:
                parts = content.split("```")
                # Take the first non-empty block after a fence
                for part in parts[1::2]:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part:
                        content = part
                        break

            result = json.loads(content)

            # Validate and normalise
            if result.get("direction") not in ("bullish", "bearish", "neutral"):
                result["direction"] = "neutral"
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            result["reasoning"] = str(result.get("reasoning", ""))

            return result

        except json.JSONDecodeError as e:
            logger.error(f"SentimentAgent JSON parse error for {asset}: {e} — content was: {repr(content)[:300]}")
            return None
        except Exception as e:
            logger.error(f"SentimentAgent query failed for {asset}: {e} — skipping (no retry)", exc_info=True)
            return None

    async def run(self):
        logger.info("SentimentAgent: querying AI for all assets...")
        success_count = 0

        for asset in ASSETS:
            indicators = self.state.indicators.get(asset, {})
            if not indicators:
                logger.warning(f"SentimentAgent: no indicator data for {asset}, skipping.")
                continue

            result = await self._query_ai(asset, indicators)
            if result is None:
                # Fallback to neutral
                result = {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "reasoning": "AI query failed.",
                }

            result["updated_at"] = datetime.now(timezone.utc).isoformat()
            await self.state.update_sentiment(asset, result)

            logger.info(
                f"{asset} sentiment: {result['direction']} "
                f"(confidence={result['confidence']:.2f}) — {result['reasoning'][:80]}..."
            )
            success_count += 1

        await self.state.update_agent_status(
            "sentiment", "ok", f"{success_count}/{len(ASSETS)} assets analyzed"
        )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
