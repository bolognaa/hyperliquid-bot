import json
import os
from datetime import datetime, timezone

import aiohttp_jinja2
import jinja2
from aiohttp import web

from core.config import DASHBOARD_HOST, DASHBOARD_PORT
from core.logger import get_logger
from core.state import BotState

logger = get_logger()


async def _index(request: web.Request) -> web.Response:
    state: BotState = request.app["state"]
    snap = await state.snapshot()
    return aiohttp_jinja2.render_template("index.html", request, {"state": snap})


async def _api_state(request: web.Request) -> web.Response:
    state: BotState = request.app["state"]
    snap = await state.snapshot()
    return web.json_response(snap, dumps=lambda o: json.dumps(o, default=str))


def _create_app(state: BotState) -> web.Application:
    app = web.Application()
    app["state"] = state

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(template_dir))

    app.router.add_get("/", _index)
    app.router.add_get("/api/state", _api_state)

    return app


async def run_dashboard(state: BotState):
    app = _create_app(state)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, DASHBOARD_HOST, DASHBOARD_PORT)
    await site.start()
    logger.info(f"Dashboard running at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    # Keep running until bot stops
    while state.running:
        import asyncio
        await asyncio.sleep(5)

    await runner.cleanup()
