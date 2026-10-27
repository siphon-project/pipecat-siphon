"""siphon-pbx's voice-session routes on loopback, for the tests that fetch, report or transfer.

Scripted per test and recording every request, headers and body included, so a test can check what
the bot sent as well as what it made of the answer. The `pbx` fixture in `conftest.py` serves it.
"""

import asyncio
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from pbx_fixtures import BRIEF_MENU, TRANSFER_COMPLETED


class FakePbx:
    """siphon-pbx's voice routes, scripted per test and recording every request."""

    def __init__(self) -> None:
        """Answer with `BRIEF_MENU` and take every report, until a test says otherwise."""
        self.brief_status = 200
        self.brief_body: bytes = BRIEF_MENU
        self.report_status = 204
        self.transfer_status = 200
        self.transfer_body: bytes = TRANSFER_COMPLETED
        self.delay = 0.0
        self.requests: list[dict[str, Any]] = []
        self.server: TestServer | None = None

    async def _brief(self, request: web.Request) -> web.StreamResponse:
        await self._record(request)
        await asyncio.sleep(self.delay)
        return web.Response(status=self.brief_status, body=self.brief_body)

    async def _report(self, request: web.Request) -> web.StreamResponse:
        await self._record(request)
        await asyncio.sleep(self.delay)
        return web.Response(status=self.report_status)

    async def _transfer(self, request: web.Request) -> web.StreamResponse:
        await self._record(request)
        await asyncio.sleep(self.delay)
        return web.Response(
            status=self.transfer_status,
            body=self.transfer_body,
            content_type="application/json",
        )

    async def _record(self, request: web.Request) -> None:
        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "body": await request.read(),
            }
        )

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/voice/sessions/{token}", self._brief)
        app.router.add_post("/voice/sessions/{token}/report", self._report)
        app.router.add_post("/voice/sessions/{token}/transfer", self._transfer)
        return app

    @property
    def url(self) -> str:
        assert self.server is not None
        return str(self.server.make_url("")).rstrip("/")
