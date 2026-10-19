"""siphon-pbx's two voice-session routes on loopback, for the tests that fetch a brief or report.

Scripted per test and recording every request, headers and body included, so a test can check what
the bot sent as well as what it made of the answer. The `pbx` fixture in `conftest.py` serves it.
"""

import asyncio
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from pbx_fixtures import BRIEF_MENU


class FakePbx:
    """siphon-pbx's two voice routes, scripted per test and recording every request."""

    def __init__(self) -> None:
        """Answer with `BRIEF_MENU` and take every report, until a test says otherwise."""
        self.brief_status = 200
        self.brief_body: bytes = BRIEF_MENU
        self.report_status = 204
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
        return app

    @property
    def url(self) -> str:
        assert self.server is not None
        return str(self.server.make_url("")).rstrip("/")
