"""A session service of your own: brief the agent bot per call, with no siphon-pbx anywhere.

The agent bot asks whoever `BOT_PBX_URL` names for a call's brief before it speaks, and tells it how
the call went when it is over. siphon-pbx is one service that answers those two routes. This is
another, in one file, for a deployment that has its own idea of who is calling and what the bot
should know: a CRM, a ticket system, an account lookup.

    caller -> siphon-sip --(routing script)--> this service: POST /calls -> {"token": ...}
                  |
                  +--> siphon-rtp --(ws_uri ...&session=<token>)--> agent bot
                                                                       |
                                     this service <-- GET  /voice/sessions/<token> (the brief)
                                     this service <-- POST /voice/sessions/<token>/report

1. Before the routing script hands a call to the bot, it opens a session here with what it knows
   about the call, and gets a token back.
2. It dials the bot with `&session=<token>` on the `ws_uri`.
3. The bot fetches the brief: instructions for this call, and the caller's facts. Here the
   instructions carry what the stand-in account lookup knows about the caller.
4. When the conversation is over the bot reports `finished` (or the option the caller chose, or
   `failed`), once. A real service would act on it; this one logs it.

Run it beside the bot, and point the bot at it::

    python examples/session_service.py --port 8090
    export BOT_PBX_URL=http://127.0.0.1:8090
    export SIPHON_PBX_BOT_SECRET=<the same --bot-secret>

The secrets come from the environment, `SESSION_SERVICE_BOT_SECRET` and
`SESSION_SERVICE_ROUTER_SECRET`, or the flags. The bot's secret only reads briefs and reports; the
router's only opens sessions, so a bot cannot mint itself a session.

The brief and report shapes are the ones `pipecat_siphon.parse_brief` and `encode_report` read and
write; the README's "Your own session service" section lists them. Transfers are siphon-pbx's: a
brief with no `destinations` and `any_number` false, as here, never offers the bot one.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web
from loguru import logger

ACCOUNTS: dict[str, dict[str, str]] = {
    "+15550142": {
        "name": "Test Caller",
        "plan": "business",
        "open_ticket": "a delivery that has not arrived",
    },
}
"""A stand-in for your own lookup, by the caller's number. Replace it with the real one."""

ORGANISATION = {
    "id": "00000000-0000-4000-8000-0000000000aa",
    "name": "Example Office",
    "time_zone": "Europe/Amsterdam",
}
"""Whose line the calls are on: the bot tells the caller the organisation's name and local time."""

INSTRUCTIONS = "Find out what the caller needs and help them, briefly."
"""What every call is for, before anything the lookup adds."""

DEADLINE_SECONDS = 600
"""How long a session waits for the bot's report before this service forgets it."""

MAX_REQUEST_BYTES = 16 * 1024
"""The largest request either route reads. A call's facts and a report are a few hundred bytes."""


@dataclass
class Session:
    """One call's session: what the router said about it, and whether the bot has reported."""

    call_id: str
    caller: str | None
    called: str | None
    reported: dict[str, Any] | None = None


@dataclass
class SessionService:
    """The two routes the bot calls, and the one the routing script opens a session with."""

    bot_secret: str
    router_secret: str
    sessions: dict[str, Session] = field(default_factory=dict)

    def open(self, call_id: str, caller: str | None, called: str | None) -> str:
        """Open a session for a call the router is about to hand to the bot, and name its token."""
        token = secrets.token_hex(16)
        self.sessions[token] = Session(call_id=call_id, caller=caller, called=called)
        return token

    def brief(self, session: Session) -> dict[str, Any]:
        """Build the brief: the call's instructions, and the caller's facts as `context`."""
        account = ACCOUNTS.get(session.caller or "")
        instructions = INSTRUCTIONS
        if account is not None:
            # Your own data, so it may go in as text. Never the caller's words: they are not yours.
            instructions += (
                f" The caller is {account['name']}, on the {account['plan']} plan, with an open"
                f" ticket about {account['open_ticket']}. Greet them by name."
            )
        context: dict[str, Any] = {"version": 1, "organisation": ORGANISATION}
        if session.called is not None:
            context["called"] = {"number": session.called, "uri": f"tel:{session.called}"}
        context["caller"] = (
            {"number": session.caller, "privacy": "none"}
            if session.caller is not None
            else {"privacy": "id"}
        )
        context["direction"] = "inbound"
        context["destinations"] = []
        context["any_number"] = False
        return {
            "call_id": session.call_id,
            "instructions": instructions,
            "options": [],
            "deadline_secs": DEADLINE_SECONDS,
            "context": context,
            "kind": "call",
        }

    def _session_for(self, request: web.Request) -> Session | None:
        """Find the session a bot's request names, if the bot proved who it is, or None.

        One answer, `404`, for every failure, as siphon-pbx gives: a bot learns nothing about which
        tokens exist. A browser (an `Origin` header) is refused the same way.
        """
        if "Origin" in request.headers:
            return None
        if not _bearer_matches(request, self.bot_secret):
            return None
        return self.sessions.get(request.match_info["token"])

    async def open_route(self, request: web.Request) -> web.StreamResponse:
        """`POST /calls {"call_id", "caller"?, "called"?}` from the router: `{"token"}`."""
        if not _bearer_matches(request, self.router_secret):
            return web.Response(status=404)
        document = await _json_body(request)
        call_id = document.get("call_id") if document is not None else None
        if document is None or not isinstance(call_id, str) or not call_id:
            return web.Response(status=400)
        caller = document.get("caller")
        called = document.get("called")
        token = self.open(
            call_id,
            caller if isinstance(caller, str) else None,
            called if isinstance(called, str) else None,
        )
        logger.info(f"opened a session for call {call_id}")
        return web.json_response({"token": token}, status=201)

    async def brief_route(self, request: web.Request) -> web.StreamResponse:
        """`GET /voice/sessions/{token}` from the bot: the brief."""
        session = self._session_for(request)
        if session is None or session.reported is not None:
            return web.Response(status=404)
        return web.json_response(self.brief(session), headers={"Cache-Control": "no-store"})

    async def report_route(self, request: web.Request) -> web.StreamResponse:
        """`POST /voice/sessions/{token}/report` from the bot: taken once, `204`."""
        session = self._session_for(request)
        if session is None or session.reported is not None:
            return web.Response(status=404)
        document = await _json_body(request)
        if document is None or document.get("outcome") not in ("chose", "finished", "failed"):
            return web.Response(status=404)
        session.reported = dict(document)
        # Act on it here: route the call on, close a ticket, schedule a callback.
        logger.info(f"call {session.call_id} reported {json.dumps(document)}")
        return web.Response(status=204)

    def app(self) -> web.Application:
        """Build the HTTP app: the router's route and the bot's two."""
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.router.add_post("/calls", self.open_route)
        app.router.add_get("/voice/sessions/{token}", self.brief_route)
        app.router.add_post("/voice/sessions/{token}/report", self.report_route)
        return app


def _bearer_matches(request: web.Request, secret: str) -> bool:
    """Whether the request carries `secret` as its bearer, compared in constant time."""
    offered = request.headers.get("Authorization", "")
    return hmac.compare_digest(offered.encode(), f"Bearer {secret}".encode())


async def _json_body(request: web.Request) -> Mapping[str, Any] | None:
    """Read a JSON object body, or None for anything that is not one."""
    try:
        document = json.loads(await request.read())
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def main() -> None:
    """Serve the session routes until stopped."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--bot-secret", default=os.environ.get("SESSION_SERVICE_BOT_SECRET", ""))
    parser.add_argument(
        "--router-secret", default=os.environ.get("SESSION_SERVICE_ROUTER_SECRET", "")
    )
    arguments = parser.parse_args()
    if not arguments.bot_secret or not arguments.router_secret:
        parser.error("set both secrets: --bot-secret and --router-secret, or their variables")
    service = SessionService(bot_secret=arguments.bot_secret, router_secret=arguments.router_secret)
    web.run_app(service.app(), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
