"""The example session service, read by the package's own client and the agent bot's own prompt.

The README documents the brief and report routes as a contract anyone can serve. This keeps the
example that serves them honest against the code that reads them: a field the parser stops
accepting, or a route the client stops calling, fails here rather than in somebody's deployment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from pipecat_siphon import Chose, Finished, VoiceSession, VoiceSessionClient

BOT_SECRET = "bot-secret-for-tests-only"
ROUTER_SECRET = "router-secret-for-tests-only"
CALL_ID = "b2b-00000000000000c1@example.invalid"


@pytest.fixture
async def service() -> AsyncIterator[tuple[Any, str]]:
    """Serve the example on loopback, and name its base URL."""
    from session_service import SessionService

    example = SessionService(bot_secret=BOT_SECRET, router_secret=ROUTER_SECRET)
    server = TestServer(example.app(), host="127.0.0.1")
    await server.start_server()
    try:
        yield example, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


async def _open(base_url: str, body: dict[str, Any], secret: str = ROUTER_SECRET) -> Any:
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"{base_url}/calls", json=body, headers={"Authorization": f"Bearer {secret}"}
        ) as reply,
    ):
        return reply.status, (await reply.json() if reply.status == 201 else None)


class TestTheExampleSessionService:
    async def test_the_bot_is_briefed_with_what_the_lookup_knows(
        self, service: tuple[Any, str]
    ) -> None:
        _example, base_url = service
        status, opened = await _open(
            base_url, {"call_id": CALL_ID, "caller": "+15550142", "called": "+15550100"}
        )
        assert status == 201

        brief = await VoiceSessionClient(base_url, BOT_SECRET).fetch_brief(opened["token"])

        assert brief is not None
        assert brief.call_id == CALL_ID
        assert brief.kind == "call"
        assert "Test Caller" in brief.instructions
        assert "a delivery that has not arrived" in brief.instructions
        assert brief.context is not None
        assert brief.context.organisation.name == "Example Office"
        assert brief.context.caller is not None
        assert brief.context.caller.number == "+15550142"
        assert brief.context.destinations == ()
        assert brief.context.any_number is False

    async def test_the_bots_prompt_carries_it(
        self, monkeypatch: pytest.MonkeyPatch, service: tuple[Any, str]
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        _example, base_url = service
        _status, opened = await _open(
            base_url, {"call_id": CALL_ID, "caller": "+15550142", "called": "+15550100"}
        )
        brief = await VoiceSessionClient(base_url, BOT_SECRET).fetch_brief(opened["token"])
        assert brief is not None

        settings: Any = agent_bot.build_llm_service(
            with_hangup=False,
            brief=brief,
            facts=agent_bot.call_facts_from_context(brief.context),
        ).settings

        assert "Greet them by name." in settings.system_instruction
        assert "This call is for Example Office." in settings.system_instruction
        assert "The caller is calling from +15550142." in settings.system_instruction
        # No destinations and no free numbers: the bot is never offered a transfer here.
        assert "transfer_call" not in settings.system_instruction

    async def test_an_unknown_caller_gets_the_plain_instructions(
        self, service: tuple[Any, str]
    ) -> None:
        from session_service import INSTRUCTIONS

        _example, base_url = service
        _status, opened = await _open(base_url, {"call_id": CALL_ID})

        brief = await VoiceSessionClient(base_url, BOT_SECRET).fetch_brief(opened["token"])

        assert brief is not None
        assert brief.instructions == INSTRUCTIONS
        assert brief.context is not None
        assert brief.context.caller is not None
        assert brief.context.caller.privacy == "id"

    async def test_the_report_is_taken_once(self, service: tuple[Any, str]) -> None:
        example, base_url = service
        _status, opened = await _open(base_url, {"call_id": CALL_ID, "caller": "+15550142"})
        client = VoiceSessionClient(base_url, BOT_SECRET)
        brief = await client.fetch_brief(opened["token"])
        assert brief is not None
        session = VoiceSession(client, opened["token"], brief)

        assert await session.report(Finished()) is True
        # A second from another bot object: the service itself refuses it.
        assert await client.report(opened["token"], Chose("sales")) is False

        assert example.sessions[opened["token"]].reported == {"outcome": "finished"}

    async def test_the_bots_secret_reads_and_nothing_else_does(
        self, service: tuple[Any, str]
    ) -> None:
        _example, base_url = service
        _status, opened = await _open(base_url, {"call_id": CALL_ID})

        assert (
            await VoiceSessionClient(base_url, "not-the-secret").fetch_brief(opened["token"])
            is None
        )
        assert await VoiceSessionClient(base_url, BOT_SECRET).fetch_brief("0" * 32) is None

    async def test_only_the_router_opens_a_session(self, service: tuple[Any, str]) -> None:
        """The bot's secret cannot mint the bot a session of its own."""
        _example, base_url = service

        status, _opened = await _open(base_url, {"call_id": CALL_ID}, secret=BOT_SECRET)

        assert status == 404
