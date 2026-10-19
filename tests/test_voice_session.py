"""Tests for `pipecat_siphon.voice_session`: a siphon-pbx voice session's brief and report.

The briefs come from `pbx_fixtures`, emitted by siphon-pbx's own serializer, and the report bodies
are checked against the bytes its route was shown to accept, so a test here checks our reading of
the PBX and not our reading of ourselves. The PBX is faked on loopback.
"""

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger

from fake_pbx import FakePbx
from pbx_fixtures import (
    BRIEF_ESCAPED,
    BRIEF_MENU,
    BRIEF_OPEN,
    REPORT_CHOSE_SALES,
    REPORT_FAILED,
    REPORT_FINISHED,
)
from pipecat_siphon import (
    BriefOption,
    Chose,
    Failed,
    Finished,
    SessionBrief,
    VoiceSession,
    VoiceSessionClient,
    encode_report,
    parse_brief,
    session_token_from,
)
from pipecat_siphon.voice_session import (
    MAX_BRIEF_BYTES,
    MAX_REPORTED_OPTION,
    MAX_REPORTED_REASON,
)

TOKEN = "9f86d081884c7d659a2feaa0c55ad015"
SECRET = "s3cret-for-tests-only"


class TestParseBrief:
    def test_an_open_conversation(self) -> None:
        assert parse_brief(BRIEF_OPEN) == SessionBrief(
            call_id="a84b4c76e66710@pc33.example.invalid",
            instructions="Answer questions about opening hours.",
            options=(),
            deadline_secs=120,
        )

    def test_a_menu_keeps_its_order_and_an_omitted_means_reads_empty(self) -> None:
        brief = parse_brief(BRIEF_MENU)
        assert brief is not None
        assert brief.options == (
            BriefOption(option="sales", means="buying something new"),
            BriefOption(option="support", means=""),
        )
        assert brief.deadline_secs == 65535

    def test_escapes_and_non_ascii_text(self) -> None:
        brief = parse_brief(BRIEF_ESCAPED)
        assert brief is not None
        assert brief.call_id == 'c"1\\@example.invalid'
        assert brief.instructions == 'Say "hallo" first.\nThen ask: wat kan ik voor u doen? é😀'
        assert brief.deadline_secs == 0

    def test_fields_it_does_not_know_are_ignored(self) -> None:
        """A hosted bot's `runtime` and a later `context` add fields; the brief still reads."""
        document = json.loads(BRIEF_MENU)
        document["runtime"] = {"llm": {"provider": "x"}}
        document["context"] = {"version": 1}
        document["options"][0]["later"] = True
        assert parse_brief(json.dumps(document).encode()) == parse_brief(BRIEF_MENU)

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"not json",
            BRIEF_OPEN[:-1],
            b"[]",
            b"null",
            b'"a string"',
            b"\xff\xfe",
        ],
    )
    def test_malformed_input_reads_as_no_brief(self, raw: bytes) -> None:
        assert parse_brief(raw) is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("call_id", None),
            ("call_id", 7),
            ("instructions", ["a"]),
            ("options", {"option": "sales"}),
            ("options", ["sales"]),
            ("options", [{"means": "no option name"}]),
            ("options", [{"option": ""}]),
            ("options", [{"option": "sales", "means": 3}]),
            ("deadline_secs", -1),
            ("deadline_secs", 65536),
            ("deadline_secs", 1.5),
            ("deadline_secs", True),
            ("deadline_secs", "120"),
        ],
    )
    def test_a_field_of_the_wrong_shape_reads_as_no_brief(self, field: str, value: Any) -> None:
        document = json.loads(BRIEF_MENU)
        document[field] = value
        assert parse_brief(json.dumps(document).encode()) is None

    @pytest.mark.parametrize("field", ["call_id", "instructions", "options", "deadline_secs"])
    def test_a_missing_field_reads_as_no_brief(self, field: str) -> None:
        document = json.loads(BRIEF_MENU)
        del document[field]
        assert parse_brief(json.dumps(document).encode()) is None

    def test_an_oversized_body_is_not_parsed(self) -> None:
        document = json.loads(BRIEF_OPEN)
        document["instructions"] = "x" * MAX_BRIEF_BYTES
        assert parse_brief(json.dumps(document).encode()) is None


class TestEncodeReport:
    def test_the_three_reports_are_the_bytes_the_route_accepts(self) -> None:
        assert encode_report(Chose("sales")) == REPORT_CHOSE_SALES
        assert encode_report(Finished()) == REPORT_FINISHED
        assert encode_report(Failed("no speech service")) == REPORT_FAILED

    def test_text_is_sent_as_utf8_not_escaped(self) -> None:
        assert (
            encode_report(Chose("verkoop é")) == '{"outcome":"chose","option":"verkoop é"}'.encode()
        )

    def test_an_overlong_option_and_reason_are_cut_where_the_pbx_cuts_them(self) -> None:
        chose = json.loads(encode_report(Chose("o" * (MAX_REPORTED_OPTION + 5))))
        failed = json.loads(encode_report(Failed("r" * (MAX_REPORTED_REASON + 5))))
        assert chose["option"] == "o" * MAX_REPORTED_OPTION
        assert failed["reason"] == "r" * MAX_REPORTED_REASON

    def test_the_cut_counts_characters_not_bytes(self) -> None:
        chose = json.loads(encode_report(Chose("é" * (MAX_REPORTED_OPTION + 1))))
        assert chose["option"] == "é" * MAX_REPORTED_OPTION


class TestSessionTokenFrom:
    def test_reads_the_session_parameter(self) -> None:
        assert session_token_from({"call": "abc", "session": TOKEN}) == TOKEN

    @pytest.mark.parametrize(
        "query",
        [
            {},
            {"session": ""},
            {"session": "short"},
            {"session": "a" * 257},
            {"session": "../../api/v1/x" + "a" * 16},
            {"session": TOKEN + "/report"},
            {"session": TOKEN + "?x=1"},
            {"session": TOKEN + " "},
        ],
    )
    def test_anything_but_a_plain_token_is_no_session(self, query: dict[str, str]) -> None:
        assert session_token_from(query) is None


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink)


class TestVoiceSessionClient:
    async def test_fetches_the_brief_with_the_secret_as_a_bearer(self, pbx: FakePbx) -> None:
        client = VoiceSessionClient(pbx.url, SECRET)
        assert await client.fetch_brief(TOKEN) == parse_brief(BRIEF_MENU)
        (request,) = pbx.requests
        assert request["method"] == "GET"
        assert request["path"] == f"/voice/sessions/{TOKEN}"
        assert request["headers"]["Authorization"] == f"Bearer {SECRET}"

    async def test_sends_no_origin_header_which_the_pbx_reads_as_a_browser(
        self, pbx: FakePbx
    ) -> None:
        client = VoiceSessionClient(pbx.url, SECRET)
        await client.fetch_brief(TOKEN)
        await client.report(TOKEN, Finished())
        assert all("Origin" not in request["headers"] for request in pbx.requests)

    async def test_a_trailing_slash_on_the_base_url_does_not_double_up(self, pbx: FakePbx) -> None:
        assert await VoiceSessionClient(pbx.url + "/", SECRET).fetch_brief(TOKEN) is not None
        assert pbx.requests[-1]["path"] == f"/voice/sessions/{TOKEN}"

    @pytest.mark.parametrize("status", [404, 500, 302])
    async def test_a_refused_brief_is_none(self, pbx: FakePbx, status: int) -> None:
        pbx.brief_status = status
        assert await VoiceSessionClient(pbx.url, SECRET).fetch_brief(TOKEN) is None

    async def test_a_malformed_brief_is_none(self, pbx: FakePbx) -> None:
        pbx.brief_body = b'{"call_id":'
        assert await VoiceSessionClient(pbx.url, SECRET).fetch_brief(TOKEN) is None

    async def test_an_oversized_brief_is_none(self, pbx: FakePbx) -> None:
        pbx.brief_body = b" " * (MAX_BRIEF_BYTES + 1) + BRIEF_OPEN
        assert await VoiceSessionClient(pbx.url, SECRET).fetch_brief(TOKEN) is None

    async def test_a_slow_pbx_is_given_up_on(self, pbx: FakePbx) -> None:
        pbx.delay = 1.0
        client = VoiceSessionClient(pbx.url, SECRET, timeout_seconds=0.1)
        assert await client.fetch_brief(TOKEN) is None
        assert await client.report(TOKEN, Finished()) is False

    async def test_an_unreachable_pbx_is_none(self) -> None:
        client = VoiceSessionClient("http://127.0.0.1:9", SECRET, timeout_seconds=0.5)
        assert await client.fetch_brief(TOKEN) is None
        assert await client.report(TOKEN, Finished()) is False

    async def test_posts_the_report_as_json(self, pbx: FakePbx) -> None:
        client = VoiceSessionClient(pbx.url, SECRET)
        assert await client.report(TOKEN, Chose("sales")) is True
        (request,) = pbx.requests
        assert request["method"] == "POST"
        assert request["path"] == f"/voice/sessions/{TOKEN}/report"
        assert request["headers"]["Authorization"] == f"Bearer {SECRET}"
        assert request["headers"]["Content-Type"] == "application/json"
        assert request["body"] == REPORT_CHOSE_SALES

    async def test_a_report_nobody_was_waiting_for_is_false(self, pbx: FakePbx) -> None:
        pbx.report_status = 404
        assert await VoiceSessionClient(pbx.url, SECRET).report(TOKEN, Finished()) is False

    async def test_neither_the_secret_nor_the_token_is_ever_logged(
        self, pbx: FakePbx, log_lines: list[str]
    ) -> None:
        client = VoiceSessionClient(pbx.url, SECRET)
        pbx.brief_status = 404
        await client.fetch_brief(TOKEN)
        pbx.brief_status = 200
        pbx.brief_body = b"junk"
        await client.fetch_brief(TOKEN)
        pbx.report_status = 404
        await client.report(TOKEN, Finished())
        await VoiceSessionClient("http://127.0.0.1:9", SECRET).fetch_brief(TOKEN)
        assert log_lines
        assert not any(SECRET in line or TOKEN in line for line in log_lines)

    def test_a_base_url_that_is_not_http_is_refused(self) -> None:
        with pytest.raises(ValueError):
            VoiceSessionClient("ftp://pbx.example.invalid", SECRET)
        with pytest.raises(ValueError):
            VoiceSessionClient("http://pbx.example.invalid", "")


class TestVoiceSession:
    def test_nothing_is_reported_until_something_is(self) -> None:
        brief = parse_brief(BRIEF_OPEN)
        assert brief is not None
        session = VoiceSession(VoiceSessionClient("http://127.0.0.1:9", SECRET), TOKEN, brief)
        assert session.reported is None
        assert session.brief is brief

    async def test_reports_once_and_drops_what_comes_after(
        self, pbx: FakePbx, log_lines: list[str]
    ) -> None:
        brief = parse_brief(BRIEF_MENU)
        assert brief is not None
        session = VoiceSession(VoiceSessionClient(pbx.url, SECRET), TOKEN, brief)
        assert await session.report(Chose("sales")) is True
        assert await session.report(Finished()) is False
        assert await session.report(Failed("late")) is False
        assert session.reported == Chose("sales")
        assert [request["body"] for request in pbx.requests] == [REPORT_CHOSE_SALES]
        assert any("already reported" in line for line in log_lines)

    async def test_two_reports_at_once_still_send_one(self, pbx: FakePbx) -> None:
        pbx.delay = 0.05
        brief = parse_brief(BRIEF_OPEN)
        assert brief is not None
        session = VoiceSession(VoiceSessionClient(pbx.url, SECRET), TOKEN, brief)
        results = await asyncio.gather(session.report(Finished()), session.report(Failed("x")))
        assert sorted(results) == [False, True]
        assert len(pbx.requests) == 1

    async def test_a_report_the_pbx_did_not_take_still_counts_as_the_one(
        self, pbx: FakePbx
    ) -> None:
        """The PBX takes one report per call; retrying another would not make the first land."""
        pbx.report_status = 404
        brief = parse_brief(BRIEF_OPEN)
        assert brief is not None
        session = VoiceSession(VoiceSessionClient(pbx.url, SECRET), TOKEN, brief)
        assert await session.report(Finished()) is False
        assert session.reported == Finished()
        assert await session.report(Failed("x")) is False
        assert len(pbx.requests) == 1
