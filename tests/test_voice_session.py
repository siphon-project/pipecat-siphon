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
    BRIEF_CONTEXT_FULL,
    BRIEF_CONTEXT_NATIONAL,
    BRIEF_CONTEXT_ORGANISATION,
    BRIEF_CONTEXT_UNSET,
    BRIEF_ESCAPED,
    BRIEF_MENU,
    BRIEF_OPEN,
    REPORT_CHOSE_SALES,
    REPORT_FAILED,
    REPORT_FINISHED,
)
from pipecat_siphon import (
    BriefOption,
    CallContext,
    Chose,
    ContextCalled,
    ContextCaller,
    ContextOrganisation,
    ContextTrunk,
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
CLUSTERED_TOKEN = TOKEN + ".pbx-node-2"
"""What a clustered siphon-pbx node mints: the token, `.`, and the node that holds the session."""
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


ORGANISATION = ContextOrganisation(
    id="00000000-0000-4000-8000-00000000000a",
    name="Example Office",
    time_zone="Europe/Amsterdam",
    external_ref="CUST-0042",
    locale="nl",
)


class TestCallContext:
    """`context`, as siphon-pbx's own `CallContext` serializer puts it in the brief."""

    def test_a_brief_without_one_has_none(self) -> None:
        brief = parse_brief(BRIEF_MENU)
        assert brief is not None
        assert brief.context is None

    def test_full(self) -> None:
        brief = parse_brief(BRIEF_CONTEXT_FULL)
        assert brief is not None
        assert brief.context == CallContext(
            organisation=ORGANISATION,
            called=ContextCalled(
                uri="sip:+15550100@pbx.example.invalid;user=phone", number="+15550100"
            ),
            caller=ContextCaller(privacy="none", number="+15550142", display_name="Test Caller"),
            direction="inbound",
            trunk=ContextTrunk(id="22222222-2222-4222-8222-222222222222", name="Carrier"),
        )

    def test_a_national_form_beside_each_number(self) -> None:
        brief = parse_brief(BRIEF_CONTEXT_NATIONAL)
        assert brief is not None
        context = brief.context
        assert context is not None
        assert context.called == ContextCalled(
            uri="sip:05550100@pbx.example.invalid;user=phone",
            number="+995550100",
            national="05550100",
        )
        assert context.caller == ContextCaller(
            privacy="none", number="+995550142", national="05550142", display_name="Test Caller"
        )

    def test_a_pbx_from_before_the_national_form_reads_as_none(self) -> None:
        brief = parse_brief(BRIEF_CONTEXT_FULL)
        assert brief is not None
        assert brief.context is not None
        assert brief.context.called is not None
        assert brief.context.caller is not None
        assert brief.context.called.national is None
        assert brief.context.caller.national is None

    def test_a_withheld_caller_keeps_no_national_form_either(self) -> None:
        document = json.loads(BRIEF_CONTEXT_NATIONAL)
        document["context"]["caller"]["privacy"] = "id"
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None
        assert brief.context is not None
        assert brief.context.caller == ContextCaller(privacy="id")

    @pytest.mark.parametrize("part", ["called", "caller"])
    def test_a_national_form_that_is_not_text_drops_the_context(self, part: str) -> None:
        document = json.loads(BRIEF_CONTEXT_NATIONAL)
        document["context"][part]["national"] = 5550142
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None
        assert brief.context is None

    def test_organisation_only(self) -> None:
        brief = parse_brief(BRIEF_CONTEXT_ORGANISATION)
        assert brief is not None
        assert brief.context == CallContext(
            organisation=ContextOrganisation(
                id="00000000-0000-4000-8000-00000000000a",
                name="Example Office",
                time_zone="Europe/Amsterdam",
                external_ref="CUST-0042",
                locale="en",
            )
        )

    def test_unset_values_and_a_withheld_caller(self) -> None:
        brief = parse_brief(BRIEF_CONTEXT_UNSET)
        assert brief is not None
        context = brief.context
        assert context is not None
        assert context.organisation.external_ref is None
        assert context.organisation.locale is None
        assert context.called == ContextCalled(uri="sip:5550199@pbx.example.invalid")
        assert context.caller == ContextCaller(privacy="id")
        assert context.direction == "outbound"
        assert context.trunk is None

    def test_a_withheld_caller_never_carries_a_number_even_if_sent_one(self) -> None:
        document = json.loads(BRIEF_CONTEXT_FULL)
        document["context"]["caller"]["privacy"] = "id"
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None and brief.context is not None
        assert brief.context.caller == ContextCaller(privacy="id")

    def test_a_privacy_it_does_not_know_reads_as_withheld(self) -> None:
        document = json.loads(BRIEF_CONTEXT_FULL)
        document["context"]["caller"]["privacy"] = "header"
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None and brief.context is not None
        assert brief.context.caller == ContextCaller(privacy="id")

    def test_a_direction_it_does_not_know_reads_as_none(self) -> None:
        document = json.loads(BRIEF_CONTEXT_FULL)
        document["context"]["direction"] = "sideways"
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None and brief.context is not None
        assert brief.context.direction is None
        assert brief.context.organisation == ORGANISATION

    @pytest.mark.parametrize("version", [0, 2, "1", None, True])
    def test_a_version_it_does_not_know_is_no_context(self, version: Any) -> None:
        document = json.loads(BRIEF_CONTEXT_FULL)
        document["context"]["version"] = version
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None
        assert brief.context is None
        assert brief.call_id == "b2b-00000000000000a1@example.invalid"

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("context",), "a string"),
            (("context", "organisation"), None),
            (("context", "organisation", "name"), 7),
            (("context", "organisation", "external_ref"), 42),
            (("context", "called", "uri"), None),
            (("context", "called", "number"), 15550100),
            (("context", "caller", "display_name"), ["Test"]),
            (("context", "trunk"), "Carrier"),
        ],
    )
    def test_a_malformed_context_drops_only_the_context(
        self, path: tuple[str, ...], value: Any
    ) -> None:
        document = json.loads(BRIEF_CONTEXT_FULL)
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        brief = parse_brief(json.dumps(document).encode())
        assert brief is not None
        assert brief.context is None
        assert brief.instructions == "Ask what the caller wants."

    def test_fields_it_does_not_know_are_ignored(self) -> None:
        document = json.loads(BRIEF_CONTEXT_FULL)
        document["context"]["later"] = {"x": 1}
        document["context"]["caller"]["verified"] = True
        document["context"]["called"]["label"] = "Sales line"
        assert parse_brief(json.dumps(document).encode()) == parse_brief(BRIEF_CONTEXT_FULL)


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
        "token",
        [
            CLUSTERED_TOKEN,
            TOKEN + ".a",
            TOKEN + ".node.eu-west_1",
            TOKEN + ".n" + "x" * 62,
            TOKEN + ".7",
        ],
    )
    def test_a_node_suffix_is_kept_whole(self, token: str) -> None:
        """siphon-pbx routes on the suffix, so the token is passed on exactly as minted."""
        assert session_token_from({"session": token}) == token

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
            # A node suffix as siphon-pbx's `NodeId` refuses it: empty, not starting with a
            # letter or digit, too long, or carrying anything that walks a path.
            {"session": TOKEN + "."},
            {"session": TOKEN + ".."},
            {"session": TOKEN + "..node"},
            {"session": TOKEN + ".-node"},
            {"session": TOKEN + "._node"},
            {"session": TOKEN + ".n" + "x" * 63},
            {"session": TOKEN + ".node/report"},
            {"session": TOKEN + ".node/../x"},
            {"session": TOKEN + ".node%2F"},
            {"session": "." + TOKEN},
            {"session": "short.node"},
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

    async def test_a_clustered_token_reaches_the_pbx_unchanged(self, pbx: FakePbx) -> None:
        client = VoiceSessionClient(pbx.url, SECRET)
        assert await client.fetch_brief(CLUSTERED_TOKEN) is not None
        assert await client.report(CLUSTERED_TOKEN, Finished()) is True
        assert [request["path"] for request in pbx.requests] == [
            f"/voice/sessions/{CLUSTERED_TOKEN}",
            f"/voice/sessions/{CLUSTERED_TOKEN}/report",
        ]

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
