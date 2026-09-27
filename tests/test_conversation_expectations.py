"""Judging one call against its scenario, from evidence assembled by hand.

Every failure carries a category, because "the bot made the wrong decision", "the recognizer
misheard the name" and "the harness talked over the bot" call for different fixes, and a pass rate
that cannot tell them apart cannot say what to fix.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tools.conversation.bot_events import BotEvent, BotLog, TransferVerdict
from tools.conversation.downlink_activity import Span
from tools.conversation.evidence import CallEvidence
from tools.conversation.expectations import CheckResult, SignalRules, evaluate_call
from tools.conversation.scenario import Scenario, parse_scenario
from tools.conversation.script_planner import TimelineEvent
from tools.conversation.sip_events import SipMessage

CALL = "conversation-scenario-1-1@172.28.8.40"
CALLER = "172.28.8.40"
PROXY = "172.28.8.20"
ENGINE = "172.28.8.10"
TARGET = "sip:manager@harness.example"
RULES = SignalRules(
    minimum_farewell_seconds=0.4, silent_before_signal_seconds=0.2, maximum_signal_delay_seconds=3.0
)


def _scenario(document: dict[str, Any]) -> Scenario:
    return parse_scenario(document, source="test.yaml", pause_bounds_ms=(600, 1200))


GOODBYE = _scenario(
    {
        "name": "goodbye",
        "lines": [
            {
                "id": "goodbye",
                "wait_for": "greeting",
                "segments": [{"text": "Sorry, wrong number. Goodbye."}],
            }
        ],
        "expectations": {
            "outcome": {"kind": "bot_bye"},
            "tools": {"called": ["end_call"], "not_called": ["transfer_call"]},
            "user_turns": {"exactly": 1},
            "bot_said": [{"after_line": "goodbye", "any_of": ["bye"]}],
        },
    }
)

TRANSFER = _scenario(
    {
        "name": "transfer_named_manager",
        "lines": [
            {
                "id": "ask",
                "wait_for": "greeting",
                "segments": [{"text": "Could I speak to the manager, please? My name is Karen."}],
                "recognition_keywords": ["manager", "karen"],
            }
        ],
        "expectations": {
            "outcome": {"kind": "refer", "refer_to": TARGET},
            "tools": {"called": ["transfer_call"], "not_called": ["end_call"]},
            "user_turns": {"exactly": 1},
            "bot_said": [{"after_line": "ask", "any_of": ["putting you through", "transfer"]}],
        },
    }
)

PAUSE = _scenario(
    {
        "name": "pause_mid_sentence",
        "lines": [
            {
                "id": "ask",
                "wait_for": "greeting",
                "segments": [
                    {
                        "text": "Could I speak to the manager, please? My name is",
                        "pause_after_ms": 900,
                    },
                    {"text": "Karen."},
                ],
                "recognition_keywords": ["manager", "karen"],
            }
        ],
        "expectations": {
            "outcome": {"kind": "refer", "refer_to": TARGET},
            "user_turns": {"exactly": 1},
            "engine_speech_segments_in_turn": [{"turn": 1, "at_least": 2}],
            "llm_user_message": [{"turn": 1, "contains_all": ["name", "karen"]}],
            "bot_audio_during_line": "forbidden",
        },
    }
)

MESSAGE = _scenario(
    {
        "name": "message_for_other_name",
        "lines": [
            {
                "id": "ask",
                "wait_for": "greeting",
                "segments": [{"text": "Could I speak to the manager, please? My name is David."}],
            },
            {
                "id": "goodbye",
                "wait_for": "reply",
                "segments": [{"text": "No, that's alright, thank you. Goodbye."}],
            },
        ],
        "expectations": {"outcome": {"kind": "bot_bye"}},
    }
)


def _sip(
    time: float,
    source: str,
    destination: str,
    *,
    method: str | None = None,
    status: int | None = None,
    cseq: str = "INVITE",
    refer_to: str | None = None,
    sdp_address: str | None = None,
) -> SipMessage:
    return SipMessage(
        time=time,
        source=source,
        destination=destination,
        method=method,
        status=status,
        cseq_method=cseq,
        call_id=CALL,
        refer_to=refer_to,
        to_tag=None,
        sdp_address=sdp_address,
        sdp_port=None,
    )


def _answered() -> list[SipMessage]:
    return [
        _sip(0.0, CALLER, PROXY, method="INVITE", sdp_address=CALLER),
        _sip(0.1, PROXY, CALLER, status=200, sdp_address=ENGINE),
        _sip(0.12, CALLER, PROXY, method="ACK", cseq="ACK"),
    ]


def _bot_bye(time: float) -> list[SipMessage]:
    return [
        _sip(time, PROXY, CALLER, method="BYE", cseq="BYE"),
        _sip(time + 0.01, CALLER, PROXY, status=200, cseq="BYE"),
    ]


def _caller_bye(time: float) -> list[SipMessage]:
    return [
        _sip(time, CALLER, PROXY, method="BYE", cseq="BYE"),
        _sip(time + 0.01, PROXY, CALLER, status=200, cseq="BYE"),
    ]


def _refer(time: float, target: str = TARGET) -> list[SipMessage]:
    return [
        _sip(time, PROXY, CALLER, method="REFER", cseq="REFER", refer_to=target),
        _sip(time + 0.01, CALLER, PROXY, status=202, cseq="REFER"),
        _sip(time + 0.02, CALLER, PROXY, method="NOTIFY", cseq="NOTIFY"),
        _sip(time + 0.03, PROXY, CALLER, status=200, cseq="NOTIFY"),
        *_caller_bye(time + 0.04),
    ]


def _event(event: str, time: float, **fields: object) -> BotEvent:
    return BotEvent(event=event, call_id=CALL, wall_time=time, fields=fields)


def _greeting() -> list[BotEvent]:
    return [_event("bot_started_speaking", 1.0), _event("bot_stopped_speaking", 6.0)]


def _turn(speech_starts: list[float], text: str, stop: float) -> list[BotEvent]:
    return [
        *(_event("engine_speech_started", start) for start in speech_starts),
        _event("user_turn_started", stop - 0.3),
        _event("transcript_heard", stop - 0.2, text=text),
        _event("transcript_accepted", stop - 0.2, text=text),
        _event("user_turn_stopped", stop),
        _event("llm_run", stop + 0.001, user_text=text),
    ]


def _reply(tool: str | None, text: str, start: float, end: float) -> list[BotEvent]:
    events = [] if tool is None else [_event("tool_called", start - 0.4, name=tool)]
    return [
        *events,
        _event("bot_text", start - 0.35, text=text),
        _event("bot_started_speaking", start),
        _event("bot_stopped_speaking", end),
    ]


def _line(line_id: str, start: float, end: float) -> list[TimelineEvent]:
    return [
        TimelineEvent(time=start, kind="line_started", line_id=line_id),
        TimelineEvent(time=start, kind="segment_started", line_id=line_id),
        TimelineEvent(time=end, kind="segment_ended", line_id=line_id),
        TimelineEvent(time=end, kind="line_ended", line_id=line_id),
    ]


def _evidence(
    *,
    sip: list[SipMessage],
    spans: list[Span],
    timeline: list[TimelineEvent],
    events: list[BotEvent],
    transfers: tuple[TransferVerdict, ...] = (),
) -> CallEvidence:
    return CallEvidence(
        call_id=CALL,
        caller_address=CALLER,
        proxy_address=PROXY,
        engine_address=ENGINE,
        sip=tuple(sip),
        downlink_spans=tuple(spans),
        timeline=tuple(timeline),
        bot=BotLog(events=tuple(events), transfers=transfers, unreadable=0),
        sipp_exit_code=0,
        engine_control_errors_delta=0,
        engine_sessions_after=0,
    )


def _goodbye_call() -> CallEvidence:
    return _evidence(
        sip=[*_answered(), *_bot_bye(11.9)],
        spans=[Span(1.0, 6.0), Span(10.6, 11.3)],
        timeline=_line("goodbye", 8.0, 9.5),
        events=[
            *_greeting(),
            *_turn([8.1], "Sorry, wrong number. Goodbye.", 9.9),
            *_reply("end_call", "Goodbye, take care.", 10.6, 11.3),
        ],
    )


def _transfer_call(
    *, text: str = "Could I speak to the manager, please? My name is Karen."
) -> CallEvidence:
    return _evidence(
        sip=[*_answered(), *_refer(11.9)],
        spans=[Span(1.0, 6.0), Span(10.6, 11.4)],
        timeline=_line("ask", 8.0, 10.0),
        events=[
            *_greeting(),
            *_turn([8.1], text, 10.2),
            *_reply("transfer_call", "I am putting you through now.", 10.6, 11.4),
        ],
        transfers=(TransferVerdict(call_id=CALL, completed=True, detail="stage=transferred"),),
    )


def _failures(results: list[CheckResult]) -> dict[str, str]:
    return {result.name: result.category for result in results if not result.passed}


class TestCleanCalls:
    def test_a_goodbye_call_passes_every_check(self) -> None:
        assert _failures(evaluate_call(GOODBYE, _goodbye_call(), RULES)) == {}

    def test_a_transfer_call_passes_every_check(self) -> None:
        assert _failures(evaluate_call(TRANSFER, _transfer_call(), RULES)) == {}


class TestTheEndOfTheCall:
    def test_hanging_up_over_the_goodbye_is_a_media_failure(self) -> None:
        evidence = dataclasses.replace(_goodbye_call(), sip=(*_answered(), *_bot_bye(11.2)))

        assert _failures(evaluate_call(GOODBYE, evidence, RULES)) == {
            "farewell_before_signal": "media"
        }

    def test_hanging_up_without_saying_goodbye_is_a_decision(self) -> None:
        evidence = _evidence(
            sip=[*_answered(), *_bot_bye(10.5)],
            spans=[Span(1.0, 6.0)],
            timeline=_line("goodbye", 8.0, 9.5),
            events=[
                *_greeting(),
                *_turn([8.1], "Sorry, wrong number. Goodbye.", 9.9),
                _event("tool_called", 10.2, name="end_call"),
            ],
        )

        failures = _failures(evaluate_call(GOODBYE, evidence, RULES))

        assert failures["farewell_before_signal"] == "decision"
        assert failures["bot_said:goodbye"] == "decision"

    def test_a_transfer_in_the_goodbye_scenario_is_a_wrong_decision(self) -> None:
        evidence = dataclasses.replace(_goodbye_call(), sip=(*_answered(), *_refer(11.9)))

        assert _failures(evaluate_call(GOODBYE, evidence, RULES))["outcome"] == "decision"

    def test_the_wrong_transfer_target_is_a_signalling_failure(self) -> None:
        evidence = dataclasses.replace(
            _transfer_call(), sip=(*_answered(), *_refer(11.9, "sip:someone@harness.example"))
        )

        assert _failures(evaluate_call(TRANSFER, evidence, RULES))["outcome"] == "signalling"

    def test_a_transfer_the_bot_never_saw_complete_is_a_signalling_failure(self) -> None:
        evidence = dataclasses.replace(
            _transfer_call(),
            bot=BotLog(
                events=_transfer_call().bot.events,
                transfers=(TransferVerdict(call_id=CALL, completed=False, detail="status=503"),),
                unreadable=0,
            ),
        )

        assert _failures(evaluate_call(TRANSFER, evidence, RULES))["outcome"] == "signalling"

    def test_a_sipp_failure_is_a_signalling_failure(self) -> None:
        evidence = dataclasses.replace(_goodbye_call(), sipp_exit_code=1)

        assert _failures(evaluate_call(GOODBYE, evidence, RULES)) == {"sipp_flow": "signalling"}


class TestTurns:
    def test_a_name_that_reached_the_model_in_two_turns_is_turn_taking(self) -> None:
        evidence = _evidence(
            sip=[*_answered(), *_refer(13.9)],
            spans=[Span(1.0, 6.0), Span(10.3, 10.9), Span(12.6, 13.4)],
            timeline=[
                TimelineEvent(time=8.0, kind="line_started", line_id="ask"),
                TimelineEvent(time=8.0, kind="segment_started", line_id="ask"),
                TimelineEvent(time=9.5, kind="segment_ended", line_id="ask"),
                TimelineEvent(time=10.4, kind="segment_started", line_id="ask"),
                TimelineEvent(time=10.9, kind="segment_ended", line_id="ask"),
                TimelineEvent(time=10.9, kind="line_ended", line_id="ask"),
            ],
            events=[
                *_greeting(),
                *_turn([8.1], "Could I speak to the manager, please? My name is", 9.9),
                *_reply(None, "Sorry, who is calling?", 10.3, 10.9),
                *_turn([10.5], "Karen.", 11.3),
                *_reply("transfer_call", "Putting you through.", 12.6, 13.4),
            ],
            transfers=(TransferVerdict(call_id=CALL, completed=True, detail="stage=transferred"),),
        )

        failures = _failures(evaluate_call(PAUSE, evidence, RULES))

        assert failures["user_turns"] == "turn_taking"
        assert failures["llm_user_message:turn1"] == "turn_taking"
        assert failures["no_overlap"] == "decision"

    def test_a_pause_that_did_not_split_the_speech_is_a_harness_problem(self) -> None:
        evidence = _transfer_call(text="Could I speak to the manager, please? My name is Karen.")

        failures = _failures(evaluate_call(PAUSE, evidence, RULES))

        assert failures == {"engine_speech_segments:turn1": "harness"}

    def test_a_misheard_name_is_recognition_rather_than_a_wrong_decision(self) -> None:
        evidence = _evidence(
            sip=[*_answered(), *_caller_bye(40.0)],
            spans=[Span(1.0, 6.0), Span(10.6, 12.0)],
            timeline=_line("ask", 8.0, 10.0),
            events=[
                *_greeting(),
                *_turn([8.1], "Could I speak to the manager, please? My name is Kara.", 10.2),
                *_reply(None, "The manager is not available. Can I take a message?", 10.6, 12.0),
            ],
        )

        failures = _failures(evaluate_call(TRANSFER, evidence, RULES))

        assert failures["recognition:ask"] == "recognition"
        assert failures["outcome"] == "recognition"
        assert failures["tools"] == "recognition"
        assert "decision" not in failures.values()


class TestInterference:
    def test_a_transcript_the_echo_guard_dropped_is_timing_interference(self) -> None:
        call = _goodbye_call()
        evidence = dataclasses.replace(
            call,
            bot=BotLog(
                events=(*call.bot.events, _event("transcript_heard", 3.0, text="hello")),
                transfers=(),
                unreadable=0,
            ),
        )

        assert _failures(evaluate_call(GOODBYE, evidence, RULES)) == {
            "no_echo_guard_drop": "timing_interference"
        }

    def test_talking_over_the_bot_is_timing_interference(self) -> None:
        call = _goodbye_call()
        evidence = dataclasses.replace(
            call,
            timeline=(*call.timeline, TimelineEvent(time=8.5, kind="overlap", line_id="goodbye")),
        )

        assert _failures(evaluate_call(GOODBYE, evidence, RULES)) == {
            "no_overlap": "timing_interference"
        }


class TestLinesThatNeverHappened:
    def test_no_reply_to_a_line_is_a_decision(self) -> None:
        evidence = _evidence(
            sip=[*_answered(), *_caller_bye(40.0)],
            spans=[Span(1.0, 6.0)],
            timeline=[
                *_line("ask", 8.0, 10.0),
                TimelineEvent(time=30.0, kind="wait_timed_out", line_id="goodbye"),
            ],
            events=[*_greeting(), *_turn([8.1], "My name is David.", 10.2)],
        )

        assert _failures(evaluate_call(MESSAGE, evidence, RULES))["script_completed"] == "decision"

    def test_no_greeting_at_all_is_infrastructure(self) -> None:
        evidence = _evidence(
            sip=[*_answered(), *_caller_bye(40.0)],
            spans=[],
            timeline=[TimelineEvent(time=30.0, kind="wait_timed_out", line_id="goodbye")],
            events=[],
        )

        assert _failures(evaluate_call(GOODBYE, evidence, RULES))["script_completed"] == (
            "infrastructure"
        )


class TestInfrastructure:
    def test_a_pipeline_error_and_a_dirty_engine_are_infrastructure(self) -> None:
        call = _goodbye_call()
        evidence = dataclasses.replace(
            call,
            bot=BotLog(
                events=(*call.bot.events, _event("pipeline_error", 5.0, message="boom")),
                transfers=(),
                unreadable=0,
            ),
            engine_sessions_after=1,
        )

        assert _failures(evaluate_call(GOODBYE, evidence, RULES)) == {
            "no_pipeline_error": "infrastructure",
            "engine_clean": "infrastructure",
        }


class TestWhatTheBotSaid:
    def test_phrases_ignore_case_and_punctuation(self) -> None:
        call = _transfer_call()
        events = [
            event
            if event.event != "bot_text"
            else _event("bot_text", event.wall_time, text="Okay -- PUTTING you through!")
            for event in call.bot.events
        ]
        evidence = dataclasses.replace(
            call, bot=BotLog(events=tuple(events), transfers=call.bot.transfers, unreadable=0)
        )

        assert _failures(evaluate_call(TRANSFER, evidence, RULES)) == {}
