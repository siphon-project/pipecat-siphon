"""Judge one call against its scenario, check by check, and put every failure in a category.

The categories exist because the fixes differ. A wrong decision is the agent's prompt or model; a
misheard name is the recognizer; talking over the bot is the harness's timing or the bot's echo
guard; a stack that never answered is neither. A pass rate that lumps them together cannot say what
to change. When recognition failed, the decisions that followed from it are counted as recognition
too: a bot that did not transfer "Kara" behaved correctly for what it heard.

Everything here is pure: the evidence comes in already assembled, so every rule is tested on calls
built by hand.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Literal

from tools.conversation.bot_events import BotEvent
from tools.conversation.evidence import CallEvidence
from tools.conversation.scenario import BotSaid, Scenario, TurnCount, TurnText
from tools.conversation.sip_events import SipMessage

Category = Literal[
    "signalling",
    "media",
    "turn_taking",
    "recognition",
    "decision",
    "timing_interference",
    "infrastructure",
    "harness",
]


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict. `category` says what a failure points at; it is kept on a pass too."""

    name: str
    passed: bool
    category: Category
    message: str


@dataclass(frozen=True)
class SignalRules:
    """What "the goodbye played before the hangup" means, in seconds."""

    minimum_farewell_seconds: float
    """How much the bot must say between the caller's last line and its BYE or REFER."""
    silent_before_signal_seconds: float
    """How long the bot must already be quiet when the BYE or REFER goes out. Less is a cut-off."""
    maximum_signal_delay_seconds: float
    """How long after the bot falls quiet the BYE or REFER may come."""


def evaluate_call(
    scenario: Scenario, evidence: CallEvidence, rules: SignalRules
) -> list[CheckResult]:
    """Run every check that applies to `scenario` on `evidence`."""
    call = _Call(evidence)
    results = [
        _answered_by_engine(call),
        _sipp_flow(evidence),
        _script_completed(scenario, call),
        _no_overlap(scenario, call),
        _no_echo_guard_drop(call),
        _no_pipeline_error(call),
        _engine_clean(evidence),
        *_recognition(scenario, call),
        _outcome(scenario, call),
        *_farewell_before_signal(scenario, call, rules),
        *_tools(scenario, call),
        *_user_turns(scenario, call),
        *(_engine_speech_segments(expectation, call) for expectation in _segment_counts(scenario)),
        *(_llm_user_message(expectation, call) for expectation in _user_messages(scenario)),
        *(_bot_said(expectation, scenario, call) for expectation in scenario.expectations.bot_said),
    ]
    if any(result.category == "recognition" and not result.passed for result in results):
        results = [
            replace(result, category="recognition")
            if not result.passed and result.category == "decision"
            else result
            for result in results
        ]
    return results


class _Call:
    """The evidence, indexed the ways the checks read it."""

    def __init__(self, evidence: CallEvidence) -> None:
        self.evidence = evidence
        self.events = sorted(evidence.bot.events, key=lambda event: event.wall_time)
        self.turn_stops = [
            event.wall_time for event in self.events if event.event == "user_turn_stopped"
        ]
        self.refer = self._first_request_from_proxy("REFER")
        self.proxy_bye = self._first_request_from_proxy("BYE")
        self.line_starts = {
            event.line_id: event.time
            for event in reversed(evidence.timeline)
            if event.kind == "line_started"
        }
        self.line_ends = {
            event.line_id: event.time for event in evidence.timeline if event.kind == "line_ended"
        }

    def named(self, name: str, start: float = -math.inf, end: float = math.inf) -> list[BotEvent]:
        """Return the bot's `name` events with `start <= time < end`."""
        return [
            event for event in self.events if event.event == name and start <= event.wall_time < end
        ]

    def turn_window(self, turn: int) -> tuple[float, float] | None:
        """From just after the previous turn ended to this turn's end, or None if no such turn."""
        if turn > len(self.turn_stops):
            return None
        start = self.turn_stops[turn - 2] if turn >= 2 else -math.inf
        return start, self.turn_stops[turn - 1]

    def line_window(self, scenario: Scenario, line_id: str) -> tuple[float, float] | None:
        """From this line's start to the next line's start, or None if the line never started."""
        start = self.line_starts.get(line_id)
        if start is None:
            return None
        ids = [line.id for line in scenario.lines]
        later = [
            self.line_starts[other]
            for other in ids[ids.index(line_id) + 1 :]
            if other in self.line_starts
        ]
        return start, later[0] if later else math.inf

    def _first_request_from_proxy(self, method: str) -> SipMessage | None:
        proxy = self.evidence.proxy_address
        return next(
            (
                message
                for message in self.evidence.sip
                if message.method == method and message.source == proxy
            ),
            None,
        )


def _pass(name: str, category: Category, message: str) -> CheckResult:
    return CheckResult(name=name, passed=True, category=category, message=message)


def _fail(name: str, category: Category, message: str) -> CheckResult:
    return CheckResult(name=name, passed=False, category=category, message=message)


def _answered_by_engine(call: _Call) -> CheckResult:
    name = "answered_by_engine"
    evidence = call.evidence
    answer = next(
        (
            message
            for message in evidence.sip
            if message.status == 200
            and message.cseq_method == "INVITE"
            and message.source == evidence.proxy_address
        ),
        None,
    )
    if answer is None:
        return _fail(name, "infrastructure", "the call was never answered")
    if answer.sdp_address != evidence.engine_address:
        return _fail(
            name,
            "signalling",
            f"answered with media at {answer.sdp_address}, not at the engine's "
            f"{evidence.engine_address}",
        )
    return _pass(name, "signalling", "answered with the engine's media address")


def _sipp_flow(evidence: CallEvidence) -> CheckResult:
    if evidence.sipp_exit_code != 0:
        return _fail("sipp_flow", "signalling", f"SIPp exited {evidence.sipp_exit_code}")
    return _pass("sipp_flow", "signalling", "SIPp completed its scenario")


def _script_completed(scenario: Scenario, call: _Call) -> CheckResult:
    name = "script_completed"
    timeline = call.evidence.timeline
    timeout = next((event for event in timeline if event.kind == "wait_timed_out"), None)
    missing = [line for line in scenario.lines if line.id not in call.line_ends]
    if timeout is None and not missing:
        return _pass(name, "decision", "every line was said")
    if timeout is not None:
        line = next((line for line in scenario.lines if line.id == timeout.line_id), None)
        if line is not None and line.wait_for == "greeting":
            return _fail(name, "infrastructure", f"no greeting was heard before {line.id!r}")
        return _fail(name, "decision", f"the bot did not reply before {timeout.line_id!r}")
    first_missing = missing[0]
    if call.proxy_bye is not None or call.refer is not None:
        return _fail(
            name, "decision", f"the bot ended the call before the caller said {first_missing.id!r}"
        )
    return _fail(name, "harness", f"the caller never said {first_missing.id!r}")


def _no_overlap(scenario: Scenario, call: _Call) -> CheckResult:
    name = "no_overlap"
    category: Category = (
        "decision"
        if scenario.expectations.bot_audio_during_line == "forbidden"
        else "timing_interference"
    )
    noticed = [event.line_id for event in call.evidence.timeline if event.kind == "overlap"]
    heard = [
        line.id
        for line in scenario.lines
        if line.id in call.line_starts
        and line.id in call.line_ends
        and any(
            span.start < call.line_ends[line.id] and span.end > call.line_starts[line.id]
            for span in call.evidence.downlink_spans
        )
    ]
    lines = sorted(set(noticed) | set(heard))
    if lines:
        return _fail(name, category, f"the bot was audible while the caller said {lines}")
    return _pass(name, category, "the bot was quiet during every line")


def _no_echo_guard_drop(call: _Call) -> CheckResult:
    name = "no_echo_guard_drop"
    heard = len(call.named("transcript_heard"))
    accepted = len(call.named("transcript_accepted"))
    if heard > accepted:
        return _fail(
            name,
            "timing_interference",
            f"{heard - accepted} transcript(s) were dropped by the echo guard",
        )
    return _pass(name, "timing_interference", "every transcript reached the turn logic")


def _no_pipeline_error(call: _Call) -> CheckResult:
    errors = call.named("pipeline_error")
    if errors:
        messages = [str(event.fields.get("message", "")) for event in errors]
        return _fail("no_pipeline_error", "infrastructure", f"pipeline errors: {messages}")
    return _pass("no_pipeline_error", "infrastructure", "no pipeline errors")


def _engine_clean(evidence: CallEvidence) -> CheckResult:
    name = "engine_clean"
    if evidence.engine_control_errors_delta is None or evidence.engine_sessions_after is None:
        return _fail(name, "harness", "the engine's metrics could not be read")
    if evidence.engine_control_errors_delta or evidence.engine_sessions_after:
        return _fail(
            name,
            "infrastructure",
            f"{evidence.engine_control_errors_delta} control error(s) during the call, "
            f"{evidence.engine_sessions_after} session(s) left after it",
        )
    return _pass(name, "infrastructure", "no control errors and no sessions left behind")


def _recognition(scenario: Scenario, call: _Call) -> list[CheckResult]:
    results: list[CheckResult] = []
    for line in scenario.lines:
        if not line.recognition_keywords:
            continue
        name = f"recognition:{line.id}"
        window = call.line_window(scenario, line.id)
        if window is None:
            continue
        heard = " ".join(
            str(event.fields.get("text", ""))
            for event in call.named("transcript_accepted", *window)
        )
        missing = [word for word in line.recognition_keywords if not _contains(heard, word)]
        if missing:
            results.append(
                _fail(name, "recognition", f"{line.id!r} was heard as {heard!r}, without {missing}")
            )
        else:
            results.append(_pass(name, "recognition", f"{line.id!r} was heard as {heard!r}"))
    return results


def _outcome(scenario: Scenario, call: _Call) -> CheckResult:
    name = "outcome"
    outcome = scenario.expectations.outcome
    refer, bye = call.refer, call.proxy_bye
    if outcome.kind == "bot_bye":
        if refer is not None:
            return _fail(
                name, "decision", f"the bot transferred to {refer.refer_to} instead of hanging up"
            )
        if bye is None:
            return _fail(name, "decision", "the bot never hung up")
        return _pass(name, "decision", "the bot hung up")
    if outcome.kind == "none":
        if refer is not None or bye is not None:
            return _fail(name, "decision", "the bot ended the call, which it should not have")
        return _pass(name, "decision", "the bot neither hung up nor transferred")

    if refer is None:
        return _fail(name, "decision", "the bot never transferred the call")
    if bye is not None and bye.time < refer.time:
        return _fail(name, "decision", "the bot hung up before transferring")
    if refer.refer_to != outcome.refer_to:
        return _fail(
            name, "signalling", f"the REFER named {refer.refer_to}, not {outcome.refer_to}"
        )
    evidence = call.evidence
    accepted = any(
        message.status == 202
        and message.cseq_method == "REFER"
        and message.source == evidence.caller_address
        for message in evidence.sip
    )
    notified = any(
        message.status == 200
        and message.cseq_method == "NOTIFY"
        and message.source == evidence.proxy_address
        for message in evidence.sip
    )
    if not (accepted and notified):
        return _fail(name, "signalling", "the REFER's 202 and NOTIFY exchange did not complete")
    verdicts = evidence.bot.transfers
    if not any(verdict.completed for verdict in verdicts):
        details = [verdict.detail for verdict in verdicts] or ["no verdict logged"]
        return _fail(name, "signalling", f"the bot did not see the transfer complete: {details}")
    return _pass(name, "decision", f"the bot transferred the call to {refer.refer_to}")


def _farewell_before_signal(
    scenario: Scenario, call: _Call, rules: SignalRules
) -> list[CheckResult]:
    name = "farewell_before_signal"
    kind = scenario.expectations.outcome.kind
    signal_message = (
        call.proxy_bye if kind == "bot_bye" else call.refer if kind == "refer" else None
    )
    if signal_message is None:
        return []
    signal = signal_message.time
    after = max(call.line_ends.values(), default=-math.inf)
    spans = [
        span for span in call.evidence.downlink_spans if span.end > after and span.start < signal
    ]
    spoken = sum(min(span.end, signal) - max(span.start, after) for span in spans)
    if spoken < rules.minimum_farewell_seconds:
        said = call.named("bot_text", start=after)
        category: Category = "media" if said else "decision"
        return [
            _fail(
                name,
                category,
                f"the bot said {spoken:.2f}s between the caller's last line and its "
                f"{signal_message.method}" + ("" if said else ", and produced no words to say"),
            )
        ]
    cut_off = signal - rules.silent_before_signal_seconds
    if any(span.end > cut_off for span in spans):
        return [
            _fail(
                name,
                "media",
                f"the bot was still audible within {rules.silent_before_signal_seconds}s of its "
                f"{signal_message.method}",
            )
        ]
    delay = signal - max(span.end for span in spans)
    if delay > rules.maximum_signal_delay_seconds:
        return [
            _fail(
                name,
                "media",
                f"the {signal_message.method} came {delay:.2f}s after the bot fell quiet",
            )
        ]
    return [_pass(name, "media", f"the bot spoke {spoken:.2f}s, then sent {signal_message.method}")]


def _tools(scenario: Scenario, call: _Call) -> list[CheckResult]:
    expected = scenario.expectations.tools
    if not expected.called and not expected.not_called:
        return []
    used = {str(event.fields.get("name")) for event in call.named("tool_called")}
    missing = [tool for tool in expected.called if tool not in used]
    forbidden = [tool for tool in expected.not_called if tool in used]
    if missing or forbidden:
        return [
            _fail(
                "tools",
                "decision",
                f"called {sorted(used)}; missing {missing}, should not have called {forbidden}",
            )
        ]
    return [_pass("tools", "decision", f"called {sorted(used)}")]


def _user_turns(scenario: Scenario, call: _Call) -> list[CheckResult]:
    expected = scenario.expectations.user_turns
    if expected is None:
        return []
    counted = len(call.turn_stops)
    if counted != expected:
        return [
            _fail("user_turns", "turn_taking", f"{counted} caller turn(s), expected {expected}")
        ]
    return [_pass("user_turns", "turn_taking", f"{counted} caller turn(s)")]


def _segment_counts(scenario: Scenario) -> tuple[TurnCount, ...]:
    return scenario.expectations.engine_speech_segments_in_turn


def _user_messages(scenario: Scenario) -> tuple[TurnText, ...]:
    return scenario.expectations.llm_user_message


def _engine_speech_segments(expectation: TurnCount, call: _Call) -> CheckResult:
    name = f"engine_speech_segments:turn{expectation.turn}"
    window = call.turn_window(expectation.turn)
    if window is None:
        return _fail(name, "turn_taking", f"there was no caller turn {expectation.turn}")
    start, end = window
    counted = len(
        [
            event
            for event in call.events
            if event.event == "engine_speech_started" and start < event.wall_time <= end
        ]
    )
    if counted < expectation.at_least:
        return _fail(
            name,
            "harness",
            f"the engine heard {counted} stretch(es) of speech in the turn, fewer than "
            f"{expectation.at_least}: the scripted pause did not split the caller's speech",
        )
    return _pass(name, "harness", f"the engine heard {counted} stretches of speech in the turn")


def _llm_user_message(expectation: TurnText, call: _Call) -> CheckResult:
    name = f"llm_user_message:turn{expectation.turn}"
    window = call.turn_window(expectation.turn)
    if window is None:
        return _fail(name, "turn_taking", f"there was no caller turn {expectation.turn}")
    stop = window[1]
    runs = call.named("llm_run", start=stop)
    if not runs:
        return _fail(
            name, "turn_taking", f"the model was never asked after turn {expectation.turn}"
        )
    text = str(runs[0].fields.get("user_text", ""))
    missing = [word for word in expectation.contains_all if not _contains(text, word)]
    if missing:
        return _fail(name, "turn_taking", f"the model was given {text!r}, without {missing}")
    return _pass(name, "turn_taking", f"the model was given {text!r}")


def _bot_said(expectation: BotSaid, scenario: Scenario, call: _Call) -> CheckResult:
    name = f"bot_said:{expectation.after_line}"
    end = call.line_ends.get(expectation.after_line)
    if end is None:
        return _fail(name, "decision", f"the caller never finished {expectation.after_line!r}")
    ids = [line.id for line in scenario.lines]
    later = [
        call.line_starts[other]
        for other in ids[ids.index(expectation.after_line) + 1 :]
        if other in call.line_starts
    ]
    said = " ".join(
        str(event.fields.get("text", ""))
        for event in call.named("bot_text", start=end, end=later[0] if later else math.inf)
    )
    if expectation.any_of and not any(_contains(said, phrase) for phrase in expectation.any_of):
        return _fail(name, "decision", f"the bot said {said!r}, none of {list(expectation.any_of)}")
    spoken = [phrase for phrase in expectation.none_of if _contains(said, phrase)]
    if spoken:
        return _fail(name, "decision", f"the bot said {said!r}, including {spoken}")
    return _pass(name, "decision", f"the bot said {said!r}")


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.lower()).split())


def _contains(text: str, phrase: str) -> bool:
    """Whether `phrase` occurs in `text`, ignoring case and punctuation, inside words too."""
    return _normalized(phrase) in _normalized(text)
