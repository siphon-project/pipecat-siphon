"""Scenario files: what a scripted caller says, and what must come of the call.

A scenario is read strictly. Every key is either understood or refused, and every refusal names
the file and the key path, because a harness that skipped a key it did not recognise would pass a
scenario whose expectation was never checked -- the failure mode siphon-sip has with its own
configuration, and exactly the one a test harness cannot afford.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

# PyYAML ships no type information. Nothing it returns is trusted: `parse_scenario` checks the type
# of every value it reads.
import yaml  # type: ignore[import-untyped]

from tools.conversation.yaml_fields import ConfigError, Fields

WaitFor = Literal["greeting", "reply"]
OutcomeKind = Literal["bot_bye", "refer", "none"]
AudioPolicy = Literal["interference", "forbidden"]

FRAME_MILLISECONDS = 20
"""The caller's ptime. A scripted pause is a whole number of silent frames."""


class ScenarioError(ConfigError):
    """A scenario that cannot be run as written. The message names the file and the key path."""


@dataclass(frozen=True)
class Segment:
    """One stretch of caller speech, and the silence the caller leaves after it."""

    text: str
    pause_after_ms: int


@dataclass(frozen=True)
class Line:
    """What the caller says in one turn, and what it waits for before saying it."""

    id: str
    wait_for: WaitFor
    segments: tuple[Segment, ...]
    recognition_keywords: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    """How the call must end: the bot hangs up, the bot transfers, or neither."""

    kind: OutcomeKind
    refer_to: str | None


@dataclass(frozen=True)
class ToolExpectations:
    """Tools the model must call, and tools it must not."""

    called: tuple[str, ...]
    not_called: tuple[str, ...]


@dataclass(frozen=True)
class TurnCount:
    """A minimum count of engine speech segments inside one caller turn."""

    turn: int
    at_least: int


@dataclass(frozen=True)
class TurnText:
    """Words the model must have been given for one caller turn."""

    turn: int
    contains_all: tuple[str, ...]


@dataclass(frozen=True)
class BotSaid:
    """Phrases the bot must, or must not, say in its reply to one line."""

    after_line: str
    any_of: tuple[str, ...]
    none_of: tuple[str, ...]


@dataclass(frozen=True)
class Expectations:
    """Everything a run of the scenario is checked against. Left out means not checked."""

    outcome: Outcome
    tools: ToolExpectations
    user_turns: int | None
    engine_speech_segments_in_turn: tuple[TurnCount, ...]
    llm_user_message: tuple[TurnText, ...]
    bot_said: tuple[BotSaid, ...]
    bot_audio_during_line: AudioPolicy


@dataclass(frozen=True)
class Scenario:
    """A scripted call."""

    name: str
    description: str
    lines: tuple[Line, ...]
    expectations: Expectations


def load_scenario(path: Path, *, pause_bounds_ms: tuple[int, int]) -> Scenario:
    """Read and check the scenario file at `path`."""
    with path.open(encoding="utf-8") as handle:
        document: object = yaml.safe_load(handle)
    return parse_scenario(document, source=path.name, pause_bounds_ms=pause_bounds_ms)


def parse_scenario(document: object, *, source: str, pause_bounds_ms: tuple[int, int]) -> Scenario:
    """Check a scenario document read from `source`.

    Args:
        document: The document as YAML loads it.
        source: The file name errors are reported against.
        pause_bounds_ms: The shortest and longest pause between two segments. Shorter, and the
            engine may not end the first stretch of speech; longer, and the bot's hold for an
            unfinished sentence runs out, so the pause scenario would test something else.

    Raises:
        ScenarioError: For anything missing, unknown, of the wrong type or inconsistent.

    """
    root = Fields(document, source=source, path="", error_type=ScenarioError)
    name = root.text("name")
    description = root.optional_text("description") or ""

    line_fields = root.mappings("lines")
    if not line_fields:
        raise root.error("at least one line is needed", "lines")
    lines: list[Line] = []
    for fields in line_fields:
        line = _parse_line(fields, pause_bounds_ms)
        if any(earlier.id == line.id for earlier in lines):
            raise fields.error(f"duplicate line id {line.id!r}", "id")
        lines.append(line)

    expectations = _parse_expectations(root.mapping("expectations"), tuple(lines))
    root.finish()
    return Scenario(
        name=name, description=description, lines=tuple(lines), expectations=expectations
    )


def _parse_line(fields: Fields, pause_bounds_ms: tuple[int, int]) -> Line:
    line_id = fields.text("id")
    wait_for = cast(WaitFor, fields.choice("wait_for", get_args(WaitFor)))

    segment_fields = fields.mappings("segments")
    if not segment_fields:
        raise fields.error("at least one segment is needed", "segments")
    segments: list[Segment] = []
    for index, segment in enumerate(segment_fields):
        text = segment.text("text")
        pause = segment.optional_integer("pause_after_ms")
        last = index == len(segment_fields) - 1
        if last and pause is not None:
            raise segment.error("only a segment followed by another can pause", "pause_after_ms")
        if not last:
            if pause is None:
                raise segment.error("a segment followed by another needs a pause", "pause_after_ms")
            if pause % FRAME_MILLISECONDS:
                raise segment.error(
                    f"must be a multiple of {FRAME_MILLISECONDS} ms, the caller's frame",
                    "pause_after_ms",
                )
            low, high = pause_bounds_ms
            if not low <= pause <= high:
                raise segment.error(f"must be between {low} and {high} ms", "pause_after_ms")
        segment.finish()
        segments.append(Segment(text=text, pause_after_ms=pause or 0))

    keywords = fields.texts("recognition_keywords")
    fields.finish()
    return Line(
        id=line_id, wait_for=wait_for, segments=tuple(segments), recognition_keywords=keywords
    )


def _parse_expectations(fields: Fields, lines: tuple[Line, ...]) -> Expectations:
    outcome_fields = fields.mapping("outcome")
    kind = cast(OutcomeKind, outcome_fields.choice("kind", get_args(OutcomeKind)))
    refer_to = outcome_fields.optional_text("refer_to")
    if kind == "refer" and refer_to is None:
        raise outcome_fields.error("a refer outcome needs the transfer target", "refer_to")
    if kind != "refer" and refer_to is not None:
        raise outcome_fields.error("only a refer outcome names a transfer target", "refer_to")
    outcome_fields.finish()

    tools = ToolExpectations(called=(), not_called=())
    tools_fields = fields.optional_mapping("tools")
    if tools_fields is not None:
        tools = ToolExpectations(
            called=tools_fields.texts("called"), not_called=tools_fields.texts("not_called")
        )
        tools_fields.finish()

    user_turns: int | None = None
    turns_fields = fields.optional_mapping("user_turns")
    if turns_fields is not None:
        user_turns = turns_fields.positive_integer("exactly")
        turns_fields.finish()

    segment_counts: list[TurnCount] = []
    for item in fields.mappings("engine_speech_segments_in_turn"):
        segment_counts.append(
            TurnCount(turn=_turn(item, lines), at_least=item.positive_integer("at_least"))
        )
        item.finish()

    user_messages: list[TurnText] = []
    for item in fields.mappings("llm_user_message"):
        turn = _turn(item, lines)
        words = item.texts("contains_all")
        if not words:
            raise item.error("expected at least one word", "contains_all")
        user_messages.append(TurnText(turn=turn, contains_all=words))
        item.finish()

    phrases: list[BotSaid] = []
    for item in fields.mappings("bot_said"):
        after_line = item.text("after_line")
        if not any(line.id == after_line for line in lines):
            raise item.error(f"no line {after_line!r} in this scenario", "after_line")
        any_of = item.texts("any_of")
        none_of = item.texts("none_of")
        if not any_of and not none_of:
            raise item.error("expected any_of, none_of or both")
        phrases.append(BotSaid(after_line=after_line, any_of=any_of, none_of=none_of))
        item.finish()

    policy = cast(
        AudioPolicy,
        fields.choice("bot_audio_during_line", get_args(AudioPolicy), default="interference"),
    )
    fields.finish()
    return Expectations(
        outcome=Outcome(kind=kind, refer_to=refer_to),
        tools=tools,
        user_turns=user_turns,
        engine_speech_segments_in_turn=tuple(segment_counts),
        llm_user_message=tuple(user_messages),
        bot_said=tuple(phrases),
        bot_audio_during_line=policy,
    )


def _turn(fields: Fields, lines: tuple[Line, ...]) -> int:
    """Read a caller turn number, which cannot exceed the number of lines: one turn per line."""
    turn = fields.positive_integer("turn")
    if turn > len(lines):
        raise fields.error(f"the scenario has {len(lines)} line(s), so no turn {turn}", "turn")
    return turn
