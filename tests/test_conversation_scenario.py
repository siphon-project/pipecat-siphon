"""The scenario files: what a call says, and what must come of it, loaded strictly."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from tools.conversation.scenario import (
    BotSaid,
    Outcome,
    Scenario,
    ScenarioError,
    TurnCount,
    TurnText,
    parse_scenario,
)

PAUSE_BOUNDS_MS = (600, 1200)
SOURCE = "pause_mid_sentence.yaml"


def _valid() -> dict[str, Any]:
    return {
        "name": "pause_mid_sentence",
        "description": "The caller stops after 'my name is' and then gives the name.",
        "lines": [
            {
                "id": "ask_for_manager_with_pause",
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
            "outcome": {"kind": "refer", "refer_to": "sip:manager@harness.example"},
            "tools": {"called": ["transfer_call"], "not_called": ["end_call"]},
            "user_turns": {"exactly": 1},
            "engine_speech_segments_in_turn": [{"turn": 1, "at_least": 2}],
            "llm_user_message": [{"turn": 1, "contains_all": ["name", "karen"]}],
            "bot_said": [
                {"after_line": "ask_for_manager_with_pause", "any_of": ["putting you through"]}
            ],
            "bot_audio_during_line": "forbidden",
        },
    }


def _parse(document: object) -> Scenario:
    return parse_scenario(document, source=SOURCE, pause_bounds_ms=PAUSE_BOUNDS_MS)


class TestAValidScenario:
    def test_every_field_arrives(self) -> None:
        scenario = _parse(_valid())

        assert scenario.name == "pause_mid_sentence"
        (line,) = scenario.lines
        assert line.id == "ask_for_manager_with_pause"
        assert line.wait_for == "greeting"
        assert [segment.text for segment in line.segments] == [
            "Could I speak to the manager, please? My name is",
            "Karen.",
        ]
        assert [segment.pause_after_ms for segment in line.segments] == [900, 0]
        assert line.recognition_keywords == ("manager", "karen")

        expectations = scenario.expectations
        assert expectations.outcome == Outcome(kind="refer", refer_to="sip:manager@harness.example")
        assert expectations.tools.called == ("transfer_call",)
        assert expectations.tools.not_called == ("end_call",)
        assert expectations.user_turns == 1
        assert expectations.engine_speech_segments_in_turn == (TurnCount(turn=1, at_least=2),)
        assert expectations.llm_user_message == (TurnText(turn=1, contains_all=("name", "karen")),)
        assert expectations.bot_said == (
            BotSaid(
                after_line="ask_for_manager_with_pause",
                any_of=("putting you through",),
                none_of=(),
            ),
        )
        assert expectations.bot_audio_during_line == "forbidden"

    def test_expectations_left_out_are_not_checked(self) -> None:
        document = _valid()
        document["expectations"] = {"outcome": {"kind": "bot_bye"}}

        expectations = _parse(document).expectations

        assert expectations.outcome == Outcome(kind="bot_bye", refer_to=None)
        assert expectations.tools.called == ()
        assert expectations.tools.not_called == ()
        assert expectations.user_turns is None
        assert expectations.engine_speech_segments_in_turn == ()
        assert expectations.llm_user_message == ()
        assert expectations.bot_said == ()
        assert expectations.bot_audio_during_line == "interference"


def _set(path: tuple[Any, ...], value: object) -> Callable[[dict[str, Any]], None]:
    """Build a mutation that sets the value at `path` inside a scenario document."""

    def mutate(document: dict[str, Any]) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _duplicate_the_line(document: dict[str, Any]) -> None:
    document["lines"].append(copy.deepcopy(document["lines"][0]))


MISTAKES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("an unknown top-level key", _set(("expectation",), {}), "unknown key 'expectation'"),
    (
        "an unknown key in a segment",
        _set(("lines", 0, "segments", 0, "pause_ms"), 900),
        "lines[0].segments[0]: unknown key 'pause_ms'",
    ),
    ("a duplicate line id", _duplicate_the_line, "duplicate line id"),
    (
        "a pause after the last segment",
        _set(("lines", 0, "segments", 1, "pause_after_ms"), 600),
        "lines[0].segments[1].pause_after_ms",
    ),
    (
        "a pause off the frame grid",
        _set(("lines", 0, "segments", 0, "pause_after_ms"), 910),
        "multiple of 20",
    ),
    (
        "a pause the hold would not survive",
        _set(("lines", 0, "segments", 0, "pause_after_ms"), 2500),
        "between 600 and 1200",
    ),
    (
        "a transfer target on a hangup",
        _set(("expectations", "outcome"), {"kind": "bot_bye", "refer_to": "sip:x@y.example"}),
        "expectations.outcome.refer_to",
    ),
    (
        "a transfer without a target",
        _set(("expectations", "outcome"), {"kind": "refer"}),
        "expectations.outcome.refer_to",
    ),
    ("an unknown outcome", _set(("expectations", "outcome", "kind"), "hangup"), "kind"),
    ("an unknown wait", _set(("lines", 0, "wait_for"), "silence"), "wait_for"),
    (
        "a phrase check after a line that does not exist",
        _set(("expectations", "bot_said", 0, "after_line"), "nope"),
        "after_line",
    ),
    (
        "a turn the call does not have",
        _set(("expectations", "engine_speech_segments_in_turn", 0, "turn"), 3),
        "turn",
    ),
    ("empty caller text", _set(("lines", 0, "segments", 0, "text"), ""), "text"),
    ("no lines at all", _set(("lines",), []), "lines"),
    (
        "a word list written as one string",
        _set(("lines", 0, "recognition_keywords"), "manager"),
        "recognition_keywords",
    ),
    (
        "an unknown audio policy",
        _set(("expectations", "bot_audio_during_line"), "sometimes"),
        "bot_audio_during_line",
    ),
]


class TestMistakesAreNamed:
    """Every mistake in a scenario file stops the run, naming the file and the key.

    siphon-sip drops an unknown key without a word. A harness that did the same would pass a
    scenario whose expectation was never read.
    """

    @pytest.mark.parametrize(
        ("mutate", "fragment"),
        [(mutate, fragment) for _, mutate, fragment in MISTAKES],
        ids=[name for name, _, _ in MISTAKES],
    )
    def test_the_mistake_is_refused(
        self, mutate: Callable[[dict[str, Any]], None], fragment: str
    ) -> None:
        document = _valid()
        mutate(document)

        with pytest.raises(ScenarioError) as raised:
            _parse(document)

        assert str(raised.value).startswith(f"{SOURCE}: ")
        assert fragment in str(raised.value)

    def test_a_document_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ScenarioError, match="expected a mapping"):
            _parse(["not", "a", "scenario"])
