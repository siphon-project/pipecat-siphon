"""What the bot reported about a call: its event lines, and the verdict on a transfer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.conversation.bot_events import TransferVerdict, parse_bot_log

CALL = "conversation-goodbye-1-1@172.28.8.40"
OTHER = "conversation-goodbye-2-1@172.28.8.40"


def _event(event: str, *, call_id: str = CALL, wall_time: float = 100.0, **fields: object) -> str:
    record = {"event": event, "call_id": call_id, "wall_time": wall_time, **fields}
    return "conversation-event " + json.dumps(record)


class TestEventLines:
    def test_event_lines_are_read_in_order_among_ordinary_log_lines(self) -> None:
        text = "\n".join(
            [
                "2026-09-15 10:41:19.167 | INFO     | pipecat:<module>:54 - Pipecat 1.10.0",
                _event("user_turn_started", wall_time=10.0),
                "2026-09-15 10:41:19.500 | DEBUG    | pipecat.services.whisper.stt - loaded",
                _event("transcript_accepted", wall_time=10.5, text="My name is Karen."),
                _event("tool_called", wall_time=11.0, name="transfer_call"),
            ]
        )

        log = parse_bot_log(text)

        assert [event.event for event in log.events] == [
            "user_turn_started",
            "transcript_accepted",
            "tool_called",
        ]
        assert log.events[1].fields["text"] == "My name is Karen."
        assert log.events[2].wall_time == 11.0
        assert log.unreadable == 0

    def test_a_line_prefixed_by_the_container_log_is_still_read(self) -> None:
        log = parse_bot_log("bot-1  | " + _event("bot_started_speaking"))

        assert [event.event for event in log.events] == ["bot_started_speaking"]

    @pytest.mark.parametrize(
        "line",
        [
            "conversation-event {not json",
            "conversation-event " + json.dumps(["a", "list"]),
            "conversation-event " + json.dumps({"event": "tool_called"}),
            "conversation-event " + json.dumps({"event": 3, "call_id": CALL, "wall_time": 1.0}),
            "conversation-event "
            + json.dumps({"event": "tool_called", "call_id": CALL, "wall_time": "soon"}),
        ],
        ids=["not json", "not an object", "no call or time", "event not text", "time not a number"],
    )
    def test_an_unreadable_event_line_is_counted_not_trusted(self, line: str) -> None:
        log = parse_bot_log(line)

        assert log.events == ()
        assert log.unreadable == 1

    def test_events_are_kept_to_one_call(self) -> None:
        text = "\n".join(
            [
                _event("tool_called", call_id=CALL, name="end_call"),
                _event("tool_called", call_id=OTHER, name="transfer_call"),
            ]
        )

        log = parse_bot_log(text).for_call(CALL)

        assert [event.fields["name"] for event in log.events] == ["end_call"]


class TestTransferVerdicts:
    def test_a_completed_transfer(self) -> None:
        line = (
            "2026-09-15 10:00:05.844 | INFO     | __main__:_handle_call:590 - "
            f"transfer completed on call {CALL}: stage=transferred status=None reason=OK"
        )

        log = parse_bot_log(line)

        assert log.transfers == (
            TransferVerdict(
                call_id=CALL, completed=True, detail="stage=transferred status=None reason=OK"
            ),
        )

    def test_a_failed_transfer(self) -> None:
        line = (
            "2026-09-15 10:00:05.844 | WARNING  | __main__:_handle_call:592 - transfer FAILED on "
            f"call {CALL}: stage=rejected status=486 reason=Busy Here -- the caller was told they "
            "were being put through"
        )

        log = parse_bot_log(line)

        assert log.transfers == (
            TransferVerdict(
                call_id=CALL, completed=False, detail="stage=rejected status=486 reason=Busy Here"
            ),
        )

    def test_verdicts_are_kept_to_one_call(self) -> None:
        text = "\n".join(
            [
                f"transfer completed on call {OTHER}: stage=transferred",
                f"transfer completed on call {CALL}: stage=transferred",
            ]
        )

        assert [verdict.call_id for verdict in parse_bot_log(text).for_call(CALL).transfers] == [
            CALL
        ]

    def test_the_bot_still_logs_the_lines_this_reads(self) -> None:
        """The verdict comes from the bot's own log lines, so their wording is pinned here."""
        source = (Path(__file__).resolve().parent.parent / "examples" / "agent_bot.py").read_text(
            encoding="utf-8"
        )

        assert 'f"transfer completed on call {sip_call_id}: {detail}"' in source
        assert 'f"transfer FAILED on call {sip_call_id}: {detail} "' in source
