"""What the bot reported about a call: the observer's event lines, and the verdict on a transfer.

Two sources, both read from the bot container's output. The harness's `CallEventObserver` writes
one line per pipeline event, marked with `EVENT_MARKER` and carrying JSON. The transfer verdict is
not a pipeline event -- it arrives on the control channel after the REFER -- so it comes from the
bot's own log lines, whose wording a test pins against `examples/agent_bot.py`.

An event line that does not parse is counted rather than dropped quietly: a run whose evidence was
partly unreadable should say so, not look like a bot that did less.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

EVENT_MARKER = "conversation-event "
"""The prefix of every line the observer writes. Everything after it is one JSON object."""

_TRANSFER_LINE = re.compile(
    r"transfer (?P<verdict>completed|FAILED) on call (?P<call_id>\S+): (?P<detail>.*?)(?: -- .*)?$"
)

_ENVELOPE_KEYS = frozenset({"event", "call_id", "wall_time"})


@dataclass(frozen=True)
class BotEvent:
    """One event the observer saw in the bot's pipeline."""

    event: str
    call_id: str
    wall_time: float
    fields: Mapping[str, object]


@dataclass(frozen=True)
class TransferVerdict:
    """The bot's verdict on a transfer it asked for."""

    call_id: str
    completed: bool
    detail: str


@dataclass(frozen=True)
class BotLog:
    """Everything read from the bot's output."""

    events: tuple[BotEvent, ...]
    transfers: tuple[TransferVerdict, ...]
    unreadable: int
    """Event lines that carried the marker but not a usable event."""

    def for_call(self, call_id: str) -> BotLog:
        """Keep only what belongs to the call with SIP Call-ID `call_id`."""
        return BotLog(
            events=tuple(event for event in self.events if event.call_id == call_id),
            transfers=tuple(verdict for verdict in self.transfers if verdict.call_id == call_id),
            unreadable=self.unreadable,
        )


def parse_bot_log(text: str) -> BotLog:
    """Read events and transfer verdicts out of the bot container's output."""
    events: list[BotEvent] = []
    transfers: list[TransferVerdict] = []
    unreadable = 0
    for line in text.splitlines():
        marker = line.find(EVENT_MARKER)
        if marker >= 0:
            event = _parse_event(line[marker + len(EVENT_MARKER) :])
            if event is None:
                unreadable += 1
            else:
                events.append(event)
            continue
        match = _TRANSFER_LINE.search(line)
        if match is not None:
            transfers.append(
                TransferVerdict(
                    call_id=match["call_id"],
                    completed=match["verdict"] == "completed",
                    detail=match["detail"].strip(),
                )
            )
    return BotLog(events=tuple(events), transfers=tuple(transfers), unreadable=unreadable)


def _parse_event(payload: str) -> BotEvent | None:
    try:
        record: object = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    event = record.get("event")
    call_id = record.get("call_id")
    wall_time = record.get("wall_time")
    if not isinstance(event, str) or not isinstance(call_id, str):
        return None
    if isinstance(wall_time, bool) or not isinstance(wall_time, int | float):
        return None
    fields: dict[str, object] = {
        str(key): value for key, value in record.items() if key not in _ENVELOPE_KEYS
    }
    return BotEvent(event=event, call_id=call_id, wall_time=float(wall_time), fields=fields)
