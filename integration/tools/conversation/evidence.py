"""The evidence for one call: what the capture, the caller and the bot each say happened.

Three independent witnesses, kept apart so that no one of them is taken on trust about another:
the capture on the caller's interface (SIP and the bot's audio, dissected by tshark), the caller's
own timeline (when each line started and ended, and any wait that ran out), and the bot's output
(its pipeline events and its verdict on a transfer). Times are wall-clock seconds from the same
host's kernel clock, so the three line up without translation.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.conversation.bot_events import BotLog
from tools.conversation.downlink_activity import Span
from tools.conversation.script_planner import TimelineEvent
from tools.conversation.sip_events import SipMessage


@dataclass(frozen=True)
class CallEvidence:
    """Everything one call is judged on."""

    call_id: str
    """The call's SIP Call-ID, which the bot's events are keyed on too."""
    caller_address: str
    proxy_address: str
    engine_address: str
    sip: tuple[SipMessage, ...]
    """Every SIP message of the call, from the capture, in capture order."""
    downlink_spans: tuple[Span, ...]
    """When the bot was audible to the caller, from the capture's RTP."""
    timeline: tuple[TimelineEvent, ...]
    """What the caller did: lines, segments, wait timeouts, and overlaps it noticed live."""
    bot: BotLog
    """The bot's events and transfer verdicts for this call."""
    sipp_exit_code: int
    engine_control_errors_delta: int | None
    """Control errors the engine counted during the call; None if the metric could not be read."""
    engine_sessions_after: int | None
    """Media sessions the engine still held after the call; None if it could not be read."""
