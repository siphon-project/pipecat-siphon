"""Placing a call through siphon-sip's control plane and learning how it went.

siphon-sip's `originate` with `media=True` anchors the placed leg on siphon-rtp, and when the callee
answers the engine dials the bot's WebSocket exactly as it does for a call coming in. Two things
are left to the bot, and this module is them:

- **Which dial-in is which call.** The engine's dial carries no headers and its `start` envelope
  carries only the SIP Call-ID, which arrives after the socket opens and so after a bot that speaks
  first has to know who it called and why. The channel id this module mints rides on the URI the
  engine dials instead (`placed=<channel>`), so the call is known at accept time. It is a random
  UUID, which is also what keeps a stranger from claiming the call's context with a guessed one.
- **How the call went.** Answered, rejected (with the SIP status), unanswered, or failed, from the
  events siphon sends on the channel. Every path resolves: siphon times an unanswered call out
  itself, and the guards here hang up and report a call siphon never reports on, and a call that
  was answered but never reached the bot -- a callee answering into dead air.

Nothing here imports the control SDK. The client and call handle are protocols, satisfied by a
`siphon_control.ControlClient` whose `originate` returns the placed `Call` (older releases return a
dict of ids and drop the call's events, so cannot follow it) and by anything else shaped like it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol
from urllib.parse import parse_qs, urlsplit, urlunsplit

from loguru import logger

OutcomeKind = Literal["answered", "rejected", "timed_out", "failed", "ended"]
"""How a placed call went, so far.

`answered` is the only one that is not an ending: the call goes on to end in `ended` (either side
hung up), or in `failed` if the bot never got it. `rejected`, `timed_out` and `failed` can end a
call that was never answered."""

QUERY_PARAMETER = "placed"
"""The `ws_uri` query parameter naming the placed call the engine is dialling in for."""

EventListener = Callable[[Mapping[str, Any]], object]


@dataclasses.dataclass(frozen=True)
class CallOutcome:
    """How a placed call went: the kind, and the SIP status and reason siphon gave, if any."""

    kind: OutcomeKind
    code: int | None = None
    response: str | None = None
    reason: str | None = None

    def __str__(self) -> str:
        """Say it the way a log line wants it: `rejected 486 Busy Here`, `ended (bye)`."""
        parts: list[str] = [self.kind]
        if self.code is not None:
            parts.append(str(self.code))
        if self.response:
            parts.append(self.response)
        if self.code is None and not self.response and self.reason and self.reason != self.kind:
            parts.append(f"({self.reason})")
        return " ".join(parts)


class ControlCall(Protocol):
    """The handle `originate` resolves to: the placed call's events, and a way to end it."""

    @property
    def channel_id(self) -> str:
        """The channel id the call was placed under."""
        ...

    @property
    def sip_call_id(self) -> str | None:
        """The placed leg's SIP Call-ID, which the engine's `start` envelope carries as `callId`."""
        ...

    async def next_event(self) -> Mapping[str, Any] | None:
        """Return the next `{"kind", "payload"}` event on the channel, None once it ends."""
        ...

    async def hangup(self) -> object:
        """CANCEL the call while it rings, BYE it once answered."""
        ...


class PlacingClient(Protocol):
    """A control client that can place a call: `siphon_control.ControlClient`'s `originate`."""

    async def originate(
        self,
        channel: str,
        to: str | None = None,
        *,
        media: bool,
        profile: str | None,
        ws_uri: str | None,
        from_uri: str | None,
        timeout: int | None,
    ) -> ControlCall:
        """Place a call under `channel` and resolve, once its INVITE is out, to its handle."""
        ...


def outcome_from_event(event: Mapping[str, Any]) -> CallOutcome | None:
    """Read what one control event says about how a placed call went, or None if nothing.

    The kinds are read off the event's reason, not its SIP status. siphon reports the CANCEL's 487
    when a call rings out, so the status cannot tell a callee who declined from one who never
    picked up; the reason can. Malformed events are read as far as they go and never raise.
    """
    kind = event.get("kind")
    payload = event.get("payload")
    fields: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
    code = fields.get("code")
    status = code if isinstance(code, int) and not isinstance(code, bool) else None
    response = fields.get("response")
    response_text = response if isinstance(response, str) else None

    if kind == "ChannelStateChange":
        if fields.get("state") == "answered":
            return CallOutcome("answered", code=status)
        return None
    if kind != "StasisEnd":
        return None

    reason = fields.get("reason")
    reason_text = reason if isinstance(reason, str) else None
    ending: OutcomeKind
    if reason_text == "rejected":
        ending = "rejected"
    elif reason_text == "ring timeout":
        ending = "timed_out"
    elif reason_text == "media_failed":
        ending = "failed"
    else:
        ending = "ended"
    return CallOutcome(ending, code=status, response=response_text, reason=reason_text)


def with_placed_channel(ws_uri: str, channel: str) -> str:
    """Add `placed=<channel>` to the URI the engine dials, leaving the rest exactly as written.

    The rest is siphon's to expand -- `{call_id}` and its siblings -- so it is joined as text rather
    than re-encoded.

    Raises:
        ValueError: If the URI already names a placed call.

    """
    parts = urlsplit(ws_uri)
    if QUERY_PARAMETER in parse_qs(parts.query):
        raise ValueError(f"ws_uri already carries {QUERY_PARAMETER}=: {ws_uri}")
    added = f"{QUERY_PARAMETER}={channel}"
    query = f"{parts.query}&{added}" if parts.query else added
    return urlunsplit(parts._replace(query=query))


class PlacedCall:
    """One call placed through the control plane, from the originate to its outcome."""

    def __init__(
        self,
        channel: str,
        to: str,
        context: Mapping[str, str],
        dial_in_timeout_seconds: float,
        release: Callable[[str], None],
    ) -> None:
        """Reserve the call under `channel`. `release` drops it from its registry once it ends."""
        self.channel = channel
        self.to = to
        self.context = dict(context)
        self._dial_in_timeout_seconds = dial_in_timeout_seconds
        self._release = release
        self._call: ControlCall | None = None
        loop = asyncio.get_running_loop()
        self._answered: asyncio.Future[CallOutcome] = loop.create_future()
        self._ended: asyncio.Future[CallOutcome] = loop.create_future()
        self._dialled_in = False
        self._ring_guard: asyncio.Task[None] | None = None
        self._dial_in_guard: asyncio.Task[None] | None = None
        self._pump: asyncio.Task[None] | None = None

    @property
    def call(self) -> ControlCall | None:
        """The control handle, once `originate` has returned it."""
        return self._call

    @property
    def sip_call_id(self) -> str | None:
        """The placed leg's SIP Call-ID, once known: the key a handed-over call is joined on too."""
        return self._call.sip_call_id if self._call is not None else None

    @property
    def outcome(self) -> CallOutcome | None:
        """The latest outcome: the ending if there is one, else the answer, else None."""
        if self._ended.done():
            return self._ended.result()
        if self._answered.done():
            return self._answered.result()
        return None

    async def answered(self) -> CallOutcome:
        """Wait for the answer, or for whatever ended the call without one."""
        return await asyncio.shield(self._answered)

    async def ended(self) -> CallOutcome:
        """Wait for the call to end, however it does."""
        return await asyncio.shield(self._ended)

    def mark_dialled_in(self) -> None:
        """Record that the engine reached the bot for this call; the dial-in guard stands down."""
        self._dialled_in = True
        if self._dial_in_guard is not None:
            self._dial_in_guard.cancel()

    def _start(
        self, call: ControlCall, ring_deadline_seconds: float, on_event: EventListener | None
    ) -> None:
        """Take over the call's events and start the guard for a call nobody reports on."""
        self._call = call
        self._pump = asyncio.create_task(self._read_events(call, on_event))
        self._ring_guard = asyncio.create_task(self._guard_ringing(ring_deadline_seconds))

    async def _read_events(self, call: ControlCall, on_event: EventListener | None) -> None:
        """Read the channel's events until it ends. This task is the only reader of the stream."""
        try:
            while (event := await call.next_event()) is not None:
                if on_event is not None:
                    try:
                        on_event(event)
                    except Exception:
                        logger.exception(f"placed call {self.channel}: event listener failed")
                outcome = outcome_from_event(event)
                if outcome is None:
                    continue
                if outcome.kind == "answered":
                    self._on_answered(outcome)
                    continue
                # `StasisEnd` is the channel's last event that reaches a call, so the read is over.
                self._finish(outcome)
                return
        except Exception as error:
            logger.error(f"placed call {self.channel}: reading its control events failed: {error}")
            self._finish(CallOutcome("failed", reason=f"control events failed: {error}"))
            return
        # The SDK ends the stream after `StasisEnd`; one that ends first lost the call's ending,
        # and a call whose ending nobody will report is not one to wait on.
        self._finish(
            CallOutcome("failed", reason="the control stream closed before the call ended")
        )

    def _on_answered(self, outcome: CallOutcome) -> None:
        if self._answered.done():
            return
        self._answered.set_result(outcome)
        if self._ring_guard is not None:
            self._ring_guard.cancel()
        logger.info(f"placed call {self.channel} to {self.to}: {outcome}")
        if not self._dialled_in:
            self._dial_in_guard = asyncio.create_task(self._guard_dial_in())

    def _finish(self, outcome: CallOutcome) -> None:
        if self._ended.done():
            return
        if not self._answered.done():
            self._answered.set_result(outcome)
        self._ended.set_result(outcome)
        for guard in (self._ring_guard, self._dial_in_guard):
            if guard is not None and guard is not asyncio.current_task():
                guard.cancel()
        self._release(self.channel)
        logger.info(f"placed call {self.channel} to {self.to} over: {outcome}")

    async def _guard_ringing(self, deadline_seconds: float) -> None:
        await asyncio.sleep(deadline_seconds)
        if self._answered.done():
            return
        logger.error(
            f"placed call {self.channel} to {self.to}: no answer and no ending from siphon after "
            f"{deadline_seconds:g}s, past its own ring timeout -- hanging it up"
        )
        await self._hang_up()
        self._finish(CallOutcome("timed_out", reason="siphon reported no outcome"))

    async def _guard_dial_in(self) -> None:
        await asyncio.sleep(self._dial_in_timeout_seconds)
        if self._dialled_in or self._ended.done():
            return
        logger.error(
            f"placed call {self.channel} to {self.to} was answered, but the engine has not dialled "
            f"the bot after {self._dial_in_timeout_seconds:g}s -- hanging up rather than leaving "
            "the callee on a silent line"
        )
        await self._hang_up()
        self._finish(CallOutcome("failed", reason="answered, but the engine never dialled the bot"))

    async def _hang_up(self) -> None:
        if self._call is None:
            return
        try:
            await self._call.hangup()
        except Exception as error:
            # The call may already be gone; either way the outcome is reported by the caller.
            logger.error(f"placed call {self.channel}: hangup failed: {error}")


class PlacedCalls:
    """The calls a bot has placed and not yet matched to the engine's dial-in."""

    def __init__(
        self, *, ring_guard_seconds: float = 10.0, dial_in_timeout_seconds: float = 10.0
    ) -> None:
        """Set the two guards, each of which hangs a call up when it runs out.

        `ring_guard_seconds` is how long past its own ring timeout a call may go unreported by
        siphon. `dial_in_timeout_seconds` is how long an answered call may wait for the engine to
        reach the bot.
        """
        self._ring_guard_seconds = ring_guard_seconds
        self._dial_in_timeout_seconds = dial_in_timeout_seconds
        self._pending: dict[str, PlacedCall] = {}

    async def place(
        self,
        client: PlacingClient,
        *,
        to: str,
        ws_uri: str,
        from_uri: str | None = None,
        profile: str | None = None,
        ring_timeout: int = 30,
        context: Mapping[str, str] | None = None,
        on_event: EventListener | None = None,
    ) -> PlacedCall:
        """Place a call to `to` whose media the engine brings to `ws_uri` when it is answered.

        `ws_uri` is the bot's own address as the engine reaches it, siphon's placeholders and all;
        `placed=<channel>` is added to it. `ring_timeout` is siphon's, in whole seconds. `context`
        travels with the call to whoever claims its dial-in. `on_event` sees every control event
        on the call, since this is the only reader of its stream.

        Raises:
            ValueError: If `ring_timeout` is under one second (siphon reads 0 as "ring forever")
                or `ws_uri` already names a placed call.
            TypeError: If the client's `originate` returns no call handle, as control SDKs from
                before it returned one do.
            Exception: Whatever the client raises when siphon refuses the originate; nothing is
                left pending.

        """
        if ring_timeout < 1:
            raise ValueError(f"ring_timeout must be at least 1 second, not {ring_timeout}")
        channel = f"placed-{uuid.uuid4().hex}"
        dialled = with_placed_channel(ws_uri, channel)
        placed = PlacedCall(
            channel, to, context or {}, self._dial_in_timeout_seconds, self._release
        )
        # Pending before the INVITE goes out: an instant answer can bring the engine's dial-in
        # ahead of the originate's reply, and it must find its call.
        self._pending[channel] = placed
        try:
            call = await client.originate(
                channel,
                to,
                media=True,
                profile=profile,
                ws_uri=dialled,
                from_uri=from_uri,
                timeout=ring_timeout,
            )
        except BaseException as error:
            self._release(channel)
            logger.error(f"placing a call to {to} failed: {error}")
            raise
        if not callable(getattr(call, "next_event", None)) or not callable(
            getattr(call, "hangup", None)
        ):
            self._release(channel)
            message = (
                f"originate returned a {type(call).__name__}, not the placed call: this control "
                f"client cannot follow a call it places. The call to {to} rings on until siphon's "
                "ring timeout; upgrade siphon-control to a release whose originate returns a Call"
            )
            logger.error(message)
            raise TypeError(message)
        placed._start(call, ring_timeout + self._ring_guard_seconds, on_event)
        logger.info(f"placed call {channel} to {to}: calling, sip call id {call.sip_call_id}")
        return placed

    def claim(self, query: Mapping[str, str]) -> PlacedCall | None:
        """Match a dial-in's query to the call it belongs to, once. None for anything else."""
        channel = query.get(QUERY_PARAMETER)
        if channel is None:
            return None
        placed = self._pending.pop(channel, None)
        if placed is None:
            logger.warning(f"dial-in names placed call {channel}, which is not pending here")
            return None
        placed.mark_dialled_in()
        return placed

    def pending(self) -> list[PlacedCall]:
        """List the calls placed and neither dialled in for nor ended."""
        return list(self._pending.values())

    def _release(self, channel: str) -> None:
        self._pending.pop(channel, None)


__all__ = [
    "QUERY_PARAMETER",
    "CallOutcome",
    "ControlCall",
    "EventListener",
    "OutcomeKind",
    "PlacedCall",
    "PlacedCalls",
    "PlacingClient",
    "outcome_from_event",
    "with_placed_channel",
]
