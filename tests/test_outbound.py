"""Tests for `pipecat_siphon.outbound`: placing a call and learning how it went.

The control client is faked, with the call's events taken from `control_event_fixtures`, which
are transcribed from siphon-sip's source rather than from the mapping under test.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from control_event_fixtures import (
    ANSWERED,
    BUSY,
    FAR_END_HUNG_UP,
    MEDIA_FAILED,
    RING_TIMEOUT,
    RINGING,
)
from pipecat_siphon import CallOutcome, PlacedCalls, outcome_from_event
from pipecat_siphon.outbound import with_placed_channel

# Long enough not to race the event loop, short enough that a guard test is quick.
GUARD_SECONDS = 0.05
SETTLE_SECONDS = 0.2


class _FakeCall:
    """The handle `originate` resolves to: events pushed by the test, hangups recorded."""

    def __init__(self, channel_id: str, sip_call_id: str | None) -> None:
        self.channel_id = channel_id
        self.sip_call_id = sip_call_id
        self.hangups = 0
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def push(self, *events: dict[str, Any] | None) -> None:
        for event in events:
            self._events.put_nowait(event)

    async def next_event(self) -> dict[str, Any] | None:
        return await self._events.get()

    async def hangup(self) -> None:
        self.hangups += 1


class _FakeClient:
    """Records each `originate` and hands back a `_FakeCall`, or raises `error`."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.originated: list[dict[str, Any]] = []
        self.calls: list[_FakeCall] = []

    async def originate(self, channel: str, to: str | None = None, **options: Any) -> _FakeCall:
        self.originated.append({"channel": channel, "to": to, **options})
        if self.error is not None:
            raise self.error
        call = _FakeCall(channel, f"sip-{len(self.calls)}@host")
        self.calls.append(call)
        return call


def _registry() -> PlacedCalls:
    return PlacedCalls(ring_guard_seconds=GUARD_SECONDS, dial_in_timeout_seconds=GUARD_SECONDS)


class TestOutcomeFromEvent:
    """What each event siphon sends for a placed call says about how it went."""

    def test_the_answer(self) -> None:
        assert outcome_from_event(ANSWERED) == CallOutcome("answered", code=200)

    def test_a_provisional_is_not_an_outcome(self) -> None:
        assert outcome_from_event(RINGING) is None

    def test_a_rejection_carries_its_status(self) -> None:
        outcome = outcome_from_event(BUSY)
        assert outcome == CallOutcome("rejected", code=486, response="Busy Here", reason="rejected")
        assert str(outcome) == "rejected 486 Busy Here"

    def test_the_ring_timeout_is_read_from_the_reason_not_the_code(self) -> None:
        """A rung-out call carries the CANCEL's 487, not a 408.

        So the code cannot tell a callee who declined from one who never picked up.
        """
        outcome = outcome_from_event(RING_TIMEOUT)
        assert outcome is not None
        assert outcome.kind == "timed_out"
        assert outcome.code == 487

    def test_a_media_failure_is_a_failure(self) -> None:
        outcome = outcome_from_event(MEDIA_FAILED)
        assert outcome is not None
        assert outcome.kind == "failed"

    def test_any_other_end_is_an_ending(self) -> None:
        assert outcome_from_event(FAR_END_HUNG_UP) == CallOutcome("ended", reason="bye")

    @pytest.mark.parametrize(
        "event",
        [
            {},
            {"kind": "StasisEnd"},
            {"kind": "StasisEnd", "payload": None},
            {"kind": "StasisEnd", "payload": ["rejected"]},
            {"kind": "ChannelStateChange", "payload": {"state": 7}},
            {"kind": 3, "payload": {}},
        ],
    )
    def test_a_malformed_event_is_not_an_outcome_and_never_raises(
        self, event: dict[str, Any]
    ) -> None:
        outcome = outcome_from_event(event)
        assert outcome is None or outcome.kind == "ended"

    def test_a_code_that_is_not_a_number_is_dropped(self) -> None:
        outcome = outcome_from_event(
            {"kind": "StasisEnd", "payload": {"reason": "rejected", "code": "486"}}
        )
        assert outcome == CallOutcome("rejected", reason="rejected")


class TestWithPlacedChannel:
    """The channel id rides on the URI the engine dials, so the dial-in names its call."""

    def test_it_joins_an_existing_query_and_leaves_siphon_placeholders_alone(self) -> None:
        assert (
            with_placed_channel("ws://198.51.100.10:9001/stream?call={call_id}", "placed-1")
            == "ws://198.51.100.10:9001/stream?call={call_id}&placed=placed-1"
        )

    def test_it_starts_a_query_when_there_is_none(self) -> None:
        assert (
            with_placed_channel("ws://198.51.100.10:9001/stream", "placed-1")
            == "ws://198.51.100.10:9001/stream?placed=placed-1"
        )

    def test_a_uri_that_already_names_a_placed_call_is_refused(self) -> None:
        with pytest.raises(ValueError, match="placed"):
            with_placed_channel("ws://198.51.100.10:9001/?placed=other", "placed-1")


class TestPlace:
    """Placing a call: what goes to siphon, and what comes back."""

    async def test_it_originates_an_anchored_call_naming_its_own_channel(self) -> None:
        client = _FakeClient()
        placed = await _registry().place(
            client,
            to="sip:+15550100@trunk.example",
            ws_uri="ws://198.51.100.10:9001/stream?call={call_id}",
            from_uri="sip:bot@pbx.example",
            profile="agent",
            ring_timeout=20,
            context={"reason": "a callback"},
        )

        (sent,) = client.originated
        assert sent["channel"] == placed.channel
        assert placed.channel.startswith("placed-")
        assert sent == {
            "channel": placed.channel,
            "to": "sip:+15550100@trunk.example",
            "media": True,
            "profile": "agent",
            "ws_uri": f"ws://198.51.100.10:9001/stream?call={{call_id}}&placed={placed.channel}",
            "from_uri": "sip:bot@pbx.example",
            "timeout": 20,
        }
        assert placed.sip_call_id == "sip-0@host"
        assert placed.context == {"reason": "a callback"}
        assert placed.to == "sip:+15550100@trunk.example"

    async def test_two_calls_get_two_channels(self) -> None:
        client = _FakeClient()
        registry = _registry()
        first = await registry.place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        second = await registry.place(client, to="sip:b@example.com", ws_uri="ws://192.0.2.1/")
        assert first.channel != second.channel

    async def test_an_originate_siphon_refuses_raises_and_leaves_nothing_pending(self) -> None:
        registry = _registry()
        refused = RuntimeError("not_found: no route")
        with pytest.raises(RuntimeError, match="no route"):
            await registry.place(
                _FakeClient(error=refused), to="sip:a@example.com", ws_uri="ws://192.0.2.1/"
            )
        assert registry.pending() == []

    async def test_a_ring_timeout_siphon_reads_as_forever_is_refused(self) -> None:
        """Siphon takes `timeout: 0` as no ring timeout at all: a call that could ring forever."""
        client = _FakeClient()
        with pytest.raises(ValueError, match="ring_timeout"):
            await _registry().place(
                client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/", ring_timeout=0
            )
        assert client.originated == []

    async def test_a_client_whose_originate_returns_no_call_is_refused_loudly(self) -> None:
        """Control SDKs before the placed `Call` return a dict of ids and drop its events."""

        class _OldClient:
            async def originate(self, channel: str, to: str | None = None, **_: Any) -> Any:
                return {"channel": channel, "call_id": "uuid", "sip_call_id": "sip@host"}

        registry = _registry()
        with pytest.raises(TypeError, match="siphon-control"):
            await registry.place(_OldClient(), to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        assert registry.pending() == []


class TestOutcome:
    """A placed call always resolves: answered, rejected, timed out, failed or ended."""

    async def test_answered_then_the_far_end_hangs_up(self) -> None:
        client = _FakeClient()
        registry = _registry()
        placed = await registry.place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        call = client.calls[0]

        call.push(RINGING, ANSWERED)
        assert await placed.answered() == CallOutcome("answered", code=200)
        assert registry.claim({"placed": placed.channel}) is placed
        call.push(FAR_END_HUNG_UP, None)

        assert await placed.ended() == CallOutcome("ended", reason="bye")
        assert call.hangups == 0

    async def test_busy(self) -> None:
        client = _FakeClient()
        placed = await _registry().place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        client.calls[0].push(RINGING, BUSY, None)

        answered = await placed.answered()
        assert str(answered) == "rejected 486 Busy Here"
        assert await placed.ended() == answered

    async def test_no_answer(self) -> None:
        client = _FakeClient()
        placed = await _registry().place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        client.calls[0].push(RINGING, RING_TIMEOUT, None)

        assert (await placed.answered()).kind == "timed_out"

    async def test_every_event_reaches_the_listener(self) -> None:
        client = _FakeClient()
        seen: list[Mapping[str, Any]] = []
        placed = await _registry().place(
            client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/", on_event=seen.append
        )
        client.calls[0].push(RINGING, BUSY, None)
        await placed.ended()
        assert seen == [RINGING, BUSY]

    async def test_a_listener_that_raises_does_not_stop_the_call_resolving(self) -> None:
        def broken(_event: Mapping[str, Any]) -> None:
            raise RuntimeError("listener bug")

        client = _FakeClient()
        placed = await _registry().place(
            client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/", on_event=broken
        )
        client.calls[0].push(BUSY, None)
        assert (await placed.ended()).kind == "rejected"

    async def test_siphon_saying_nothing_is_timed_out_and_hung_up(self) -> None:
        """Siphon times the ring out itself; this is the guard for when it never says so."""
        client = _FakeClient()
        placed = await _registry().place(
            client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/", ring_timeout=1
        )

        outcome = await asyncio.wait_for(placed.answered(), 1 + SETTLE_SECONDS * 5)
        assert outcome.kind == "timed_out"
        assert client.calls[0].hangups == 1

    async def test_answered_with_no_bot_dialled_in_is_a_failure_and_hung_up(self) -> None:
        """A callee who answers into dead air is worse than one who was never rung."""
        client = _FakeClient()
        placed = await _registry().place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        client.calls[0].push(ANSWERED)

        assert (await placed.answered()).kind == "answered"
        ended = await asyncio.wait_for(placed.ended(), SETTLE_SECONDS * 5)
        assert ended.kind == "failed"
        assert client.calls[0].hangups == 1

    async def test_a_dial_in_ahead_of_the_answer_event_still_counts(self) -> None:
        """A dial-in ahead of the answer event is not a missing one.

        The engine dials on the 2xx, and siphon reports the answer after the ACK.
        """
        client = _FakeClient()
        registry = _registry()
        placed = await registry.place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")

        assert registry.claim({"placed": placed.channel}) is placed
        client.calls[0].push(ANSWERED)
        await placed.answered()
        await asyncio.sleep(SETTLE_SECONDS)

        assert placed.outcome == CallOutcome("answered", code=200)
        assert client.calls[0].hangups == 0

    async def test_a_control_stream_that_closes_without_an_end_still_resolves(self) -> None:
        client = _FakeClient()
        placed = await _registry().place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        client.calls[0].push(RINGING, None)

        assert (await placed.answered()).kind == "failed"
        assert (await placed.ended()).kind == "failed"


class TestClaim:
    """Matching the engine's dial-in to the call it belongs to."""

    async def test_an_inbound_call_names_no_placed_call(self) -> None:
        assert _registry().claim({"call": "inbound@host"}) is None

    async def test_an_unknown_channel_is_nobody_s(self) -> None:
        assert _registry().claim({"placed": "placed-unknown"}) is None

    async def test_a_placed_call_is_claimed_once(self) -> None:
        client = _FakeClient()
        registry = _registry()
        placed = await registry.place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")

        assert registry.claim({"placed": placed.channel}) is placed
        assert registry.claim({"placed": placed.channel}) is None

    async def test_an_ended_call_is_no_longer_pending(self) -> None:
        client = _FakeClient()
        registry = _registry()
        placed = await registry.place(client, to="sip:a@example.com", ws_uri="ws://192.0.2.1/")
        client.calls[0].push(BUSY, None)
        await placed.ended()

        assert registry.pending() == []
        assert registry.claim({"placed": placed.channel}) is None
