"""Tests for the parts of `examples/agent_bot.py` that carry real logic.

The example's vendor wiring is not worth testing -- it is three constructors -- but the two
pieces the control plane rests on are: waiting for the farewell to finish before hanging up, and
joining a media session to its control channel. Both fail silently on the phone if they are
wrong, which is exactly the class of bug this example exists to stop reproducing.
"""

import asyncio
from typing import Any

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from agent_bot import BotSpeechMonitor, ControlPlane

# Short enough that the suite stays fast, long enough not to race the event loop.
START_SECONDS = 0.3
QUIET_SECONDS = 0.05
TIMEOUT_SECONDS = 1.0


async def _speak(monitor: BotSpeechMonitor, speaking: bool) -> None:
    """Drive one bot-speaking edge through the monitor."""
    frame = BotStartedSpeakingFrame() if speaking else BotStoppedSpeakingFrame()
    await monitor.process_frame(frame, FrameDirection.DOWNSTREAM)


async def _wait(monitor: BotSpeechMonitor) -> bool:
    """Run the farewell wait with test-scale timings."""
    return await monitor.wait_until_finished(
        start_seconds=START_SECONDS,
        quiet_seconds=QUIET_SECONDS,
        timeout_seconds=TIMEOUT_SECONDS,
    )


class TestBotSpeechMonitor:
    """The farewell wait: hang up after the goodbye has played, not during it."""

    async def test_waits_for_speech_that_has_not_started_yet(self) -> None:
        """The tool fires before the farewell is spoken, so quiet-now must not end the wait."""
        monitor = BotSpeechMonitor()
        waiter = asyncio.create_task(_wait(monitor))

        # The bot is silent at this point: a naive "wait for quiet" would already be satisfied.
        await asyncio.sleep(QUIET_SECONDS * 2)
        assert not waiter.done()

        await _speak(monitor, True)
        await asyncio.sleep(QUIET_SECONDS * 2)
        assert not waiter.done(), "hung up while the bot was still speaking"

        await _speak(monitor, False)
        assert await waiter is True

    async def test_returns_when_the_bot_never_speaks(self) -> None:
        """A farewell that never arrives still has to release the call, once the grace is up."""
        monitor = BotSpeechMonitor()
        assert await _wait(monitor) is True

    async def test_gives_up_on_a_bot_that_will_not_stop(self) -> None:
        """The cap exists so a runaway monologue cannot hold the line open forever."""
        monitor = BotSpeechMonitor()
        await _speak(monitor, True)
        assert await _wait(monitor) is False

    async def test_a_second_sentence_defers_the_hangup(self) -> None:
        """Speech resuming inside the quiet window restarts it, rather than counting as done."""
        monitor = BotSpeechMonitor()
        await _speak(monitor, True)
        waiter = asyncio.create_task(_wait(monitor))

        await _speak(monitor, False)
        # Interrupt the quiet window before it elapses: the bot is talking again.
        await asyncio.sleep(QUIET_SECONDS / 2)
        await _speak(monitor, True)
        await asyncio.sleep(QUIET_SECONDS * 2)
        assert not waiter.done(), "hung up between two sentences of the farewell"

        await _speak(monitor, False)
        assert await waiter is True

    async def test_reset_forgets_the_previous_call(self) -> None:
        """The server outlives a call, so speech from the last one must not satisfy this one."""
        monitor = BotSpeechMonitor()
        await _speak(monitor, True)
        await _speak(monitor, False)

        monitor.reset()
        waiter = asyncio.create_task(_wait(monitor))
        # Without the reset the wait would return immediately on the stale "has spoken".
        await asyncio.sleep(QUIET_SECONDS * 2)
        assert not waiter.done()

        await _speak(monitor, True)
        await _speak(monitor, False)
        assert await waiter is True

    async def test_passes_every_frame_through(self) -> None:
        """It is a monitor, not a gate: unrelated frames must not be swallowed."""
        monitor = BotSpeechMonitor()
        pushed: list[Any] = []

        async def capture(
            frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
        ) -> Any:
            pushed.append(frame)

        monitor.push_frame = capture  # type: ignore[method-assign]
        frame = TextFrame(text="goodbye")
        await monitor.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert pushed == [frame]


class _FakeControlCall:
    """Stands in for a `siphon_control.Call`, which needs a live engine to obtain.

    A real handle stays open for the length of the call, so this one blocks in `next_event`
    until the test ends it -- otherwise the handler would deregister the call before the
    assertions ran, and the tests would pass for the wrong reason.
    """

    def __init__(self, sip_call_id: str | None, channel_id: str = "channel-1") -> None:
        self.sip_call_id = sip_call_id
        self.channel_id = channel_id
        self.hangups = 0
        self._ended = asyncio.Event()

    def end(self) -> None:
        """Let the call finish, as a BYE from either side would."""
        self._ended.set()

    async def next_event(self) -> dict[str, str] | None:
        """Block until the call ends, then report it as over."""
        await self._ended.wait()
        return None

    async def hangup(self) -> None:
        """Record the hangup."""
        self.hangups += 1


class TestControlPlaneCorrelation:
    """Joining a media session to its control channel, on the right id.

    The engine expands `{call_id}` in a `ws_uri` to the SIP Call-ID, and that is what the media
    socket's `start` envelope carries -- while a control frame's own `call_id` is siphon's
    internal UUID. Joining on the matching name silently never matches.
    """

    @pytest.fixture
    def control(self) -> ControlPlane:
        """Build a control plane with nothing dialled."""
        return ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9092/x")

    async def test_correlates_on_the_sip_call_id(self, control: ControlPlane) -> None:
        """The handle comes back for the id the media side actually knows."""
        call = _FakeControlCall(sip_call_id="abc@example.invalid")
        holder = asyncio.create_task(control._handle_call(call))
        await asyncio.sleep(0)

        assert control.call_for("abc@example.invalid") is call

        call.end()
        await holder

    async def test_unknown_and_missing_ids_resolve_to_nothing(self, control: ControlPlane) -> None:
        """A media session with no match must not reach some other caller's control channel."""
        call = _FakeControlCall(sip_call_id="abc@example.invalid")
        holder = asyncio.create_task(control._handle_call(call))
        await asyncio.sleep(0)

        assert control.call_for("someone-else@example.invalid") is None
        assert control.call_for(None) is None

        call.end()
        await holder

    async def test_the_call_is_forgotten_once_it_ends(self, control: ControlPlane) -> None:
        """A stale handle would let a later call hang up a number that already went away."""
        call = _FakeControlCall(sip_call_id="abc@example.invalid")
        holder = asyncio.create_task(control._handle_call(call))
        await asyncio.sleep(0)
        assert control.call_for("abc@example.invalid") is call

        call.end()
        await holder
        assert control.call_for("abc@example.invalid") is None

    async def test_a_call_without_a_sip_id_is_ignored(self, control: ControlPlane) -> None:
        """Nothing could ever join it, so registering it would only risk a wrong match."""
        call = _FakeControlCall(sip_call_id=None)
        await control._handle_call(call)
        assert control.call_for(None) is None
