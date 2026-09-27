"""Tests for the parts of `examples/agent_bot.py` that carry real logic.

The example's vendor wiring is not worth testing -- it is three constructors -- but the two
pieces the control plane rests on are: waiting for the farewell to finish before hanging up, and
joining a media session to its control channel. Both fail silently on the phone if they are
wrong, which is exactly the class of bug this example exists to stop reproducing.
"""

import asyncio
import importlib.util
import time
from pathlib import Path
from typing import Any

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMContextFrame,
    STTMetadataFrame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from agent_bot import (
    EMPTY_TURN_RECOVERY,
    BotSpeechMonitor,
    ControlPlane,
    SharedConstructor,
    build_user_aggregator_params,
    ends_mid_sentence,
    is_english,
    missing_environment,
    refuse_an_unknown_backend,
)

# Short enough that the suite stays fast, long enough not to race the event loop.
START_SECONDS = 0.3
QUIET_SECONDS = 0.05
TIMEOUT_SECONDS = 1.0

# How long `run_test` may take to get a pipeline running before it gives up. Its own default is one
# second, which is ample alone and not ample under a full suite: starting a pipeline links every
# processor and starts their tasks, and on a loaded machine that occasionally ran past the second,
# so `asyncio.wait_for(Event.wait(), 1.0)` was cancelled and surfaced as a bare TimeoutError in
# whichever timing test happened to be running. It is a budget for *starting*, not a measurement
# window -- the tests below time what they care about themselves -- so a generous one costs nothing
# and buys a suite that does not fail on how busy the machine is.
PIPELINE_START_TIMEOUT_SECONDS = 15.0


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

    async def test_a_stop_while_already_quiet_does_not_reopen_the_echo_tail(self) -> None:
        """A stop edge with no speech before it is not the bot stopping.

        The engine acknowledges every `clear` with a mark, the serializer reports each mark as the
        bot stopping, and every caller turn sends a clear -- with the bot long silent. Measured on
        a call: the caller said "my name is", paused, said "Karen", and the echo guard dropped
        "Karen" because that acknowledgement had restarted the echo tail half a second earlier.
        """
        monitor = BotSpeechMonitor()
        await _speak(monitor, True)
        await _speak(monitor, False)
        await asyncio.sleep(QUIET_SECONDS * 2)
        assert not monitor.line_is_ours(QUIET_SECONDS)

        await _speak(monitor, False)

        assert not monitor.line_is_ours(QUIET_SECONDS)

    async def test_speech_earlier_in_the_call_is_not_the_farewell(self) -> None:
        """With the tool called before the goodbye, the bot is silent when the wait begins.

        Measured on a call: the model called end_call, pipecat asked it again on the result, and the
        goodbye started playing 0.55 s later. The greeting had already counted as speech, the wait
        took the silence in between for the farewell being over, and the line went 40 ms before the
        goodbye would have played.
        """
        monitor = BotSpeechMonitor()
        await _speak(monitor, True)
        await _speak(monitor, False)
        waiter = asyncio.create_task(_wait(monitor))

        await asyncio.sleep(QUIET_SECONDS * 3)
        assert not waiter.done(), "hung up before the goodbye started"

        await _speak(monitor, True)
        await asyncio.sleep(QUIET_SECONDS * 2)
        assert not waiter.done(), "hung up while the goodbye was playing"

        await _speak(monitor, False)
        assert await waiter is True


class _FakeControlCall:
    """Stands in for a `siphon_control.Call`, which needs a live engine to obtain.

    A real handle stays open for the length of the call, so this one blocks in `next_event`
    until the test ends it -- otherwise the handler would deregister the call before the
    assertions ran, and the tests would pass for the wrong reason.
    """

    def __init__(
        self,
        sip_call_id: str | None,
        channel_id: str = "channel-1",
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sip_call_id = sip_call_id
        self.channel_id = channel_id
        self.hangups = 0
        self._events = list(events or [])
        self._ended = asyncio.Event()

    def end(self) -> None:
        """Let the call finish, as a BYE from either side would."""
        self._ended.set()

    async def next_event(self) -> dict[str, Any] | None:
        """Deliver the scripted events, then block until the call ends and report it as over."""
        if self._events:
            return self._events.pop(0)
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


class TestSharedConstructor:
    """One loaded model per process, however many calls build a service around it.

    The example builds its services per call, and the local ones load their model in their
    constructor. Without sharing, every call loads Whisper and Kokoro again right before the
    greeting, and two calls at once hold two copies of each in memory.
    """

    def test_the_same_model_is_built_once(self) -> None:
        """Keyword order is an accident of the call site, not a different model."""
        built: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def construct(*arguments: Any, **keyword_arguments: Any) -> object:
            built.append((arguments, keyword_arguments))
            return object()

        shared = SharedConstructor(construct)
        first = shared("small", device="cuda", compute_type="float16")
        second = shared("small", compute_type="float16", device="cuda")

        assert first is second
        assert len(built) == 1
        assert shared.built() == [first]

    def test_a_different_model_is_a_different_instance(self) -> None:
        """Sharing must never hand a call the wrong model because something else matched."""
        shared = SharedConstructor(lambda *arguments, **keyword_arguments: object())

        assert shared("small", device="cuda") is not shared("medium", device="cuda")
        assert shared("small", device="cuda") is not shared("small", device="cpu")
        assert len(shared.built()) == 3

    def test_sharing_is_installed_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrapping twice would still work, but hide the first cache behind the second."""
        import sys
        import types

        import agent_bot

        module = types.ModuleType("fake_pipecat_service")
        module.Model = lambda *arguments: object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_pipecat_service", module)
        monkeypatch.setattr(
            agent_bot, "LOCAL_MODEL_CONSTRUCTORS", {"stt": ("fake_pipecat_service", "Model")}
        )
        # `share_local_models` only touches the roles a local model actually fills.
        monkeypatch.setattr(agent_bot, "STT_PROVIDER", "local")

        agent_bot.share_local_models()
        wrapped = module.Model
        agent_bot.share_local_models()

        assert isinstance(wrapped, SharedConstructor)
        assert module.Model is wrapped


class TestLocalModelHooks:
    """The two names the local path wraps must still be how pipecat builds its models.

    The sharing works by replacing the constructor name a pipecat service module calls, not by
    subclassing, so a pipecat release that moves construction elsewhere would silently turn it back
    into a load per call. The modules themselves cannot be imported here -- their model libraries
    are not in the dev dependencies -- so the check reads the installed source instead, which is
    also the only thing that can see whether a call site still goes through the name.
    """

    def test_pipecat_still_constructs_through_the_wrapped_names(self) -> None:
        from agent_bot import LOCAL_MODEL_CONSTRUCTORS

        spec = importlib.util.find_spec("pipecat")
        assert spec is not None and spec.submodule_search_locations
        package_directory = Path(next(iter(spec.submodule_search_locations)))

        for module_name, attribute in LOCAL_MODEL_CONSTRUCTORS.values():
            relative = module_name.split(".")[1:]
            source = package_directory.joinpath(*relative).with_suffix(".py").read_text()
            assert f"import {attribute}" in source, f"{module_name} no longer imports {attribute}"
            assert f"= {attribute}(" in source, (
                f"{module_name} no longer builds through {attribute}"
            )


CLOUD_PROVIDERS = {"llm": "anthropic", "stt": "deepgram", "tts": "cartesia"}
LOCAL_PROVIDERS = {"llm": "local", "stt": "local", "tts": "local"}


class TestMissingEnvironment:
    """What a selection needs before the bot may start, checked at startup rather than per call."""

    def test_the_cloud_path_names_every_missing_vendor_value(self) -> None:
        missing = missing_environment(CLOUD_PROVIDERS, {"ANTHROPIC_API_KEY": "key"})
        assert missing == ["DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID"]

    def test_an_empty_value_is_missing(self) -> None:
        """Compose passes an unset variable through as an empty string, not as an absent one."""
        environment = dict.fromkeys(
            ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID"), ""
        )
        assert len(missing_environment(CLOUD_PROVIDERS, environment)) == 4

    def test_the_local_path_needs_no_vendor_keys(self) -> None:
        assert missing_environment(LOCAL_PROVIDERS, {}) == []

    def test_an_unknown_backend_is_refused(self) -> None:
        """A typo must stop the bot, not quietly fall back to one of the real backends."""
        with pytest.raises(ValueError, match="gpu"):
            refuse_an_unknown_backend("gpu")

    def test_an_unknown_provider_is_refused(self) -> None:
        """Same reason, per role: a typo must not silently bill somebody for a different vendor."""
        with pytest.raises(ValueError, match="BOT_LLM_PROVIDER 'antropic'"):
            missing_environment({**CLOUD_PROVIDERS, "llm": "antropic"}, {})

    def test_gemini_needs_its_own_key_and_google_speech_needs_a_service_account(self) -> None:
        """One vendor, two products, two credentials: an API key is not a service account."""
        assert missing_environment({"llm": "google", "stt": "deepgram", "tts": "cartesia"}, {}) == [
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "CARTESIA_API_KEY",
            "CARTESIA_VOICE_ID",
        ]
        assert missing_environment({"llm": "anthropic", "stt": "google", "tts": "google"}, {}) == [
            "ANTHROPIC_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ]

    def test_one_provider_in_two_roles_asks_for_its_key_once(self) -> None:
        """OpenAI for the model and the voice is one key, so the startup error must say it once."""
        assert missing_environment({"llm": "openai", "stt": "openai", "tts": "openai"}, {}) == [
            "OPENAI_API_KEY"
        ]

    def test_a_mixed_selection_asks_for_exactly_what_it_uses(self) -> None:
        """The shape the roles exist for: a hosted model with local speech needs only the key."""
        assert missing_environment({"llm": "google", "stt": "local", "tts": "local"}, {}) == [
            "GOOGLE_API_KEY"
        ]


class TestServiceImportsAreLazy:
    """Each backend's services are imported only when that backend is built.

    Found on the first local image: the cloud services were imported at module level, the local
    image carries none of their libraries, and the bot died at import with no module named
    anthropic before it could load a single model. The dev environment installs every extra, so no
    other test in this file can see the difference between the two images.
    """

    def test_no_service_module_is_imported_at_module_level(self) -> None:
        import ast

        import agent_bot

        # Derived from the provider tuples rather than listed, so a provider added without a line
        # here fails this test instead of quietly escaping it. `local` is two libraries, one per
        # role; every other provider is one package serving whichever roles it fills.
        packages_by_provider = {
            "anthropic": ("pipecat.services.anthropic",),
            "cartesia": ("pipecat.services.cartesia",),
            "deepgram": ("pipecat.services.deepgram",),
            "google": ("pipecat.services.google",),
            "openai": ("pipecat.services.openai",),
            "local": ("pipecat.services.whisper", "pipecat.services.kokoro"),
        }
        providers = set(agent_bot.LLM_PROVIDERS + agent_bot.STT_PROVIDERS + agent_bot.TTS_PROVIDERS)
        unmapped = providers - packages_by_provider.keys()
        assert not unmapped, f"provider(s) with no pipecat package mapped here: {sorted(unmapped)}"
        service_packages = tuple(
            package for name in providers for package in packages_by_provider[name]
        )
        assert agent_bot.__file__ is not None
        tree = ast.parse(Path(agent_bot.__file__).read_text())
        module_level: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                module_level.append(node.module)
            elif isinstance(node, ast.Import):
                module_level.extend(alias.name for alias in node.names)

        imported = [name for name in module_level if name.startswith(service_packages)]
        assert imported == [], f"imported at module level, so every image needs them: {imported}"


class _Stopwatch(FrameProcessor):
    """Note when a frame of one type first passes, and pass every frame on."""

    def __init__(self, watched: type[Frame]) -> None:
        super().__init__()
        self._watched = watched
        self.seen_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self.seen_at is None and isinstance(frame, self._watched):
            self.seen_at = time.monotonic()
        await self.push_frame(frame, direction)


async def _seconds_from_transcript_to_model(
    user_params: LLMUserAggregatorParams, *, settle_seconds: float
) -> float | None:
    """Replay a turn whose transcript lands after the speech has ended, and time the model's cue.

    The shape a recognizer that transcribes a whole segment produces, with the bot silent: its
    latency announcement when the call starts, the engine's speech edges, then the transcript once
    the segment is done. Returns how long after the transcript arrived the aggregator asked the
    model, or None if it never did.
    """
    transcript_arrived = _Stopwatch(TranscriptionFrame)
    model_asked = _Stopwatch(LLMContextFrame)
    aggregators = LLMContextAggregatorPair(LLMContext(), user_params=user_params)
    await run_test(
        Pipeline([transcript_arrived, aggregators.user(), model_asked]),
        start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
        frames_to_send=[
            # What pipecat announces for a recognizer that sets no latency of its own. It arms the
            # end-of-turn timer whose cancellation opened the gap; without it the stall never
            # reproduced.
            STTMetadataFrame(service_name="recognizer", ttfs_p99_latency=1.0),
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.4),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
            TranscriptionFrame("Goodbye.", "caller", time_now_iso8601()),
            SleepFrame(sleep=settle_seconds),
        ],
    )
    if transcript_arrived.seen_at is None or model_asked.seen_at is None:
        return None
    return model_asked.seen_at - transcript_arrived.seen_at


# Well clear of the processor's own 200 ms endpoint, and of the half-second recheck the aggregator
# makes for a transcript that arrives late; far below pipecat's five-second fallback.
MODEL_CUE_SECONDS = 1.0


class TestTheAggregatorDecidesTheTurn:
    """The processor that releases the context to the model decides the caller's turn, alone.

    Measured on a call: a turn processor in front of the aggregator ended the turn 200 ms after the
    caller said goodbye, and the model was asked 5.0 s after that, by pipecat's fallback. Two
    processors were deciding the same turn from the same frames, and a gap between them -- see
    `build_user_aggregator_params` -- left the aggregator unable to end it.
    """

    async def test_a_late_transcript_reaches_the_model_without_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        # The word count opens the turn on the transcript, as it does on a phone deployment, and
        # here the transcript arrives after the speech edge that should have ended the turn.
        monkeypatch.setattr(agent_bot, "MIN_INTERRUPT_WORDS", 2)

        seconds = await _seconds_from_transcript_to_model(
            build_user_aggregator_params(), settle_seconds=MODEL_CUE_SECONDS * 1.5
        )

        assert seconds is not None, "the model was never asked"
        assert seconds < MODEL_CUE_SECONDS

    def test_nothing_in_front_of_it_decides_the_turn_too(self) -> None:
        """A turn processor ahead of the aggregator is the second opinion that stalled the call.

        The replay above cannot see one, because it builds the aggregator on its own; the example's
        pipeline is only assembled per call, around a live socket.
        """
        import ast

        import agent_bot

        assert agent_bot.__file__ is not None
        tree = ast.parse(Path(agent_bot.__file__).read_text())
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "UserTurnProcessor" not in referenced


class TestEndsMidSentence:
    """The words a sentence cannot end on, which is what holds a local turn open."""

    @pytest.mark.parametrize(
        "text",
        ["My name is", "I would like to speak to the", "and", "Can I talk to your", " my name is "],
    )
    def test_a_sentence_left_hanging_is_unfinished(self, text: str) -> None:
        assert ends_mid_sentence(text)

    @pytest.mark.parametrize(
        "text",
        [
            "My name is Karen.",
            "Goodbye.",
            "Karen",
            "Thank you, that's all.",
            "I'd like to speak to the manager please.",
            "",
        ],
    )
    def test_a_finished_sentence_is_finished(self, text: str) -> None:
        assert not ends_mid_sentence(text)


class _Recorder(FrameProcessor):
    """Keep every frame of one type with the time it passed, and pass every frame on."""

    def __init__(self, watched: type[Frame]) -> None:
        super().__init__()
        self._watched = watched
        self.seen: list[tuple[float, Frame]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, self._watched):
            self.seen.append((time.monotonic(), frame))
        await self.push_frame(frame, direction)


def _segment(text: str, *, language: Language | None = None) -> list[Frame]:
    """One stretch of speech as the engine and a segment-at-a-time recognizer deliver it."""
    return [
        VADUserStartedSpeakingFrame(),
        SleepFrame(sleep=0.3),
        VADUserStoppedSpeakingFrame(),
        # The recognizer's delay after the speech edge, measured on calls at 0.17 to 0.26 s.
        SleepFrame(sleep=0.2),
        # A recognizer that transcribes whole segments marks every transcript final.
        TranscriptionFrame(text, "caller", time_now_iso8601(), language, finalized=True),
    ]


def _streamed_segment(text: str, *, language: Language | None) -> list[Frame]:
    """One stretch of speech as the engine and a streaming recognizer deliver it.

    The transcript lands while the caller is still speaking, and is not marked final: a streaming
    recognizer's own endpointing decides its segments, not the engine's speech edge.
    """
    return [
        VADUserStartedSpeakingFrame(),
        SleepFrame(sleep=0.2),
        TranscriptionFrame(text, "caller", time_now_iso8601(), language),
        SleepFrame(sleep=0.1),
        VADUserStoppedSpeakingFrame(),
    ]


RECOGNIZER_P99_SECONDS = {"local": 1.0, "deepgram": 0.35}
"""What each recognizer announces as its transcript latency: pipecat's default for a service that
sets none (Whisper), and its measured figure for Deepgram.

Keyed by recognizer rather than by backend, because the latency is a property of the recognizer
and the recognizer is now chosen on its own."""


async def _replay_turns(
    frames: list[Frame],
    monkeypatch: pytest.MonkeyPatch,
    *,
    recognizer: str = "local",
    configured_language: str = "en",
) -> tuple[list[float], list[tuple[float, str]]]:
    """Replay `frames` through the aggregator, as `recognizer`'s announced latency shapes it.

    Returns when each transcript arrived, and each time the model was asked with the caller's
    words it was asked about.
    """
    import agent_bot

    monkeypatch.setattr(agent_bot, "STT_PROVIDER", recognizer)
    # Read at import, for the recognizer selected then.
    monkeypatch.setattr(agent_bot, "STT_LANGUAGE", configured_language)
    monkeypatch.setattr(agent_bot, "MIN_INTERRUPT_WORDS", 2)

    transcripts = _Recorder(TranscriptionFrame)
    contexts = _Recorder(LLMContextFrame)
    aggregators = LLMContextAggregatorPair(LLMContext(), user_params=build_user_aggregator_params())
    await run_test(
        Pipeline([transcripts, aggregators.user(), contexts]),
        start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
        frames_to_send=[
            STTMetadataFrame(
                service_name="recognizer", ttfs_p99_latency=RECOGNIZER_P99_SECONDS[recognizer]
            ),
            *frames,
        ],
    )

    cues: list[tuple[float, str]] = []
    for seen_at, frame in contexts.seen:
        assert isinstance(frame, LLMContextFrame)
        said = [
            str(message.get("content"))
            for message in frame.context.get_messages()
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        cues.append((seen_at, said[-1] if said else ""))
    return [seen_at for seen_at, _ in transcripts.seen], cues


# The short silence after a finished sentence, plus scheduling.
TURN_CUE_SECONDS = 0.5


class TestTurnsWaitForAFinishedSentence:
    """A sentence left hanging holds the turn open for the rest of it, on every backend.

    Measured on a call: the caller said "my name is", paused, and said "Karen". The turn ended
    0.2 s into the pause, the bot answered that it had not caught the name, and "Karen" arrived
    while it was talking, where the echo guard dropped it.
    """

    async def test_a_pause_inside_a_sentence_does_not_end_the_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = [*_segment("My name is"), SleepFrame(sleep=0.8), *_segment("Karen.")]

        transcripts, cues = await _replay_turns([*frames, SleepFrame(sleep=1.0)], monkeypatch)

        assert len(cues) == 1, f"asked the model {len(cues)} times: {[said for _, said in cues]}"
        seen_at, said = cues[0]
        assert "My name is" in said and "Karen" in said, said
        assert seen_at - transcripts[-1] < TURN_CUE_SECONDS

    async def test_a_finished_sentence_is_answered_promptly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transcripts, cues = await _replay_turns(
            [*_segment("Goodbye."), SleepFrame(sleep=1.0)], monkeypatch
        )

        assert len(cues) == 1
        assert cues[0][0] - transcripts[-1] < TURN_CUE_SECONDS

    async def test_a_sentence_left_hanging_is_still_answered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hold is bounded: a caller who never finishes is answered with what they said."""
        import agent_bot

        hold = 0.6
        monkeypatch.setattr(agent_bot, "TURN_INCOMPLETE_HOLD_SECONDS", hold)

        transcripts, cues = await _replay_turns(
            [*_segment("My name is"), SleepFrame(sleep=hold + 1.0)], monkeypatch
        )

        assert len(cues) == 1
        waited = cues[0][0] - transcripts[-1]
        assert hold - 0.1 <= waited < hold + TURN_CUE_SECONDS, waited

    async def test_a_streaming_recognizer_gets_the_same_hold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cloud path's recognizer transcribes while the caller speaks; the hold still applies.

        Its language is `multi`, so it is the language the recognizer reports for the transcript
        that says whether the English word list applies.
        """
        frames = [
            *_streamed_segment("My name is", language=Language.EN),
            SleepFrame(sleep=0.8),
            *_streamed_segment("Karen.", language=Language.EN),
        ]

        _, cues = await _replay_turns(
            [*frames, SleepFrame(sleep=1.0)],
            monkeypatch,
            recognizer="deepgram",
            configured_language="multi",
        )

        assert len(cues) == 1, f"asked the model {len(cues)} times: {[said for _, said in cues]}"
        assert "My name is" in cues[0][1] and "Karen" in cues[0][1], cues[0][1]

    async def test_a_transcript_in_another_language_is_not_held(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The word list is English: "is" ends plenty of sentences in other languages."""
        transcripts, cues = await _replay_turns(
            [*_segment("Mijn naam is", language=Language.NL), SleepFrame(sleep=1.0)],
            monkeypatch,
        )

        assert len(cues) == 1
        assert cues[0][0] - transcripts[-1] < TURN_CUE_SECONDS

    async def test_without_a_reported_language_the_configured_one_decides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recognizer set to `multi` that reports no language could be hearing anything."""
        transcripts, cues = await _replay_turns(
            [*_segment("My name is"), SleepFrame(sleep=1.0)],
            monkeypatch,
            recognizer="deepgram",
            configured_language="multi",
        )

        assert len(cues) == 1
        assert cues[0][0] - transcripts[-1] < TURN_CUE_SECONDS


class TestIsEnglish:
    """Which transcript languages the English word list applies to."""

    @pytest.mark.parametrize("language", ["en", "en-US", "EN-gb", "en_AU"])
    def test_english_and_its_regional_variants(self, language: str) -> None:
        assert is_english(language)

    @pytest.mark.parametrize("language", ["multi", "nl", "de-DE", "", "eng"])
    def test_anything_else(self, language: str) -> None:
        assert not is_english(language)


class TestTransferVerdict:
    """The log says what happened to a transfer, and the verdict's kind is what decides it.

    Found on a call: siphon reported the transfer completed, with reason OK and no status code,
    and the log called it FAILED while the caller was already hearing the destination.
    """

    @staticmethod
    async def _log_for(event: dict[str, Any]) -> list[str]:
        """Run one control call that delivers `event`, and return the transfer lines it logged."""
        from loguru import logger

        control = ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9092/x")
        control._transfer_final = lambda kind: kind in {"TransferCompleted", "TransferFailed"}
        control._transfer_outcome = lambda raw: raw.get("payload")
        call = _FakeControlCall(sip_call_id="abc@example.invalid", events=[event])
        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
        try:
            holder = asyncio.create_task(control._handle_call(call))
            await asyncio.sleep(QUIET_SECONDS)
            call.end()
            await holder
        finally:
            logger.remove(sink)
        return [line for line in lines if "transfer" in line]

    async def test_a_completed_transfer_without_a_status_is_logged_as_completed(self) -> None:
        lines = await self._log_for(
            {
                "kind": "TransferCompleted",
                "payload": {"stage": "transferred", "status": None, "reason": "OK"},
            }
        )
        assert any("transfer completed" in line for line in lines), lines
        assert not any("FAILED" in line for line in lines), lines

    async def test_a_failed_transfer_is_logged_as_failed(self) -> None:
        lines = await self._log_for(
            {
                "kind": "TransferFailed",
                "payload": {"stage": "rejected", "status": 486, "reason": "Busy Here"},
            }
        )
        assert any("FAILED" in line and "486" in line for line in lines), lines


class TestPinnedSampling:
    """Sampling pinned in the environment reaches the model, and is the server's choice otherwise.

    A call repeated a hundred times measures the agent only if the model samples the same way each
    time; left unset, the example keeps each model server's own defaults.
    """

    def test_the_local_model_is_asked_with_the_pinned_temperature_and_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
        from pipecat.services.openai.llm import OpenAILLMService

        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        monkeypatch.setattr(agent_bot, "LLM_TEMPERATURE", 0.0)
        monkeypatch.setattr(agent_bot, "LLM_SEED", 7)

        service = agent_bot.build_llm_service(with_hangup=False)

        assert isinstance(service, OpenAILLMService)
        params = service.build_chat_completion_params(
            OpenAILLMInvocationParams(messages=[], tools=[], tool_choice="none")
        )
        assert params["temperature"] == 0.0
        assert params["seed"] == 7

    def test_unpinned_sampling_is_not_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from openai import NOT_GIVEN
        from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
        from pipecat.services.openai.llm import OpenAILLMService

        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        monkeypatch.setattr(agent_bot, "LLM_TEMPERATURE", None)
        monkeypatch.setattr(agent_bot, "LLM_SEED", None)

        service = agent_bot.build_llm_service(with_hangup=False)

        assert isinstance(service, OpenAILLMService)
        params = service.build_chat_completion_params(
            OpenAILLMInvocationParams(messages=[], tools=[], tool_choice="none")
        )
        assert params["temperature"] is NOT_GIVEN
        assert params["seed"] is NOT_GIVEN

    def test_claude_gets_the_pinned_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Claude takes no seed, so only the temperature applies there."""
        from pipecat.services.anthropic.llm import AnthropicLLMService

        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "anthropic")
        monkeypatch.setattr(agent_bot, "LLM_TEMPERATURE", 0.0)
        monkeypatch.setattr(agent_bot, "LLM_SEED", 7)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")

        service = agent_bot.build_llm_service(with_hangup=False)

        assert isinstance(service, AnthropicLLMService)
        # The request that carries it is built inside the streaming call, so the settings are read
        # back instead. `settings` is public since pipecat 1.12 (this reached into `_settings`
        # before) but typed as the base class, so the per-provider fields need a loose binding.
        settings: Any = service.settings
        assert settings.temperature == 0.0

    def test_a_seed_is_withheld_from_the_provider_that_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic's API has no seed. Sending one is an error, so a pinned seed is dropped."""
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_SEED", 7)
        monkeypatch.setattr(agent_bot, "LLM_TEMPERATURE", 0.0)

        assert "seed" not in agent_bot._sampling_settings("anthropic")
        for provider in ("google", "openai", "local"):
            assert agent_bot._sampling_settings(provider)["seed"] == 7, provider


class TestEmptyTurnRecovery:
    """A caller turn that ends with nothing recognised in it.

    pipecat 1.12 runs the model on one of those rather than leaving the bot silent, which is right
    on a phone: silence after somebody speaks reads as a dropped call. What it ships is a prompt
    written for a chat window, and this is a voice route.
    """

    def test_the_aggregator_is_given_the_recovery_config(self) -> None:
        params = build_user_aggregator_params()

        assert params.empty_user_turn is EMPTY_TURN_RECOVERY

    def test_the_recovery_line_is_short_enough_to_be_said(self) -> None:
        """The shipped default is three sentences and asks the model to re-ask its question."""
        from pipecat.turns.empty_user_turn import EmptyUserTurnConfig

        shipped = EmptyUserTurnConfig().interrupted_prompt
        ours = EMPTY_TURN_RECOVERY.interrupted_prompt

        assert ours is not None and shipped is not None
        assert len(ours.split()) < len(shipped.split()) / 2

    def test_a_caller_who_goes_quiet_is_not_prompted(self) -> None:
        """Same reasoning that leaves `idle_timeout_secs` unset: a pause is a caller thinking."""
        assert EMPTY_TURN_RECOVERY.idle_prompt is None

    def test_recovery_cannot_repeat(self) -> None:
        """Cap it at one, because an echo can look exactly like a caller who was not heard.

        On a leg whose echo the engine cannot cancel, the bot's own voice can open a turn whose
        transcript the echo guard then drops: an interrupted turn with no words in it. One
        recovery caps that at a single stray ask rather than a loop.
        """
        assert EMPTY_TURN_RECOVERY.max_consecutive_recoveries == 1


@pytest.fixture
def _vendor_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set every vendor key to a value, so a builder gets past its own startup check."""
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "CARTESIA_VOICE_ID",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.setenv(name, "test-value")


def _select(monkeypatch: pytest.MonkeyPatch, role: str, provider: str) -> None:
    """Point one role at `provider`, with that provider's own pinned defaults."""
    import agent_bot

    monkeypatch.setattr(agent_bot, f"{role.upper()}_PROVIDER", provider)
    if role == "llm":
        monkeypatch.setattr(agent_bot, "LLM_MODEL", agent_bot.DEFAULT_LLM_MODEL[provider])
        monkeypatch.setattr(
            agent_bot,
            "LLM_BASE_URL",
            "http://127.0.0.1:8090/v1" if provider == "local" else "",
        )
    elif role == "stt":
        monkeypatch.setattr(agent_bot, "STT_MODEL", agent_bot.DEFAULT_STT_MODEL[provider])
        monkeypatch.setattr(agent_bot, "STT_LANGUAGE", agent_bot.DEFAULT_STT_LANGUAGE[provider])
    else:
        monkeypatch.setattr(agent_bot, "TTS_MODEL", agent_bot.DEFAULT_TTS_MODEL[provider])
        monkeypatch.setattr(agent_bot, "TTS_VOICE", agent_bot.DEFAULT_TTS_VOICE[provider])


@pytest.mark.usefixtures("_vendor_credentials")
class TestEveryProviderBuilds:
    """Each provider in each role builds the service it says it does.

    The three roles are chosen separately, so these branches multiply: four ways to answer a turn,
    four to transcribe one and four to speak it. They had no coverage at all while there were two
    of each, which is how a builder that names a settings field the service does not have stays
    green until somebody runs a real call on it.

    The local recognizer and voice are left out on purpose: constructing them downloads and loads a
    model, which is minutes and a network. `TestLocalModelHooks` covers those names instead.
    """

    @pytest.mark.parametrize("provider", ["anthropic", "google", "openai", "local"])
    def test_a_model_provider_builds(self, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
        import agent_bot

        _select(monkeypatch, "llm", provider)

        service = agent_bot.build_llm_service(with_hangup=False)

        settings: Any = service.settings
        assert settings.model == agent_bot.DEFAULT_LLM_MODEL[provider]
        # The system prompt reaches every provider, under the one name they all share.
        assert settings.system_instruction.startswith("You are a voice assistant")

    @pytest.mark.parametrize("provider", ["deepgram", "openai"])
    def test_a_recognizer_provider_builds(
        self, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        import agent_bot

        _select(monkeypatch, "stt", provider)

        service = agent_bot.build_speech_to_text()

        assert service.settings.model == agent_bot.DEFAULT_STT_MODEL[provider]
        # `sample_rate` is resolved at setup; what the constructor was told is kept here.
        assert service._init_sample_rate == agent_bot.PIPELINE_SAMPLE_RATE

    @pytest.mark.parametrize("provider", ["cartesia", "openai"])
    def test_a_voice_provider_builds(self, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
        import agent_bot

        _select(monkeypatch, "tts", provider)

        service = agent_bot.build_text_to_speech()

        assert service.settings.model == agent_bot.DEFAULT_TTS_MODEL[provider]

    def test_the_openai_voice_is_built_at_the_rate_it_actually_emits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI's speech endpoint has one rate and takes no rate parameter.

        Asking it for the pipeline's rate does not change the bytes, it only mislabels them, and
        24 kHz audio tagged 16 kHz plays a third slow and a fifth flat. pipecat warns and carries
        on, so the guard has to be here: the service is told the truth and the output transport
        resamples on the way to the wire.
        """
        import agent_bot

        _select(monkeypatch, "tts", "openai")

        service = agent_bot.build_text_to_speech()

        assert service._init_sample_rate == agent_bot.OPENAI_TTS_SAMPLE_RATE
        assert agent_bot.OPENAI_TTS_SAMPLE_RATE != agent_bot.PIPELINE_SAMPLE_RATE, (
            "the rates coincide, so this test no longer proves anything"
        )

    def test_a_voice_that_belongs_to_an_account_still_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cartesia is the one provider whose voice cannot have a default."""
        import agent_bot

        _select(monkeypatch, "tts", "cartesia")
        monkeypatch.setenv("CARTESIA_VOICE_ID", "a-voice-on-this-account")

        service = agent_bot.build_text_to_speech()

        settings: Any = service.settings
        assert settings.voice == "a-voice-on-this-account"


class TestObserverHook:
    """A deployment can watch each call's pipeline without editing this file.

    The conversation harness reads what the bot heard, decided and said from an observer. The
    observer is built per call, from the serializer, because the serializer is what learns the
    call's SIP Call-ID.
    """

    @staticmethod
    def _stub_the_call(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[list[object], list[dict[str, Any]]]:
        """Build calls around stand-ins for the socket and the services.

        Returns the serializers the transports were built with, and the keyword arguments of every
        worker constructed.
        """
        from pipecat.pipeline.worker import PipelineWorker

        import agent_bot

        serializers: list[object] = []

        class _Transport:
            def __init__(self, serializer: object) -> None:
                serializers.append(serializer)
                self._input = FrameProcessor()
                self._output = FrameProcessor()

            def input(self) -> FrameProcessor:
                return self._input

            def output(self) -> FrameProcessor:
                return self._output

            def event_handler(self, _name: str) -> Any:
                return lambda handler: handler

        workers: list[dict[str, Any]] = []

        class _RecordingWorker(PipelineWorker):
            def __init__(self, *arguments: Any, **keyword_arguments: Any) -> None:
                # A copy of the list: the worker appends its own observers to the one it is given.
                observers = list(keyword_arguments.get("observers") or [])
                workers.append({**keyword_arguments, "observers": observers})
                super().__init__(*arguments, **keyword_arguments)

        monkeypatch.setattr(
            agent_bot, "build_transport", lambda _websocket, serializer: _Transport(serializer)
        )
        monkeypatch.setattr(agent_bot, "build_speech_to_text", FrameProcessor)
        monkeypatch.setattr(agent_bot, "build_llm_service", lambda with_hangup: FrameProcessor())
        monkeypatch.setattr(agent_bot, "build_text_to_speech", FrameProcessor)
        monkeypatch.setattr(agent_bot, "PipelineWorker", _RecordingWorker)
        return serializers, workers

    async def test_each_call_carries_the_observer_built_for_its_serializer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.observers.base_observer import BaseObserver

        import agent_bot

        serializers, workers = self._stub_the_call(monkeypatch)
        observer = BaseObserver()
        given: list[object] = []

        def factory(serializer: object) -> BaseObserver:
            given.append(serializer)
            return observer

        websocket: Any = object()
        agent_bot.build_worker(websocket, None, observer_factory=factory)

        assert given == serializers
        assert workers[0]["observers"] == [observer]

    async def test_without_a_factory_nothing_observes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        _, workers = self._stub_the_call(monkeypatch)

        websocket: Any = object()
        agent_bot.build_worker(websocket, None)

        assert workers[0]["observers"] == []
