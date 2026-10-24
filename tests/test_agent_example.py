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
from types import SimpleNamespace
from typing import Any, cast

import pytest
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
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
    DEFAULT_SYSTEM_PROMPT,
    EMPTY_TURN_RECOVERY,
    KICKOFF,
    BotSpeechMonitor,
    CallRequest,
    ControlPlane,
    SharedConstructor,
    TransferVerdict,
    build_user_aggregator_params,
    call_request_from_arguments,
    ends_mid_sentence,
    is_english,
    kickoff_for,
    missing_environment,
    parse_arguments,
    place_one_call,
    refuse_an_unknown_backend,
    refuse_an_unknown_transfer_mode,
    refuse_missing_google_cloud_credentials,
    resolve_system_prompt,
)
from control_event_fixtures import ANSWERED, BUSY, FAR_END_HUNG_UP, RINGING

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


class _PlacingClient:
    """Stands in for a `siphon_control.ControlClient` that places calls.

    Each originate hands back a `_FakeControlCall` scripted with `events`, then holding the call
    open until the test ends it -- or raises `error`, as a refused originate does.
    """

    def __init__(
        self, events: list[dict[str, Any]] | None = None, error: BaseException | None = None
    ) -> None:
        self.events = events or []
        self.error = error
        self.calls: list[_FakeControlCall] = []
        self.originated: list[dict[str, Any]] = []

    async def originate(
        self, channel: str, to: str | None = None, **options: Any
    ) -> _FakeControlCall:
        self.originated.append({"channel": channel, "to": to, **options})
        if self.error is not None:
            raise self.error
        call = _FakeControlCall(
            sip_call_id=f"placed-{len(self.calls)}@example.invalid",
            channel_id=channel,
            events=list(self.events),
        )
        self.calls.append(call)
        return call


def _placing_control(client: _PlacingClient) -> ControlPlane:
    control = ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9092/x")
    control.attach_client(client)
    return control


def _call_request(**overrides: Any) -> CallRequest:
    fields: dict[str, Any] = {
        "to": "sip:+15550100@trunk.example",
        "ws_uri": "ws://198.51.100.10:9001/stream?call={call_id}",
        "from_uri": None,
        "profile": "agent",
        "ring_timeout": 30,
        "context": {},
    }
    return CallRequest(**{**fields, **overrides})


class TestPlacingACall:
    """`--call`: the bot rings a number, greets first on answer, and says how the call went."""

    def test_no_call_asked_for_is_no_request(self) -> None:
        assert call_request_from_arguments(parse_arguments([])) is None

    def test_a_call_needs_the_control_plane(self) -> None:
        with pytest.raises(ValueError, match="--control-url"):
            call_request_from_arguments(
                parse_arguments(
                    ["--call", "sip:+15550100@trunk.example", "--call-ws-uri", "ws://192.0.2.1/"]
                )
            )

    def test_a_call_needs_the_address_the_engine_reaches_the_bot_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BOT_CALL_WS_URI", raising=False)
        with pytest.raises(ValueError, match="BOT_CALL_WS_URI"):
            call_request_from_arguments(
                parse_arguments(
                    ["--call", "sip:+15550100@trunk.example", "--control-url", "ws://192.0.2.1/"]
                )
            )

    def test_a_ring_timeout_under_a_second_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--call-timeout"):
            call_request_from_arguments(
                parse_arguments(
                    [
                        "--call",
                        "sip:+15550100@trunk.example",
                        "--control-url",
                        "ws://192.0.2.1/",
                        "--call-ws-uri",
                        "ws://192.0.2.1/",
                        "--call-timeout",
                        "0",
                    ]
                )
            )

    def test_the_request_carries_what_was_asked_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BOT_CALL_WS_URI", "ws://198.51.100.10:9001/stream?call={call_id}")
        request = call_request_from_arguments(
            parse_arguments(
                [
                    "--call",
                    "sip:+15550100@trunk.example",
                    "--control-url",
                    "ws://192.0.2.1/",
                    "--call-from",
                    "sip:bot@pbx.example",
                    "--call-context",
                    "the parcel delivery they asked about",
                    "--call-timeout",
                    "20",
                ]
            )
        )
        assert request == _call_request(
            from_uri="sip:bot@pbx.example",
            ring_timeout=20,
            context={"reason": "the parcel delivery they asked about"},
        )

    def test_an_inbound_call_opens_on_the_caller_kickoff(self) -> None:
        assert kickoff_for(None) == KICKOFF

    def test_a_placed_call_opens_knowing_the_bot_rang_them_and_why(self) -> None:
        kickoff = kickoff_for({"reason": "the parcel delivery they asked about"})
        assert kickoff != KICKOFF
        assert "placed this call" in kickoff
        assert "the parcel delivery they asked about" in kickoff
        # Without a reason it still says who called whom.
        assert "placed this call" in kickoff_for({})

    async def test_a_placed_call_is_joined_like_a_handed_over_one(self) -> None:
        """`end_call` and `transfer_call` find a placed call by its SIP Call-ID, as any other."""
        client = _PlacingClient()
        control = _placing_control(client)

        placed = await control.place(_call_request())
        call = client.calls[0]
        assert control.call_for(call.sip_call_id) is call
        assert client.originated[0]["ws_uri"].endswith(f"&placed={placed.channel}")

        call.end()
        await placed.ended()
        await asyncio.sleep(0)
        assert control.call_for(call.sip_call_id) is None

    async def test_a_placed_call_s_transfer_verdict_still_reaches_the_tool(self) -> None:
        client = _PlacingClient(
            events=[
                {
                    "kind": "TransferCompleted",
                    "payload": {"stage": "transferred", "status": None, "reason": "OK"},
                }
            ]
        )
        control = _placing_control(client)
        control._transfer_final = lambda kind: kind == "TransferCompleted"
        control._transfer_outcome = lambda raw: raw.get("payload")

        verdict = control.expect_transfer_verdict("placed-0@example.invalid")
        await control.place(_call_request())

        assert (await asyncio.wait_for(verdict, TIMEOUT_SECONDS)).ending == "completed"
        client.calls[0].end()

    async def test_a_busy_callee_is_reported_and_fails_the_run(self) -> None:
        control = _placing_control(_PlacingClient(events=[RINGING, BUSY]))
        assert await place_one_call(control, _call_request()) == 1

    async def test_an_answered_call_that_ends_normally_is_a_clean_run(self) -> None:
        control = _placing_control(_PlacingClient(events=[RINGING, ANSWERED, FAR_END_HUNG_UP]))
        assert await place_one_call(control, _call_request()) == 0

    async def test_a_refused_originate_fails_the_run(self) -> None:
        control = _placing_control(_PlacingClient(error=RuntimeError("not_found: no route")))
        assert await place_one_call(control, _call_request()) == 1


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

    def test_gemini_needs_its_own_key_and_google_speech_needs_no_variable(self) -> None:
        """One vendor, two products: Gemini takes a key, the speech APIs an identity.

        The identity is application default credentials, which on a Google Cloud VM come from the
        metadata server with no variable set at all, so no variable may be demanded for it.
        `TestGoogleCloudCredentials` covers the check that replaces one.
        """
        assert missing_environment({"llm": "google", "stt": "deepgram", "tts": "cartesia"}, {}) == [
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "CARTESIA_API_KEY",
            "CARTESIA_VOICE_ID",
        ]
        assert missing_environment({"llm": "anthropic", "stt": "google", "tts": "google"}, {}) == [
            "ANTHROPIC_API_KEY",
        ]

    def test_elevenlabs_needs_its_key_and_a_voice_from_the_account(self) -> None:
        """An ElevenLabs voice id belongs to an account, so like Cartesia's it has no default."""
        environment = {"ANTHROPIC_API_KEY": "key", "DEEPGRAM_API_KEY": "key"}
        assert missing_environment({**CLOUD_PROVIDERS, "tts": "elevenlabs"}, environment) == [
            "ELEVENLABS_API_KEY",
            "BOT_TTS_VOICE",
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


class TestGoogleCloudCredentials:
    """Google's speech APIs start on whatever application default credentials resolve to.

    `google.auth.default` is replaced rather than run: off Google Cloud it probes the metadata
    server, which is a network call and seconds of timeout, and its answer depends on the host.
    """

    @staticmethod
    def _resolve_to(monkeypatch: pytest.MonkeyPatch, project: str | None) -> list[dict[str, Any]]:
        """Make application default credentials resolve, to `project`; return the calls made."""
        import google.auth

        calls: list[dict[str, Any]] = []

        def default(**keyword_arguments: Any) -> tuple[object, str | None]:
            calls.append(keyword_arguments)
            return object(), project

        monkeypatch.setattr(google.auth, "default", default)
        return calls

    @staticmethod
    def _resolve_to_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        def default(**_keyword_arguments: Any) -> tuple[object, str | None]:
            raise DefaultCredentialsError("Your default credentials were not found.")  # type: ignore[no-untyped-call]

        monkeypatch.setattr(google.auth, "default", default)

    def test_a_metadata_server_identity_is_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A VM's service account, with no variable and no key file anywhere, starts the bot."""
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        calls = self._resolve_to(monkeypatch, "test-project")

        refuse_missing_google_cloud_credentials(
            {"llm": "anthropic", "stt": "google", "tts": "google"}
        )

        # Asked once for both roles, with the scope pipecat's services ask for.
        assert calls == [{"scopes": ["https://www.googleapis.com/auth/cloud-platform"]}]

    def test_no_credentials_names_both_ways_to_provide_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._resolve_to_nothing(monkeypatch)

        with pytest.raises(ValueError) as raised:
            refuse_missing_google_cloud_credentials(
                {"llm": "anthropic", "stt": "deepgram", "tts": "google"}
            )

        message = str(raised.value)
        assert "GOOGLE_APPLICATION_CREDENTIALS" in message
        assert "metadata server" in message
        assert "tts=google" in message

    def test_the_recognizer_also_needs_a_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The recognizer refuses credentials that name no project; the voice does not."""
        self._resolve_to(monkeypatch, None)

        refuse_missing_google_cloud_credentials(
            {"llm": "anthropic", "stt": "deepgram", "tts": "google"}
        )
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            refuse_missing_google_cloud_credentials(
                {"llm": "anthropic", "stt": "google", "tts": "cartesia"}
            )

    def test_a_selection_without_google_speech_never_asks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gemini is not the speech APIs: its key is all it needs, and the probe costs seconds."""
        calls = self._resolve_to(monkeypatch, "test-project")

        refuse_missing_google_cloud_credentials({"llm": "google", "stt": "local", "tts": "local"})

        assert calls == []


class TestResolveSystemPrompt:
    """Where the persona comes from: the example's own, an inline value, or a file."""

    def test_nothing_set_is_the_example_default(self) -> None:
        assert resolve_system_prompt({}) == DEFAULT_SYSTEM_PROMPT

    def test_empty_values_are_unset(self) -> None:
        """Compose passes an unset variable through as an empty string, not as an absent one."""
        environment = {"BOT_SYSTEM_PROMPT": "", "BOT_SYSTEM_PROMPT_FILE": "  "}
        assert resolve_system_prompt(environment) == DEFAULT_SYSTEM_PROMPT

    def test_an_inline_prompt_replaces_the_default(self) -> None:
        environment = {"BOT_SYSTEM_PROMPT": "You are the front desk of a bike shop."}
        assert resolve_system_prompt(environment) == "You are the front desk of a bike shop."

    def test_a_prompt_file_replaces_the_default(self, tmp_path: Path) -> None:
        prompt = tmp_path / "persona.txt"
        prompt.write_text("You are the front desk of a bike shop.\n\nAsk one question at a time.\n")
        environment = {"BOT_SYSTEM_PROMPT_FILE": str(prompt)}
        assert resolve_system_prompt(environment) == (
            "You are the front desk of a bike shop.\n\nAsk one question at a time."
        )

    def test_both_at_once_is_refused(self, tmp_path: Path) -> None:
        """Neither silently winning: a stale line in .env would otherwise override the file."""
        prompt = tmp_path / "persona.txt"
        prompt.write_text("From the file.")
        environment = {"BOT_SYSTEM_PROMPT": "Inline.", "BOT_SYSTEM_PROMPT_FILE": str(prompt)}
        with pytest.raises(ValueError, match="BOT_SYSTEM_PROMPT and BOT_SYSTEM_PROMPT_FILE"):
            resolve_system_prompt(environment)

    def test_a_missing_file_is_refused_by_name(self, tmp_path: Path) -> None:
        """A typo in the path must stop the bot, not fall back to the default persona."""
        missing = tmp_path / "no-such-persona.txt"
        with pytest.raises(ValueError, match=r"no-such-persona\.txt"):
            resolve_system_prompt({"BOT_SYSTEM_PROMPT_FILE": str(missing)})

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        prompt = tmp_path / "persona.txt"
        prompt.write_text("\n  \n")
        with pytest.raises(ValueError, match="empty"):
            resolve_system_prompt({"BOT_SYSTEM_PROMPT_FILE": str(prompt)})

    def test_the_shipped_demo_persona_loads(self) -> None:
        demo = Path(__file__).parent.parent / "examples" / "prompts" / "escalation_demo.txt"
        prompt = resolve_system_prompt({"BOT_SYSTEM_PROMPT_FILE": str(demo)})
        # A replacement persona has to keep the telephone rules the default carries, or the
        # synthesizer reads markdown aloud.
        assert "formatting" in prompt
        assert "interrupt" in prompt


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
            "elevenlabs": ("pipecat.services.elevenlabs",),
            "deepgram": ("pipecat.services.deepgram",),
            "google": ("pipecat.services.google",),
            "openai": ("pipecat.services.openai",),
            "local": ("pipecat.services.whisper", "pipecat.services.kokoro"),
            "gemini-live": ("pipecat.services.google",),
            "openai-realtime": ("pipecat.services.openai",),
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


SIP_CALL_ID = "abc@example.invalid"
TRANSFER_DESTINATION = "sip:desk@example.invalid"


def _verdict(kind: str, stage: str, code: int | None, reason: str | None) -> dict[str, Any]:
    """One transfer event as `next_event()` returns it, in the payload shape siphon sends."""
    return {"kind": kind, "payload": {"stage": stage, "code": code, "reason": reason}}


ACCEPTED = _verdict("TransferProgress", "accepted", 202, "Accepted")
COMPLETED = _verdict("TransferCompleted", "transferred", 200, "OK")
REFUSED_BY_THE_PHONE = _verdict("TransferFailed", "rejected", 405, "Method Not Allowed")
ENDED_WITH_THE_CALL = _verdict("TransferFailed", "call_ended", None, None)


class _FakeTransferringCall(_FakeControlCall):
    """A control call whose `transfer()` is answered on the event stream, as siphon answers one.

    `transfer()` resolves once the REFER is sent and the verdict follows as events (RFC 3515
    §2.4.4), so the fake queues `verdicts` when it is asked to transfer. `None` in the script is
    the call ending: the stream closes, as `next_event()` does when siphon tears the call down.
    """

    def __init__(
        self,
        verdicts: list[dict[str, Any] | None],
        transfer_error: BaseException | None = None,
    ) -> None:
        super().__init__(sip_call_id=SIP_CALL_ID)
        self.transfers: list[str] = []
        self._verdicts = verdicts
        self._transfer_error = transfer_error
        self._stream: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def deliver(self, event: dict[str, Any] | None) -> None:
        """Put one more event on the stream, later than the REFER."""
        self._stream.put_nowait(event)

    def end(self) -> None:
        """Close the stream, as the end of the call does."""
        self._stream.put_nowait(None)

    async def next_event(self) -> dict[str, Any] | None:
        """Deliver whatever has arrived, in order, blocking until something does."""
        return await self._stream.get()

    async def transfer(self, target: str) -> None:
        """Record the REFER, then answer it with the scripted verdict."""
        self.transfers.append(target)
        if self._transfer_error is not None:
            raise self._transfer_error
        for event in self._verdicts:
            self._stream.put_nowait(event)


class _FakeWorker:
    """Record what the tool queues into the pipeline: the end of the call, or a model note."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []

    async def queue_frame(self, frame: Frame) -> None:
        self.frames.append(frame)

    def ended(self) -> bool:
        from pipecat.frames.frames import EndWorkerFrame

        return any(isinstance(frame, EndWorkerFrame) for frame in self.frames)

    def notes_to_the_model(self) -> list[str]:
        from pipecat.frames.frames import LLMMessagesAppendFrame

        return [
            str(message["content"])  # type: ignore[index]
            for frame in self.frames
            if isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm
            for message in frame.messages
        ]


class _FinishedSpeech:
    """The handover line has already played: the transfer tests are about what follows it."""

    async def wait_until_finished(self) -> bool:
        return True


class _Transfer:
    """One `transfer_call` tool call on a call wired to a control plane, and what it did."""

    def __init__(self, call: _FakeTransferringCall) -> None:
        import agent_bot

        self.call = call
        self.control = ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9/x")
        self.control._transfer_final = lambda kind: kind in {"TransferCompleted", "TransferFailed"}
        self.control._transfer_outcome = lambda event: event.get("payload")
        self.worker = _FakeWorker()
        serializer: Any = SimpleNamespace(call_id=SIP_CALL_ID)
        speech: Any = _FinishedSpeech()
        worker: Any = self.worker
        self.resources = agent_bot.CallResources(
            serializer=serializer, speech=speech, control=self.control, worker=worker
        )
        self.results: list[dict[str, Any]] = []
        self._holder: asyncio.Task[None] | None = None

    async def attach(self) -> None:
        """Hand the call to the control plane, as siphon's handover does."""
        self._holder = asyncio.create_task(self.control._handle_call(self.call))
        await asyncio.sleep(0)

    async def request(self) -> None:
        """Run the tool as pipecat would, recording every result it hands the model."""
        import agent_bot

        async def result_callback(result: dict[str, Any], **_: Any) -> None:
            self.results.append(result)

        params: Any = SimpleNamespace(app_resources=self.resources, result_callback=result_callback)
        await agent_bot.transfer_call(params)

    async def hang_up(self) -> None:
        """End the call and let the control plane let go of it."""
        self.call.end()
        if self._holder is not None:
            await self._holder


@pytest.fixture
def transfer_settings(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Configure a destination, and a verdict wait short enough to run in a test."""
    import agent_bot

    monkeypatch.setattr(agent_bot, "TRANSFER_TARGET", TRANSFER_DESTINATION)
    monkeypatch.setattr(agent_bot, "TRANSFER_VERDICT_TIMEOUT_SECONDS", TIMEOUT_SECONDS)
    return monkeypatch


@pytest.mark.usefixtures("transfer_settings")
class TestTransferKeepsTheCallerUntilTheVerdict:
    """The bot lets go of a call only once its transfer has actually happened.

    `transfer()` resolves when the REFER is sent, and many phones refuse a REFER from the network
    (`405`, `403`). Ending the bot's leg on the send left such a caller on a silent line until the
    node gave up on the call; measured on a test rig, that is the common failure, not the rare one.
    """

    async def test_a_completed_transfer_ends_the_bot(self) -> None:
        transfer = _Transfer(_FakeTransferringCall([ACCEPTED, COMPLETED]))
        await transfer.attach()

        await transfer.request()

        assert transfer.call.transfers == [TRANSFER_DESTINATION]
        assert transfer.worker.ended(), "the transfer completed and the bot is still on the call"
        assert transfer.worker.notes_to_the_model() == []
        await transfer.hang_up()

    async def test_a_phone_that_refuses_the_refer_keeps_the_caller_and_tells_the_model(
        self,
    ) -> None:
        transfer = _Transfer(_FakeTransferringCall([REFUSED_BY_THE_PHONE]))
        await transfer.attach()

        await transfer.request()

        assert not transfer.worker.ended(), "the bot left a caller whose transfer never started"
        notes = transfer.worker.notes_to_the_model()
        assert len(notes) == 1, transfer.worker.frames
        assert "still on the line" in notes[0]
        assert not transfer.resources.transfer_hold.holding, "the caller is still not heard"
        await transfer.hang_up()

    async def test_no_verdict_within_the_wait_counts_as_failed(
        self, transfer_settings: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        transfer_settings.setattr(agent_bot, "TRANSFER_VERDICT_TIMEOUT_SECONDS", QUIET_SECONDS)
        transfer = _Transfer(_FakeTransferringCall([ACCEPTED]))
        await transfer.attach()

        started = time.monotonic()
        await transfer.request()

        assert time.monotonic() - started < TIMEOUT_SECONDS, "the wait was not bounded"
        assert not transfer.worker.ended()
        assert len(transfer.worker.notes_to_the_model()) == 1
        assert not transfer.resources.transfer_hold.holding
        await transfer.hang_up()

    async def test_a_completion_after_the_wait_still_ends_the_bot(
        self, transfer_settings: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        transfer_settings.setattr(agent_bot, "TRANSFER_VERDICT_TIMEOUT_SECONDS", QUIET_SECONDS)
        transfer = _Transfer(_FakeTransferringCall([ACCEPTED]))
        await transfer.attach()
        await transfer.request()
        assert not transfer.worker.ended()

        # The destination answers late: the caller is gone, so the bot's leg is finished after all.
        transfer.call.deliver(COMPLETED)
        for _ in range(100):
            if transfer.worker.ended():
                break
            await asyncio.sleep(0.01)

        assert transfer.worker.ended(), "the caller was transferred and the bot stayed on the call"
        await transfer.hang_up()

    async def test_the_call_ending_during_the_transfer_ends_the_bot_without_recovering(
        self,
    ) -> None:
        # A phone may BYE its old leg as soon as it has sent the INVITE to the destination (TS
        # 24.629 allows it), and siphon then reports call_ended although the new call may connect.
        transfer = _Transfer(_FakeTransferringCall([ACCEPTED, ENDED_WITH_THE_CALL, None]))
        await transfer.attach()

        await transfer.request()

        assert transfer.worker.ended()
        assert transfer.worker.notes_to_the_model() == [], "the bot tried to talk to nobody"
        await transfer.hang_up()

    async def test_the_stream_closing_without_a_verdict_counts_as_the_call_ending(self) -> None:
        transfer = _Transfer(_FakeTransferringCall([ACCEPTED, None]))
        await transfer.attach()

        await transfer.request()

        assert transfer.worker.ended()
        assert transfer.worker.notes_to_the_model() == []

    async def test_a_second_request_while_one_is_pending_is_refused(self) -> None:
        transfer = _Transfer(_FakeTransferringCall([ACCEPTED]))
        await transfer.attach()

        first = asyncio.create_task(transfer.request())
        for _ in range(100):
            if transfer.call.transfers:
                break
            await asyncio.sleep(0.01)
        assert transfer.resources.transfer_hold.holding, "the caller is heard over the transfer"

        await transfer.request()

        assert transfer.call.transfers == [TRANSFER_DESTINATION], "a second REFER was sent"
        assert "error" in transfer.results[-1], transfer.results

        transfer.call.deliver(REFUSED_BY_THE_PHONE)
        await first
        assert not transfer.worker.ended()
        await transfer.hang_up()

    async def test_a_transfer_the_control_plane_refuses_keeps_the_caller(self) -> None:
        transfer = _Transfer(_FakeTransferringCall([], transfer_error=RuntimeError("not_found")))
        await transfer.attach()

        await transfer.request()

        assert not transfer.worker.ended()
        assert len(transfer.worker.notes_to_the_model()) == 1
        assert not transfer.resources.transfer_hold.holding
        await transfer.hang_up()

    async def test_without_a_control_plane_the_tool_refuses_and_nothing_is_held(self) -> None:
        import agent_bot

        results: list[dict[str, Any]] = []

        async def result_callback(result: dict[str, Any], **_: Any) -> None:
            results.append(result)

        params: Any = SimpleNamespace(app_resources=None, result_callback=result_callback)
        await agent_bot.transfer_call(params)

        assert results == [{"error": "this call cannot be transferred from here"}]


HANDED_OVER_VARS = {
    "transfer_next_hop": "sip:proxy.example.invalid:5060;transport=tcp",
    "transfer_from": "sip:+15550100001@example.invalid;user=phone",
    "transfer_p_asserted_identity": "tel:+15550100001",
    "transfer_privacy": "restricted",
}
BRIDGED: dict[str, Any] = {
    "kind": "ChannelBridged",
    "payload": {"role": "anchor", "anchored": True},
}
DIAL_BUSY: dict[str, Any] = {
    "kind": "DialFailed",
    "payload": {"cause": "rejected", "code": 486, "reason": "Busy Here", "timed_out": False},
}
DIAL_NOT_ANSWERED: dict[str, Any] = {
    "kind": "DialFailed",
    "payload": {"cause": "timeout", "code": 408, "reason": "Request Timeout", "timed_out": True},
}
DIAL_CALLER_GONE: dict[str, Any] = {
    "kind": "DialFailed",
    "payload": {"cause": "caller_hangup", "code": 487, "reason": "Request Terminated"},
}


class _FakeBridgingCall(_FakeTransferringCall):
    """A control call that answers a `dial` on the event stream, as siphon answers a bridge dial.

    The reply only says the destination is ringing; whether it answered arrives afterwards as
    `ChannelBridged` or `DialFailed`, so the fake queues `verdicts` when the dial is sent.
    """

    def __init__(
        self,
        verdicts: list[dict[str, Any] | None],
        handed_over: dict[str, str] | None = None,
        dial_error: BaseException | None = None,
    ) -> None:
        super().__init__(verdicts)
        self.payload: dict[str, Any] = {"vars": dict(handed_over or {})}
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self._dial_error = dial_error

    async def command(self, verb: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record the command, then answer it with the scripted verdict."""
        self.commands.append((verb, dict(args or {})))
        if self._dial_error is not None:
            raise self._dial_error
        for event in self._verdicts:
            self._stream.put_nowait(event)
        return {"state": "dialing", "on_answer": "bridge"}


@pytest.fixture
def bridge_settings(transfer_settings: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Transfer by bridging, with a stream grace short enough to run in a test."""
    import agent_bot

    transfer_settings.setattr(agent_bot, "TRANSFER_MODE", "bridge")
    transfer_settings.setattr(agent_bot, "TRANSFER_RING_TIMEOUT_SECONDS", 20)
    transfer_settings.setattr(agent_bot, "BRIDGED_STREAM_GRACE_SECONDS", QUIET_SECONDS)
    return transfer_settings


@pytest.mark.usefixtures("bridge_settings")
class TestTransferByBridging:
    """`BOT_TRANSFER_MODE=bridge`: siphon rings the destination and joins the caller to it.

    The caller's phone is never asked to do anything, so a phone that accepts a REFER and then does
    nothing with it cannot strand the transfer. The verdict the tool waits for is the dial's.
    """

    async def test_the_destination_is_dialled_with_what_the_call_was_handed(self) -> None:
        call = _FakeBridgingCall([BRIDGED], handed_over=HANDED_OVER_VARS)
        transfer = _Transfer(call)
        await transfer.attach()

        await transfer.request()

        assert call.transfers == [], "a REFER went to the caller's phone"
        assert call.commands == [
            (
                "dial",
                {
                    "on_answer": "bridge",
                    "targets": [
                        {
                            "uri": TRANSFER_DESTINATION,
                            "next_hop": "sip:proxy.example.invalid:5060;transport=tcp",
                        }
                    ],
                    "timeout": 20,
                    "from": "sip:+15550100001@example.invalid;user=phone",
                    "p_asserted_identity": "tel:+15550100001",
                    "privacy": "restricted",
                },
            )
        ]
        await transfer.hang_up()

    async def test_with_nothing_handed_over_the_dial_names_only_the_destination(self) -> None:
        call = _FakeBridgingCall([BRIDGED])
        transfer = _Transfer(call)
        await transfer.attach()

        await transfer.request()

        assert call.commands == [
            (
                "dial",
                {"on_answer": "bridge", "targets": [{"uri": TRANSFER_DESTINATION}], "timeout": 20},
            )
        ]
        await transfer.hang_up()

    async def test_a_bridged_caller_is_let_go_once_siphon_has_had_time_to_close_the_stream(
        self,
    ) -> None:
        """The stream is siphon's to close when it bridges; closing it first reads as leaving."""
        transfer = _Transfer(_FakeBridgingCall([BRIDGED]))
        await transfer.attach()

        await transfer.request()

        assert not transfer.worker.ended(), "the bot closed its stream ahead of siphon"
        assert transfer.worker.notes_to_the_model() == []
        assert not transfer.resources.transfer_hold.holding
        watch = transfer.resources.bridged_stream_watch
        assert watch is not None
        await asyncio.wait_for(watch, TIMEOUT_SECONDS)
        assert transfer.worker.ended(), "a stream siphon never closed was left open"
        await transfer.hang_up()

    @pytest.mark.parametrize(
        ("failure", "explanation"),
        [
            (DIAL_BUSY, "could not be reached"),
            (DIAL_NOT_ANSWERED, "no answer came back in time"),
        ],
    )
    async def test_a_destination_that_refuses_or_does_not_answer_keeps_the_caller(
        self, failure: dict[str, Any], explanation: str
    ) -> None:
        transfer = _Transfer(_FakeBridgingCall([failure]))
        await transfer.attach()

        await transfer.request()

        assert not transfer.worker.ended(), "the bot left a caller nobody was bridged to"
        notes = transfer.worker.notes_to_the_model()
        assert len(notes) == 1 and explanation in notes[0], notes
        assert not transfer.resources.transfer_hold.holding, "the caller is still not heard"
        assert transfer.resources.bridged_stream_watch is None
        await transfer.hang_up()

    async def test_a_caller_who_hangs_up_while_it_rings_ends_the_bot_without_recovering(
        self,
    ) -> None:
        transfer = _Transfer(_FakeBridgingCall([DIAL_CALLER_GONE]))
        await transfer.attach()

        await transfer.request()

        assert transfer.worker.ended()
        assert transfer.worker.notes_to_the_model() == [], "the bot tried to talk to nobody"
        await transfer.hang_up()

    async def test_a_dial_the_control_plane_refuses_keeps_the_caller(self) -> None:
        transfer = _Transfer(_FakeBridgingCall([], dial_error=RuntimeError("invalid_state")))
        await transfer.attach()

        await transfer.request()

        assert not transfer.worker.ended()
        notes = transfer.worker.notes_to_the_model()
        assert len(notes) == 1 and "could not be started" in notes[0], notes
        assert not transfer.resources.transfer_hold.holding
        await transfer.hang_up()


class TestBridgeVerdict:
    """A bridge dial's end, read into the verdict a REFER's end would give."""

    def test_bridged_is_completed(self) -> None:
        verdict = TransferVerdict.from_bridge_event("ChannelBridged", BRIDGED["payload"])
        assert verdict.ending == "completed" and verdict.leaves_the_call

    @pytest.mark.parametrize(
        ("cause", "stage"),
        [
            ("rejected", "refused"),
            ("timeout", "timeout"),
            ("ring timeout", "timeout"),
            ("unsent", "not_sent"),
            ("bridge_failed", "refused"),
            ("gave_up", "gave_up"),
        ],
    )
    def test_a_failed_dial_keeps_the_caller(self, cause: str, stage: str) -> None:
        verdict = TransferVerdict.from_bridge_event(
            "DialFailed", {"cause": cause, "code": 486, "reason": "Busy Here"}
        )
        assert (verdict.ending, verdict.stage, verdict.status) == ("failed", stage, 486)
        assert not verdict.leaves_the_call

    def test_the_caller_hanging_up_while_it_rings_has_left_the_call(self) -> None:
        verdict = TransferVerdict.from_bridge_event("DialFailed", DIAL_CALLER_GONE["payload"])
        assert verdict.ending == "ended" and verdict.leaves_the_call


class TestTransferModeConfiguration:
    @pytest.mark.parametrize("mode", ["refer", "bridge"])
    def test_the_two_modes_are_accepted(self, mode: str) -> None:
        refuse_an_unknown_transfer_mode(mode)

    def test_anything_else_stops_the_bot_naming_the_variable(self) -> None:
        with pytest.raises(ValueError, match="BOT_TRANSFER_MODE"):
            refuse_an_unknown_transfer_mode("replace")


class TestTransferHold:
    """While a transfer is being decided the caller's words do not reach the model.

    The phone is busy with the REFER and the destination, so a reply from the bot would talk over
    the transfer; the cleanest way to have none is to start no turn at all.
    """

    async def test_holding_drops_the_caller_and_releasing_lets_them_through(self) -> None:
        import agent_bot

        hold = agent_bot.TransferHold()
        hold.engage()
        await run_test(
            hold,
            start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                TranscriptionFrame("hello?", "caller", time_now_iso8601()),
                VADUserStoppedSpeakingFrame(),
                TextFrame("not the caller"),
            ],
            expected_down_frames=[TextFrame],
        )
        hold.release()
        await run_test(
            hold,
            start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
            frames_to_send=[TranscriptionFrame("hello?", "caller", time_now_iso8601())],
            expected_down_frames=[TranscriptionFrame],
        )

    def test_the_hold_is_in_the_pipeline_only_with_a_control_plane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        TestObserverHook._stub_the_call(monkeypatch)
        built: list[list[FrameProcessor]] = []

        def recording_pipeline(processors: list[FrameProcessor]) -> Pipeline:
            built.append(processors)
            return Pipeline(processors)

        monkeypatch.setattr(agent_bot, "Pipeline", recording_pipeline)
        websocket: Any = object()
        agent_bot.build_worker(websocket, None)
        agent_bot.build_worker(
            websocket, ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9/x")
        )

        holds = [
            [processor for processor in processors if isinstance(processor, agent_bot.TransferHold)]
            for processors in built
        ]
        assert [len(found) for found in holds] == [0, 1]


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
        "ELEVENLABS_API_KEY",
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

    def test_elevenlabs_streams_over_its_websocket_at_the_pipeline_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The WebSocket service, with the account voice, asked for PCM at the wire's own rate.

        ElevenLabs serves `pcm_16000`, so nothing between the voice and the wire resamples. A rate
        it does not serve falls back to 24 kHz with only a warning, so the mapping is pinned too.
        """
        from pipecat.services.elevenlabs.tts import (
            ElevenLabsTTSService,
            output_format_from_sample_rate,
        )

        import agent_bot

        _select(monkeypatch, "tts", "elevenlabs")
        monkeypatch.setenv("BOT_TTS_VOICE", "a-voice-on-this-account")

        service = agent_bot.build_text_to_speech()

        assert isinstance(service, ElevenLabsTTSService)
        settings: Any = service.settings
        assert settings.model == "eleven_flash_v2_5"
        assert settings.voice == "a-voice-on-this-account"
        assert service._init_sample_rate == agent_bot.PIPELINE_SAMPLE_RATE
        assert output_format_from_sample_rate(agent_bot.PIPELINE_SAMPLE_RATE) == "pcm_16000"

    def test_the_pinned_elevenlabs_model_takes_a_dutch_language_code(self) -> None:
        """A language the model does not take is dropped by pipecat with only a warning."""
        from pipecat.services.elevenlabs.tts_base import elevenlabs_language_code

        import agent_bot

        model = agent_bot.DEFAULT_TTS_MODEL["elevenlabs"]
        assert elevenlabs_language_code(model, "nl") == "nl"


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
        monkeypatch.setattr(
            agent_bot,
            "build_llm_service",
            lambda with_hangup, **_settings: FrameProcessor(),
        )
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


SPEAKING_VARIABLES = (
    "BOT_TTS_LANGUAGE",
    "BOT_TTS_LANGUAGES",
    "BOT_TTS_VOICES",
    "BOT_TTS_FOLLOW_CALLER",
)


@pytest.fixture
def _no_speaking_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every speaking-language test from a deployment that sets none of it."""
    for name in SPEAKING_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("_no_speaking_language")
class TestSpeakingLanguageConfiguration:
    """What the deployment says the bot may speak, read and checked before any call."""

    def test_nothing_set_is_no_language_at_all(self) -> None:
        """Unset has to mean today's behaviour: the synthesizer's own default, no tool."""
        from agent_bot import resolve_speaking_languages

        languages = resolve_speaking_languages({}, "cartesia", "voice")

        assert languages.starting is None
        assert languages.available == ()
        assert not languages.switchable

    def test_one_language_is_spoken_from_the_start_with_no_switch(self) -> None:
        from agent_bot import resolve_speaking_languages

        languages = resolve_speaking_languages({"BOT_TTS_LANGUAGE": "nl"}, "cartesia", "voice")

        assert languages.starting == "nl"
        assert languages.available == ("nl",)
        assert not languages.switchable

    def test_a_list_starts_in_its_first_language_unless_told_otherwise(self) -> None:
        from agent_bot import resolve_speaking_languages

        listed = resolve_speaking_languages({"BOT_TTS_LANGUAGES": "nl, en"}, "cartesia", "voice")
        chosen = resolve_speaking_languages(
            {"BOT_TTS_LANGUAGES": "nl,en", "BOT_TTS_LANGUAGE": "en"}, "cartesia", "voice"
        )

        assert listed.starting == "nl" and listed.available == ("nl", "en")
        assert listed.switchable
        assert chosen.starting == "en"

    def test_a_starting_language_outside_the_list_is_refused(self) -> None:
        from agent_bot import resolve_speaking_languages

        with pytest.raises(ValueError, match="BOT_TTS_LANGUAGE 'de'"):
            resolve_speaking_languages(
                {"BOT_TTS_LANGUAGES": "nl,en", "BOT_TTS_LANGUAGE": "de"}, "cartesia", "voice"
            )

    def test_the_voice_map_gives_each_language_its_own_voice(self) -> None:
        from agent_bot import resolve_speaking_languages

        languages = resolve_speaking_languages(
            {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_VOICES": "nl=dutch-voice, en=english-voice"},
            "cartesia",
            "account-voice",
        )

        assert languages.voices == {"nl": "dutch-voice", "en": "english-voice"}

    @pytest.mark.parametrize("voices", ["nl", "nl=", "=voice", "nl=a,nl=b"])
    def test_a_malformed_voice_map_is_refused(self, voices: str) -> None:
        from agent_bot import resolve_speaking_languages

        with pytest.raises(ValueError, match="BOT_TTS_VOICES"):
            resolve_speaking_languages(
                {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_VOICES": voices}, "cartesia", "voice"
            )

    def test_a_voice_for_a_language_nobody_configured_is_refused(self) -> None:
        """A typo in the map would otherwise leave the language it meant on the wrong voice."""
        from agent_bot import resolve_speaking_languages

        with pytest.raises(ValueError, match="'de'"):
            resolve_speaking_languages(
                {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_VOICES": "de=voice"}, "cartesia", "voice"
            )

    def test_following_the_caller_needs_a_second_language(self) -> None:
        from agent_bot import resolve_speaking_languages

        with pytest.raises(ValueError, match="BOT_TTS_FOLLOW_CALLER"):
            resolve_speaking_languages(
                {"BOT_TTS_LANGUAGE": "nl", "BOT_TTS_FOLLOW_CALLER": "true"}, "cartesia", "voice"
            )
        following = resolve_speaking_languages(
            {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_FOLLOW_CALLER": "true"}, "cartesia", "voice"
        )
        assert following.follow_caller

    def test_the_local_voice_refuses_a_language_it_cannot_speak(self) -> None:
        """Kokoro has no Dutch: refusing at start beats an English voice reading Dutch."""
        from agent_bot import resolve_speaking_languages

        with pytest.raises(ValueError) as raised:
            resolve_speaking_languages({"BOT_TTS_LANGUAGES": "en,nl"}, "local", "af_heart")

        message = str(raised.value)
        assert "'nl'" in message
        assert "BOT_TTS_PROVIDER" in message

    def test_the_local_voice_refuses_a_voice_of_another_language(self) -> None:
        """Kokoro can speak Spanish, but not with its American English voice."""
        from agent_bot import resolve_speaking_languages

        with pytest.raises(ValueError, match="BOT_TTS_VOICES"):
            resolve_speaking_languages({"BOT_TTS_LANGUAGES": "en,es"}, "local", "af_heart")
        languages = resolve_speaking_languages(
            {"BOT_TTS_LANGUAGES": "en,es", "BOT_TTS_VOICES": "es=ef_dora"}, "local", "af_heart"
        )
        assert languages.available == ("en", "es")

    def test_the_languages_the_local_voice_accepts_are_ones_pipecat_maps(self) -> None:
        """Read from pipecat's installed source: its Kokoro module cannot be imported here."""
        import ast

        from agent_bot import KOKORO_VOICE_PREFIXES

        spec = importlib.util.find_spec("pipecat")
        assert spec is not None and spec.submodule_search_locations
        source = (
            Path(next(iter(spec.submodule_search_locations))) / "services" / "kokoro" / "tts.py"
        ).read_text()
        mapped: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == "language_to_kokoro_language":
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Dict):
                        for key in inner.keys:
                            if isinstance(key, ast.Attribute):
                                mapped.add(getattr(Language, key.attr).value.split("-")[0])
        assert mapped, "language_to_kokoro_language no longer holds a map"
        assert set(KOKORO_VOICE_PREFIXES) <= mapped
        assert "nl" not in mapped, "pipecat's Kokoro maps Dutch now; KOKORO_VOICE_PREFIXES can too"


def _record_settings(monkeypatch: pytest.MonkeyPatch, service_class: Any) -> list[dict[str, Any]]:
    """Record the keyword arguments every `service_class.Settings(...)` is built with.

    The builder's own call is the first: the service's constructor builds its defaults through the
    same name afterwards.
    """
    real_settings = service_class.Settings
    recorded: list[dict[str, Any]] = []

    def recording(**keyword_arguments: Any) -> Any:
        recorded.append(keyword_arguments)
        return real_settings(**keyword_arguments)

    monkeypatch.setattr(service_class, "Settings", staticmethod(recording))
    return recorded


class _FakeVoiceService:
    """Stands in for a voice whose constructor needs credentials or a model on disk."""

    Settings: Any = None
    built: list[dict[str, Any]]

    def __init__(self, **keyword_arguments: Any) -> None:
        type(self).built.append(keyword_arguments)
        self.settings = keyword_arguments["settings"]


@pytest.mark.usefixtures("_vendor_credentials", "_no_speaking_language")
class TestSpeakingLanguageReachesEveryVoice:
    """The configured language reaches each synthesizer that takes one, and only when set."""

    def test_cartesia_is_built_exactly_as_before_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.services.cartesia.tts import CartesiaTTSService

        import agent_bot

        _select(monkeypatch, "tts", "cartesia")
        recorded = _record_settings(monkeypatch, CartesiaTTSService)

        service = agent_bot.build_text_to_speech()

        assert recorded[0] == {
            "voice": "test-value",
            "model": agent_bot.DEFAULT_TTS_MODEL["cartesia"],
        }
        # Cartesia's own default, which is the accent this change exists to remove.
        settings: Any = service.settings
        assert settings.language == "en"

    def test_cartesia_speaks_the_configured_language_with_its_voice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        _select(monkeypatch, "tts", "cartesia")
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")
        monkeypatch.setenv("BOT_TTS_VOICES", "nl=a-dutch-voice")

        service = agent_bot.build_text_to_speech()

        settings: Any = service.settings
        assert settings.language == "nl"
        assert settings.voice == "a-dutch-voice"

    def test_cartesia_keeps_the_account_voice_when_the_map_names_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multilingual voice with the language set is the other way to lose the accent."""
        import agent_bot

        _select(monkeypatch, "tts", "cartesia")
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")
        monkeypatch.setenv("CARTESIA_VOICE_ID", "a-multilingual-voice")

        settings: Any = agent_bot.build_text_to_speech().settings

        assert settings.language == "nl"
        assert settings.voice == "a-multilingual-voice"

    def test_elevenlabs_is_built_without_a_language_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        import agent_bot

        _select(monkeypatch, "tts", "elevenlabs")
        monkeypatch.setenv("BOT_TTS_VOICE", "an-account-voice")
        recorded = _record_settings(monkeypatch, ElevenLabsTTSService)

        service = agent_bot.build_text_to_speech()

        assert recorded[0] == {
            "voice": "an-account-voice",
            "model": agent_bot.DEFAULT_TTS_MODEL["elevenlabs"],
        }
        settings: Any = service.settings
        assert settings.language is None

    def test_elevenlabs_asks_for_the_configured_language_with_its_voice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Read back from the URL the service connects to, which is where the language travels."""
        from pipecat.services.elevenlabs.tts import output_format_from_sample_rate

        import agent_bot

        _select(monkeypatch, "tts", "elevenlabs")
        monkeypatch.setenv("BOT_TTS_VOICE", "an-account-voice")
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")
        monkeypatch.setenv("BOT_TTS_VOICES", "nl=a-dutch-voice")

        service: Any = agent_bot.build_text_to_speech()
        # Set in `setup` from the resolved rate, which a test without a pipeline does not run.
        service._output_format = output_format_from_sample_rate(agent_bot.PIPELINE_SAMPLE_RATE)
        url = service._build_websocket_url()

        assert service.settings.language == "nl"
        assert "/v1/text-to-speech/a-dutch-voice/" in url
        assert "language_code=nl" in url
        assert "output_format=pcm_16000" in url

    def test_openai_is_built_exactly_as_before_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.services.openai.tts import OpenAITTSService

        import agent_bot

        _select(monkeypatch, "tts", "openai")
        recorded = _record_settings(monkeypatch, OpenAITTSService)

        agent_bot.build_text_to_speech()

        assert recorded[0] == {"voice": "alloy", "model": agent_bot.DEFAULT_TTS_MODEL["openai"]}

    def test_openai_is_told_the_language_and_asked_for_a_native_accent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI's endpoint takes no language: its instructions are the only accent control."""
        import agent_bot

        _select(monkeypatch, "tts", "openai")
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")

        settings: Any = agent_bot.build_text_to_speech().settings

        assert settings.language == Language.NL
        assert "nl" in settings.instructions and "native" in settings.instructions

    @staticmethod
    def _fake_google(monkeypatch: pytest.MonkeyPatch) -> type[_FakeVoiceService]:
        import pipecat.services.google.tts as google_tts

        fake = type(
            "FakeGoogleTTSService",
            (_FakeVoiceService,),
            {"Settings": google_tts.GoogleTTSService.Settings, "built": []},
        )
        monkeypatch.setattr(google_tts, "GoogleTTSService", fake)
        return fake

    def test_google_is_built_exactly_as_before_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        _select(monkeypatch, "tts", "google")
        fake = self._fake_google(monkeypatch)

        agent_bot.build_text_to_speech()

        settings: Any = fake.built[0]["settings"]
        assert settings.voice == "en-US-Chirp3-HD-Charon"
        assert not isinstance(settings.language, (str, Language))

    def test_google_gets_the_language_and_the_same_voice_in_that_locale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Google's voice names carry their locale, and a voice of another one is refused."""
        import agent_bot

        _select(monkeypatch, "tts", "google")
        fake = self._fake_google(monkeypatch)
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")

        agent_bot.build_text_to_speech()

        settings: Any = fake.built[0]["settings"]
        assert settings.language == Language.NL
        assert settings.voice == "nl-NL-Chirp3-HD-Charon"

    @staticmethod
    def _fake_kokoro(monkeypatch: pytest.MonkeyPatch) -> type[_FakeVoiceService]:
        import sys
        import types

        from pipecat.services.settings import TTSSettings

        module = types.ModuleType("pipecat.services.kokoro.tts")
        fake = type(
            "FakeKokoroTTSService", (_FakeVoiceService,), {"Settings": TTSSettings, "built": []}
        )
        module.KokoroTTSService = fake  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pipecat.services.kokoro.tts", module)
        return fake

    def test_kokoro_is_built_exactly_as_before_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        _select(monkeypatch, "tts", "local")
        fake = self._fake_kokoro(monkeypatch)

        agent_bot.build_text_to_speech()

        settings: Any = fake.built[0]["settings"]
        assert settings.voice == "af_heart"
        assert not isinstance(settings.language, (str, Language))

    def test_kokoro_gets_the_language_and_its_voice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        _select(monkeypatch, "tts", "local")
        fake = self._fake_kokoro(monkeypatch)
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "es")
        monkeypatch.setenv("BOT_TTS_VOICES", "es=ef_dora")

        agent_bot.build_text_to_speech()

        settings: Any = fake.built[0]["settings"]
        assert settings.language == Language.ES
        assert settings.voice == "ef_dora"


class _FakeLanguageModel:
    """Stands in for the LLM service a tool handler pushes its frames through."""

    def __init__(self) -> None:
        self.pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append((frame, direction))


class _ToolCall:
    """The parts of `FunctionCallParams` the language tool reads."""

    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments
        self.llm = _FakeLanguageModel()
        self.results: list[Any] = []

    async def result_callback(self, result: Any, **_keyword_arguments: Any) -> None:
        self.results.append(result)


def _controller(
    environment: dict[str, str], *, provider: str = "cartesia", recognizer_language: str = "multi"
) -> Any:
    from agent_bot import SpeakingLanguageController, resolve_speaking_languages

    languages = resolve_speaking_languages(environment, provider, "account-voice")
    return SpeakingLanguageController(
        languages,
        provider=provider,
        default_voice="account-voice",
        recognizer_language=recognizer_language,
    )


@pytest.mark.usefixtures("_no_speaking_language")
class TestSpeakingLanguageTool:
    """The model switches the bot's language when the caller asks, and only to a configured one."""

    def test_the_tool_exists_only_with_a_language_to_switch_to(self) -> None:
        from agent_bot import build_tool_schemas

        two = _controller({"BOT_TTS_LANGUAGES": "en,nl"})
        offered = {schema.name: schema for schema in build_tool_schemas(False, two)}
        assert set(offered) == {"set_speaking_language"}
        assert offered["set_speaking_language"].properties["language"]["enum"] == ["en", "nl"]

        assert build_tool_schemas(False, None) == []
        with_control = [schema.name for schema in build_tool_schemas(True, None)]
        assert "set_speaking_language" not in with_control
        assert "end_call" in with_control

    def test_one_configured_language_builds_no_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")
        assert agent_bot.build_speaking_language_controller() is None
        monkeypatch.setenv("BOT_TTS_LANGUAGES", "nl,en")
        assert agent_bot.build_speaking_language_controller() is not None

    async def test_the_tool_changes_the_voice_for_the_rest_of_the_call(self) -> None:
        from pipecat.frames.frames import STTUpdateSettingsFrame, TTSUpdateSettingsFrame

        controller = _controller(
            {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_VOICES": "nl=a-dutch-voice"}
        )
        call = _ToolCall({"language": "nl"})

        await controller.set_speaking_language(call)

        assert controller.current == "nl"
        assert len(call.llm.pushed) == 1
        frame, direction = call.llm.pushed[0]
        assert isinstance(frame, TTSUpdateSettingsFrame)
        assert direction == FrameDirection.DOWNSTREAM
        assert frame.delta is not None
        assert frame.delta.language == Language.NL
        assert frame.delta.voice == "a-dutch-voice"
        assert not any(isinstance(pushed, STTUpdateSettingsFrame) for pushed, _ in call.llm.pushed)
        assert call.results == [{"speaking_language": "nl"}]

    async def test_a_pinned_recognizer_is_switched_too(self) -> None:
        """A recognizer on `multi` hears every language already; one pinned to English does not."""
        from pipecat.frames.frames import STTUpdateSettingsFrame

        controller = _controller({"BOT_TTS_LANGUAGES": "en,nl"}, recognizer_language="en-US")
        call = _ToolCall({"language": "nl"})

        await controller.set_speaking_language(call)

        upstream = [
            frame for frame, direction in call.llm.pushed if direction == FrameDirection.UPSTREAM
        ]
        assert len(upstream) == 1
        assert isinstance(upstream[0], STTUpdateSettingsFrame)
        assert upstream[0].delta is not None and upstream[0].delta.language == Language.NL

    async def test_a_language_nobody_configured_is_refused(self) -> None:
        controller = _controller({"BOT_TTS_LANGUAGES": "en,nl"})
        call = _ToolCall({"language": "de"})

        await controller.set_speaking_language(call)

        assert controller.current == "en"
        assert call.llm.pushed == []
        assert "error" in call.results[0]

    async def test_the_real_voice_service_takes_the_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Applied to a built Cartesia service, not only shaped like a frame it would take."""
        import agent_bot

        monkeypatch.setenv("CARTESIA_API_KEY", "test-value")
        monkeypatch.setenv("CARTESIA_VOICE_ID", "account-voice")
        _select(monkeypatch, "tts", "cartesia")
        service = agent_bot.build_text_to_speech()
        controller = _controller(
            {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_VOICES": "nl=a-dutch-voice"}
        )
        call = _ToolCall({"language": "nl"})

        await controller.set_speaking_language(call)
        frame: Any = call.llm.pushed[0][0]
        await service._update_settings(frame.delta)

        settings: Any = service.settings
        assert settings.language == "nl"
        assert settings.voice == "a-dutch-voice"

    def test_the_prompt_says_how_to_switch_only_when_it_can(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        unset: Any = agent_bot.build_llm_service(with_hangup=False).settings
        assert unset.system_instruction == agent_bot.resolve_system_prompt({})

        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")
        single: Any = agent_bot.build_llm_service(with_hangup=False).settings
        assert "nl" in single.system_instruction
        assert "set_speaking_language" not in single.system_instruction

        monkeypatch.setenv("BOT_TTS_LANGUAGES", "nl,en")
        switchable: Any = agent_bot.build_llm_service(with_hangup=False).settings
        assert "set_speaking_language" in switchable.system_instruction


async def _hear(controller: Any, stretches: list[Language | None]) -> list[Frame]:
    """Feed one transcript per stretch of caller speech, in `stretches`' languages."""
    pushed: list[Frame] = []

    async def capture(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    controller.push_frame = capture
    for language in stretches:
        await controller.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await controller.process_frame(
            TranscriptionFrame("words", "caller", time_now_iso8601(), language),
            FrameDirection.DOWNSTREAM,
        )
    return pushed


def _switches(pushed: list[Frame]) -> list[Any]:
    from pipecat.frames.frames import TTSUpdateSettingsFrame

    return [
        frame.delta.language
        for frame in pushed
        if isinstance(frame, TTSUpdateSettingsFrame) and frame.delta is not None
    ]


FOLLOWING = {"BOT_TTS_LANGUAGES": "en,nl", "BOT_TTS_FOLLOW_CALLER": "true"}


@pytest.mark.usefixtures("_no_speaking_language")
class TestFollowTheCaller:
    """The voice follows the language the recognizer hears, once it has heard it consistently."""

    async def test_consistent_turns_switch_the_voice(self) -> None:
        from agent_bot import FOLLOW_CALLER_CONSISTENT_TURNS

        controller = _controller(FOLLOWING)

        pushed = await _hear(controller, [Language.NL] * FOLLOW_CALLER_CONSISTENT_TURNS)

        assert _switches(pushed) == [Language.NL]
        assert controller.current == "nl"
        # The switch goes ahead of the transcript, so the reply to it is already in the new voice.
        from pipecat.frames.frames import TTSUpdateSettingsFrame

        first_switch = next(
            index for index, frame in enumerate(pushed) if isinstance(frame, TTSUpdateSettingsFrame)
        )
        assert isinstance(pushed[first_switch + 1], TranscriptionFrame)

    async def test_one_misheard_turn_does_not_flip_the_voice(self) -> None:
        controller = _controller(FOLLOWING)

        pushed = await _hear(controller, [Language.NL, Language.EN, Language.NL, Language.EN])

        assert _switches(pushed) == []
        assert controller.current == "en"

    async def test_several_transcripts_in_one_stretch_count_once(self) -> None:
        controller = _controller(FOLLOWING)
        pushed: list[Frame] = []

        async def capture(
            frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
        ) -> None:
            pushed.append(frame)

        controller.push_frame = capture
        await controller.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        for _ in range(3):
            await controller.process_frame(
                TranscriptionFrame("woord", "caller", time_now_iso8601(), Language.NL),
                FrameDirection.DOWNSTREAM,
            )

        assert _switches(pushed) == []

    async def test_a_regional_report_matches_its_configured_language(self) -> None:
        controller = _controller(FOLLOWING)

        pushed = await _hear(controller, [Language.NL_BE, Language.NL_NL])

        assert _switches(pushed) == [Language.NL]

    async def test_unconfigured_and_unreported_languages_are_ignored(self) -> None:
        controller = _controller(FOLLOWING)

        pushed = await _hear(controller, [Language.DE, None, Language.DE])

        assert _switches(pushed) == []

    async def test_without_the_option_the_voice_never_follows(self) -> None:
        controller = _controller({"BOT_TTS_LANGUAGES": "en,nl"})

        pushed = await _hear(controller, [Language.NL] * 5)

        assert _switches(pushed) == []

    async def test_an_explicit_request_wins_over_following(self) -> None:
        """Told to speak Dutch by an English speaker, the bot must not drift back to English."""
        controller = _controller(FOLLOWING)
        await controller.set_speaking_language(_ToolCall({"language": "nl"}))

        pushed = await _hear(controller, [Language.EN] * 5)

        assert _switches(pushed) == []
        assert controller.current == "nl"


# The caller's number on these calls. Country code 999 is unassigned, so this is nobody's number.
CALLER = "99900000001"


class TestCallerNumberFromTheStreamUrl:
    """The number in the media socket's `caller` query parameter, and nothing that is not one."""

    @pytest.mark.parametrize("value", ["999000", CALLER, "999000000000001"])
    def test_six_to_fifteen_digits_is_a_number(self, value: str) -> None:
        from agent_bot import parse_caller

        assert parse_caller(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "99900",
            "9990000000000001",
            "+99900000001",
            "999 000 00001",
            "99900000001\n",
            "9990000000a",
            # Arabic-Indic digits: `str.isdigit` says yes, and a URL path should never see them.
            "٩٩٩٠٠٠٠٠٠٠١",
        ],
    )
    def test_anything_else_is_absent(self, value: str | None) -> None:
        from agent_bot import parse_caller

        assert parse_caller(value) is None

    def test_a_logged_number_shows_only_its_last_digits(self) -> None:
        from agent_bot import mask_number

        masked = mask_number(CALLER)
        assert masked.endswith("01")
        assert CALLER[:-2] not in masked
        assert len(masked) == len(CALLER)

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ({"call": "abc", "caller": CALLER}, CALLER),
            ({"call": "abc", "caller": "+1"}, None),
            ({}, None),
        ],
    )
    async def test_the_media_route_hands_the_call_its_caller(
        self, monkeypatch: pytest.MonkeyPatch, query: dict[str, str], expected: str | None
    ) -> None:
        from fastapi.routing import APIWebSocketRoute

        import agent_bot

        served: list[str | None] = []

        async def serve_call(
            _websocket: object,
            _control: object,
            _observer_factory: object = None,
            caller: str | None = None,
            _placed: object = None,
            **_rest: object,
        ) -> None:
            served.append(caller)

        monkeypatch.setattr(agent_bot, "serve_call", serve_call)
        app = agent_bot.build_app(None)
        route = next(route for route in app.routes if isinstance(route, APIWebSocketRoute))
        websocket: Any = SimpleNamespace(query_params=query)

        await route.endpoint(websocket)

        assert served == [expected]


def _account(**overrides: Any) -> dict[str, Any]:
    """Build a lookup answer for a made-up account, plus `overrides`."""
    body: dict[str, Any] = {
        "name": "Test User",
        "plan": "Basic",
        "balance": 12.5,
        "open_tickets": 1,
    }
    body.update(overrides)
    return body


class _FakeHttpService:
    """A loopback HTTP server answering each path with a configured status, body and delay."""

    def __init__(self) -> None:
        self.answers: dict[str, tuple[int, Any, float]] = {}
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.base_url = ""
        self._server: Any = None

    def answer(
        self, path: str, status: int, body: Any = None, delay: float = 0.0, raw: bytes | None = None
    ) -> None:
        self.answers[path] = (status, raw if raw is not None else body, delay)

    async def start(self) -> str:
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def handle(request: web.Request) -> web.StreamResponse:
            self.requests.append((request.path, dict(request.query)))
            status, body, delay = self.answers.get(request.path, (404, None, 0.0))
            if delay:
                await asyncio.sleep(delay)
            if body is None:
                return web.Response(status=status)
            if isinstance(body, bytes):
                return web.Response(status=status, body=body, content_type="text/html")
            return web.json_response(body, status=status)

        app = web.Application()
        app.router.add_get("/{tail:.*}", handle)
        self._server = TestServer(app, host="127.0.0.1")
        await self._server.start_server()
        return str(self._server.make_url("")).rstrip("/")

    async def close(self) -> None:
        if self._server is not None:
            await self._server.close()


@pytest.fixture
async def http_service() -> Any:
    """Serve fake answers on loopback for the duration of one test."""
    service = _FakeHttpService()
    service.base_url = await service.start()
    yield service
    await service.close()


class _ResultOnly:
    """The parts of `FunctionCallParams` a lookup tool reads."""

    def __init__(self, arguments: dict[str, Any] | None = None) -> None:
        self.arguments = arguments or {}
        self.results: list[Any] = []

    async def result_callback(self, result: Any, **_keyword_arguments: Any) -> None:
        self.results.append(result)


ACCOUNT_PATH = f"/customers/{CALLER}"


async def _ask_caller_lookup(base_url: str, caller: str | None) -> dict[str, Any]:
    from agent_bot import CallerLookup

    call = _ResultOnly()
    await CallerLookup(base_url + "/customers/{caller}", caller).caller_lookup(call)  # type: ignore[arg-type]
    assert len(call.results) == 1
    result: dict[str, Any] = call.results[0]
    return result


class TestCallerLookupTool:
    """The model can ask for the caller's own details, and only theirs."""

    async def test_no_caller_on_the_call_is_not_available(self, http_service: Any) -> None:
        result = await _ask_caller_lookup(http_service.base_url, None)

        assert result["available"] is False
        assert http_service.requests == [], "looked somebody up without a number"

    async def test_the_answer_is_passed_through(self, http_service: Any) -> None:
        http_service.answer(ACCOUNT_PATH, 200, _account())

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert http_service.requests[0][0] == ACCOUNT_PATH
        assert result == {"available": True, "details": _account()}

    async def test_one_level_of_lists_and_objects_is_kept(self, http_service: Any) -> None:
        http_service.answer(
            ACCOUNT_PATH,
            200,
            _account(
                tags=["new", 3, True, None, {"too": "deep"}, ["deeper"]],
                address={"city": "Testville", "floor": 2, "extra": {"too": "deep"}, "none": None},
                note=None,
                history={"deep": {"er": 1}},
            ),
        )

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        details = result["details"]
        assert details["tags"] == ["new", 3, True]
        assert details["address"] == {"city": "Testville", "floor": 2}
        assert "note" not in details
        assert details["history"] == {}

    async def test_an_answer_over_the_cap_is_not_available(self, http_service: Any) -> None:
        http_service.answer(ACCOUNT_PATH, 200, _account(notes="x" * 3000))

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert result["available"] is False

    async def test_an_unknown_caller(self, http_service: Any) -> None:
        http_service.answer(ACCOUNT_PATH, 404)

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert result["available"] is False
        assert "not known" in result["reason"]

    @pytest.mark.parametrize("status", [401, 500, 503])
    async def test_any_other_failure_status_is_not_available(
        self, http_service: Any, status: int
    ) -> None:
        http_service.answer(ACCOUNT_PATH, status, {"error": "boom"})

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert result["available"] is False

    async def test_an_answer_that_is_not_an_object_is_not_available(
        self, http_service: Any
    ) -> None:
        http_service.answer(ACCOUNT_PATH, 200, ["not", "an", "object"])

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert result["available"] is False

    async def test_an_answer_that_is_not_json_is_not_available(self, http_service: Any) -> None:
        http_service.answer(ACCOUNT_PATH, 200, raw=b"<html>not json</html>")

        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert result["available"] is False

    async def test_a_slow_answer_is_given_up_on(
        self, http_service: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_TIMEOUT_SECONDS", 0.1)
        http_service.answer(ACCOUNT_PATH, 200, _account(), delay=1.0)

        started = time.monotonic()
        result = await _ask_caller_lookup(http_service.base_url, CALLER)

        assert time.monotonic() - started < 0.9, "the timeout did not bound the lookup"
        assert result["available"] is False

    async def test_nothing_listening_is_not_available(self) -> None:
        # Port 9 on loopback: discard, which nothing in a test environment serves.
        result = await _ask_caller_lookup("http://127.0.0.1:9", CALLER)

        assert result["available"] is False

    async def test_the_number_is_never_logged_whole(self, http_service: Any) -> None:
        from loguru import logger

        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
        try:
            for status in (200, 404, 503):
                http_service.answer(ACCOUNT_PATH, status, _account() if status == 200 else None)
                await _ask_caller_lookup(http_service.base_url, CALLER)
            await _ask_caller_lookup("http://127.0.0.1:9", CALLER)
        finally:
            logger.remove(sink)

        assert lines, "nothing was logged at all, so this test proves nothing"
        assert not [line for line in lines if CALLER in line]

    def test_the_tool_takes_no_arguments_and_says_what_it_is_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_DESCRIPTION", "Look up the caller's plan.")
        lookup = agent_bot.CallerLookup("http://127.0.0.1:9/customers/{caller}", CALLER)
        offered = {
            schema.name: schema
            for schema in agent_bot.build_tool_schemas(False, None, caller_lookup=lookup)
        }

        assert set(offered) == {"caller_lookup"}
        assert offered["caller_lookup"].properties == {}
        assert offered["caller_lookup"].required == []
        assert offered["caller_lookup"].description == "Look up the caller's plan."

    def test_a_url_with_nowhere_to_put_the_number_is_refused(self) -> None:
        from agent_bot import refuse_a_caller_lookup_url_without_the_caller

        with pytest.raises(ValueError, match="BOT_CALLER_LOOKUP_URL"):
            refuse_a_caller_lookup_url_without_the_caller("https://accounts.example.invalid/me")
        refuse_a_caller_lookup_url_without_the_caller("")
        refuse_a_caller_lookup_url_without_the_caller(
            "https://accounts.example.invalid/customers/{caller}"
        )


def _record_tool_lists(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Build calls around stand-ins, and record the name of every tool each call offers."""
    import agent_bot

    TestObserverHook._stub_the_call(monkeypatch)
    offered: list[list[str]] = []
    original = agent_bot.build_tool_schemas

    def recording(*arguments: Any, **keyword_arguments: Any) -> Any:
        tools = original(*arguments, **keyword_arguments)
        offered.append([tool.name for tool in tools])
        return tools

    monkeypatch.setattr(agent_bot, "build_tool_schemas", recording)
    return offered


@pytest.mark.usefixtures("_no_speaking_language")
class TestLookupToolsAreOfferedOnlyWhenConfigured:
    """Neither lookup is offered, nor promised in the prompt, unless the deployment turned it on."""

    def test_without_a_lookup_url_there_is_no_caller_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        offered = _record_tool_lists(monkeypatch)
        websocket: Any = object()

        agent_bot.build_worker(websocket, None, caller=CALLER)

        assert offered == [[]]

    def test_with_both_configured_both_are_offered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "http://127.0.0.1:9/customers/{caller}")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", True)
        offered = _record_tool_lists(monkeypatch)
        websocket: Any = object()

        # Offered without a caller too: the tool then says it cannot see the line, which is what
        # the caller should hear, rather than the bot not knowing it could ever look.
        agent_bot.build_worker(websocket, None, caller=None)

        assert offered == [["caller_lookup", "weather"]]

    def test_the_prompt_mentions_only_what_is_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        neither: Any = agent_bot.build_llm_service(with_hangup=False).settings
        assert neither.system_instruction == agent_bot.resolve_system_prompt({})

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "http://127.0.0.1:9/customers/{caller}")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", True)
        both: Any = agent_bot.build_llm_service(with_hangup=False).settings
        assert "caller_lookup" in both.system_instruction
        assert "weather" in both.system_instruction

    @pytest.mark.parametrize(
        ("value", "enabled"),
        [("", False), ("off", False), ("false", False), ("on", True), ("true", True)],
    )
    def test_the_weather_switch(self, value: str, enabled: bool) -> None:
        from agent_bot import is_switched_on

        assert is_switched_on(value) is enabled


GEOCODING_PATH = "/v1/search"
FORECAST_PATH = "/v1/forecast"


@pytest.fixture
def weather_service(http_service: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the weather tool's two endpoints at the loopback fake."""
    import agent_bot

    monkeypatch.setattr(agent_bot, "WEATHER_GEOCODING_URL", http_service.base_url + GEOCODING_PATH)
    monkeypatch.setattr(agent_bot, "WEATHER_FORECAST_URL", http_service.base_url + FORECAST_PATH)
    return http_service


async def _ask_weather(place: Any) -> dict[str, Any]:
    from agent_bot import report_weather

    call = _ResultOnly({"place": place})
    await report_weather(call)  # type: ignore[arg-type]
    assert len(call.results) == 1
    result: dict[str, Any] = call.results[0]
    return result


class TestWeatherTool:
    """The model can say what the weather is like in a place it names."""

    async def test_the_current_weather_in_a_place(self, weather_service: Any) -> None:
        weather_service.answer(
            GEOCODING_PATH,
            200,
            {
                "results": [
                    {
                        "name": "Testville",
                        "country": "Testland",
                        "latitude": 1.5,
                        "longitude": -2.25,
                    }
                ]
            },
        )
        weather_service.answer(
            FORECAST_PATH,
            200,
            {
                "current": {
                    "time": "2026-10-13T09:30",
                    "temperature_2m": 17.6,
                    "weather_code": 61,
                    "wind_speed_10m": 12.4,
                }
            },
        )

        result = await _ask_weather("Testville")

        assert result == {
            "available": True,
            "place": "Testville, Testland",
            "temperature_celsius": 18,
            "conditions": "light rain",
            "wind_kilometres_per_hour": 12,
            "observed_at": "2026-10-13T09:30",
        }
        (geocoding_path, geocoding_query), (forecast_path, forecast_query) = (
            weather_service.requests
        )
        assert geocoding_path == GEOCODING_PATH
        assert geocoding_query == {"name": "Testville", "count": "1"}
        assert forecast_path == FORECAST_PATH
        assert forecast_query == {
            "latitude": "1.5",
            "longitude": "-2.25",
            "current": "temperature_2m,weather_code,wind_speed_10m",
        }

    async def test_an_unknown_code_still_gives_the_temperature(self, weather_service: Any) -> None:
        weather_service.answer(
            GEOCODING_PATH,
            200,
            {"results": [{"name": "Testville", "latitude": 1.0, "longitude": 2.0}]},
        )
        weather_service.answer(
            FORECAST_PATH, 200, {"current": {"temperature_2m": -3.2, "weather_code": 42}}
        )

        result = await _ask_weather("Testville")

        assert result == {"available": True, "place": "Testville", "temperature_celsius": -3}

    async def test_a_place_nobody_knows(self, weather_service: Any) -> None:
        weather_service.answer(GEOCODING_PATH, 200, {"generationtime_ms": 0.1})

        result = await _ask_weather("Nowhere At All")

        assert result["available"] is False
        assert len(weather_service.requests) == 1, "asked for a forecast with nowhere to ask about"

    @pytest.mark.parametrize("place", ["", "   ", None, 42])
    async def test_no_place_is_not_available(self, weather_service: Any, place: Any) -> None:
        result = await _ask_weather(place)

        assert result["available"] is False
        assert weather_service.requests == []

    async def test_a_failing_forecast_is_not_available(self, weather_service: Any) -> None:
        weather_service.answer(
            GEOCODING_PATH,
            200,
            {"results": [{"name": "Testville", "latitude": 1.0, "longitude": 2.0}]},
        )
        weather_service.answer(FORECAST_PATH, 502)

        result = await _ask_weather("Testville")

        assert result["available"] is False

    async def test_a_slow_service_is_given_up_on(
        self, weather_service: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "WEATHER_TIMEOUT_SECONDS", 0.1)
        weather_service.answer(GEOCODING_PATH, 200, {"results": []}, delay=1.0)

        started = time.monotonic()
        result = await _ask_weather("Testville")

        assert time.monotonic() - started < 0.9
        assert result["available"] is False

    async def test_nothing_listening_is_not_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "WEATHER_GEOCODING_URL", "http://127.0.0.1:9/v1/search")

        result = await _ask_weather("Testville")

        assert result["available"] is False

    def test_the_model_names_only_a_place(self) -> None:
        from agent_bot import build_tool_schemas

        offered = {schema.name: schema for schema in build_tool_schemas(False, None, weather=True)}

        assert set(offered) == {"weather"}
        assert set(offered["weather"].properties) == {"place"}
        assert offered["weather"].required == ["place"]


PBX_TOKEN = "9f86d081884c7d659a2feaa0c55ad015"
PBX_SECRET = "s3cret-for-tests-only"


def _menu_brief() -> Any:
    from pbx_fixtures import BRIEF_MENU
    from pipecat_siphon import parse_brief

    brief = parse_brief(BRIEF_MENU)
    assert brief is not None
    return brief


def _open_brief() -> Any:
    from pbx_fixtures import BRIEF_OPEN
    from pipecat_siphon import parse_brief

    brief = parse_brief(BRIEF_OPEN)
    assert brief is not None
    return brief


def _briefed_call(pbx: Any, brief: Any) -> tuple[Any, _FakeWorker]:
    """Hand over a call with `brief`, its line already played and its worker recorded."""
    from agent_bot import BriefedCall
    from pipecat_siphon import VoiceSession, VoiceSessionClient

    session = VoiceSession(VoiceSessionClient(pbx.url, PBX_SECRET), PBX_TOKEN, brief)
    worker = _FakeWorker()
    briefed = BriefedCall(session=session, speech=_FinishedSpeech())  # type: ignore[arg-type]
    briefed.worker = worker  # type: ignore[assignment]
    return briefed, worker


class TestThePbxBriefIsFetchedBeforeTheBotSpeaks:
    """The media route opens the PBX session the dial-in URL names, before the call is served."""

    @staticmethod
    def _route(monkeypatch: pytest.MonkeyPatch, pbx: Any | None) -> tuple[Any, list[Any]]:
        from fastapi.routing import APIWebSocketRoute

        import agent_bot
        from pipecat_siphon import VoiceSessionClient

        served: list[Any] = []

        async def serve_call(*_arguments: Any, session: Any = None, **_rest: Any) -> None:
            # What the PBX had been asked by the time the call is served.
            served.append((session, len(pbx.requests) if pbx is not None else 0))

        monkeypatch.setattr(agent_bot, "serve_call", serve_call)
        client = VoiceSessionClient(pbx.url, PBX_SECRET) if pbx is not None else None
        app = agent_bot.build_app(None, voice_sessions=client)
        route = next(route for route in app.routes if isinstance(route, APIWebSocketRoute))
        return route, served

    async def test_the_brief_is_in_hand_when_the_call_is_served(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any
    ) -> None:
        route, served = self._route(monkeypatch, pbx)
        websocket: Any = SimpleNamespace(query_params={"call": "abc", "session": PBX_TOKEN})

        await route.endpoint(websocket)

        ((session, asked),) = served
        assert asked == 1
        assert session.brief == _menu_brief()
        (request,) = pbx.requests
        assert request["path"] == f"/voice/sessions/{PBX_TOKEN}"
        assert request["headers"]["Authorization"] == f"Bearer {PBX_SECRET}"

    @pytest.mark.parametrize(
        "token",
        [
            # A single siphon-pbx node, or one from before tokens carried a node.
            PBX_TOKEN,
            # A clustered node's: the node holding the session, which the PBX routes on.
            PBX_TOKEN + ".pbx-node-2",
        ],
    )
    async def test_the_token_reaches_the_pbx_exactly_as_minted(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any, token: str
    ) -> None:
        route, served = self._route(monkeypatch, pbx)

        await route.endpoint(SimpleNamespace(query_params={"call": "abc", "session": token}))

        ((session, _asked),) = served
        assert session.brief == _menu_brief()
        assert [request["path"] for request in pbx.requests] == [f"/voice/sessions/{token}"]

    async def test_a_dial_in_without_a_session_is_answered_quietly(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any
    ) -> None:
        """siphon-rtp dialling the bot directly, with no siphon-pbx at all, is the ordinary case."""
        route, served = self._route(monkeypatch, pbx)
        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
        try:
            await route.endpoint(SimpleNamespace(query_params={"call": "abc"}))
        finally:
            logger.remove(sink)

        assert served == [(None, 0)]
        assert pbx.requests == []
        assert warnings == []

    @pytest.mark.parametrize(
        "query",
        [
            {"call": "abc"},
            {"call": "abc", "session": "../../api/v1/organisations"},
            {"call": "abc", "session": PBX_TOKEN + ".node/../x"},
            {"call": "abc", "session": ""},
        ],
    )
    async def test_no_plain_session_token_asks_the_pbx_nothing(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any, query: dict[str, str]
    ) -> None:
        route, served = self._route(monkeypatch, pbx)

        await route.endpoint(SimpleNamespace(query_params=query))

        assert served == [(None, 0)]
        assert pbx.requests == []

    async def test_a_refused_brief_still_answers_the_call(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any
    ) -> None:
        pbx.brief_status = 404
        route, served = self._route(monkeypatch, pbx)

        await route.endpoint(SimpleNamespace(query_params={"session": PBX_TOKEN}))

        assert served == [(None, 1)]

    async def test_a_session_without_a_pbx_configured_answers_without_a_brief(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        route, served = self._route(monkeypatch, None)

        await route.endpoint(SimpleNamespace(query_params={"session": PBX_TOKEN}))

        assert served == [(None, 0)]


class TestThePbxConfiguration:
    def test_neither_set_is_no_pbx(self) -> None:
        from agent_bot import build_voice_session_client

        assert build_voice_session_client("", "") is None

    @pytest.mark.parametrize(
        ("url", "secret"),
        [
            ("http://127.0.0.1:8080", ""),
            ("", PBX_SECRET),
            ("pbx.example.invalid", PBX_SECRET),
        ],
    )
    def test_half_a_configuration_is_refused(self, url: str, secret: str) -> None:
        from agent_bot import build_voice_session_client

        with pytest.raises(ValueError):
            build_voice_session_client(url, secret)

    def test_both_set_is_a_client(self) -> None:
        from agent_bot import build_voice_session_client
        from pipecat_siphon import VoiceSessionClient

        client = build_voice_session_client("https://pbx.example.invalid", PBX_SECRET)
        assert isinstance(client, VoiceSessionClient)


@pytest.mark.usefixtures("_no_speaking_language")
class TestTheBriefShapesTheConversation:
    """The block's instructions reach the model, and its options become the one way to choose."""

    def test_the_options_are_the_only_choices_offered(self, pbx: Any) -> None:
        from agent_bot import build_tool_schemas

        briefed, _ = _briefed_call(pbx, _menu_brief())
        offered = {
            schema.name: schema for schema in build_tool_schemas(False, None, briefed=briefed)
        }

        assert set(offered) == {"choose_option", "end_call"}
        option = offered["choose_option"].properties["option"]
        assert option["enum"] == ["sales", "support"]
        assert "buying something new" in offered["choose_option"].description
        assert offered["end_call"].handler == briefed.end_call

    def test_an_open_conversation_has_nothing_to_choose(self, pbx: Any) -> None:
        from agent_bot import build_tool_schemas

        briefed, _ = _briefed_call(pbx, _open_brief())
        offered = [schema.name for schema in build_tool_schemas(False, None, briefed=briefed)]

        assert offered == ["end_call"]

    def test_no_transfer_without_a_control_plane_even_when_briefed(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "TRANSFER_TARGET", "sip:desk@example.invalid")
        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        briefed, _ = _briefed_call(pbx, _menu_brief())
        offered = [
            schema.name for schema in agent_bot.build_tool_schemas(False, None, briefed=briefed)
        ]
        prompt: Any = agent_bot.build_llm_service(with_hangup=False, brief=briefed.session.brief)

        assert "transfer_call" not in offered
        assert "transfer_call" not in prompt.settings.system_instruction

    def test_the_prompt_carries_the_instructions_and_says_how_to_choose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        menu: Any = agent_bot.build_llm_service(with_hangup=False, brief=_menu_brief()).settings
        open_: Any = agent_bot.build_llm_service(with_hangup=False, brief=_open_brief()).settings
        unbriefed: Any = agent_bot.build_llm_service(with_hangup=False).settings

        assert "Ask what the caller needs and route them." in menu.system_instruction
        assert "choose_option" in menu.system_instruction
        assert "end_call" in menu.system_instruction
        assert "Answer questions about opening hours." in open_.system_instruction
        assert "choose_option" not in open_.system_instruction
        assert unbriefed.system_instruction == agent_bot.resolve_system_prompt({})

    def test_a_briefed_call_offers_its_tools_from_the_worker(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any
    ) -> None:
        """`build_worker` is where a session becomes tools; the route only hands it over."""
        import agent_bot
        from pipecat_siphon import VoiceSession, VoiceSessionClient

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        offered = _record_tool_lists(monkeypatch)
        session = VoiceSession(VoiceSessionClient(pbx.url, PBX_SECRET), PBX_TOKEN, _menu_brief())
        websocket: Any = object()

        agent_bot.build_worker(websocket, None, session=session)

        assert offered == [["end_call", "choose_option"]]


class TestTheOutcomeIsReported:
    """Every way the bot finishes a briefed call tells the PBX, once, and lets go of the stream."""

    @pytest.fixture(autouse=True)
    def _short_grace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "PBX_STREAM_GRACE_SECONDS", 0.05)

    async def test_choosing_an_option_reports_it_after_the_line_has_played(self, pbx: Any) -> None:
        from pbx_fixtures import REPORT_CHOSE_SALES

        briefed, worker = _briefed_call(pbx, _menu_brief())
        call = _ResultOnly({"option": "sales"})

        await briefed.choose_option(call)

        assert call.results == [{"chosen": "sales"}]
        assert [request["body"] for request in pbx.requests] == [REPORT_CHOSE_SALES]
        # The PBX moves the call on, which closes the stream; the bot only waits for that.
        assert not worker.ended()
        await asyncio.sleep(0.1)
        assert worker.ended()

    async def test_the_stream_closing_in_time_leaves_nothing_to_end(self, pbx: Any) -> None:
        briefed, worker = _briefed_call(pbx, _menu_brief())

        await briefed.choose_option(_ResultOnly({"option": "support"}))
        briefed.stream_closed()
        await asyncio.sleep(0.1)

        assert not worker.ended()

    async def test_an_option_not_on_the_menu_is_refused_and_not_reported(self, pbx: Any) -> None:
        briefed, worker = _briefed_call(pbx, _menu_brief())
        call = _ResultOnly({"option": "billing"})

        await briefed.choose_option(call)

        assert "error" in call.results[0]
        assert "sales" in call.results[0]["error"]
        assert pbx.requests == []
        assert not worker.ended()

    async def test_ending_the_conversation_reports_it_finished(self, pbx: Any) -> None:
        from pbx_fixtures import REPORT_FINISHED

        briefed, _ = _briefed_call(pbx, _open_brief())
        call = _ResultOnly()

        await briefed.end_call(call)

        assert call.results == [{"ending": True}]
        assert [request["body"] for request in pbx.requests] == [REPORT_FINISHED]

    async def test_after_one_report_the_tools_refuse_another(self, pbx: Any) -> None:
        briefed, _ = _briefed_call(pbx, _menu_brief())
        await briefed.choose_option(_ResultOnly({"option": "sales"}))
        again = _ResultOnly({"option": "support"})
        ending = _ResultOnly()

        await briefed.choose_option(again)
        await briefed.end_call(ending)

        assert "error" in again.results[0]
        assert "error" in ending.results[0]
        assert len(pbx.requests) == 1

    async def test_a_report_the_pbx_did_not_take_ends_the_stream_now(self, pbx: Any) -> None:
        """Nobody on the other side will move the call, so waiting would only be silence."""
        pbx.report_status = 404
        briefed, worker = _briefed_call(pbx, _open_brief())

        await briefed.end_call(_ResultOnly())

        assert worker.ended()

    async def test_a_failed_pipeline_reports_failed_and_ends_the_stream(self, pbx: Any) -> None:
        briefed, worker = _briefed_call(pbx, _open_brief())

        await briefed.fail("the voice could not be reached")

        (request,) = pbx.requests
        assert request["body"] == b'{"outcome":"failed","reason":"the voice could not be reached"}'
        assert worker.ended()

    async def test_a_pipeline_error_on_a_briefed_call_reports_it(
        self, monkeypatch: pytest.MonkeyPatch, pbx: Any
    ) -> None:
        """Wired, not only written: the worker's own error handler is what reports."""
        from pipecat.frames.frames import ErrorFrame

        import agent_bot
        from pipecat_siphon import Failed, VoiceSession, VoiceSessionClient

        TestObserverHook._stub_the_call(monkeypatch)
        handlers: dict[str, Any] = {}
        queued: list[Any] = []
        recording_worker: Any = vars(agent_bot)["PipelineWorker"]

        class _HandlerRecordingWorker(recording_worker):  # type: ignore[misc]
            def event_handler(self, name: str) -> Any:
                register = super().event_handler(name)

                def recording(handler: Any) -> Any:
                    handlers[name] = handler
                    return register(handler)

                return recording

            async def queue_frame(self, frame: Any, *_arguments: Any, **_keywords: Any) -> None:
                queued.append(frame)

        monkeypatch.setattr(agent_bot, "PipelineWorker", _HandlerRecordingWorker)
        session = VoiceSession(VoiceSessionClient(pbx.url, PBX_SECRET), PBX_TOKEN, _open_brief())
        websocket: Any = object()
        agent_bot.build_worker(websocket, None, session=session)

        await handlers["on_pipeline_error"](None, ErrorFrame(error="synthesizer refused"))

        (request,) = pbx.requests
        assert b'"outcome":"failed"' in request["body"]
        assert any(type(frame).__name__ == "EndWorkerFrame" for frame in queued)
        assert isinstance(session.reported, Failed)


CALL_BACK_TEMPLATE = "sip:{number}@trunk.example.invalid;user=phone"


class TestAssertedNumber:
    """The number a callback rings is the one the network asserted, and only that."""

    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            ("ASSERTED_SIP", "+15550123"),
            ("ASSERTED_TEL", "+15550124"),
            ("ASSERTED_TWICE", "+15550125"),
            ("ASSERTED_IN_ONE_LINE", "+15550126"),
            ("ASSERTED_LOWER_CASE", "+15550127"),
            ("ASSERTED_BUT_WITHHELD", None),
            ("NOT_ASSERTED", None),
        ],
    )
    def test_reads_the_asserted_identity(self, fixture: str, expected: str | None) -> None:
        import control_event_fixtures
        from agent_bot import asserted_number

        assert asserted_number(getattr(control_event_fixtures, fixture)) == expected

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "a string",
            {},
            {"invite": None},
            {"invite": {"headers": "P-Asserted-Identity: <tel:+15550123>"}},
            {"invite": {"headers": [["P-Asserted-Identity"]]}},
            {"invite": {"headers": [["P-Asserted-Identity", 15550123]]}},
            {"invite": {"headers": [[None, "<tel:+15550123>"]]}},
            {"invite": {"headers": [["P-Asserted-Identity", "<tel:+1555>"]]}},
            {"invite": {"headers": [["P-Asserted-Identity", "<sip:+15550123>"]]}},
            {"invite": {"headers": [["P-Asserted-Identity", "<tel:+15550123;ext=1"]]}},
            {"invite": {"headers": [["P-Asserted-Identity", "<mailto:a@example.invalid>"]]}},
        ],
    )
    def test_anything_else_is_no_number_and_never_raises(self, payload: Any) -> None:
        from agent_bot import asserted_number

        assert asserted_number(payload) is None


class _HandedOverCall(_FakeControlCall):
    """A handed-over call, with the `StasisStart` payload siphon gives it."""

    def __init__(self, sip_call_id: str, payload: Any) -> None:
        super().__init__(sip_call_id=sip_call_id, channel_id="handed-over-1")
        self.payload = payload


class _CallBackRequest(_ResultOnly):
    """One `call_back` tool call, on the call `app_resources` describes."""

    def __init__(self, arguments: dict[str, Any], app_resources: Any) -> None:
        super().__init__(arguments)
        self.app_resources = app_resources


class _CallingBack:
    """A bot with call_back configured, on one handed-over call, and what it placed."""

    SIP_CALL_ID = "handed-over@example.invalid"

    def __init__(self, payload: Any, client: _PlacingClient | None = None) -> None:
        from agent_bot import CallBacks, CallBackSettings, CallResources

        self.client = client or _PlacingClient()
        self.control = _placing_control(self.client)
        self.settings = CallBackSettings(
            uri_template=CALL_BACK_TEMPLATE,
            ws_uri="ws://198.51.100.10:9001/stream?call={call_id}",
            from_uri="sip:service@example.invalid",
            profile="agent",
            maximum_delay_minutes=60,
        )
        self.control.call_backs = CallBacks(self.control, self.settings)
        self.call = _HandedOverCall(self.SIP_CALL_ID, payload)
        self.resources = CallResources(
            serializer=SimpleNamespace(call_id=self.SIP_CALL_ID),  # type: ignore[arg-type]
            speech=_FinishedSpeech(),  # type: ignore[arg-type]
            control=self.control,
        )
        self.holding: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "_CallingBack":
        self.holding = asyncio.create_task(self.control._handle_call(self.call))
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_exception: object) -> None:
        self.call.end()
        if self.holding is not None:
            await self.holding
        assert self.control.call_backs is not None
        self.control.call_backs.cancel_all()

    async def ask(self, **arguments: Any) -> Any:
        from agent_bot import call_back

        request = _CallBackRequest(arguments, self.resources)
        await call_back(request)  # type: ignore[arg-type]
        (result,) = request.results
        return result

    async def hang_up_and_settle(self) -> None:
        self.call.end()
        if self.holding is not None:
            await self.holding
        for _ in range(5):
            await asyncio.sleep(0)


@pytest.fixture
def _quick_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bot

    monkeypatch.setattr(agent_bot, "CALL_BACK_SECONDS_PER_MINUTE", 0.05)


@pytest.mark.usefixtures("_quick_minutes")
class TestCallingBack:
    """The model arranges a callback; the bot rings the asserted number once the call is over."""

    async def test_rings_the_asserted_number_once_this_call_is_over(self) -> None:
        from control_event_fixtures import ASSERTED_SIP

        async with _CallingBack(ASSERTED_SIP) as bot:
            result = await bot.ask(minutes=0, reason="the invoice they asked about")
            assert result == {"scheduled": True, "in_minutes": 0}
            await asyncio.sleep(0.1)
            assert bot.client.originated == [], "rang back while the caller was still on"

            await bot.hang_up_and_settle()

            (originate,) = bot.client.originated
            assert originate["to"] == "sip:+15550123@trunk.example.invalid;user=phone"
            assert originate["from_uri"] == "sip:service@example.invalid"
            assert originate["ws_uri"].startswith("ws://198.51.100.10:9001/stream?call={call_id}")
            assert originate["profile"] == "agent"
            assert originate["timeout"] == 30
            (placed,) = bot.control.placed_calls.pending()
            assert placed.context == {"reason": "the invoice they asked about"}

    async def test_waits_the_minutes_it_was_asked_to(self) -> None:
        from control_event_fixtures import ASSERTED_TEL

        async with _CallingBack(ASSERTED_TEL) as bot:
            assert await bot.ask(minutes=3, reason="x") == {"scheduled": True, "in_minutes": 3}
            await bot.hang_up_and_settle()
            await asyncio.sleep(0.1)
            assert bot.client.originated == []
            await asyncio.sleep(0.15)
            assert [originate["to"] for originate in bot.client.originated] == [
                "sip:+15550124@trunk.example.invalid;user=phone"
            ]

    @pytest.mark.parametrize("fixture", ["ASSERTED_BUT_WITHHELD", "NOT_ASSERTED"])
    async def test_no_asserted_number_arranges_nothing(self, fixture: str) -> None:
        import control_event_fixtures

        async with _CallingBack(getattr(control_event_fixtures, fixture)) as bot:
            result = await bot.ask(minutes=0, reason="x")
            await bot.hang_up_and_settle()

        assert result["available"] is False
        assert bot.client.originated == []

    async def test_a_call_with_no_control_handle_arranges_nothing(self) -> None:
        from control_event_fixtures import ASSERTED_SIP

        async with _CallingBack(ASSERTED_SIP) as bot:
            bot.resources.serializer = SimpleNamespace(call_id="unknown@example.invalid")  # type: ignore[assignment]
            result = await bot.ask(minutes=0, reason="x")

        assert result["available"] is False

    @pytest.mark.parametrize("minutes", [-1, 61, "5", True, 2.5, None])
    async def test_minutes_out_of_range_are_refused(self, minutes: Any) -> None:
        from control_event_fixtures import ASSERTED_SIP

        async with _CallingBack(ASSERTED_SIP) as bot:
            result = await bot.ask(minutes=minutes, reason="x")
            await bot.hang_up_and_settle()

        assert "error" in result
        assert bot.client.originated == []

    async def test_one_callback_per_number(self) -> None:
        from control_event_fixtures import ASSERTED_SIP

        async with _CallingBack(ASSERTED_SIP) as bot:
            await bot.ask(minutes=0, reason="first")
            second = await bot.ask(minutes=5, reason="second")
            await bot.hang_up_and_settle()

            assert "error" in second
            assert len(bot.client.originated) == 1

    async def test_the_reason_is_cut_to_its_cap(self) -> None:
        import agent_bot
        from control_event_fixtures import ASSERTED_SIP

        async with _CallingBack(ASSERTED_SIP) as bot:
            await bot.ask(minutes=0, reason="  " + "r" * 500 + "  ")
            await bot.hang_up_and_settle()
            (placed,) = bot.control.placed_calls.pending()

        assert placed.context["reason"] == "r" * agent_bot.CALL_BACK_REASON_MAXIMUM_LENGTH

    async def test_a_refused_originate_is_logged_masked_and_frees_the_number(self) -> None:
        from control_event_fixtures import ASSERTED_SIP

        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
        try:
            async with _CallingBack(
                ASSERTED_SIP, _PlacingClient(error=RuntimeError("refused"))
            ) as bot:
                await bot.ask(minutes=0, reason="x")
                await bot.hang_up_and_settle()
                assert len(bot.client.originated) == 1
                assert bot.control.call_backs is not None
                assert bot.control.call_backs.pending() == []
        finally:
            logger.remove(sink)

        assert any("callback" in line and "failed" in line for line in lines)
        assert not any("5550123" in line for line in lines)

    async def test_stopping_the_bot_logs_every_callback_it_loses(self) -> None:
        from control_event_fixtures import ASSERTED_SIP

        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(str(message)), level="ERROR")
        try:
            async with _CallingBack(ASSERTED_SIP) as bot:
                await bot.ask(minutes=30, reason="x")
                assert bot.control.call_backs is not None
                bot.control.call_backs.cancel_all()
                await bot.hang_up_and_settle()
        finally:
            logger.remove(sink)

        assert bot.client.originated == []
        assert any("lost" in line for line in lines)
        assert not any("5550123" in line for line in lines)

    async def test_nothing_logged_shows_the_whole_number(self) -> None:
        from control_event_fixtures import ASSERTED_SIP

        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
        try:
            async with _CallingBack(ASSERTED_SIP) as bot:
                await bot.ask(minutes=0, reason="x")
                await bot.hang_up_and_settle()
                assert len(bot.client.originated) == 1
        finally:
            logger.remove(sink)

        assert any("callback" in line for line in lines)
        assert not any("5550123" in line for line in lines)


class TestCallBackIsOfferedOnlyWhenConfigured:
    def test_offered_with_a_control_plane_that_can_call_back(self) -> None:
        from agent_bot import build_tool_schemas

        offered = {
            schema.name: schema for schema in build_tool_schemas(True, None, call_back_minutes=60)
        }
        assert "call_back" in offered
        minutes = offered["call_back"].properties["minutes"]
        assert (minutes["minimum"], minutes["maximum"]) == (0, 60)
        assert offered["call_back"].required == ["minutes", "reason"]

    def test_not_offered_otherwise(self) -> None:
        from agent_bot import build_tool_schemas

        assert "call_back" not in [schema.name for schema in build_tool_schemas(True, None)]
        assert "call_back" not in [
            schema.name for schema in build_tool_schemas(False, None, call_back_minutes=60)
        ]

    def test_the_prompt_mentions_it_only_when_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        with_it: Any = agent_bot.build_llm_service(with_hangup=True, call_back=True).settings
        without: Any = agent_bot.build_llm_service(with_hangup=True).settings
        assert "call_back" in with_it.system_instruction
        assert "call_back" not in without.system_instruction

    def test_the_worker_offers_it_when_the_control_plane_can_call_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        monkeypatch.setattr(agent_bot, "TRANSFER_TARGET", "")
        monkeypatch.setattr(agent_bot, "build_speaking_language_controller", lambda: None)
        offered = _record_tool_lists(monkeypatch)
        control = _placing_control(_PlacingClient())
        websocket: Any = object()

        agent_bot.build_worker(websocket, control)
        control.call_backs = agent_bot.CallBacks(
            control,
            agent_bot.CallBackSettings(
                uri_template=CALL_BACK_TEMPLATE,
                ws_uri="ws://198.51.100.10:9001/stream?call={call_id}",
                from_uri=None,
                profile="agent",
                maximum_delay_minutes=60,
            ),
        )
        agent_bot.build_worker(websocket, control)

        assert offered == [["end_call"], ["end_call", "call_back"]]


class TestCallBackSettings:
    @staticmethod
    def _arguments(**overrides: Any) -> Any:
        fields = {
            "control_url": "ws://127.0.0.1:9092/control",
            "call_ws_uri": "ws://198.51.100.10:9001/stream?call={call_id}",
            "call_profile": "agent",
        }
        return SimpleNamespace(**{**fields, **overrides})

    def test_unset_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALL_BACK_URI", "")
        assert agent_bot.call_back_settings_from_arguments(self._arguments()) is None

    def test_set_reads_the_rest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALL_BACK_URI", CALL_BACK_TEMPLATE)
        monkeypatch.setattr(agent_bot, "CALL_BACK_FROM", "sip:service@example.invalid")
        monkeypatch.setattr(agent_bot, "CALL_BACK_MAXIMUM_DELAY_MINUTES", 15)
        settings = agent_bot.call_back_settings_from_arguments(self._arguments())
        assert settings == agent_bot.CallBackSettings(
            uri_template=CALL_BACK_TEMPLATE,
            ws_uri="ws://198.51.100.10:9001/stream?call={call_id}",
            from_uri="sip:service@example.invalid",
            profile="agent",
            maximum_delay_minutes=15,
        )

    @pytest.mark.parametrize(
        ("template", "overrides", "maximum"),
        [
            ("sip:+15550100@trunk.example.invalid", {}, 60),
            (CALL_BACK_TEMPLATE, {"control_url": None}, 60),
            (CALL_BACK_TEMPLATE, {"call_ws_uri": None}, 60),
            (CALL_BACK_TEMPLATE, {}, -1),
        ],
    )
    def test_half_a_configuration_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        template: str,
        overrides: dict[str, Any],
        maximum: int,
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CALL_BACK_URI", template)
        monkeypatch.setattr(agent_bot, "CALL_BACK_MAXIMUM_DELAY_MINUTES", maximum)
        with pytest.raises(ValueError):
            agent_bot.call_back_settings_from_arguments(self._arguments(**overrides))


def _write(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8")


class TestCallConfigurationLoader:
    """One YAML file per configuration key, read per call; anything wrong falls back loudly."""

    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        monkeypatch.delenv("BOT_SYSTEM_PROMPT", raising=False)
        monkeypatch.delenv("BOT_SYSTEM_PROMPT_FILE", raising=False)
        monkeypatch.setattr(agent_bot, "TRANSFER_TARGET", "sip:default-desk@example.invalid")
        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")

    def test_no_directory_or_no_key_is_the_bots_own(self, tmp_path: Path) -> None:
        import agent_bot

        defaults = agent_bot.default_configuration()
        assert defaults.key is None
        assert defaults.transfer_target == "sip:default-desk@example.invalid"
        assert defaults.system_prompt == agent_bot.resolve_system_prompt({})
        assert agent_bot.load_configuration("", "CUST-0042") == defaults
        assert agent_bot.load_configuration(str(tmp_path), None) == defaults

    def test_a_file_overrides_what_it_names(self, tmp_path: Path) -> None:
        import agent_bot

        _write(
            tmp_path,
            "CUST-0042.yaml",
            "system_prompt: You answer the phone for a dental practice.\n"
            "transfer_target: sip:front-desk@example.invalid\n"
            "caller_lookup_url: https://accounts.example.invalid/patients/{caller}\n"
            "caller_lookup_description: Look up the caller's next appointment.\n",
        )
        loaded = agent_bot.load_configuration(str(tmp_path), "CUST-0042")
        assert loaded == agent_bot.CallConfiguration(
            key="CUST-0042",
            system_prompt="You answer the phone for a dental practice.",
            transfer_target="sip:front-desk@example.invalid",
            caller_lookup_url="https://accounts.example.invalid/patients/{caller}",
            caller_lookup_description="Look up the caller's next appointment.",
        )

    def test_what_a_file_leaves_out_stays_the_bots_own(self, tmp_path: Path) -> None:
        import agent_bot

        _write(tmp_path, "quiet.yaml", "transfer_target: ''\n")
        loaded = agent_bot.load_configuration(str(tmp_path), "quiet")
        defaults = agent_bot.default_configuration()
        assert loaded.transfer_target == ""
        assert loaded.system_prompt == defaults.system_prompt
        assert loaded.key == "quiet"

    def test_a_prompt_file_is_read_beside_the_configuration(self, tmp_path: Path) -> None:
        import agent_bot

        _write(tmp_path, "office-prompt.md", "You answer for an office.\n")
        _write(tmp_path, "office.yaml", "system_prompt_file: office-prompt.md\n")
        loaded = agent_bot.load_configuration(str(tmp_path), "office")
        assert loaded.system_prompt == "You answer for an office."

    @pytest.mark.parametrize(
        "text",
        [
            "system_prompt: [not, text]\n",
            "transfer_target: 42\n",
            "colour: blue\n",
            "- a list\n",
            "system_prompt: a\nsystem_prompt_file: b.md\n",
            "caller_lookup_url: https://accounts.example.invalid/patients\n",
            "system_prompt_file: ../outside.md\n",
            "system_prompt_file: missing.md\n",
            "system_prompt: '   '\n",
            "{unclosed\n",
        ],
    )
    def test_a_file_that_is_wrong_falls_back_and_says_so(self, tmp_path: Path, text: str) -> None:
        import agent_bot

        (tmp_path.parent / "outside.md").write_text("escaped", encoding="utf-8")
        _write(tmp_path, "broken.yaml", text)
        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(str(message)), level="ERROR")
        try:
            loaded = agent_bot.load_configuration(str(tmp_path), "broken")
        finally:
            logger.remove(sink)
        assert loaded == agent_bot.default_configuration()
        assert any("broken" in line for line in lines)

    def test_a_key_with_no_file_falls_back(self, tmp_path: Path) -> None:
        import agent_bot

        assert agent_bot.load_configuration(str(tmp_path), "nobody") == (
            agent_bot.default_configuration()
        )

    @pytest.mark.parametrize("key", ["../etc/passwd", "a/b", "", "x" * 65, "CUST 42", ".hidden"])
    def test_a_key_that_is_not_a_plain_name_reads_nothing(self, tmp_path: Path, key: str) -> None:
        import agent_bot

        _write(tmp_path, ".hidden.yaml", "transfer_target: sip:x@example.invalid\n")
        assert agent_bot.load_configuration(str(tmp_path), key) == (
            agent_bot.default_configuration()
        )

    def test_startup_names_every_file_that_would_fall_back(self, tmp_path: Path) -> None:
        import agent_bot

        _write(tmp_path, "good.yaml", "transfer_target: sip:x@example.invalid\n")
        _write(tmp_path, "bad-one.yaml", "colour: blue\n")
        _write(tmp_path, "bad-two.yaml", "transfer_target: 42\n")
        with pytest.raises(ValueError) as refused:
            agent_bot.check_configurations(str(tmp_path))
        assert "bad-one" in str(refused.value)
        assert "bad-two" in str(refused.value)
        assert "good" not in str(refused.value).replace("bad", "")

    def test_startup_refuses_a_directory_that_is_not_there(self, tmp_path: Path) -> None:
        import agent_bot

        with pytest.raises(ValueError):
            agent_bot.check_configurations(str(tmp_path / "absent"))
        agent_bot.check_configurations("")


class TestCallFacts:
    """Whose call it is: from siphon-pbx's brief, or from siphon-sip's handover."""

    def test_from_a_full_pbx_context(self) -> None:
        import agent_bot
        from pbx_fixtures import BRIEF_CONTEXT_FULL
        from pipecat_siphon import parse_brief

        brief = parse_brief(BRIEF_CONTEXT_FULL)
        assert brief is not None
        facts = agent_bot.call_facts_from_context(brief.context)
        assert facts.configuration_key == "CUST-0042"
        assert facts.caller == "15550142"
        assert facts.organisation is not None
        assert facts.organisation.name == "Example Office"

    def test_from_an_organisation_only_context(self) -> None:
        import agent_bot
        from pbx_fixtures import BRIEF_CONTEXT_ORGANISATION
        from pipecat_siphon import parse_brief

        brief = parse_brief(BRIEF_CONTEXT_ORGANISATION)
        assert brief is not None
        facts = agent_bot.call_facts_from_context(brief.context)
        assert facts.configuration_key == "CUST-0042"
        assert facts.caller is None

    def test_an_unset_reference_and_a_withheld_caller(self) -> None:
        import agent_bot
        from pbx_fixtures import BRIEF_CONTEXT_UNSET
        from pipecat_siphon import parse_brief

        brief = parse_brief(BRIEF_CONTEXT_UNSET)
        assert brief is not None
        facts = agent_bot.call_facts_from_context(brief.context)
        assert facts.configuration_key is None
        assert facts.caller is None

    def test_no_context_is_no_facts(self) -> None:
        import agent_bot

        assert agent_bot.call_facts_from_context(None) == agent_bot.CallFacts()

    def test_from_a_handover(self) -> None:
        import agent_bot
        from control_event_fixtures import ASSERTED_SIP

        payload = {**ASSERTED_SIP, "vars": {"bot_configuration": "dialog-office"}}
        facts = agent_bot.call_facts_from_handover(payload)
        assert facts.configuration_key == "dialog-office"
        assert facts.caller == "15550123"
        assert facts.organisation is None

    @pytest.mark.parametrize("variables", [{}, {"bot_configuration": 7}, None, "not a mapping"])
    def test_a_handover_without_the_variable_names_no_configuration(self, variables: Any) -> None:
        import agent_bot
        from control_event_fixtures import ASSERTED_BUT_WITHHELD

        facts = agent_bot.call_facts_from_handover({**ASSERTED_BUT_WITHHELD, "vars": variables})
        assert facts.configuration_key is None
        assert facts.caller is None

    def test_a_handover_that_is_not_a_payload_is_no_facts(self) -> None:
        import agent_bot

        assert agent_bot.call_facts_from_handover(None) == agent_bot.CallFacts()


class TestWaitingForTheHandover:
    """Under IMS the engine dials the bot before siphon hands the call to the control plane."""

    async def test_a_call_already_handed_over_is_returned_at_once(self) -> None:
        control = ControlPlane(application="a", token="t", url="ws://127.0.0.1:9/x")
        call = _FakeControlCall(sip_call_id="early@example.invalid")
        holding = asyncio.create_task(control._handle_call(call))
        await asyncio.sleep(0)
        assert await control.handed_over("early@example.invalid", 0.5) is call
        call.end()
        await holding

    async def test_a_handover_that_arrives_while_waiting_is_returned(self) -> None:
        control = ControlPlane(application="a", token="t", url="ws://127.0.0.1:9/x")
        call = _FakeControlCall(sip_call_id="late@example.invalid")
        waiting = asyncio.create_task(control.handed_over("late@example.invalid", 2.0))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        holding = asyncio.create_task(control._handle_call(call))
        assert await waiting is call
        call.end()
        await holding

    async def test_no_handover_in_time_is_none(self) -> None:
        control = ControlPlane(application="a", token="t", url="ws://127.0.0.1:9/x")
        assert await control.handed_over("never@example.invalid", 0.05) is None


class TestResolvingACallsConfiguration:
    """The route picks the configuration before the pipeline is built, from whichever source."""

    @pytest.fixture(autouse=True)
    def _directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        import agent_bot

        monkeypatch.delenv("BOT_SYSTEM_PROMPT", raising=False)
        monkeypatch.delenv("BOT_SYSTEM_PROMPT_FILE", raising=False)
        monkeypatch.setattr(agent_bot, "CONFIGURATIONS_DIRECTORY", str(tmp_path))
        monkeypatch.setattr(agent_bot, "HANDOVER_WAIT_SECONDS", 0.2)
        _write(tmp_path, "CUST-0042.yaml", "system_prompt: For customer 42.\n")
        _write(tmp_path, "dialog-office.yaml", "system_prompt: For the office line.\n")
        return tmp_path

    async def test_a_briefed_call_takes_its_organisations(self, pbx: Any) -> None:
        import agent_bot
        from pbx_fixtures import BRIEF_CONTEXT_FULL
        from pipecat_siphon import VoiceSession, VoiceSessionClient, parse_brief

        brief = parse_brief(BRIEF_CONTEXT_FULL)
        assert brief is not None
        session = VoiceSession(VoiceSessionClient(pbx.url, PBX_SECRET), PBX_TOKEN, brief)
        configuration, facts = await agent_bot.resolve_call(None, {}, session)
        assert configuration.system_prompt == "For customer 42."
        assert facts.caller == "15550142"

    async def test_a_handed_over_call_takes_the_routing_scripts(self) -> None:
        import agent_bot
        from control_event_fixtures import ASSERTED_SIP

        control = ControlPlane(application="a", token="t", url="ws://127.0.0.1:9/x")
        call = _HandedOverCall(
            "ims@example.invalid", {**ASSERTED_SIP, "vars": {"bot_configuration": "dialog-office"}}
        )

        async def hand_over_later() -> None:
            await asyncio.sleep(0.05)
            await control._handle_call(call)

        holding = asyncio.create_task(hand_over_later())
        configuration, facts = await agent_bot.resolve_call(
            control, {"call": "ims@example.invalid"}, None
        )
        assert configuration.system_prompt == "For the office line."
        assert facts.caller == "15550123"
        call.end()
        await holding

    async def test_no_handover_in_time_is_the_bots_own(self) -> None:
        import agent_bot

        control = ControlPlane(application="a", token="t", url="ws://127.0.0.1:9/x")
        configuration, facts = await agent_bot.resolve_call(
            control, {"call": "nobody@example.invalid"}, None
        )
        assert configuration == agent_bot.default_configuration()
        assert facts == agent_bot.CallFacts()

    async def test_without_a_directory_nothing_is_waited_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "CONFIGURATIONS_DIRECTORY", "")
        monkeypatch.setattr(agent_bot, "HANDOVER_WAIT_SECONDS", 30.0)
        control = ControlPlane(application="a", token="t", url="ws://127.0.0.1:9/x")
        started = time.monotonic()
        configuration, _facts = await agent_bot.resolve_call(
            control, {"call": "nobody@example.invalid"}, None
        )
        assert time.monotonic() - started < 1.0
        assert configuration == agent_bot.default_configuration()


@pytest.mark.usefixtures("_no_speaking_language")
class TestAConfigurationShapesTheCall:
    """What a configuration and the call's facts change in the pipeline built for the call."""

    def test_the_prompt_is_the_configurations_and_names_the_organisation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot
        from pipecat_siphon import ContextOrganisation

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        configuration = agent_bot.CallConfiguration(
            key="CUST-0042",
            system_prompt="For customer 42.",
            transfer_target="sip:front-desk@example.invalid",
            caller_lookup_url="",
            caller_lookup_description=agent_bot.DEFAULT_CALLER_LOOKUP_DESCRIPTION,
        )
        facts = agent_bot.CallFacts(
            organisation=ContextOrganisation(
                id="o", name="Example Office", time_zone="Europe/Amsterdam"
            )
        )
        settings: Any = agent_bot.build_llm_service(
            with_hangup=True, configuration=configuration, facts=facts
        ).settings
        assert settings.system_instruction.startswith("For customer 42.")
        assert "transfer_call" in settings.system_instruction
        assert "Example Office" in settings.system_instruction
        assert "Europe/Amsterdam" in settings.system_instruction

    def test_a_time_zone_it_does_not_know_is_left_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot
        from pipecat_siphon import ContextOrganisation

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "local")
        facts = agent_bot.CallFacts(
            organisation=ContextOrganisation(id="o", name="Example Office", time_zone="Mars/Base")
        )
        settings: Any = agent_bot.build_llm_service(with_hangup=False, facts=facts).settings
        assert "Example Office" in settings.system_instruction
        assert "Mars/Base" not in settings.system_instruction

    def test_the_configurations_transfer_target_is_the_one_offered_and_dialled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "TRANSFER_TARGET", "")
        monkeypatch.setattr(agent_bot, "CALLER_LOOKUP_URL", "")
        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        offered = _record_tool_lists(monkeypatch)
        control = _placing_control(_PlacingClient())
        websocket: Any = object()
        configuration = agent_bot.CallConfiguration(
            key="office",
            system_prompt="x",
            transfer_target="sip:front-desk@example.invalid",
            caller_lookup_url="",
            caller_lookup_description=agent_bot.DEFAULT_CALLER_LOOKUP_DESCRIPTION,
        )
        agent_bot.build_worker(websocket, control)
        agent_bot.build_worker(websocket, control, configuration=configuration)
        assert offered == [["end_call"], ["end_call", "transfer_call"]]

    async def test_a_transfer_rings_the_configurations_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "TRANSFER_TARGET", "sip:default-desk@example.invalid")
        monkeypatch.setattr(agent_bot, "TRANSFER_MODE", "refer")
        sent: list[str] = []

        class _Call:
            async def transfer(self, target: str) -> None:
                sent.append(target)

        await agent_bot._start_the_transfer(_Call(), "sip:front-desk@example.invalid")
        assert sent == ["sip:front-desk@example.invalid"]
        assert agent_bot._bridge_dial_arguments(
            SimpleNamespace(payload={}), "sip:b@example.invalid"
        )["targets"] == [{"uri": "sip:b@example.invalid"}]

    def test_a_trusted_caller_wins_over_the_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "WEATHER_ENABLED", False)
        TestObserverHook._stub_the_call(monkeypatch)
        lookups: list[Any] = []
        original = agent_bot.CallerLookup

        def recording(url: str, caller: str | None) -> Any:
            lookups.append((url, caller))
            return original(url, caller)

        monkeypatch.setattr(agent_bot, "CallerLookup", recording)
        configuration = agent_bot.CallConfiguration(
            key="office",
            system_prompt="x",
            transfer_target="",
            caller_lookup_url="https://accounts.example.invalid/c/{caller}",
            caller_lookup_description="d",
        )
        websocket: Any = object()
        agent_bot.build_worker(
            websocket,
            None,
            caller="99900000001",
            configuration=configuration,
            facts=agent_bot.CallFacts(caller="15550142"),
        )
        assert lookups == [("https://accounts.example.invalid/c/{caller}", "15550142")]


GEMINI_LIVE_PROVIDERS = {"llm": "gemini-live", "stt": "gemini-live", "tts": "gemini-live"}


def _select_speech_to_speech(
    monkeypatch: pytest.MonkeyPatch, provider: str = "gemini-live"
) -> None:
    """Point all three roles at one speech-to-speech model, as `BOT_BACKEND=<provider>` does."""
    for role in ("llm", "stt", "tts"):
        _select(monkeypatch, role, provider)


class TestSpeechToSpeechSelection:
    """Gemini Live hears, answers and speaks in one model, so it fills all three roles or none."""

    def test_the_preset_is_a_backend(self) -> None:
        refuse_an_unknown_backend("gemini-live")

    def test_it_needs_only_the_gemini_key(self) -> None:
        assert missing_environment(GEMINI_LIVE_PROVIDERS, {}) == ["GOOGLE_API_KEY"]

    @pytest.mark.parametrize(
        "providers",
        [
            {"llm": "gemini-live", "stt": "deepgram", "tts": "cartesia"},
            {"llm": "anthropic", "stt": "gemini-live", "tts": "cartesia"},
            {"llm": "gemini-live", "stt": "gemini-live", "tts": "google"},
        ],
    )
    def test_a_split_selection_is_refused(self, providers: dict[str, str]) -> None:
        """A recognizer or a voice beside a speech-to-speech model would never be in the call."""
        import agent_bot

        with pytest.raises(ValueError, match="BOT_BACKEND=gemini-live"):
            agent_bot.refuse_a_split_speech_to_speech(providers)

    def test_a_cascade_is_not_refused(self) -> None:
        import agent_bot

        agent_bot.refuse_a_split_speech_to_speech(CLOUD_PROVIDERS)

    def test_more_than_one_speaking_language_is_refused(self) -> None:
        """The voice is the model's own, so there is no voice to switch mid-call."""
        import agent_bot

        with pytest.raises(ValueError, match="gemini-live"):
            agent_bot.resolve_speaking_languages(
                {"BOT_TTS_LANGUAGES": "nl,en"}, "gemini-live", "Charon"
            )

    def test_one_speaking_language_is_accepted(self) -> None:
        import agent_bot

        languages = agent_bot.resolve_speaking_languages(
            {"BOT_TTS_LANGUAGE": "nl"}, "gemini-live", "Charon"
        )
        assert languages.starting == "nl"


@pytest.mark.usefixtures("_vendor_credentials")
class TestGeminiLiveBuilds:
    """The speech-to-speech service is built with what the cascade's three services were told."""

    def test_the_service_carries_model_voice_language_prompt_and_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

        import agent_bot

        _select_speech_to_speech(monkeypatch)
        tool = FunctionSchema(name="end_call", description="d", properties={}, required=[])

        service = agent_bot.build_llm_service(with_hangup=True, tools=[tool])

        assert isinstance(service, GeminiLiveLLMService)
        settings: Any = service.settings
        assert settings.model == agent_bot.DEFAULT_LLM_MODEL["gemini-live"]
        assert settings.voice == agent_bot.DEFAULT_TTS_VOICE["gemini-live"]
        assert settings.language == "en-US"
        assert settings.system_instruction.startswith("You are a voice assistant")
        # On the constructor, not the context: tools arriving on the context make the service
        # reconnect before it can greet the caller.
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools_from_init = service._tools_from_init
        assert isinstance(tools_from_init, ToolsSchema)
        assert [schema.name for schema in tools_from_init.standard_tools] == ["end_call"]

    def test_the_speaking_language_is_sent_as_a_locale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gemini wants `nl-NL`; pipecat 1.12 sends an enum's bare value, `nl`, unconverted."""
        import agent_bot

        _select_speech_to_speech(monkeypatch)
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl")

        service = agent_bot.build_llm_service(with_hangup=False)

        settings: Any = service.settings
        assert settings.language == "nl-NL"


OPENAI_REALTIME_PROVIDERS = {
    "llm": "openai-realtime",
    "stt": "openai-realtime",
    "tts": "openai-realtime",
}


class TestOpenAIRealtimeSelection:
    """OpenAI Realtime is selected whole, like Gemini Live, under the OpenAI key alone."""

    def test_the_preset_is_a_backend(self) -> None:
        refuse_an_unknown_backend("openai-realtime")

    def test_it_needs_only_the_openai_key(self) -> None:
        assert missing_environment(OPENAI_REALTIME_PROVIDERS, {}) == ["OPENAI_API_KEY"]

    def test_a_split_selection_is_refused(self) -> None:
        import agent_bot

        with pytest.raises(ValueError, match="BOT_BACKEND=openai-realtime"):
            agent_bot.refuse_a_split_speech_to_speech({**CLOUD_PROVIDERS, "llm": "openai-realtime"})

    def test_more_than_one_speaking_language_is_refused(self) -> None:
        import agent_bot

        with pytest.raises(ValueError, match="openai-realtime"):
            agent_bot.resolve_speaking_languages(
                {"BOT_TTS_LANGUAGES": "nl,en"}, "openai-realtime", "marin"
            )


@pytest.mark.usefixtures("_vendor_credentials", "_no_speaking_language")
class TestOpenAIRealtimeBuilds:
    """The Realtime service carries what the cascade's three services were told."""

    def test_the_session_carries_model_voice_prompt_tools_and_a_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

        import agent_bot

        _select_speech_to_speech(monkeypatch, "openai-realtime")
        tool = FunctionSchema(name="end_call", description="d", properties={}, required=[])

        service = agent_bot.build_llm_service(with_hangup=True, tools=[tool])

        assert isinstance(service, OpenAIRealtimeLLMService)
        settings: Any = service.settings
        assert settings.model == "gpt-realtime-2.1"
        assert settings.system_instruction.startswith("You are a voice assistant")
        session = settings.session_properties
        assert session.model == "gpt-realtime-2.1"
        assert session.instructions == settings.system_instruction
        assert session.audio.output.voice == agent_bot.DEFAULT_TTS_VOICE["openai-realtime"]
        assert session.max_output_tokens == agent_bot.MAXIMUM_RESPONSE_TOKENS
        # On the session, not the context, as with Gemini Live.
        assert isinstance(session.tools, ToolsSchema)
        assert [schema.name for schema in session.tools.standard_tools] == ["end_call"]
        # Off unless asked for, and without it the caller's side of the call is never written
        # down: no transcript reaches the context, the observer or the log.
        transcription = session.audio.input.transcription
        assert transcription.model == "gpt-4o-mini-transcribe"
        assert transcription.language is None

    def test_the_speaking_language_reaches_the_prompt_and_the_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The transcription API takes ISO-639-1, so a regional code is sent as its base."""
        import agent_bot

        _select_speech_to_speech(monkeypatch, "openai-realtime")
        monkeypatch.setenv("BOT_TTS_LANGUAGE", "nl-BE")

        settings: Any = agent_bot.build_llm_service(with_hangup=False).settings

        assert "nl-BE" in settings.system_instruction
        assert settings.session_properties.audio.input.transcription.language == "nl"

    def test_a_pinned_temperature_and_seed_are_not_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Withheld rather than sent, since the Realtime session has neither.

        pipecat would only warn on every call that it dropped them; the startup line says n/a.
        """
        import agent_bot

        _select_speech_to_speech(monkeypatch, "openai-realtime")
        monkeypatch.setattr(agent_bot, "LLM_TEMPERATURE", 0.2)
        monkeypatch.setattr(agent_bot, "LLM_SEED", 7)

        assert agent_bot._sampling_settings("openai-realtime") == {}


class TestOpenAIRealtimeTurns:
    """The turn follows OpenAI's own speech detection, the way the service asks for it to."""

    def test_the_services_proposals_decide_the_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit strategies override the ones a service recommends, in pipecat.

        The Realtime service only *proposes* a caller's start of speech. Under the strategies
        Gemini Live is given, nothing takes the proposal up, so no interruption reaches the output
        transport and a caller who talks over the bot hears it finish what was already buffered.
        """
        from pipecat.turns.user_start import ExternalUserTurnStartStrategy
        from pipecat.turns.user_stop import ExternalUserTurnStopStrategy

        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "openai-realtime")

        strategies = agent_bot.build_user_aggregator_params().user_turn_strategies

        assert strategies is not None
        assert [type(strategy) for strategy in strategies.start or []] == [
            ExternalUserTurnStartStrategy
        ]
        assert [type(strategy) for strategy in strategies.stop or []] == [
            ExternalUserTurnStopStrategy
        ]
        assert getattr(strategies, "enable_interruptions", False)


class TestSpeechToSpeechInputRate:
    """What the pipeline hands a speech-to-speech model is at the rate the model takes in.

    pipecat appends the caller's audio to OpenAI's Realtime buffer as it arrives, unconverted, and
    that buffer is 24 kHz PCM and nothing else. Handed the wire's 16 kHz, the model hears the caller
    half again as fast. The serializer converts from the wire to whatever the pipeline asks for, so
    asking for the model's rate is the whole fix.
    """

    def test_the_constant_is_the_rate_pipecat_declares(self) -> None:
        from pipecat.services.openai.realtime.events import PCMAudioFormat

        import agent_bot

        assert PCMAudioFormat().rate == agent_bot.OPENAI_REALTIME_SAMPLE_RATE == 24000

    @pytest.mark.parametrize(
        ("provider", "expected"),
        [("openai-realtime", 24000), ("gemini-live", 16000), ("anthropic", 16000)],
    )
    def test_the_input_rate_follows_the_model(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, expected: int
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", provider)

        assert agent_bot.pipeline_input_sample_rate() == expected

    def test_the_transport_and_the_worker_take_it_and_the_output_stays_on_the_wire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_bot

        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "openai-realtime")
        transport = agent_bot.build_transport(cast(Any, object()), agent_bot.build_serializer())
        assert transport._params.audio_in_sample_rate == 24000
        assert transport._params.audio_out_sample_rate == agent_bot.PIPELINE_SAMPLE_RATE

        _serializers, workers = TestObserverHook._stub_the_call(monkeypatch)
        monkeypatch.setattr(agent_bot, "build_text_to_speech", FrameProcessor)
        agent_bot.build_worker(cast(Any, object()), None)
        params = workers[0]["params"]
        assert params.audio_in_sample_rate == 24000
        assert params.audio_out_sample_rate == agent_bot.PIPELINE_SAMPLE_RATE


class TestSpeechToSpeechCall:
    """A Gemini Live call is the model between the wire's two ends, and the turn is the model's."""

    @staticmethod
    def _build(
        monkeypatch: pytest.MonkeyPatch, control: ControlPlane | None
    ) -> tuple[list[FrameProcessor], list[object], dict[str, Any]]:
        import agent_bot

        serializers, _workers = TestObserverHook._stub_the_call(monkeypatch)
        monkeypatch.setattr(agent_bot, "LLM_PROVIDER", "gemini-live")

        def refuse() -> FrameProcessor:
            raise AssertionError("a speech-to-speech call builds no recognizer or voice")

        monkeypatch.setattr(agent_bot, "build_speech_to_text", refuse)
        monkeypatch.setattr(agent_bot, "build_text_to_speech", refuse)
        llm_arguments: dict[str, Any] = {}

        def recording_llm(with_hangup: bool, **settings: Any) -> FrameProcessor:
            llm_arguments.update(settings, with_hangup=with_hangup)
            return FrameProcessor()

        monkeypatch.setattr(agent_bot, "build_llm_service", recording_llm)
        built: list[list[FrameProcessor]] = []

        def recording_pipeline(processors: list[FrameProcessor]) -> Pipeline:
            built.append(processors)
            return Pipeline(processors)

        monkeypatch.setattr(agent_bot, "Pipeline", recording_pipeline)
        websocket: Any = object()
        agent_bot.build_worker(websocket, control)
        return built[0], serializers, llm_arguments

    def test_no_echo_guard_and_the_engine_speech_edges_are_not_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model detects speech itself. A second detector would interrupt it on its own.

        And the echo guard drops transcripts whatever their direction, which here are the model's
        own record of what the caller said, travelling upstream.
        """
        import agent_bot

        processors, serializers, _llm = self._build(monkeypatch, None)

        assert not any(isinstance(p, agent_bot.EchoGuard) for p in processors)
        serializer: Any = serializers[0]
        assert serializer._params.speech_frames == "none"

    def test_the_tools_go_to_the_service_and_not_the_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processors, _serializers, llm = self._build(
            monkeypatch, ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9/x")
        )

        assert [schema.name for schema in llm["tools"]][:1] == ["end_call"]
        from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

        aggregator = next(p for p in processors if isinstance(p, LLMUserAggregator))
        assert not aggregator.context.tools

    def test_a_transfer_hold_also_holds_the_callers_audio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a recognizer the model hears the audio itself, so audio is what must be held."""
        import agent_bot

        processors, _serializers, _llm = self._build(
            monkeypatch, ControlPlane(application="agent-app", token="t", url="ws://127.0.0.1:9/x")
        )

        hold = next(p for p in processors if isinstance(p, agent_bot.TransferHold))
        assert hold.holds_audio

    async def test_holding_audio_drops_it_and_releasing_lets_it_through(self) -> None:
        import agent_bot

        hold = agent_bot.TransferHold(hold_audio=True)
        hold.engage()
        await run_test(
            hold,
            start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
            frames_to_send=[
                InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1),
                TextFrame("not the caller"),
            ],
            expected_down_frames=[TextFrame],
        )
        hold.release()
        await run_test(
            hold,
            start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
            frames_to_send=[
                InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
            ],
            expected_down_frames=[InputAudioRawFrame],
        )

    async def test_the_cascade_hold_keeps_passing_audio(self) -> None:
        """The recognizer sits in front of the hold there and has already heard the audio."""
        import agent_bot

        hold = agent_bot.TransferHold()
        hold.engage()
        await run_test(
            hold,
            start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
            frames_to_send=[
                InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
            ],
            expected_down_frames=[InputAudioRawFrame],
        )


class _SpeakingNow(BotSpeechMonitor):
    """A monitor that says the bot holds the line, or does not, as told."""

    def __init__(self, ours: bool) -> None:
        super().__init__()
        self.ours = ours

    def line_is_ours(self, tail_seconds: float) -> bool:
        return self.ours


class _AudioRecorder(FrameProcessor):
    """Keep the audio that reaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.audio: list[bytes] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.audio.append(frame.audio)
        await self.push_frame(frame, direction)


class TestEchoGate:
    """While the bot holds the line, a speech-to-speech model hears silence, not its own echo.

    It detects the caller's speech in the audio itself, so an echo the engine could not cancel is
    the caller interrupting, and the bot cuts itself off a syllable into every sentence.
    """

    async def _pass(self, ours: bool) -> list[bytes]:
        import agent_bot

        recorder = _AudioRecorder()
        gate = agent_bot.EchoGate(_SpeakingNow(ours), tail_seconds=0.6)
        await run_test(
            Pipeline([gate, recorder]),
            start_timeout=PIPELINE_START_TIMEOUT_SECONDS,
            frames_to_send=[
                InputAudioRawFrame(audio=b"\x10\x20" * 160, sample_rate=16000, num_channels=1)
            ],
            expected_down_frames=[InputAudioRawFrame],
        )
        return recorder.audio

    async def test_the_bots_turn_is_silenced_at_the_same_length(self) -> None:
        """Silence rather than nothing, so the model's audio clock keeps running."""
        assert await self._pass(ours=True) == [b"\x00\x00" * 160]

    async def test_the_callers_turn_passes_untouched(self) -> None:
        assert await self._pass(ours=False) == [b"\x10\x20" * 160]

    def test_the_gate_is_in_a_speech_to_speech_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_bot

        processors, _serializers, _llm = TestSpeechToSpeechCall._build(monkeypatch, None)

        gates = [p for p in processors if isinstance(p, agent_bot.EchoGate)]
        assert len(gates) == 1
        assert processors.index(gates[0]) < next(
            index for index, p in enumerate(processors) if type(p).__name__ == "LLMUserAggregator"
        )

    def test_no_tail_means_no_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A leg whose echo is cancelled can have its barge-in back."""
        import agent_bot

        monkeypatch.setattr(agent_bot, "ECHO_GUARD_TAIL_SECONDS", 0.0)
        processors, _serializers, _llm = TestSpeechToSpeechCall._build(monkeypatch, None)

        assert not any(isinstance(p, agent_bot.EchoGate) for p in processors)


@pytest.mark.usefixtures("_vendor_credentials")
class TestGeminiLiveSpeechDetection:
    def test_the_start_of_speech_is_detected_at_low_sensitivity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a phone line the residual echo and the line noise are not the caller starting."""
        from google.genai.types import StartSensitivity

        import agent_bot

        _select_speech_to_speech(monkeypatch)

        service = agent_bot.build_llm_service(with_hangup=False)

        settings: Any = service.settings
        assert settings.vad.start_sensitivity == StartSensitivity.START_SENSITIVITY_LOW
