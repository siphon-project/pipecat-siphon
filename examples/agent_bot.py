"""Conversational demo bot: a phone caller talks to Claude, over the siphon-rtp media WebSocket.

The parrot in `echo_bot.py` proves the media path. This one proves the product: speech in,
speech out, with the turn taking done by the media engine rather than in the pipeline, and the
call itself driven over the control plane rather than left as an anonymous audio stream.

    caller -> siphon-sip -> siphon-rtp --(media WebSocket)--> this bot
                    |                                         STT -> Claude -> TTS
                    +------(control WebSocket)---------------> hangup, transfer, hold

Three vendors, three keys, all read from the environment:

    export ANTHROPIC_API_KEY=...        # console.anthropic.com -- an API key, not a subscription
    export DEEPGRAM_API_KEY=...         # streaming speech to text
    export CARTESIA_API_KEY=...         # streaming text to speech
    export CARTESIA_VOICE_ID=...        # a voice from the Cartesia dashboard

    pip install "pipecat-ai[anthropic,deepgram,cartesia]"
    python examples/agent_bot.py --host 0.0.0.0 --port 9001

To let the bot end the call itself, add the control channel (see "The control plane" below)::

    pip install siphon-control
    export SIPHON_CONTROL_TOKEN=...
    python examples/agent_bot.py --control-url ws://127.0.0.1:9092/control/ws

Without those flags the bot still runs; it simply has no way to hang up, and the system prompt
does not claim otherwise.

The platform side
-----------------

The engine's built-in `voice_ai` profile is a starting point, not a complete answer. It sets
`transport_protocol`, `ice`, `dtls`, `replace`, `noise_suppression`, `echo_cancellation`,
`ws_vad` and `ws_barge_in` -- and none of `received_from`, `ws_vad_engine`,
`ws_vad_min_speech_ms` or `ws_sample_rate`. The two omissions that decide whether this works at
all are the first and the last. Define your own profile::

    media:
      backend: siphon-rtp          # rtpengine and rtpproxy have no WebSocket bridge
      profiles:
        agent:
          offer: {}                # answer_local and answer-first handover read the answer half
          answer:
            transport_protocol: "RTP/AVP"
            ice: "remove"
            dtls: "off"
            replace: ["origin"]
            received_from: true            # or a NATed caller is gated out entirely
            noise_suppression: true
            echo_cancellation: true        # not optional under barge-in
            ws_vad: true
            ws_barge_in: true
            ws_vad_engine: neural          # energy cannot tell the bot's echo from the caller
            ws_vad_min_speech_ms: 100      # leading edge, swallows a single transient
            ws_sample_rate: 16000          # match PIPELINE_SAMPLE_RATE exactly

`received_from` is the one people lose a day to. An app client behind NAT advertises an
unroutable address in `c=`, the ingress gate defaults to an exact match on that address, and
every packet the handset sends is dropped -- on a call whose signalling is clean from end to end.
The bot hears silence and nothing anywhere reports an error.

`ws_vad_engine: neural` matters for the same reason `echo_cancellation` does. The energy detector
answers "is something loud here", so a cough, a door or one burst of the bot's own echo fires the
speech-start edge and cuts the prompt off. A speech classifier cannot rescue that on its own --
echo of speech is still speech, detected down to roughly -36 dB of echo-return loss -- which is
why the canceller and the neural detector are one feature, not two. Note that the canceller and
the noise suppressor exist only at 8 and 16 kHz: a `ws_sample_rate` outside that pair silently
leaves the uplink uncancelled and barge-in then has nothing holding the echo under the detector.

`ws_sample_rate` lives on the profile rather than being passed per call because `call.handover()`
takes a profile and a `ws_uri` and no media knobs. Once the call is handed to a controller a
per-call `rtpengine.answer_local(ws_sample_rate=...)` has nowhere to go, so this belongs in one
place, not two.

The control plane
-----------------

Answer-first handover parks the call with an out-of-process control app: the engine sends the
200, anchors the media to this bot's WebSocket and hands over an already-connected channel. The
routing-script side is one call::

    @b2bua.on_invite
    async def route(call):
        call.handover(
            "agent-app",
            answer=True,
            profile="agent",
            ws_uri="ws://198.51.100.10:9001/stream?call={call_id}",
            on_lost="hangup",
        )

and siphon's own configuration registers the app::

    control:
      listen: "127.0.0.1:9092"     # loopback: this channel can hang up any call on the node
      apps:
        - name: "agent-app"        # must equal the handover target and the client's hello
          token: "${AGENT_APP_TOKEN}"
          on_lost: hangup          # a controller that dies hangs the caller up

Without handover, `rtpengine.answer_local(call, profile="agent", ws_uri=...)` bridges the audio
just as well -- there is simply no control channel, and no hangup.

Swapping a vendor is a one-line change -- pipecat ships around seventy service integrations, and
nothing below depends on which three you pick.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_siphon import SiphonFrameSerializer

PIPELINE_SAMPLE_RATE = int(os.environ.get("BOT_SAMPLE_RATE", "") or 16000)
"""What the pipeline runs at. Match it to `ws_sample_rate` on the profile and nothing in this
process resamples: the engine converts once, at the RTP boundary, where it is frame-exact."""

WIRE_PTIME_MS = 20
"""The engine's packetization time, announced in its `start` envelope.

One WebSocket frame is one ptime. Pipecat's transport writes output in 10 ms chunks and defaults
to four of them, so left alone it sends 40 ms per frame. An engine that stamps each frame it
receives as exactly one ptime then advances the RTP timestamp half as fast as the audio, and
every packet overlaps its predecessor: a stream that is impeccable packet by packet, and silence
on the handset. Recent engine builds drain the downlink by samples and packetise an oversized
frame correctly, but one frame per ptime is what the protocol asks for and it is the
lower-latency shape."""

MODEL = "claude-haiku-4-5"
"""pipecat defaults its Anthropic service to an older model, so this is set explicitly."""

EFFORT_BY_MODEL: dict[str, str | None] = {
    # Effort is the thinking-depth dial. A phone turn is answered in one or two sentences and the
    # caller is listening to silence while the model works, so this is the end of the range that
    # belongs on a voice route.
    "claude-opus-5": "low",
    "claude-sonnet-5": "low",
    # No `effort` parameter on this model: sending one is an HTTP 400, not a no-op.
    "claude-haiku-4-5": None,
}
"""Which effort level each model takes.

Bound to the model rather than left as a free-standing constant beside it, because the two drift:
`output_config.effort` is valid on the model pinned above and rejected outright by others, so an
adopter switching `MODEL` for latency would otherwise get `This model does not support the effort
parameter` on every turn. An unlisted model sends no `effort` at all, which is always accepted."""

MAXIMUM_RESPONSE_TOKENS = 400
"""A spoken answer is short. This is a ceiling against a runaway monologue on the phone, not a
target -- 400 tokens is a couple of minutes of speech, far past anything the prompt asks for."""

TURN_END_SILENCE_SECONDS = 0.2
"""How long pipecat waits after the engine reports end-of-speech before running the model.

The engine's detector has already applied its own trailing hold before it sends `speech_stopped`
(on the energy detector that is `ws_vad_hangover_ms`, ~200 ms by default; the neural detector
holds on its own and the knob does not apply), so pipecat's own 0.6 s default would be a second
silence budget stacked on the first one. This still waits for the recognizer's final transcript,
which is the part that must not be cut."""

KICKOFF = "[The caller has just come on the line and has not spoken yet.]"
"""The one message the greeting turn runs against.

The model generates its own greeting rather than reciting a canned one, which needs a turn to
answer: a request carrying a system prompt and no messages is rejected with `messages: at least
one message is required`, and the failure is invisible from the outside -- the bot simply stays
mute until the caller gives up and says "hello?", at which point there is a message in the
context and the call works normally from then on. It reads as a dead line.

Bracketed and third person on purpose. It lands in the transcript as a user turn, and a
first-person line ("hello?") gets answered literally."""

FAREWELL_START_SECONDS = 2.0
"""How long to wait for the farewell to begin before concluding there is not going to be one."""

FAREWELL_QUIET_SECONDS = 0.5
"""How long the bot must stay silent before its farewell counts as finished."""

FAREWELL_TIMEOUT_SECONDS = 20.0
"""Hard cap on waiting for the farewell. A model that will not stop talking still gets hung up."""

CONTROL_RETRY_MINIMUM_SECONDS = 1.0
CONTROL_RETRY_MAXIMUM_SECONDS = 30.0
"""Backoff bounds for the control connection.

The client supervises its own reconnects once established, but the *first* connect raises if the
engine is not listening yet. Gathered naively with the pipeline that exception takes the media
path down with it and the process exits on a routine restart-ordering race."""

SYSTEM_PROMPT = """You are a voice assistant on a telephone call. You are talking, not writing.

Keep answers to one or two sentences unless you are explicitly asked for detail. Ask one question
at a time. Never use markdown, bullet points, headings, emoji or any other formatting: everything
you write is read aloud verbatim, so write numbers, dates and units as a person would say them.

The caller can interrupt you at any time and you will be cut off mid-sentence when they do. That
is normal. When it happens, answer what they just said rather than finishing your last thought.

Open the call by greeting the caller and asking how you can help."""

SYSTEM_PROMPT = os.environ.get("BOT_SYSTEM_PROMPT") or SYSTEM_PROMPT
"""Let a deployment replace the persona without editing this file.

The prompt above is the part worth tuning per deployment and the part most likely to differ
between a demo and something answering real callers, so it should not require a fork of the
example to change. What it must not lose is the telephone constraints -- speech not prose, no
formatting, short answers -- because a replacement written as if for a chat window produces
markdown that the synthesizer then reads aloud, bullet by bullet.

The hangup instructions are appended separately below and are not overridable: they describe a
capability of this process rather than a persona, and a prompt that omits them leaves the model
telling callers it cannot end the call.
"""


LLM_MODEL = os.environ.get("BOT_LLM_MODEL", "").strip() or MODEL
"""The Claude model, overridable per deployment.

Worth choosing deliberately on a phone call. `claude-opus-5` runs adaptive thinking by *default*,
which is right for hard reasoning and wrong for a receptionist: the caller hears silence while it
thinks, on every turn. `claude-haiku-4-5` does not think unless asked to and answers this kind of
short, bounded conversation about as well. If you do want Opus on a call, pair it with a low
`output_config.effort` rather than leaving the default."""

STT_MODEL = os.environ.get("BOT_STT_MODEL", "").strip() or "nova-3-general"
"""Deepgram model. Pinned rather than left on pipecat's default, because a phone line is not the
general case the default is tuned for: 8 kHz, lossy, and frequently not in the language the default
assumes. Set `BOT_STT_MODEL` to override."""

STT_LANGUAGE = os.environ.get("BOT_STT_LANGUAGE", "").strip() or "multi"
"""Recognizer language. `multi` by default: a public number takes calls in whatever language the
caller speaks, and a recognizer pinned to one of them transcribes the rest into confident nonsense.
Pin a single language only if the line really is single-language."""

TTS_MODEL = os.environ.get("BOT_TTS_MODEL", "").strip() or "sonic-3.6"
"""Cartesia model. Pinned for the same reason as the recognizer: an example that drifts with a
vendor default is an example whose latency and voice change without anyone editing it."""

ECHO_GUARD_TAIL_SECONDS = float(os.environ.get("BOT_ECHO_GUARD_TAIL", "") or 0.6)
"""How long after the bot stops speaking its own echo is still expected to arrive.

Sized from the observed delay -- the echo of a finished sentence turned up ~280 ms later on a
mobile leg behind a carrier -- with headroom, because the cost of being slightly too long is that a
caller who interrupts immediately is missed once, and the cost of being too short is the bot
answering itself, and then answering that.

**Set it to 0 on a leg the engine's canceller has actually locked onto.** siphon-rtp 0.5.0 vetoes a
turn edge it can show is our own returning audio, which is selective where this is not -- but that
veto covers the engine's own `speech_started`, and this bot decides its turns from an always-on
recognizer instead, so it only helps here to the extent that the echo is gone from the *audio*.
That needs a confident lock, and the engine says whether it has one: run it with
`siphon_rtp::media=debug` and read the delay report. A weak-lock warning means the echo is passing
through and this guard is still carrying the call. Measured on a DID-to-mobile leg, 0.5.0 reports
confidence 5.7 against a threshold of 12."""

MIN_INTERRUPT_WORDS = int(os.environ.get("BOT_MIN_INTERRUPT_WORDS", "") or 2)
"""How many words the caller must be heard saying before they may interrupt the bot.

One word is enough to interrupt on a clean leg, and far too few on a phone line. Whatever the bot
says comes back through the handset's earpiece and into its microphone; the engine cancels that
echo only if it can find it, and on a mobile behind a carrier the round trip can exceed the
canceller's delay search window. What is left is the bot's own voice arriving as the caller, and a
single word of it stops the bot mid-sentence, every sentence.

**Set it to 1 on siphon-rtp 0.5.0 or newer with `echo_cancellation` on.** At 1 the bot interrupts
on the engine's own speech edge rather than waiting for words, and that edge is the one the
engine's canceller vetoes when it can show the audio is our returning voice -- so the echo is
handled where the far-end reference is, and a real caller is heard immediately. Above 1 the caller
waits for the recognizer: measured on a call at 3, interim transcripts arrived 1, 1, 2 then 4 words
over 3.6 seconds, and the bot talked over the caller for all of it.

This threshold applies *only while the bot is speaking*, so a caller starting a fresh turn is
unaffected. Two words is enough to swallow the leading fragment of an echo while still letting a
real "wait, stop" through. Set to 1 to restore pipecat's default behaviour."""

TRANSFER_TARGET = os.environ.get("BOT_TRANSFER_TARGET", "").strip()
"""Where the transfer_call tool sends the caller. Empty disables the tool entirely.

From the environment and never from the model. The tool therefore takes no arguments: a
destination the model could name is a destination the *caller* can talk it into naming, and the
verb on the other end will dial whatever it is given."""

TRANSFER_PROMPT = """

You can also put the caller through to a person with the transfer_call tool. Tell them you are
transferring them first: the transfer happens as soon as your words have finished playing. Once
you transfer, the call is no longer yours -- so only do it when the caller has asked for it, or
when you cannot help and a person can."""
"""Appended only when a destination is configured, for the same reason as HANGUP_PROMPT: a model
told it can transfer when it cannot promises the caller a handover that never happens."""

HANGUP_PROMPT = """

When the caller is done, say goodbye and then use the end_call tool to hang up. Say the farewell
first: the tool ends the call as soon as your words have finished playing."""
"""Appended to the system prompt only when the control plane is actually wired.

The prompt and the platform have to agree. A model told it can hang up when it cannot will
promise the caller something that never happens, exactly as a model told it can be interrupted on
a leg without barge-in will invite the caller to cut in and then never hear them."""


def _required_environment_variable(name: str, purpose: str) -> str:
    """Read an API key from the environment, or exit saying exactly which one is missing."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set. It is the {purpose}; export it and run again.")
    return value


class BotSpeechMonitor(FrameProcessor):
    """Tracks whether the bot is currently audible, so a hangup can wait for the farewell.

    `hangup()` takes effect immediately. Called the moment the model asks for it, the caller
    hears the line die two syllables into "goodbye", which reads as a crash rather than as the
    bot ending the call.
    """

    def __init__(self) -> None:
        """Start quiet, with no speech seen yet."""
        super().__init__()
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._quiet_since = time.monotonic()
        self._has_spoken = asyncio.Event()

    def reset(self) -> None:
        """Forget the previous call's speech, so a farewell wait cannot satisfy itself early."""
        self._quiet.set()
        self._quiet_since = time.monotonic()
        self._has_spoken.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Track the bot's speaking edges and pass every frame through untouched."""
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._quiet.clear()
            self._has_spoken.set()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._quiet.set()
            self._quiet_since = time.monotonic()
        await self.push_frame(frame, direction)

    def line_is_ours(self, tail_seconds: float) -> bool:
        """Whether the bot is speaking, or stopped so recently that echo is still arriving.

        The tail is the part that matters. On a leg whose echo the media engine cannot cancel, the
        bot's own voice comes back *late* -- observed at ~280 ms after it stopped talking. Any guard
        that only asks "is the bot speaking right now" is looking at the wrong moment and sees
        nothing.
        """
        if not self._quiet.is_set():
            return True
        return time.monotonic() - self._quiet_since < tail_seconds

    async def wait_until_finished(
        self,
        start_seconds: float = FAREWELL_START_SECONDS,
        quiet_seconds: float = FAREWELL_QUIET_SECONDS,
        timeout_seconds: float = FAREWELL_TIMEOUT_SECONDS,
    ) -> bool:
        """Wait for the bot to fall quiet and stay quiet. True if it did, False on the cap."""
        deadline = time.monotonic() + timeout_seconds

        # The farewell has usually not started yet when the tool fires, so waiting for quiet
        # straight away would be satisfied by the silence *before* it.
        if not self._has_spoken.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._has_spoken.wait(), start_seconds)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._quiet.wait(), remaining)
            except TimeoutError:
                return False
            # Quiet now. Hold it: a barge-in or a second sentence clears the event again.
            await asyncio.sleep(quiet_seconds)
            if self._quiet.is_set():
                return True


class ControlPlane:
    """The bot's half of the siphon control channel: the calls it is allowed to act on.

    Correlation is the detail worth reading twice. Control frames carry an id triple
    (`channel_id`, `call_id`, `sip_call_id`) where `call_id` is siphon's internal call UUID, but
    the engine expands `{call_id}` in a `ws_uri` to the *SIP* Call-ID -- and that same SIP
    Call-ID is what arrives on the media socket in the `start` envelope. So the two channels join
    on `sip_call_id`, not on the field whose name matches.
    """

    def __init__(self, application: str, token: str, url: str) -> None:
        """Record the connection parameters. Nothing is dialled or imported until `run_forever`."""
        self._application = application
        self._token = token
        self._url = url
        self._calls: dict[str, Any] = {}
        # Narrowed to the SDK's own `ControlError` once `run_forever` has imported it. Until
        # then this is the safe superset: no call can be registered before that happens, so a
        # tool handler cannot reach a hangup while it is still this wide.
        self.error_class: type[BaseException] = Exception

    def call_for(self, sip_call_id: str | None) -> Any | None:
        """Return the control handle for a media session, or None if the two never joined."""
        if sip_call_id is None:
            return None
        return self._calls.get(sip_call_id)

    async def _handle_call(self, call: Any) -> None:
        """Own one handed-over call: register it, then hold it until the call ends."""
        sip_call_id = call.sip_call_id
        if sip_call_id is None:
            # Nothing to join the media socket on, so the tool could never find this call.
            logger.warning(f"control call {call.channel_id} has no SIP Call-ID, ignoring it")
            return

        self._calls[sip_call_id] = call
        logger.info(f"control channel attached to call {sip_call_id}")
        try:
            while await call.next_event() is not None:
                pass
        finally:
            self._calls.pop(sip_call_id, None)
            logger.info(f"control channel detached from call {sip_call_id}")

    async def run_forever(self) -> None:
        """Keep a control connection up, retrying the first connect until the engine is there."""
        control_client_class, self.error_class = _import_control_sdk()
        delay = CONTROL_RETRY_MINIMUM_SECONDS
        while True:
            client = control_client_class(app=self._application, token=self._token, url=self._url)
            client.on_call(self._handle_call)
            try:
                await client.run()
            except Exception as error:  # any connect failure here is worth retrying
                logger.warning(f"control plane unreachable ({error}); retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, CONTROL_RETRY_MAXIMUM_SECONDS)
            else:
                # `run()` returned rather than raised: the client was shut down deliberately.
                return


def _import_control_sdk() -> tuple[Any, type[BaseException]]:
    """Import the optional control SDK, or exit saying how to install it."""
    try:
        from siphon_control import ControlClient, ControlError
    except ImportError:
        sys.exit("the control plane needs the siphon-control package: pip install siphon-control")
    return ControlClient, ControlError


@dataclass
class CallResources:
    """What a tool handler needs to act on the call it is running inside."""

    serializer: SiphonFrameSerializer
    speech: BotSpeechMonitor
    control: ControlPlane
    # Set after the worker exists, which is after this object is handed to it. A transfer is an
    # ending for this bot even though it is not an ending for the caller, and ending the pipeline is
    # how the media socket gets closed.
    worker: PipelineWorker | None = None


async def end_call(params: FunctionCallParams) -> None:
    """Say goodbye, wait for it to actually play, then drop the line.

    The tool deliberately takes no arguments. It acts on the call the bot is already on; a
    subscriber or channel argument would be a model-chosen string reaching a verb that can hang
    up a stranger.
    """
    resources = params.app_resources
    if not isinstance(resources, CallResources):
        await params.result_callback({"error": "this call cannot be ended from here"})
        return

    control_call = resources.control.call_for(resources.serializer.call_id)
    if control_call is None:
        await params.result_callback({"error": "this call cannot be ended from here"})
        return

    # Answer before the line goes: once the caller is hung up there is nobody to hear a result.
    await params.result_callback({"ending": True})

    if not await resources.speech.wait_until_finished():
        logger.warning("farewell did not finish within the cap, hanging up anyway")

    try:
        await control_call.hangup()
    except resources.control.error_class as error:
        # The caller hanging up first is the ordinary case, not a fault.
        logger.info(f"hangup rejected ({error}); the caller most likely hung up first")


async def transfer_call(params: FunctionCallParams) -> None:
    """Say the handover line, wait for it to play, then REFER the caller away.

    Takes no arguments, exactly like `end_call` and for the same reason: it acts on the call the
    bot is already on and sends it to the one configured destination. A model-chosen destination
    would be a caller-chosen destination.
    """
    resources = params.app_resources
    if not isinstance(resources, CallResources) or not TRANSFER_TARGET:
        await params.result_callback({"error": "this call cannot be transferred from here"})
        return

    control_call = resources.control.call_for(resources.serializer.call_id)
    if control_call is None:
        await params.result_callback({"error": "this call cannot be transferred from here"})
        return

    # Answer before the caller leaves: once they are transferred there is nobody to hear a result.
    await params.result_callback({"transferring": True})

    if not await resources.speech.wait_until_finished():
        logger.warning("handover line did not finish within the cap, transferring anyway")

    try:
        await control_call.transfer(TRANSFER_TARGET)
        logger.info(f"transferred the call to {TRANSFER_TARGET}")
    except resources.control.error_class as error:
        # The caller hanging up first is the ordinary case, not a fault.
        logger.info(f"transfer rejected ({error}); the caller most likely hung up first")
        return

    # The REFER hands the caller to someone else, so this bot's leg is finished even though the
    # call is not. Leaving the pipeline running holds the media socket open, and the leg then sits
    # there until the engine's media timeout reaps it -- half a minute of a call nobody is on,
    # during which the referee's NOTIFYs arrive on a dialog this side has stopped caring about.
    if resources.worker is not None:
        await resources.worker.queue_frame(EndWorkerFrame())


class EchoGuard(FrameProcessor):
    """Drop transcriptions that arrive while the line is the bot's own.

    The problem this solves is not barge-in tuning, it is that the bot is being transcribed as the
    caller. On a leg whose echo the media engine's canceller cannot reach, whatever the bot says
    returns through the handset, the recognizer transcribes it faithfully, and the pipeline treats
    it as the caller speaking -- interrupting the bot, and then answering it.

    Pipecat's own `MinWordsUserTurnStartStrategy` guards the *interruption*, but only while the bot
    is speaking. The echo arrives after that, so it slips past. This drops the transcription itself,
    for as long as the bot is speaking plus a tail long enough to cover the echo's flight time.

    The cost is the same as any half-duplex arrangement and should be understood: a caller who talks
    over the bot is not heard, and what they said is gone rather than queued. The engine is the
    right place for this -- siphon-rtp 0.5.0 holds the far-end reference, so it can tell the echo
    from an interruption where this cannot -- but its veto gates the engine's own turn edge, and
    this pipeline decides turns from its recognizer. So it helps here only once the echo is gone
    from the audio, which needs a confident delay lock. Set the tail to 0 when you have one.
    """

    def __init__(self, speech: BotSpeechMonitor, tail_seconds: float) -> None:
        """Guard against `speech`'s idea of who holds the line, for `tail_seconds` after it ends."""
        super().__init__()
        self._speech = speech
        self._tail = tail_seconds
        self._dropped = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass everything except a transcription that arrived while the line was ours."""
        await super().process_frame(frame, direction)

        if (
            self._tail > 0
            and isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and self._speech.line_is_ours(self._tail)
        ):
            self._dropped += 1
            # Debug, not info: on a bad leg this fires several times per sentence, and a log that
            # scrolls is a log nobody reads. The count is what matters.
            logger.debug(
                f"echo guard: dropped {frame.__class__.__name__} "
                f"{getattr(frame, 'text', '')!r} (total {self._dropped})"
            )
            return

        await self.push_frame(frame, direction)


def build_serializer() -> SiphonFrameSerializer:
    """Build the wire serializer, kept as its own object so the tools can read its `call_id`."""
    return SiphonFrameSerializer(
        params=SiphonFrameSerializer.InputParams(
            # The engine's VAD drives the turn, so there is no VAD analyzer in this pipeline at
            # all -- its edges arrive as the frames pipecat's own turn-start strategy already
            # consumes.
            speech_frames="vad",
            stereo_input="caller",
            # The wire rate is known before `start` arrives, because the profile above asks for
            # it. Saying so keeps the serializer from building a resampler for the handful of
            # frames in between and then discarding it.
            wire_sample_rate=PIPELINE_SAMPLE_RATE,
        ),
    )


def build_transport(
    websocket: WebSocket, serializer: SiphonFrameSerializer
) -> FastAPIWebsocketTransport:
    """Build the transport for one accepted call.

    One per connection, not one per process. The engine dials a fresh WebSocket per call, so the
    socket *is* the call: binding a pipeline to it and tearing both down together is what lets this
    bot answer more than one number, or the same number twice.
    """
    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
            # One WebSocket frame per ptime, rather than pipecat's default four 10 ms chunks.
            audio_out_10ms_chunks=WIRE_PTIME_MS // 10,
            serializer=serializer,
        ),
    )


def build_llm_service(with_hangup: bool) -> AnthropicLLMService:
    """Build the Claude service, tuned for a conversation happening in real time."""
    # Keyed on the model actually used, not the default: effort is per-model and Haiku 4.5
    # rejects it outright with a 400 on every turn, which is silence on every turn.
    effort = EFFORT_BY_MODEL.get(LLM_MODEL)
    extra: dict[str, Any] = {"output_config": {"effort": effort}} if effort else {}

    # Server-side refusal fallbacks belong on a voice route -- a policy refusal mid-call is
    # otherwise dead air -- but they cannot be reached from here. They are beta parameters, and
    # while pipecat does call the beta endpoint, it overwrites `betas` with its own interleaved
    # thinking flag *after* merging `extra`, so a `betas` set here never survives to the wire and
    # `fallbacks` arrives without the beta that would make it legal. Every turn is then a 400,
    # including the greeting. Fixing that means merging `betas` in pipecat's Anthropic service,
    # not working around it here.

    return AnthropicLLMService(
        api_key=_required_environment_variable("ANTHROPIC_API_KEY", "Claude API key"),
        settings=AnthropicLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=(
                SYSTEM_PROMPT
                + (HANGUP_PROMPT if with_hangup else "")
                + (TRANSFER_PROMPT if with_hangup and TRANSFER_TARGET else "")
            ),
            max_tokens=MAXIMUM_RESPONSE_TOKENS,
            # The system prompt is resent on every turn of the call; caching it is free money.
            enable_prompt_caching=True,
            extra=extra,
        ),
    )


def build_speech_to_text() -> DeepgramSTTService:
    """Build the recognizer, at the pipeline's rate."""
    # Both are pinned at import (`... or "<default>"`), so there is no unset case to branch on:
    # a phone line is 8 kHz, lossy and often not in the language a vendor default assumes, and an
    # example whose recognizer drifts with that default is an example whose accuracy changes
    # without anyone editing it.
    return DeepgramSTTService(
        api_key=_required_environment_variable("DEEPGRAM_API_KEY", "speech-to-text key"),
        settings=DeepgramSTTService.Settings(model=STT_MODEL, language=STT_LANGUAGE),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def build_text_to_speech() -> CartesiaTTSService:
    """Build the voice, at the pipeline's rate."""
    return CartesiaTTSService(
        api_key=_required_environment_variable("CARTESIA_API_KEY", "text-to-speech key"),
        settings=CartesiaTTSService.Settings(
            voice=_required_environment_variable("CARTESIA_VOICE_ID", "voice to speak with"),
            model=TTS_MODEL,
        ),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def build_user_turn_strategies() -> UserTurnStrategies:
    """How a turn opens and closes, for one processor.

    A new object per call site: these carry per-turn state, so two processors cannot share one.
    """
    return UserTurnStrategies(
        # VAD opens a turn when the line is quiet; the word threshold governs interrupting a bot
        # that is already speaking, which is where an uncancelled echo does its damage.
        start=(
            [VADUserTurnStartStrategy()]
            if MIN_INTERRUPT_WORDS <= 1
            else [MinWordsUserTurnStartStrategy(min_words=MIN_INTERRUPT_WORDS)]
        ),
        stop=[
            SpeechTimeoutUserTurnStopStrategy(
                user_speech_timeout=TURN_END_SILENCE_SECONDS,
                wait_for_transcript=True,
            )
        ],
    )


def build_worker(websocket: WebSocket, control: ControlPlane | None) -> PipelineWorker:
    """Assemble one call: transport, recognizer, model, voice, and the turn machinery.

    Everything here is per call. The services each open their own vendor connection, which is the
    honest cost of concurrency: N simultaneous calls are N recognizer sockets and N synthesizer
    sockets, not one shared pool. That is how pipecat is built, and it is why the ceiling is the
    vendors' rate limits rather than anything in this file.
    """
    serializer = build_serializer()
    transport = build_transport(websocket, serializer)
    speech_to_text = build_speech_to_text()
    llm = build_llm_service(with_hangup=control is not None)
    text_to_speech = build_text_to_speech()
    speech_monitor = BotSpeechMonitor()
    echo_guard = EchoGuard(speech_monitor, ECHO_GUARD_TAIL_SECONDS)

    # The conversation itself. The system prompt lives on the service (and is cached there), so
    # the context carries only the call.
    context = LLMContext()
    if control is not None:
        context.set_tools(
            ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name="end_call",
                        description=(
                            "Hang up the phone call you are on. Say goodbye first: the call "
                            "ends as soon as your words have finished playing."
                        ),
                        properties={},
                        required=[],
                        handler=end_call,
                    )
                ]
                + (
                    [
                        FunctionSchema(
                            name="transfer_call",
                            description=(
                                "Put the caller through to a person. Tell them you are "
                                "transferring first: it happens as soon as your words have "
                                "finished playing, and the call is not yours afterwards."
                            ),
                            properties={},
                            required=[],
                            handler=transfer_call,
                        )
                    ]
                    if TRANSFER_TARGET
                    else []
                )
            )
        )
    # Explicit strategies. Pipecat's defaults would reach for its neural smart-turn analyzer,
    # which is a model download and a second opinion about something the engine has already
    # decided. Start on the engine's speech edge, stop on its endpoint plus the transcript.
    #
    # **Both** processors need them, which is the trap. `UserTurnProcessor` is the visible one, but
    # `LLMUserAggregator` runs turn strategies of its own and it is the one that releases the
    # context to the model -- so configuring only the processor leaves the analyzer deciding when
    # the model runs. Measured on a call: the processor's endpoint fired, and the model waited
    # another 2.7 s for the analyzer to agree, on every turn where it did not agree immediately.
    # Fresh instances rather than one shared object: a strategy carries per-turn state.
    turn_processor = UserTurnProcessor(user_turn_strategies=build_user_turn_strategies())
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=build_user_turn_strategies()),
    )

    resources = (
        CallResources(serializer=serializer, speech=speech_monitor, control=control)
        if control is not None
        else None
    )

    worker = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                speech_to_text,
                # Between the recognizer and the turn logic on purpose: the echo is dropped before
                # anything can treat it as the caller starting a turn.
                echo_guard,
                turn_processor,
                aggregators.user(),
                llm,
                text_to_speech,
                transport.output(),
                speech_monitor,
                aggregators.assistant(),
            ]
        ),
        params=PipelineParams(
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        ),
        app_resources=resources,
        # A worker now lives for exactly one call, so idling is a stuck call rather than a server
        # waiting for work, and the listener no longer depends on it: uvicorn owns the socket and
        # keeps accepting whatever any individual call does. `None` still, because a caller who
        # says nothing for five minutes is a caller, not a fault -- but the blast radius is now one
        # call instead of the whole bot.
        idle_timeout_secs=None,
    )

    if resources is not None:
        resources.worker = worker

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_worker: object, frame: ErrorFrame) -> None:
        """End the call when a stage fails, rather than leaving the caller on a silent line.

        A vendor that refuses a request -- a recognizer or synthesizer over its concurrency limit,
        say -- pushes an `ErrorFrame` that is not marked permanent, so the pipeline carries on and
        the call stays up with a bot that cannot speak. The caller hears nothing, the line stays
        billed, and the only trace is a log line several layers down. Seen for real: eight
        simultaneous calls against a plan allowing two, and half of them answered in silence.

        There is no recovery a caller will wait through on a phone call, so the honest outcome is a
        clean disconnect: a hangup over the control plane if there is one, so they get a BYE rather
        than a socket closing under them.
        """
        processor = getattr(frame, "processor", None)
        logger.error(
            f"pipeline error on call {serializer.call_id or '(no call id yet)'}: {frame}"
            f" -- ending the call rather than leaving it silent"
            f" (processor {getattr(processor, 'name', '?')},"
            f" still usable: {getattr(processor, 'is_usable', '?')})"
        )

        control_call = control.call_for(serializer.call_id) if control is not None else None
        # `control is not None` is redundant at runtime -- control_call is only ever set through
        # it -- but the narrowing does not survive the conditional expression, so say it.
        if control is not None and control_call is not None:
            try:
                await control_call.hangup()
                return
            except control.error_class as error:
                # The caller hanging up first is the ordinary case, not a fault.
                logger.info(f"hangup rejected ({error}); the caller most likely hung up first")

        # No control plane, or it would not take the hangup: end the pipeline instead. The engine
        # tears the call down when the media socket closes, which is a blunter disconnect but still
        # a disconnect.
        await worker.queue_frame(EndWorkerFrame())

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: object, _client: object) -> None:
        logger.info("engine connected, greeting the caller")
        speech_monitor.reset()
        # Seeding also *clears* the previous call's transcript, which matters now that the
        # server outlives a call: `set_messages` replaces, so call two does not inherit call one.
        context.set_messages([{"role": "user", "content": KICKOFF}])
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: object, _client: object) -> None:
        logger.info("engine disconnected, call over")

    return worker


async def serve_call(websocket: WebSocket, control: ControlPlane | None) -> None:
    """Answer one call, on its own pipeline, and tear it down when the engine hangs up."""
    await websocket.accept()
    worker = build_worker(websocket, control)
    runner = WorkerRunner()
    await runner.add_workers(worker)
    # Returns when the pipeline ends, which is when the engine closes the socket. Everything the
    # call owned -- the recognizer and synthesizer connections, the context, the turn state -- goes
    # with it.
    await runner.run()


def build_app(control: ControlPlane | None) -> FastAPI:
    """Build the HTTP surface: one WebSocket route the engine dials, once per call."""
    app = FastAPI()

    @app.websocket("/{path:path}")
    async def media(websocket: WebSocket) -> None:
        # Any path. The engine's `ws_uri` carries a `{call_id}` query and whatever path the
        # deployment chose, and refusing an unexpected one would be a confusing way to fail a call
        # that is otherwise perfectly formed. The call is identified by the `start` envelope, not
        # by the URL.
        try:
            await serve_call(websocket, control)
        except WebSocketDisconnect:
            # The ordinary end of a call: the engine closed first.
            logger.info("engine closed the media socket")
        except Exception:
            # One call's failure must not take the listener with it -- the whole point of moving to
            # a per-call pipeline is that the other calls in flight are unaffected.
            logger.exception("call failed")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def main(host: str, port: int, control: ControlPlane | None) -> None:
    """Answer calls until the process is stopped."""
    # Said once, at startup, rather than per call. Without it, "why didn't it transfer" has two
    # indistinguishable answers -- the tool was never registered, or the model chose not to call
    # it -- and only one of them is a bug.
    logger.info(
        f"models: llm={LLM_MODEL} stt={STT_MODEL or 'pipecat default'}"
        f"/{STT_LANGUAGE or 'default'} tts={TTS_MODEL or 'pipecat default'}"
    )
    logger.info(
        "tools: "
        + (
            ", ".join(
                ["end_call"] + (["transfer_call -> " + TRANSFER_TARGET] if TRANSFER_TARGET else [])
            )
            if control is not None
            else "none (no control plane, so the bot can neither hang up nor transfer)"
        )
    )
    # The turn-taking posture, for the same reason: "it interrupted itself" and "it would not let me
    # interrupt" are both tuning, and neither is visible from a transcript.
    logger.info(
        "turn-taking: "
        + (
            f"echo guard {ECHO_GUARD_TAIL_SECONDS:g}s tail"
            if ECHO_GUARD_TAIL_SECONDS > 0
            else "echo guard off (the engine's canceller vetoes its own echo from 0.5.0)"
        )
        + f", interrupt after {MIN_INTERRUPT_WORDS} word(s)"
    )
    logger.info(f"listening on ws://{host}:{port}/ -- point the engine's ws_uri here")

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(control),
            host=host,
            port=port,
            log_level="warning",
            # uvicorn's own access log would add a line per call to a log that already says more.
            access_log=False,
        )
    )

    if control is None:
        await server.serve()
        return

    # The media path must survive an absent control plane: the engine and this bot restart
    # independently, so "not listening yet" is a routine race and not a reason to exit.
    control_task = asyncio.create_task(control.run_forever())
    try:
        await server.serve()
    finally:
        control_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await control_task


def parse_arguments() -> argparse.Namespace:
    """Parse the bind address and the control-plane options."""
    parser = argparse.ArgumentParser(description="Conversational siphon-rtp media bot")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9001, help="port to bind (default 9001)")
    parser.add_argument(
        "--control-url",
        help="siphon control-plane WebSocket URL; without it the bot cannot hang up",
    )
    parser.add_argument(
        "--control-app",
        default="agent-app",
        help="control application name, matching the handover target (default agent-app)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    control_plane = (
        ControlPlane(
            application=arguments.control_app,
            token=_required_environment_variable(
                "SIPHON_CONTROL_TOKEN", "control-plane bearer token"
            ),
            url=arguments.control_url,
        )
        if arguments.control_url
        else None
    )
    try:
        asyncio.run(main(arguments.host, arguments.port, control_plane))
    except KeyboardInterrupt:
        sys.exit(0)
