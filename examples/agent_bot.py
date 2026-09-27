"""Conversational demo bot: a phone caller talks to Claude, over the siphon-rtp media WebSocket.

The parrot in `echo_bot.py` proves the media path. This one proves the product: speech in,
speech out, with the turn taking done by the media engine rather than in the pipeline, and the
call itself driven over the control plane rather than left as an anonymous audio stream.

    caller -> siphon-sip -> siphon-rtp --(media WebSocket)--> this bot
                    |                                         STT -> Claude -> TTS
                    +------(control WebSocket)---------------> hangup, transfer, hold

Three jobs -- answer a turn, transcribe the caller, speak -- and a provider chosen for each:

    BOT_LLM_PROVIDER   anthropic | google | openai | local
    BOT_STT_PROVIDER   deepgram  | google | openai | local
    BOT_TTS_PROVIDER   cartesia  | google | openai | local

`BOT_BACKEND` is a preset for all three: `cloud` is Claude, Deepgram and Cartesia, `local` keeps
every model on this host. Each role can be overridden on its own, because the useful combinations
are mixed -- a hosted model with a local recognizer, say. Everything that is not a model is the same
object whichever is selected, which is what makes two providers comparable at all.

Each provider asks for its own credential, read from the environment. The default selection wants:

    export ANTHROPIC_API_KEY=...        # console.anthropic.com -- an API key, not a subscription
    export DEEPGRAM_API_KEY=...         # streaming speech to text
    export CARTESIA_API_KEY=...         # streaming text to speech
    export CARTESIA_VOICE_ID=...        # a voice from the Cartesia dashboard

    pip install "pipecat-ai[anthropic,deepgram,cartesia,websocket]" uvicorn
    python examples/agent_bot.py --host 0.0.0.0 --port 9001

Install one extra per provider you point a role at (`google`, `openai`, `whisper`, `kokoro`), and
note that Gemini and Google's speech APIs are different products: `GOOGLE_API_KEY` for the model,
`GOOGLE_APPLICATION_CREDENTIALS` for the recognizer and the voice. Whatever is missing is named at
startup, not at the first call.

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
import importlib
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

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
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import (
    BaseUserTurnStopStrategy,
    UserTurnStoppedParams,
)
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

TURN_INCOMPLETE_HOLD_SECONDS = 1.5
"""How long a turn whose English transcript stops mid-sentence is held open, on either path.

A caller who stops after "my name is" to think is still talking. This is the silence after which
they are answered anyway: long enough for the name, short enough that a finished sentence the word
list misreads ("yes, it is") does not leave the caller waiting."""

FAREWELL_START_SECONDS = 4.0
"""How long to wait for the farewell to begin before concluding there is not going to be one.

The farewell is the model's reply to the tool's result (see HANGUP_PROMPT), so this covers a second
pass through the model as well as the voice's first audio. Too short hangs up before the goodbye
plays; too long leaves a caller on a silent line for a few seconds when the model says nothing."""

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


LLM_PROVIDERS = ("anthropic", "google", "openai", "local")
"""Who answers a turn."""

STT_PROVIDERS = ("deepgram", "google", "openai", "local")
"""Who transcribes the caller."""

TTS_PROVIDERS = ("cartesia", "google", "openai", "local")
"""Who speaks for the bot."""

BACKENDS = ("cloud", "local")
"""The presets. A backend is a shorthand for one provider in each of the three roles."""

BACKEND_PRESETS: dict[str, tuple[str, str, str]] = {
    "cloud": ("anthropic", "deepgram", "cartesia"),
    "local": ("local", "local", "local"),
}
"""What each backend name means, as (LLM, recognizer, synthesizer).

The three roles are chosen separately because the combinations people actually want are mixed: a
hosted model with a local recognizer on a laptop, or a local model with a hosted voice. Folding all
three into one name would need a name per combination -- twelve of them, most never used -- so the
backend stays a *preset* and each role can be overridden on its own."""

BACKEND = os.environ.get("BOT_BACKEND", "").strip().lower() or "cloud"
"""Which preset to start from. `cloud` is the hosted vendors; `local` keeps models on this host.

Overridden per role by `BOT_LLM_PROVIDER`, `BOT_STT_PROVIDER` and `BOT_TTS_PROVIDER`. Everything
that is not a model -- the turn strategies, the echo guard, the tools, the control plane -- is the
same object whatever is selected, which is what makes two providers comparable at all.

Only the selected providers' services are imported: an image built for one provider carries none of
the others' libraries, and a deployment needs only the keys its own three providers ask for.
Checked in `main`, so a typo stops the bot at startup rather than at the first call."""

_PRESET = BACKEND_PRESETS.get(BACKEND, BACKEND_PRESETS["cloud"])
"""The preset to fall back on, per role. An unknown `BOT_BACKEND` is refused in `main` rather than
here, so that importing this module never raises and the error names the variable."""


def _selected_provider(variable: str, preset: str) -> str:
    """Read one role's provider: its own variable when set, otherwise the backend's preset."""
    return os.environ.get(variable, "").strip().lower() or preset


LLM_PROVIDER = _selected_provider("BOT_LLM_PROVIDER", _PRESET[0])
"""Who answers a turn: `anthropic`, `google`, `openai`, or `local` for a server on this host."""

STT_PROVIDER = _selected_provider("BOT_STT_PROVIDER", _PRESET[1])
"""Who transcribes the caller: `deepgram`, `google`, `openai`, or `local` for faster-whisper."""

TTS_PROVIDER = _selected_provider("BOT_TTS_PROVIDER", _PRESET[2])
"""Who speaks for the bot: `cartesia`, `google`, `openai`, or `local` for Kokoro."""

LOCAL_LLM_MODEL = "local"
"""The model name the local path asks for when `BOT_LLM_MODEL` is unset. `llama-server` serves one
model under whatever `--alias` it was started with, and the quickstart starts it with this."""

DEFAULT_LLM_MODEL: dict[str, str] = {
    "anthropic": MODEL,
    # Flash rather than Pro, for the same reason Haiku is the Anthropic default: the caller is
    # listening to silence while the model works.
    "google": "gemini-2.5-flash",
    "openai": "gpt-4.1",
    "local": LOCAL_LLM_MODEL,
}
"""Which model each provider is asked for when `BOT_LLM_MODEL` says nothing.

Pinned per provider rather than left to each service's own default, because a default that moves
changes this example's latency and cost without anyone editing it. Worth choosing deliberately on a
phone call: a model that reasons before answering is right for hard problems and wrong for a
receptionist, because every turn begins with silence. If you do want a thinking model on a call,
pair it with the lowest reasoning setting it offers rather than leaving the default.

On the local path it is only a name. What actually runs is whatever the server loaded."""

LLM_MODEL = os.environ.get("BOT_LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL.get(
    LLM_PROVIDER, MODEL
)
"""The model, overridable per deployment."""

LLM_BASE_URL = os.environ.get("BOT_LLM_BASE_URL", "").strip() or (
    "http://127.0.0.1:8090/v1" if LLM_PROVIDER == "local" else ""
)
"""Where an OpenAI-compatible LLM server answers, when it is not OpenAI's own endpoint.

Defaulted for the `local` provider, which is a server on this host. Empty for `openai`, where empty
means the vendor's own endpoint -- pointing that at localhost is the one way this variable can
silently send every turn nowhere."""

LLM_TEMPERATURE = (
    float(os.environ["BOT_LLM_TEMPERATURE"])
    if os.environ.get("BOT_LLM_TEMPERATURE", "").strip()
    else None
)
"""The model's sampling temperature, on both paths. Unset leaves it to the model server's default.

Pin it when calls are compared with each other: a scenario repeated a hundred times measures the
agent only if the model samples the same way every time."""

LLM_SEED = int(os.environ["BOT_LLM_SEED"]) if os.environ.get("BOT_LLM_SEED", "").strip() else None
"""The sampling seed. Unset lets the server choose.

Sent only by the providers that accept one -- `google`, `openai` and `local`. Anthropic's API has no
seed parameter, so a pinned seed is dropped there rather than sent and rejected; see
`SEED_IS_ACCEPTED`."""

SEED_IS_ACCEPTED: dict[str, bool] = {
    "anthropic": False,
    "google": True,
    "openai": True,
    "local": True,
}
"""Which LLM providers take a sampling seed.

A capability of the provider, not of the backend: the same pinned `BOT_LLM_SEED` has to reach an
OpenAI-compatible server and be withheld from Anthropic, and keying that on "is this the local
path" was wrong the moment a second provider spoke the same API."""

DEFAULT_STT_MODEL: dict[str, str] = {
    "deepgram": "nova-3-general",
    # Google's own model for 8 kHz telephone audio, which is exactly what arrives here. It is also
    # the one model of theirs that refuses a speech-adaptation config, so do not add one.
    "google": "telephony",
    "openai": "gpt-4o-mini-transcribe",
    # faster-whisper's distilled medium English model, pipecat's own local default today, named
    # here so the example does not drift with it.
    "local": "Systran/faster-distil-whisper-medium.en",
}
"""Which recognizer model each provider is asked for.

Pinned rather than left on each service's default, because a phone line is not the general case
those defaults are tuned for: 8 kHz, lossy, and frequently not in the language they assume."""

STT_MODEL = os.environ.get("BOT_STT_MODEL", "").strip() or DEFAULT_STT_MODEL.get(
    STT_PROVIDER, "nova-3-general"
)
"""The recognizer model. Set `BOT_STT_MODEL` to override."""

DEFAULT_STT_LANGUAGE: dict[str, str] = {
    # A public number takes calls in whatever language the caller speaks, and a recognizer pinned
    # to one of them transcribes the rest into confident nonsense. Only Deepgram offers this.
    "deepgram": "multi",
    # Google wants BCP-47, and its telephony model is per-language.
    "google": "en-US",
    "openai": "en",
    # Whisper has no `multi`, and the pinned model above is English-only. Choose a multilingual
    # model before changing this: pipecat refuses an English-only model paired with any other
    # language, and that refusal happens when the model loads, which here is startup.
    "local": "en",
}
"""Which language each recognizer is pinned to when `BOT_STT_LANGUAGE` says nothing."""

STT_LANGUAGE = os.environ.get("BOT_STT_LANGUAGE", "").strip() or DEFAULT_STT_LANGUAGE.get(
    STT_PROVIDER, "en"
)
"""Recognizer language. Pin a single language only if the line really is single-language.

Read twice over: by the recognizer, and by the hold for an unfinished sentence, which only applies
to English. A recognizer left on `multi` reports each transcript's own language and the hold decides
per transcript; see `is_english`."""

DEFAULT_TTS_MODEL: dict[str, str] = {
    "cartesia": "sonic-3.6",
    # Empty: Google's Chirp3 voices carry their own model, and naming one alongside the voice is
    # how you get a voice and a model that disagree.
    "google": "",
    "openai": "gpt-4o-mini-tts",
    # Empty: Kokoro v1.0 serves one model and is selected by voice alone.
    "local": "",
}
"""Which synthesizer model each provider is asked for. Empty means the provider takes none."""

TTS_MODEL = os.environ.get("BOT_TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL.get(TTS_PROVIDER, "")
"""The synthesizer model, where the provider has one. Pinned for the same reason as the recognizer:
an example that drifts with a vendor default is an example whose latency and voice change without
anyone editing it."""

OPENAI_TTS_SAMPLE_RATE = 24000
"""The only rate OpenAI's speech endpoint emits. `response_format: "pcm"` takes no rate parameter,
so this is not a preference -- it is what the bytes are, whatever was asked for."""

DEFAULT_TTS_VOICE: dict[str, str] = {
    # Empty: a Cartesia voice belongs to an account, so it can only come from the environment.
    "cartesia": "",
    "google": "en-US-Chirp3-HD-Charon",
    "openai": "alloy",
    "local": "af_heart",
}
"""Which voice each synthesizer is asked for when `BOT_TTS_VOICE` says nothing."""

TTS_VOICE = os.environ.get("BOT_TTS_VOICE", "").strip() or DEFAULT_TTS_VOICE.get(TTS_PROVIDER, "")
"""The voice to speak with. Empty only for Cartesia, which reads `CARTESIA_VOICE_ID` instead."""

WHISPER_DEVICE = os.environ.get("BOT_WHISPER_DEVICE", "").strip() or "auto"
"""Where faster-whisper runs: `cuda`, `cpu`, or `auto` to take a GPU when one is visible."""

WHISPER_COMPUTE_TYPE = os.environ.get("BOT_WHISPER_COMPUTE_TYPE", "").strip() or "default"
"""faster-whisper's compute type. `default` keeps the precision the model was published in, which
for the distilled models is float16 on a GPU."""

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

You can also put the caller through to a person with the transfer_call tool. Call it first, and
when it returns tell them you are transferring them: the transfer happens once your words have
finished playing. Once you transfer, the call is no longer yours -- so only do it when the caller
has asked for it, or when you cannot help and a person can."""
"""Appended only when a destination is configured, for the same reason as HANGUP_PROMPT: a model
told it can transfer when it cannot promises the caller a handover that never happens. Tool first,
words after, for the reason HANGUP_PROMPT gives."""

HANGUP_PROMPT = """

When the caller is done, call the end_call tool straight away, without saying anything first. When
it returns, say a short goodbye: the call ends once your goodbye has finished playing."""
"""Appended to the system prompt only when the control plane is actually wired.

The prompt and the platform have to agree. A model told it can hang up when it cannot will
promise the caller something that never happens, exactly as a model told it can be interrupted on
a leg without barge-in will invite the caller to cut in and then never hear them.

Tool first, words after. Told to say goodbye and then call the tool, models served locally said
goodbye and ended their turn: replayed from a call, 0 of 5 hangups and 0 of 5 transfers, on a 4B
and a 30B model alike. With the order reversed, 10 of 10 of each. The goodbye still plays before
the line goes, because pipecat runs the model again on the tool's result and `end_call` waits for
that second reply to finish."""


def _required_environment_variable(name: str, purpose: str) -> str:
    """Read an API key from the environment, or exit saying exactly which one is missing."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set. It is the {purpose}; export it and run again.")
    return value


def _provider_key(role: str, provider: str, name: str) -> str:
    """Read `name` for `provider` in `role`, taking the purpose text from `REQUIRED_ENVIRONMENT`.

    The startup check and the builder have to agree about which variables exist and what they are
    for. Reading the purpose out of the one table rather than repeating the sentence here is what
    makes that structural instead of a convention somebody has to remember.
    """
    purposes = dict(REQUIRED_ENVIRONMENT.get((role, provider), ()))
    return _required_environment_variable(name, purposes.get(name, f"{provider} {role} credential"))


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
        elif isinstance(frame, BotStoppedSpeakingFrame) and not self._quiet.is_set():
            # Only a stop that follows speech. The engine acknowledges every `clear` with a mark
            # whether or not anything was playing, the serializer reports each mark as the bot
            # stopping, and every caller turn sends a clear. Taken as an edge, that restarted the
            # echo tail with the bot long silent, and the guard dropped the caller's next words.
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
        # straight away would be satisfied by the silence *before* it. The model calls the tool
        # before it speaks and says the goodbye in its reply to the result, so a bot that is silent
        # now has not started the farewell, whatever it said earlier in the call: speech counts
        # from here. Counting the greeting instead hung up 40 ms before the goodbye would play.
        if self._quiet.is_set():
            self._has_spoken.clear()
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
        # Bound with the SDK in `run_forever`. Until then nothing is registered, so no event can
        # reach them.
        self._transfer_final: Any = lambda _kind: False
        self._transfer_outcome: Any = lambda _event: None

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
            while (event := await call.next_event()) is not None:
                # Draining events is what keeps the handle alive for the call's life; it is also
                # where a transfer's verdict lands. `transfer()` resolves as soon as siphon has
                # *sent* the REFER — RFC 3515 §2.4.4 delivers the outcome afterwards, on the
                # implicit subscription. Without reading these the bot says it transferred the
                # caller and never learns whether it did, and by then the caller is gone either
                # way, so the log is the only place it can show.
                kind = event.get("kind")
                if not self._transfer_final(kind):
                    continue
                outcome = self._transfer_outcome(event) or {}
                detail = (
                    f"stage={outcome.get('stage')} status={outcome.get('status')} "
                    f"reason={outcome.get('reason')}"
                )
                # The kind is the verdict. The status is the NOTIFY's code when siphon has one,
                # and a completed transfer has arrived without it: read as the verdict, the missing
                # code logged a caller who was already hearing the destination as a failed transfer.
                if kind == "TransferCompleted":
                    logger.info(f"transfer completed on call {sip_call_id}: {detail}")
                else:
                    logger.warning(
                        f"transfer FAILED on call {sip_call_id}: {detail} "
                        f"-- the caller was told they were being put through"
                    )
        finally:
            self._calls.pop(sip_call_id, None)
            logger.info(f"control channel detached from call {sip_call_id}")

    async def run_forever(self) -> None:
        """Keep a control connection up, retrying the first connect until the engine is there."""
        (
            control_client_class,
            self.error_class,
            self._transfer_final,
            self._transfer_outcome,
        ) = _import_control_sdk()
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


def _import_control_sdk() -> tuple[Any, type[BaseException], Any, Any]:
    """Import the optional control SDK, or exit saying how to install it."""
    try:
        from siphon_control import (
            ControlClient,
            ControlError,
            is_transfer_final,
            transfer_outcome,
        )
    except ImportError:
        sys.exit("the control plane needs the siphon-control package: pip install siphon-control")
    return ControlClient, ControlError, is_transfer_final, transfer_outcome


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

    # Answer before the line goes: once the caller is hung up there is nobody to hear a result. The
    # result is also what gets the goodbye said. The model calls this tool before speaking (see
    # HANGUP_PROMPT), pipecat runs it again on the result, and that reply is the farewell the wait
    # below listens for.
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
    # As in `end_call`, the result is what gets the handover line said, and the wait below listens
    # for that reply.
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


DANGLING_WORDS = frozenset(
    {
        # Articles and possessives: a noun is still to come.
        "a",
        "an",
        "the",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        # Prepositions a sentence does not end on, unlike "for" or "about" in a question.
        "to",
        "of",
        "with",
        "from",
        "at",
        "into",
        # Conjunctions: another clause is still to come.
        "and",
        "or",
        "but",
        "because",
        "if",
        # The verb of "my name is".
        "is",
    }
)
"""Words an English sentence does not end on: a transcript that stops on one is unfinished.

Deliberately short. A word here that a sentence can end on costs that sentence the hold ("yes, it
is" waits for more), while a word left out only gives that phrase the ordinary short silence."""


def ends_mid_sentence(text: str) -> bool:
    """Whether a transcript stops on a word an English sentence does not end on."""
    words = text.lower().split()
    return bool(words) and words[-1].strip(".,!?;:'\"…-") in DANGLING_WORDS


def is_english(language: str) -> bool:
    """Whether a language code is English, in any regional variant (`en`, `en-US`, `en_AU`)."""
    return language.replace("_", "-").split("-")[0].lower() == "en"


# pipecat's BaseUserTurnStopStrategy.__init_subclass__ is untyped and runs for every subclass, so
# strict mypy flags the class statement itself rather than anything in this file.
class DanglingWordUserTurnStopStrategy(BaseUserTurnStopStrategy):  # type: ignore[no-untyped-call]
    """End the caller's turn when pipecat's speech timeout would, unless they stopped mid-sentence.

    The end of the turn is decided by pipecat's `SpeechTimeoutUserTurnStopStrategy`, unchanged: a
    short silence after the engine's speech edge, with a transcript to show for it, including one
    the recognizer is still delivering. This only holds a turn that strategy would end on words
    that stop where an English sentence does not ("my name is"). It then waits up to
    `hold_seconds`, ends the turn as soon as the rest of the sentence arrives, and drops the wait
    when the caller speaks again.

    Whether the word list applies is decided per transcript, by the language the recognizer reports
    for it, or by `language` when it reports none. A recognizer set to `multi` that reports nothing
    could be hearing any language, and "is" ends plenty of sentences in other languages.

    pipecat's smart-turn model, which judges the audio instead, was tried first. On phone-band
    speech it misjudged both ways: "my name is" passed through an 8 kHz channel read as finished,
    and a recorded "goodbye" as unfinished.
    """

    def __init__(
        self, *, silence_seconds: float, hold_seconds: float, language: str, **kwargs: Any
    ) -> None:
        """End turns after `silence_seconds`, holding an unfinished English one `hold_seconds`."""
        super().__init__(**kwargs)
        self._hold_seconds = hold_seconds
        self._language = language
        self._text = ""
        self._text_language = language
        self._hold: asyncio.Task[None] | None = None
        self._speech_timeout = SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=silence_seconds, wait_for_transcript=True
        )
        self._speech_timeout.add_event_handler("on_user_turn_stopped", self._on_speech_timeout)

    # Typed as pipecat's own strategies type it. Their base class narrows `setup` from the task
    # manager BaseObject takes to the processor setup that carries it, which mypy reports here.
    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        """Set up this strategy and the speech timeout it wraps."""
        await super().setup(setup)
        await self._speech_timeout.setup(setup)

    # pipecat's base cleanup and the speech timeout's turn callbacks are unannotated, which strict
    # mypy reports at each call.
    async def cleanup(self) -> None:
        """Cancel a pending hold, and the speech timeout's own timers, along with the call."""
        await super().cleanup()  # type: ignore[no-untyped-call]
        await self._cancel_hold()
        await self._speech_timeout.cleanup()  # type: ignore[no-untyped-call]

    async def handle_user_turn_started(self) -> None:
        """Start the turn with no words heard and nothing held."""
        await self._forget()
        await self._speech_timeout.handle_user_turn_started()  # type: ignore[no-untyped-call]

    async def handle_user_turn_stopped(self) -> None:
        """Leave nothing held once the turn is over, however it ended."""
        await self._forget()
        await self._speech_timeout.handle_user_turn_stopped()  # type: ignore[no-untyped-call]

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Note the turn's words, then let the speech timeout judge the frame."""
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._cancel_hold()
        elif isinstance(frame, TranscriptionFrame):
            # Before the speech timeout sees it, since this transcript may be what ends the turn.
            self._text = f"{self._text} {frame.text}".strip()
            self._text_language = str(frame.language) if frame.language else self._language
        return await self._speech_timeout.process_frame(frame)

    async def _on_speech_timeout(
        self, _strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ) -> None:
        """End the turn the speech timeout ended, or hold it if the sentence is unfinished."""
        if is_english(self._text_language) and ends_mid_sentence(self._text):
            if self._hold is None:
                self._hold = self.task_manager.create_task(
                    self._end_after_hold(params.enable_user_speaking_frames), f"{self}::hold"
                )
            return
        await self._cancel_hold()
        await self.trigger_user_turn_stopped(
            enable_user_speaking_frames=params.enable_user_speaking_frames
        )

    async def _end_after_hold(self, enable_user_speaking_frames: bool) -> None:
        """End the turn with what was said once the hold passes without the rest of it."""
        await asyncio.sleep(self._hold_seconds)
        # Cleared first: ending the turn forgets the hold, and must not cancel this task.
        self._hold = None
        await self.trigger_user_turn_stopped(
            enable_user_speaking_frames=enable_user_speaking_frames
        )

    async def _cancel_hold(self) -> None:
        """Drop a pending hold, if there is one."""
        if self._hold is not None:
            await self.task_manager.cancel_task(self._hold)
            self._hold = None

    async def _forget(self) -> None:
        """Clear the turn's words and any hold."""
        self._text = ""
        self._text_language = self._language
        await self._cancel_hold()


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


_GOOGLE_CLOUD_CREDENTIALS = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "path to a Google Cloud service-account JSON file",
)
"""Google's speech APIs are Cloud services and authenticate as a service account, not with an API
key. pipecat passes no credentials of its own, so the client falls back to application default
credentials, which is this variable. The same account needs the Speech-to-Text and Text-to-Speech
APIs enabled; Gemini is a different product with a different credential, and one does not imply the
other."""

REQUIRED_ENVIRONMENT: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("llm", "anthropic"): (("ANTHROPIC_API_KEY", "Claude API key"),),
    ("llm", "google"): (("GOOGLE_API_KEY", "Gemini API key"),),
    ("llm", "openai"): (("OPENAI_API_KEY", "OpenAI API key"),),
    # Nothing: the model server's address has a default and takes no key.
    ("llm", "local"): (),
    ("stt", "deepgram"): (("DEEPGRAM_API_KEY", "speech-to-text key"),),
    ("stt", "google"): (_GOOGLE_CLOUD_CREDENTIALS,),
    ("stt", "openai"): (("OPENAI_API_KEY", "OpenAI API key"),),
    # Nothing: the model downloads itself.
    ("stt", "local"): (),
    ("tts", "cartesia"): (
        ("CARTESIA_API_KEY", "text-to-speech key"),
        ("CARTESIA_VOICE_ID", "voice to speak with"),
    ),
    ("tts", "google"): (_GOOGLE_CLOUD_CREDENTIALS,),
    ("tts", "openai"): (("OPENAI_API_KEY", "OpenAI API key"),),
    ("tts", "local"): (),
}
"""What each provider cannot start without, in each role, and what the value is for.

Keyed per role *and* provider because the same provider asks for different things depending on the
job -- Gemini takes an API key while Google's speech APIs take a service account -- and because the
one table is then the only place a key is named. The builders read their purpose text out of it
rather than repeating the pair, so the two cannot drift apart."""

_ROLE_PROVIDER_CHOICES: dict[str, tuple[str, ...]] = {
    "llm": LLM_PROVIDERS,
    "stt": STT_PROVIDERS,
    "tts": TTS_PROVIDERS,
}
_ROLE_VARIABLES: dict[str, str] = {
    "llm": "BOT_LLM_PROVIDER",
    "stt": "BOT_STT_PROVIDER",
    "tts": "BOT_TTS_PROVIDER",
}


def selected_providers() -> dict[str, str]:
    """Name the provider chosen for each role, as the module resolved it at import."""
    return {"llm": LLM_PROVIDER, "stt": STT_PROVIDER, "tts": TTS_PROVIDER}


def required_environment(providers: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """List what `providers` need between them, each variable once, in role order.

    Raises:
        ValueError: If a role is unknown, or its provider is not one the role offers.

    """
    required: dict[str, str] = {}
    for role, provider in providers.items():
        choices = _ROLE_PROVIDER_CHOICES.get(role)
        if choices is None:
            raise ValueError(f"unknown role {role!r}: expected one of {', '.join(_ROLE_VARIABLES)}")
        if provider not in choices:
            variable = _ROLE_VARIABLES[role]
            raise ValueError(
                f"unknown {variable} {provider!r}: expected one of {', '.join(choices)}"
            )
        for name, purpose in REQUIRED_ENVIRONMENT[role, provider]:
            # One provider in two roles asks for its key once, and the first purpose wins: both
            # read the same variable, so a second line about it would only be noise.
            required.setdefault(name, purpose)
    return tuple(required.items())


def missing_environment(providers: Mapping[str, str], environment: Mapping[str, str]) -> list[str]:
    """Name the variables `providers` need that `environment` does not provide.

    An empty value counts as missing, because compose passes a variable that is unset in `.env`
    through as an empty string rather than leaving it out.

    Raises:
        ValueError: If a role is unknown, or its provider is not one the role offers.

    """
    return [name for name, _purpose in required_environment(providers) if not environment.get(name)]


def _provider_summary(providers: Mapping[str, str]) -> str:
    """Name the selection in an error, as the preset when all three agree with one."""
    chosen = (providers.get("llm", ""), providers.get("stt", ""), providers.get("tts", ""))
    for name, preset in BACKEND_PRESETS.items():
        if chosen == preset:
            return f"the {name} backend"
    return "llm={}, stt={}, tts={}".format(*chosen)


def refuse_an_unknown_backend(backend: str) -> None:
    """Stop a preset nobody defined, naming the variable and what it accepts.

    Raises:
        ValueError: If `backend` is not one of `BACKENDS`.

    """
    if backend not in BACKEND_PRESETS:
        raise ValueError(f"unknown BOT_BACKEND {backend!r}: expected one of {', '.join(BACKENDS)}")


_Model = TypeVar("_Model")


class SharedConstructor(Generic[_Model]):
    """Build each distinct model once per process, however many calls construct it.

    Stands in for the constructor a pipecat service module calls. The services are built per call
    and the local ones load their model inside `__init__`, so without this every call pays a full
    model load before it can greet the caller, and two calls at once hold two copies in memory. One
    instance serves every call: both services already run inference off the event loop, and both
    runtimes underneath (ctranslate2 and ONNX Runtime) support concurrent inference on one model.
    """

    def __init__(self, construct: Callable[..., _Model]) -> None:
        """Wrap `construct`, which is then called at most once per distinct set of arguments."""
        self._construct = construct
        self._models: dict[tuple[Any, ...], _Model] = {}
        # Held across the construction itself, so two calls arriving together load the model once
        # rather than both loading it and one copy being thrown away.
        self._lock = threading.Lock()

    def __call__(self, *arguments: Any, **keyword_arguments: Any) -> _Model:
        """Return the model for these arguments, building it the first time they are seen."""
        key = (arguments, tuple(sorted(keyword_arguments.items())))
        with self._lock:
            if key not in self._models:
                self._models[key] = self._construct(*arguments, **keyword_arguments)
            return self._models[key]

    def built(self) -> list[_Model]:
        """Every model built so far, in the order each was first asked for."""
        with self._lock:
            return list(self._models.values())


LOCAL_MODEL_CONSTRUCTORS: dict[str, tuple[str, str]] = {
    "stt": ("pipecat.services.whisper.stt", "WhisperModel"),
    "tts": ("pipecat.services.kokoro.tts", "Kokoro"),
}
"""For each role a local model can fill, the pipecat module and the constructor it builds through.

The name is replaced in the module rather than the service being subclassed, because Kokoro builds
its model inline in `__init__` with nothing to override, and doing Whisper the same way keeps one
mechanism instead of two. `tests/test_agent_example.py` reads pipecat's installed source to check
that both names are still where the construction happens.

Keyed by role rather than a flat list, because the roles are chosen separately: a bot running a
hosted model with a local recognizer must not import Kokoro, and importing it is what would pull a
synthesizer the image may not even carry."""


def local_roles() -> tuple[str, ...]:
    """Which roles a model on this host fills, in `LOCAL_MODEL_CONSTRUCTORS` order."""
    providers = selected_providers()
    return tuple(role for role in LOCAL_MODEL_CONSTRUCTORS if providers.get(role) == "local")


def share_local_models() -> None:
    """Make the local services share one loaded model per process. Safe to call more than once."""
    for role in local_roles():
        module_name, attribute = LOCAL_MODEL_CONSTRUCTORS[role]
        module = importlib.import_module(module_name)
        construct = getattr(module, attribute, None)
        if construct is None:
            sys.exit(
                f"{module_name} no longer defines {attribute}, which is how this example shares "
                f"one model across calls. Update LOCAL_MODEL_CONSTRUCTORS for this pipecat version."
            )
        if not isinstance(construct, SharedConstructor):
            setattr(module, attribute, SharedConstructor(construct))


def preload_local_models() -> None:
    """Load, download and exercise the local models once, before the first call can arrive.

    Building a service is what loads its model, so this builds one of each and discards it: the
    models stay behind in the shared constructors. Each is then run once on a throwaway input,
    because the libraries load lazily -- ctranslate2 opens cuDNN at the first transcription, not
    when the model loads -- and a missing library should stop the bot at startup rather than
    silence its first caller.
    """
    import numpy
    from pipecat.transcriptions.language import Language

    roles = local_roles()
    started = time.monotonic()
    share_local_models()

    if "stt" in roles:
        build_speech_to_text()
        whisper = importlib.import_module("pipecat.services.whisper.stt")
        for model in whisper.WhisperModel.built():
            segments, _ = model.transcribe(numpy.zeros(PIPELINE_SAMPLE_RATE, dtype=numpy.float32))
            list(segments)  # transcription is lazy: iterating the segments is what runs it

    if "tts" in roles:
        build_text_to_speech()
        kokoro = importlib.import_module("pipecat.services.kokoro.tts")
        language = kokoro.language_to_kokoro_language(Language.EN)
        for synthesizer in kokoro.Kokoro.built():
            synthesizer.create("Ready.", voice=TTS_VOICE, speed=1.0, lang=language)

    logger.info(
        f"local models loaded and exercised in {time.monotonic() - started:.1f}s "
        f"({', '.join(roles)})"
    )


def build_llm_service(with_hangup: bool) -> LLMService[Any]:
    """Build the LLM service for the selected backend, tuned for a conversation in real time."""
    system_instruction = (
        SYSTEM_PROMPT
        + (HANGUP_PROMPT if with_hangup else "")
        + (TRANSFER_PROMPT if with_hangup and TRANSFER_TARGET else "")
    )
    builders: dict[str, Callable[[str], LLMService[Any]]] = {
        "anthropic": _build_anthropic_llm_service,
        "google": _build_google_llm_service,
        "openai": _build_openai_llm_service,
        "local": _build_local_llm_service,
    }
    # Deliberately no `.get(..., default)`: an unrecognised provider is refused at startup, and a
    # silent fallback to one particular vendor would bill somebody for a typo.
    return builders[LLM_PROVIDER](system_instruction)


def _sampling_settings(provider: str) -> dict[str, Any]:
    """Gather only what is pinned, and only what `provider` accepts.

    A value passed here, even one equal to the service's own default, overrides the server's, so an
    unpinned temperature has to be *absent* rather than explicit. The seed is dropped for a
    provider whose API has none, because sending it is an error rather than a no-op.
    """
    sampling: dict[str, Any] = {}
    if LLM_TEMPERATURE is not None:
        sampling["temperature"] = LLM_TEMPERATURE
    if LLM_SEED is not None and SEED_IS_ACCEPTED.get(provider, False):
        sampling["seed"] = LLM_SEED
    return sampling


def _build_anthropic_llm_service(system_instruction: str) -> LLMService[Any]:
    """Build the Claude service."""
    from pipecat.services.anthropic.llm import AnthropicLLMService

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
        api_key=_provider_key("llm", "anthropic", "ANTHROPIC_API_KEY"),
        settings=AnthropicLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_instruction,
            max_tokens=MAXIMUM_RESPONSE_TOKENS,
            # The system prompt is resent on every turn of the call; caching it is free money.
            enable_prompt_caching=True,
            extra=extra,
            **_sampling_settings("anthropic"),
        ),
    )


def _build_google_llm_service(system_instruction: str) -> LLMService[Any]:
    """Build the Gemini service."""
    from pipecat.services.google.llm import GoogleLLMService

    # Gemini's thinking dial, and the reason this is not one builder shared with Anthropic: the
    # three providers spell the same idea three ways (`output_config.effort`, `thinking`,
    # `reasoning_effort`) and each rejects the others' spelling outright. pipecat already asks
    # Gemini not to think by default, which is what a phone turn wants, so nothing is set here.

    return GoogleLLMService(
        api_key=_provider_key("llm", "google", "GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_instruction,
            # Gemini takes `max_tokens`; the OpenAI-shaped services take `max_completion_tokens`.
            max_tokens=MAXIMUM_RESPONSE_TOKENS,
            **_sampling_settings("google"),
        ),
    )


def _build_openai_llm_service(system_instruction: str) -> LLMService[Any]:
    """Build the OpenAI service."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key=_provider_key("llm", "openai", "OPENAI_API_KEY"),
        # `None` means the vendor's own endpoint. Only set for a compatible server elsewhere.
        base_url=LLM_BASE_URL or None,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_instruction,
            max_completion_tokens=MAXIMUM_RESPONSE_TOKENS,
            **_sampling_settings("openai"),
        ),
    )


def _build_local_llm_service(system_instruction: str) -> LLMService[Any]:
    """Build a client for the OpenAI-compatible LLM server on this host."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        # The server takes no key, but the OpenAI client refuses to start without one.
        api_key="unused-by-a-local-server",
        base_url=LLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_instruction,
            max_completion_tokens=MAXIMUM_RESPONSE_TOKENS,
            **_sampling_settings("local"),
        ),
    )


def build_speech_to_text() -> STTService:
    """Build the recognizer for the selected provider, at the pipeline's rate."""
    builders: dict[str, Callable[[], STTService]] = {
        "deepgram": _build_deepgram_stt_service,
        "google": _build_google_stt_service,
        "openai": _build_openai_stt_service,
        "local": _build_local_stt_service,
    }
    return builders[STT_PROVIDER]()


def _build_deepgram_stt_service() -> STTService:
    """Build Deepgram's recognizer. The one that takes `multi`."""
    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=_provider_key("stt", "deepgram", "DEEPGRAM_API_KEY"),
        # A raw language string here, unlike the two below: Deepgram's own codes include `multi`,
        # which is not a language and so has no member in pipecat's enum.
        settings=DeepgramSTTService.Settings(model=STT_MODEL, language=STT_LANGUAGE),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def _build_google_stt_service() -> STTService:
    """Build Google's recognizer, on the model meant for telephone audio."""
    from pipecat.services.google.stt import GoogleSTTService

    # No credentials passed: pipecat falls back to Google's application default credentials, which
    # is `GOOGLE_APPLICATION_CREDENTIALS`. Read here only to fail with that name rather than with
    # the library's own `No valid credentials provided.`
    _provider_key("stt", "google", "GOOGLE_APPLICATION_CREDENTIALS")
    return GoogleSTTService(
        settings=GoogleSTTService.Settings(model=STT_MODEL, language=STT_LANGUAGE),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def _build_openai_stt_service() -> STTService:
    """Build OpenAI's recognizer."""
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.transcriptions.language import Language

    return OpenAISTTService(
        api_key=_provider_key("stt", "openai", "OPENAI_API_KEY"),
        settings=OpenAISTTService.Settings(model=STT_MODEL, language=Language(STT_LANGUAGE)),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def _build_local_stt_service() -> STTService:
    """Build faster-whisper, on this host."""
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language

    return WhisperSTTService(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        settings=WhisperSTTService.Settings(model=STT_MODEL, language=Language(STT_LANGUAGE)),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def build_text_to_speech() -> TTSService:
    """Build the voice for the selected provider, at the pipeline's rate."""
    builders: dict[str, Callable[[], TTSService]] = {
        "cartesia": _build_cartesia_tts_service,
        "google": _build_google_tts_service,
        "openai": _build_openai_tts_service,
        "local": _build_local_tts_service,
    }
    return builders[TTS_PROVIDER]()


def _build_cartesia_tts_service() -> TTSService:
    """Build Cartesia's voice."""
    from pipecat.services.cartesia.tts import CartesiaTTSService

    # The only provider whose voice cannot have a default: a Cartesia voice belongs to an account.
    # `BOT_TTS_VOICE` still wins when it is set, so the two ways of saying it agree.
    voice = TTS_VOICE or _provider_key("tts", "cartesia", "CARTESIA_VOICE_ID")
    return CartesiaTTSService(
        api_key=_provider_key("tts", "cartesia", "CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(voice=voice, model=TTS_MODEL),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def _build_google_tts_service() -> TTSService:
    """Build Google's voice."""
    from pipecat.services.google.tts import GoogleTTSService

    # As with the recognizer: application default credentials, named here so the failure does.
    _provider_key("tts", "google", "GOOGLE_APPLICATION_CREDENTIALS")
    # The Chirp3 voices carry their own model, so `model` stays unset unless a deployment names
    # one. `NOT_GIVEN` rather than `None`: pipecat reads None as a value to send.
    from pipecat.utils.types import NOT_GIVEN

    return GoogleTTSService(
        settings=GoogleTTSService.Settings(voice=TTS_VOICE, model=TTS_MODEL or NOT_GIVEN),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def _build_openai_tts_service() -> TTSService:
    """Build OpenAI's voice, at its own rate rather than the pipeline's."""
    from pipecat.services.openai.tts import OpenAITTSService

    # The one service here that is not asked for `PIPELINE_SAMPLE_RATE`, and it matters. OpenAI's
    # speech endpoint takes no rate parameter: `response_format: "pcm"` is always
    # OPENAI_TTS_SAMPLE_RATE. pipecat asks for the rate anyway and then *labels the frames with it*,
    # so asking for 16 kHz yields 24 kHz bytes tagged 16 kHz -- playable, and a third slower and a
    # fifth lower than the voice actually said it. pipecat only warns.
    #
    # Declaring the true rate makes the frames honest, and the output transport resamples them to
    # the wire on the way out, which is the one place in this process that converts.
    return OpenAITTSService(
        api_key=_provider_key("tts", "openai", "OPENAI_API_KEY"),
        settings=OpenAITTSService.Settings(voice=TTS_VOICE, model=TTS_MODEL),
        sample_rate=OPENAI_TTS_SAMPLE_RATE,
    )


def _build_local_tts_service() -> TTSService:
    """Build Kokoro, on this host."""
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=TTS_VOICE),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def build_user_turn_strategies() -> UserTurnStrategies:
    """How a turn opens and closes. A new object per call: the strategies carry per-turn state."""
    # The word threshold governs interrupting a bot that is already speaking, which is where an
    # uncancelled echo does its damage.
    start: list[BaseUserTurnStartStrategy] = (
        [VADUserTurnStartStrategy()]
        if MIN_INTERRUPT_WORDS <= 1
        else [MinWordsUserTurnStartStrategy(min_words=MIN_INTERRUPT_WORDS)]
    )
    # The same on both paths, so a caller pausing mid-sentence gets the same bot from either: a
    # fixed silence ended "my name is" before the name, and the hold decides per transcript
    # whether its English word list applies.
    stop = DanglingWordUserTurnStopStrategy(
        silence_seconds=TURN_END_SILENCE_SECONDS,
        hold_seconds=TURN_INCOMPLETE_HOLD_SECONDS,
        language=STT_LANGUAGE,
    )
    return UserTurnStrategies(start=start, stop=[stop])


def build_user_aggregator_params() -> LLMUserAggregatorParams:
    """Let the aggregator, which releases the context to the model, decide the caller's turn.

    It has to be the only processor deciding. A `UserTurnProcessor` in front of it decides the
    same turn from the same frames, and that pair stalled on calls whether the aggregator held
    its own copy of these strategies or followed the processor through
    `ExternalUserTurnStrategies`. The recognizer announces a transcription latency, so the end
    of the caller's speech arms an end-of-turn timer in the processor. The word count then opens
    the turn on the transcript, and cancelling that timer yields before the processor announces
    the start. The aggregator handles the transcript in that gap, ahead of the turn it belongs
    to, and then cannot end the turn until pipecat's five-second fallback. Measured on a call:
    the processor ended the turn 200 ms after the caller said goodbye, and the model was asked
    5.0 s after that. Within one controller the strategies run in order, so there is no gap.

    Explicit rather than left unset, because an aggregator with no strategies adopts pipecat's
    defaults, which choose one way of opening and closing a turn for every backend, and the two
    backends here need different ones: see `build_user_turn_strategies`.
    """
    return LLMUserAggregatorParams(
        user_turn_strategies=build_user_turn_strategies(),
        empty_user_turn=EMPTY_TURN_RECOVERY,
    )


EMPTY_TURN_RECOVERY = EmptyUserTurnConfig(
    # Said out loud, so it is one sentence rather than pipecat's three. Its own default explains the
    # situation to the model at chat length and asks it to repeat any question the caller may have
    # missed, which on a phone turns a missed word into a speech.
    interrupted_prompt=(
        "The caller said something while you were talking and it was not recognised. "
        "Ask them to say it again, in one short sentence."
    ),
    # Left off. A caller who goes quiet is a caller thinking, not a fault, and the same reasoning
    # keeps `idle_timeout_secs` unset on the worker: prompting them to speak would talk over them
    # just as they started. pipecat leaves this off by default too.
    idle_prompt=None,
    # One. This is the guard that matters here, because on a leg whose echo the engine cannot cancel
    # the bot's own voice can open a turn, and the echo guard then drops the transcript -- which is
    # an interrupted turn with no words in it, indistinguishable from a caller who was not heard. At
    # one recovery the worst case is a single "say that again" the caller did not ask for; higher,
    # and the bot could sit there asking its own echo to repeat itself.
    max_consecutive_recoveries=1,
)
"""What to do about a caller turn that ends with nothing recognised in it.

New in pipecat 1.12 and on by default, which is the right default for a phone call: before it, an
interruption the recognizer could not make out left the bot silent, and silence after someone speaks
reads as a dropped call. The prompt is replaced rather than the feature disabled."""

ObserverFactory = Callable[[SiphonFrameSerializer], BaseObserver]
"""Builds an observer for one call. It is handed the call's serializer, which is what learns the
SIP Call-ID once the engine's `start` envelope arrives."""


def build_worker(
    websocket: WebSocket,
    control: ControlPlane | None,
    observer_factory: ObserverFactory | None = None,
) -> PipelineWorker:
    """Assemble one call: transport, recognizer, model, voice, and the turn machinery.

    Everything here is per call. The services each open their own vendor connection, which is the
    honest cost of concurrency: N simultaneous calls are N recognizer sockets and N synthesizer
    sockets, not one shared pool. That is how pipecat is built, and it is why the ceiling is the
    vendors' rate limits rather than anything in this file.

    `observer_factory`, when given, builds an observer that watches this call's pipeline without
    being part of it -- how a test harness reads what the bot heard and decided.
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
                            "Hang up the phone call. Call it as soon as the caller is done, "
                            "before saying goodbye, and say the goodbye after it returns: the "
                            "call ends once the goodbye has finished playing."
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
                                "Put the caller through to a person. Call it first, and after "
                                "it returns tell them you are transferring them: the transfer "
                                "happens once your words have finished playing, and the call is "
                                "not yours afterwards."
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
    # The aggregator decides the caller's turn, and nothing else in the pipeline does: see
    # `build_user_aggregator_params` for what a second decider in front of it cost. Its strategies
    # are explicit and differ by backend: the cloud path ends a turn on the engine's endpoint plus
    # the transcript, the local path also holds a sentence the caller stopped in the middle of.
    aggregators = LLMContextAggregatorPair(context, user_params=build_user_aggregator_params())

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
        observers=[observer_factory(serializer)] if observer_factory is not None else [],
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


async def serve_call(
    websocket: WebSocket,
    control: ControlPlane | None,
    observer_factory: ObserverFactory | None = None,
) -> None:
    """Answer one call, on its own pipeline, and tear it down when the engine hangs up."""
    await websocket.accept()
    worker = build_worker(websocket, control, observer_factory)
    runner = WorkerRunner()
    await runner.add_workers(worker)
    # Returns when the pipeline ends, which is when the engine closes the socket. Everything the
    # call owned -- the recognizer and synthesizer connections, the context, the turn state -- goes
    # with it.
    await runner.run()


def build_app(
    control: ControlPlane | None, observer_factory: ObserverFactory | None = None
) -> FastAPI:
    """Build the HTTP surface: one WebSocket route the engine dials, once per call."""
    app = FastAPI()

    @app.websocket("/{path:path}")
    async def media(websocket: WebSocket) -> None:
        # Any path. The engine's `ws_uri` carries a `{call_id}` query and whatever path the
        # deployment chose, and refusing an unexpected one would be a confusing way to fail a call
        # that is otherwise perfectly formed. The call is identified by the `start` envelope, not
        # by the URL.
        try:
            await serve_call(websocket, control, observer_factory)
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


async def main(
    host: str,
    port: int,
    control: ControlPlane | None,
    observer_factory: ObserverFactory | None = None,
) -> None:
    """Answer calls until the process is stopped, each watched by `observer_factory`'s observer."""
    # Before anything listens. A selection that cannot run should stop the bot here, naming what is
    # missing, rather than at the first call, where the caller hears it as a dead line.
    providers = selected_providers()
    try:
        refuse_an_unknown_backend(BACKEND)
        missing = missing_environment(providers, os.environ)
    except ValueError as error:
        sys.exit(str(error))
    if missing:
        purposes = dict(required_environment(providers))
        sys.exit(
            f"{_provider_summary(providers)} needs "
            + ", ".join(f"{name} (the {purposes[name]})" for name in missing)
            + "; set them and run again."
        )
    if local_roles():
        preload_local_models()

    # Said once, at startup, rather than per call. Without it, "why didn't it transfer" has two
    # indistinguishable answers -- the tool was never registered, or the model chose not to call
    # it -- and only one of them is a bug. Every segment names its provider, because with the three
    # roles chosen separately the model alone no longer says what is running.
    logger.info(
        f"models: llm={LLM_PROVIDER}/{LLM_MODEL}"
        + (f" at {LLM_BASE_URL}" if LLM_BASE_URL else "")
        + f" stt={STT_PROVIDER}/{STT_MODEL}/{STT_LANGUAGE}"
        + (f" on {WHISPER_DEVICE}" if STT_PROVIDER == "local" else "")
        + f" tts={TTS_PROVIDER}/{TTS_MODEL or TTS_VOICE or 'account voice'}"
        # A run compared against another is only comparable at the same sampling, so say which.
        + f" temperature={'default' if LLM_TEMPERATURE is None else LLM_TEMPERATURE}"
        + (
            f" seed={'default' if LLM_SEED is None else LLM_SEED}"
            if SEED_IS_ACCEPTED.get(LLM_PROVIDER, False)
            else " seed=n/a"
        )
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
            build_app(control, observer_factory),
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


def run(arguments: argparse.Namespace, observer_factory: ObserverFactory | None = None) -> None:
    """Run the bot with the parsed `arguments` until the process is stopped.

    `observer_factory` watches each call without this file being edited: the conversation harness
    passes one that writes what every call's pipeline heard, decided and said.
    """
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
        asyncio.run(main(arguments.host, arguments.port, control_plane, observer_factory))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    run(parse_arguments())
