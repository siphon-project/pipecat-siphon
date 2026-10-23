"""Conversational demo bot: a phone caller talks to Claude, over the siphon-rtp media WebSocket.

The parrot in `echo_bot.py` proves the media path. This one proves the product: speech in,
speech out, with the turn taking done by the media engine rather than in the pipeline, and the
call itself driven over the control plane rather than left as an anonymous audio stream.

    caller -> siphon-sip -> siphon-rtp --(media WebSocket)--> this bot
                    |                                         STT -> Claude -> TTS
                    +------(control WebSocket)---------------> hangup, transfer, hold

Three jobs -- answer a turn, transcribe the caller, speak -- and a provider chosen for each:

    BOT_LLM_PROVIDER   anthropic | google | openai | local | gemini-live
    BOT_STT_PROVIDER   deepgram  | google | openai | local | gemini-live
    BOT_TTS_PROVIDER   cartesia  | google | openai | local | gemini-live

`BOT_BACKEND` is a preset for all three: `cloud` is Claude, Deepgram and Cartesia, `local` keeps
every model on this host, and `gemini-live` is Gemini's native audio, one model that hears, answers
and speaks, detecting the caller's turns itself. Each role can be overridden on its own, because
the useful combinations are mixed -- a hosted model with a local recognizer, say. Everything that
is not a model is the same object whichever is selected, which is what makes two providers
comparable at all.

Each provider asks for its own credential, read from the environment. The default selection wants:

    export ANTHROPIC_API_KEY=...        # console.anthropic.com -- an API key, not a subscription
    export DEEPGRAM_API_KEY=...         # streaming speech to text
    export CARTESIA_API_KEY=...         # streaming text to speech
    export CARTESIA_VOICE_ID=...        # a voice from the Cartesia dashboard

    pip install "pipecat-ai[anthropic,deepgram,cartesia,websocket]" uvicorn
    python examples/agent_bot.py --host 0.0.0.0 --port 9001

Install one extra per provider you point a role at (`google`, `openai`, `whisper`, `kokoro`), and
note that Gemini and Google's speech APIs are different products: `GOOGLE_API_KEY` for the model,
application default credentials for the recognizer and the voice -- `GOOGLE_APPLICATION_CREDENTIALS`
pointing at a service-account JSON file, or on Google Cloud the instance's own service account with
nothing set. Whatever is missing is named at startup, not at the first call.

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
import json
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import uvicorn
import yaml  # type: ignore[import-untyped]
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
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
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
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
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

from pipecat_siphon import (
    CallContext,
    Chose,
    ContextOrganisation,
    Failed,
    Finished,
    Report,
    SessionBrief,
    SiphonFrameSerializer,
    VoiceSession,
    VoiceSessionClient,
    session_token_from,
)
from pipecat_siphon.outbound import PlacedCall, PlacedCalls
from pipecat_siphon.voice_session import QUERY_PARAMETER as SESSION_QUERY_PARAMETER

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


def kickoff_for(placed_context: Mapping[str, str] | None) -> str:
    """Return the greeting turn's one message: `KICKOFF` coming in, its twin for a placed call.

    On a placed call the bot is the caller, and the model has to know it to open the call the right
    way round: say who it is and why it rang, rather than asking how it can help.
    """
    if placed_context is None:
        return KICKOFF
    reason = placed_context.get("reason", "").strip()
    about = f", about {reason}" if reason else ""
    return f"[You placed this call{about}. They have just answered and have not spoken yet.]"


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

CONTROL_CONNECT_TIMEOUT_SECONDS = 30.0
"""How long `--call` waits for the control plane before giving up on placing its call.

The media path waits forever, since siphon restarting is routine; a one-shot call is not worth
waiting forever for, and a process that never exits would be the worst way to say so."""

CONTROL_RETRY_MINIMUM_SECONDS = 1.0
CONTROL_RETRY_MAXIMUM_SECONDS = 30.0
"""Backoff bounds for the control connection.

The client supervises its own reconnects once established, but the *first* connect raises if the
engine is not listening yet. Gathered naively with the pipeline that exception takes the media
path down with it and the process exits on a routine restart-ordering race."""

DEFAULT_SYSTEM_PROMPT = """\
You are a voice assistant on a telephone call. You are talking, not writing.

Keep answers to one or two sentences unless you are explicitly asked for detail. Ask one question
at a time. Never use markdown, bullet points, headings, emoji or any other formatting: everything
you write is read aloud verbatim, so write numbers, dates and units as a person would say them.

The caller can interrupt you at any time and you will be cut off mid-sentence when they do. That
is normal. When it happens, answer what they just said rather than finishing your last thought.

Open the call by greeting the caller and asking how you can help."""
"""The persona when the deployment does not supply one. See `resolve_system_prompt`."""


def resolve_system_prompt(environment: Mapping[str, str]) -> str:
    """Pick the persona: `BOT_SYSTEM_PROMPT`, the file at `BOT_SYSTEM_PROMPT_FILE`, or the default.

    Lets a deployment replace the persona without editing this file. The persona is the part worth
    tuning per deployment and the part most likely to differ between a demo and something answering
    real callers, so it should not require a fork of the example to change. What it must not lose
    is the telephone constraints -- speech not prose, no formatting, short answers -- because a
    replacement written as if for a chat window produces markdown that the synthesizer then reads
    aloud, bullet by bullet.

    The hangup instructions are appended separately and are not overridable: they describe a
    capability of this process rather than a persona, and a prompt that omits them leaves the model
    telling callers it cannot end the call.

    The file form exists because a persona worth writing runs to paragraphs, which a single `.env`
    line carries badly. It is read on every call rather than once, so a persona can be tuned between
    calls without a restart. `examples/prompts/escalation_demo.txt` is one.

    Raises:
        ValueError: If both are set, or the file cannot be read or holds nothing.

    """
    inline = environment.get("BOT_SYSTEM_PROMPT", "").strip()
    path = environment.get("BOT_SYSTEM_PROMPT_FILE", "").strip()
    if inline and path:
        # Neither silently wins: a line left in .env from an earlier run would otherwise override
        # the file being edited, and every change to it would appear to do nothing.
        raise ValueError("BOT_SYSTEM_PROMPT and BOT_SYSTEM_PROMPT_FILE are both set; unset one")
    if inline:
        return inline
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    try:
        prompt = Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"BOT_SYSTEM_PROMPT_FILE {path!r} cannot be read: {error}") from error
    if not prompt:
        raise ValueError(f"BOT_SYSTEM_PROMPT_FILE {path!r} is empty")
    return prompt


SPEECH_TO_SPEECH_PROVIDERS = ("gemini-live",)
"""Providers whose one model hears the caller, answers and speaks, so they fill all three roles.

Audio goes to the model and audio comes back: there is no transcript to decide a turn on and no text
to synthesize, and the model detects the caller's speech and interruptions itself. A recognizer or a
voice from another provider beside one would never be in the call, so a selection that splits one
off is refused at startup; see `refuse_a_split_speech_to_speech`."""

LLM_PROVIDERS = ("anthropic", "google", "openai", "local", *SPEECH_TO_SPEECH_PROVIDERS)
"""Who answers a turn."""

STT_PROVIDERS = ("deepgram", "google", "openai", "local", *SPEECH_TO_SPEECH_PROVIDERS)
"""Who transcribes the caller."""

TTS_PROVIDERS = ("cartesia", "google", "openai", "local", *SPEECH_TO_SPEECH_PROVIDERS)
"""Who speaks for the bot."""

BACKENDS = ("cloud", "local", *SPEECH_TO_SPEECH_PROVIDERS)
"""The presets. A backend is a shorthand for one provider in each of the three roles."""

BACKEND_PRESETS: dict[str, tuple[str, str, str]] = {
    "cloud": ("anthropic", "deepgram", "cartesia"),
    "local": ("local", "local", "local"),
    **{provider: (provider, provider, provider) for provider in SPEECH_TO_SPEECH_PROVIDERS},
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
"""Who answers a turn: `anthropic`, `google`, `openai`, `local` for a server on this host, or
`gemini-live` for Gemini's native audio."""

_ROLE_PRESET = (
    BACKEND_PRESETS[LLM_PROVIDER] if LLM_PROVIDER in SPEECH_TO_SPEECH_PROVIDERS else _PRESET
)
"""The other two roles follow a speech-to-speech model, so `BOT_LLM_PROVIDER=gemini-live` alone
selects it whole rather than leaving the backend's recognizer and voice beside it."""

STT_PROVIDER = _selected_provider("BOT_STT_PROVIDER", _ROLE_PRESET[1])
"""Who transcribes the caller: `deepgram`, `google`, `openai`, or `local` for faster-whisper."""

TTS_PROVIDER = _selected_provider("BOT_TTS_PROVIDER", _ROLE_PRESET[2])
"""Who speaks for the bot: `cartesia`, `google`, `openai`, or `local` for Kokoro."""


def speech_to_speech() -> bool:
    """Whether the selected model hears and speaks itself, read when a call is built."""
    return LLM_PROVIDER in SPEECH_TO_SPEECH_PROVIDERS


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
    # The Gemini API's name; on Vertex the same model has another. pipecat's own default moved
    # from a 2.5 native-audio preview to this between 1.12 and 1.13, which is why it is pinned.
    "gemini-live": "models/gemini-3.8-live",
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
    "gemini-live": True,
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
    # Empty: the model hears the caller itself.
    "gemini-live": "",
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
    # What the model is told it will hear and speak, when `BOT_TTS_LANGUAGE` names nothing.
    "gemini-live": "en-US",
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
    # Empty: the voice is the model's own.
    "gemini-live": "",
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
    # One of Gemini's prebuilt voices, which speak every language the model does.
    "gemini-live": "Charon",
}
"""Which voice each synthesizer is asked for when `BOT_TTS_VOICE` says nothing."""

TTS_VOICE = os.environ.get("BOT_TTS_VOICE", "").strip() or DEFAULT_TTS_VOICE.get(TTS_PROVIDER, "")
"""The voice to speak with. Empty only for Cartesia, which reads `CARTESIA_VOICE_ID` instead."""

FOLLOW_CALLER_CONSISTENT_TURNS = 2
"""How many stretches of caller speech in a row must be heard in another configured language
before `BOT_TTS_FOLLOW_CALLER` moves the voice to it.

One is not enough: a recognizer on `multi` reports a language per transcript, and a caller who says
"okay" or a name in the middle of Dutch is reported as English for that one stretch. Answering that
in an English voice and then flipping back is worse than the accent. Two keeps a single misheard
stretch from flipping the voice and still follows a caller who has really changed language by
their second sentence."""

KOKORO_VOICE_PREFIXES: dict[str, str] = {
    "en": "ab",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh": "z",
    "cmn": "z",
}
"""The languages the local voice speaks, by base code, and the first letter of its voices in each.

Kokoro v1.0 is trained on these languages only, and names every voice by its language: `af_heart` is
American English, `ef_dora` Spanish, `zf_xiaobei` Mandarin. There is no Dutch, and nothing pipecat
can do about that: its Kokoro service would fall back to espeak-ng's Dutch phonemes read by an
English voice, which is the accent this setting exists to remove. So a language outside this table,
or a voice of the wrong letter for one inside it, stops the bot at startup instead."""


def base_language(code: str) -> str:
    """Drop the region and the case from a language code: `nl` for `nl-BE`, `en` for `en_US`."""
    return code.replace("_", "-").split("-")[0].lower()


def as_pipecat_language(code: str) -> Language | str:
    """Pipecat's enum member for a BCP-47 code, or the code itself for a provider-specific one.

    Each service converts the enum to its own spelling (`nl-NL` for Google, `nl` for Cartesia), so
    the enum is what lets one setting reach four providers. A code the enum does not know is passed
    through as written, which pipecat sends to the provider unchanged.
    """
    try:
        return Language(code)
    except ValueError:
        return code


@dataclass(frozen=True)
class SpeakingLanguages:
    """What the bot may speak on a call, read from the environment by `resolve_speaking_languages`.

    `starting` is None when nothing is configured, and then nothing below reaches a service: each
    synthesizer keeps its own default, which is how the example behaved before this existed.
    """

    starting: str | None
    available: tuple[str, ...]
    voices: Mapping[str, str]
    follow_caller: bool

    @property
    def switchable(self) -> bool:
        """Whether there is another language to switch to, which is what the tool is for."""
        return len(self.available) > 1

    def configured_language_for(self, reported: str) -> str | None:
        """Find the configured language a recognizer report means, by exact code, then base code.

        A recognizer reports `nl-BE` for a caller the deployment configured as `nl`; that caller
        speaks the configured language. None when the report is no configured language at all.
        """
        normalised = reported.replace("_", "-").lower()
        for language in self.available:
            if language.replace("_", "-").lower() == normalised:
                return language
        for language in self.available:
            if base_language(language) == base_language(reported):
                return language
        return None

    def voice_for(self, language: str) -> str | None:
        """Name the voice `BOT_TTS_VOICES` gives `language`, or None when it gives none."""
        return self.voices.get(language)


def resolve_speaking_languages(
    environment: Mapping[str, str], provider: str, default_voice: str
) -> SpeakingLanguages:
    """Read what the bot speaks: `BOT_TTS_LANGUAGE`, `BOT_TTS_LANGUAGES`, `BOT_TTS_VOICES`.

    `BOT_TTS_LANGUAGES` is every language a call may switch to (`en,nl`), `BOT_TTS_LANGUAGE` the one
    it starts in, defaulting to the first of the list. `BOT_TTS_VOICES` gives a language its own
    voice (`nl=<voice>,en=<voice>`), because a voice speaks its own language natively and every
    other one with its accent; a language it names nothing for keeps `default_voice`.
    `BOT_TTS_FOLLOW_CALLER=true` lets the voice follow the language the recognizer hears.

    Checked in `main`, so a combination that cannot work stops the bot at startup, naming the
    variable, rather than at the first call, where it would be heard as an accent and not as an
    error.

    Raises:
        ValueError: For a malformed value, a starting language or a voice for a language the list
            does not hold, following the caller with nothing to follow, or a language the local
            voice cannot speak natively.

    """
    starting = environment.get("BOT_TTS_LANGUAGE", "").strip()
    listed = tuple(
        dict.fromkeys(
            language.strip()
            for language in environment.get("BOT_TTS_LANGUAGES", "").split(",")
            if language.strip()
        )
    )
    if starting and listed and starting not in listed:
        raise ValueError(
            f"BOT_TTS_LANGUAGE {starting!r} is not one of BOT_TTS_LANGUAGES ({', '.join(listed)})"
        )
    available = listed or ((starting,) if starting else ())

    voices: dict[str, str] = {}
    for entry in environment.get("BOT_TTS_VOICES", "").split(","):
        if not entry.strip():
            continue
        language, separator, voice = (part.strip() for part in entry.partition("="))
        if not separator or not language or not voice or language in voices:
            raise ValueError(
                f"BOT_TTS_VOICES entry {entry.strip()!r} is not language=voice, once per language "
                "(for example nl=<voice id>,en=<voice id>)"
            )
        if language not in available:
            raise ValueError(
                f"BOT_TTS_VOICES names a voice for {language!r}, which is not a configured "
                f"language ({', '.join(available) or 'none: set BOT_TTS_LANGUAGES'})"
            )
        voices[language] = voice

    follow_caller = environment.get("BOT_TTS_FOLLOW_CALLER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if follow_caller and len(available) < 2:
        raise ValueError(
            "BOT_TTS_FOLLOW_CALLER needs BOT_TTS_LANGUAGES to list more than one language to follow"
        )

    languages = SpeakingLanguages(
        starting=starting or (available[0] if available else None),
        available=available,
        voices=voices,
        follow_caller=follow_caller,
    )
    if provider == "local":
        _refuse_what_the_local_voice_cannot_speak(languages, default_voice)
    if provider in SPEECH_TO_SPEECH_PROVIDERS and (languages.switchable or languages.voices):
        # The language is told to the model when its session opens and the voice is its own,
        # so there is nothing a mid-call switch could send. Asked in the prompt, it still answers
        # a caller in their language; a list here would only promise a tool that does nothing.
        raise ValueError(
            f"{provider} speaks one configured language per call: set BOT_TTS_LANGUAGE alone, "
            "without BOT_TTS_LANGUAGES or BOT_TTS_VOICES, and BOT_TTS_VOICE to one of its voices"
        )
    return languages


def _refuse_what_the_local_voice_cannot_speak(
    languages: SpeakingLanguages, default_voice: str
) -> None:
    """Stop a language Kokoro has no voice for, and a Kokoro voice of another language.

    Raises:
        ValueError: Naming the language, what the local voice can speak, and what to change.

    """
    for language in languages.available:
        prefixes = KOKORO_VOICE_PREFIXES.get(base_language(language))
        if prefixes is None:
            raise ValueError(
                f"the local voice (Kokoro) cannot speak {language!r}: it speaks only "
                f"{', '.join(sorted(KOKORO_VOICE_PREFIXES))}, and would read {language!r} in an "
                "English accent. Choose a hosted voice for it with BOT_TTS_PROVIDER "
                "(cartesia, google or openai), or remove it from BOT_TTS_LANGUAGES."
            )
        voice = languages.voice_for(language) or default_voice
        if not voice or voice[0] not in prefixes:
            raise ValueError(
                f"the local voice {voice!r} is not a {language!r} voice, and would speak "
                f"{language!r} with its own accent: name one in BOT_TTS_VOICES, as "
                f"{language}=<a Kokoro voice starting with {' or '.join(prefixes)}>"
            )


GOOGLE_VOICE_LOCALE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-")
"""The locale a Google voice name starts with, as in `en-US-Chirp3-HD-Charon`."""


def speaking_voice(
    provider: str, language: str, languages: SpeakingLanguages, default_voice: str
) -> str:
    """Choose the voice that speaks `language`: the one the map names, or the configured voice.

    Google is the one provider whose voice names carry a locale, and whose API refuses a voice of
    one locale asked to speak another. Without a voice in the map, its voice is the configured one
    moved to the language's locale: the Chirp3 voices exist under the same name in every locale
    they serve, so `en-US-Chirp3-HD-Charon` speaking Dutch is `nl-NL-Chirp3-HD-Charon`.
    """
    voice = languages.voice_for(language)
    if voice:
        return voice
    if provider == "google" and GOOGLE_VOICE_LOCALE.match(default_voice):
        from pipecat.services.google.tts import language_to_google_tts_language

        pipecat_language = as_pipecat_language(language)
        locale = (
            language_to_google_tts_language(pipecat_language)
            if isinstance(pipecat_language, Language)
            else language
        )
        return GOOGLE_VOICE_LOCALE.sub(f"{locale}-", default_voice, count=1)
    return default_voice


OPENAI_ACCENT_INSTRUCTIONS = (
    "Speak in the language whose BCP-47 code is {language}, with the accent of a native speaker."
)
"""What OpenAI's voice is told when a language is set. Its speech endpoint takes no language at all
and reads every text in its voice's own accent, so the instructions, which only its `gpt-` speech
models accept, are the one place a language can be asked for."""


def speaking_settings(
    provider: str, language: str | None, languages: SpeakingLanguages, default_voice: str
) -> dict[str, Any]:
    """Gather the settings that make `provider` speak `language`. Empty when it is None.

    Used twice: as constructor settings for the starting language, and as the delta a switch sends
    mid-call, so the two cannot disagree about what speaking a language means for a provider.
    """
    if language is None:
        return {}
    settings: dict[str, Any] = {
        "language": as_pipecat_language(language),
        "voice": speaking_voice(provider, language, languages, default_voice),
    }
    if provider == "openai" and TTS_MODEL.startswith("gpt-"):
        settings["instructions"] = OPENAI_ACCENT_INSTRUCTIONS.format(language=language)
    return settings


def default_tts_voice() -> str:
    """Name the voice the selected synthesizer speaks when the voice map names none."""
    if TTS_PROVIDER == "cartesia":
        return TTS_VOICE or os.environ.get("CARTESIA_VOICE_ID", "")
    return TTS_VOICE


def configured_speaking_languages() -> SpeakingLanguages:
    """Read the speaking languages for the selected synthesizer, as the environment says now."""
    return resolve_speaking_languages(os.environ, TTS_PROVIDER, default_tts_voice())


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

TRANSFER_VERDICT_TIMEOUT_SECONDS = float(os.environ.get("BOT_TRANSFER_VERDICT_TIMEOUT", "") or 60)
"""How long the bot keeps a caller it has asked to transfer before it takes the transfer as failed.

`transfer()` resolves when siphon has sent the REFER. Whether it happened arrives later, on the
REFER's implicit subscription (RFC 3515 §2.4.4), and RFC 3515 gives that subscription no duration of
its own: the phone picks it. siphon's docs name no deadline for it either, so a phone that answers
`202` and never sends the terminating NOTIFY would keep the bot waiting for the rest of the call.

Sixty seconds is siphon's own default for a transfer to report (`accept_refer` in `controller` mode,
after which siphon tells the referrer `503`), and it is well past 64*T1 = 32 s, the most a REFER can
go unanswered (RFC 3261 Timer F), so a phone that refuses or never answers the REFER is always
heard from first. A refusal itself takes a round trip: a `405` from a phone arrives in milliseconds,
and only a phone that accepted and is ringing the destination uses this much of it."""

TRANSFER_MODES = ("refer", "bridge")
TRANSFER_MODE = os.environ.get("BOT_TRANSFER_MODE", "").strip().lower() or "refer"
"""How the transfer_call tool moves the caller: `refer` or `bridge`.

`refer` asks the caller's own phone to place the new call (an in-dialog REFER, RFC 3515), which
works only where the caller's phone carries a REFER out. Many accept one with a `202` and then do
nothing, so the transfer never happens.

`bridge` leaves the caller's phone out of it. siphon rings the destination itself (the control
plane's `dial` with `on_answer: "bridge"`) while the caller stays on this call, and when the
destination answers it joins the two and drops this bot's media stream. A destination that refuses
or does not answer within BOT_TRANSFER_RING_TIMEOUT leaves the caller with the bot, as a refused
REFER does. Where the new leg goes and whom it presents are the call's own vars, set by the
routing script that handed the call over: `transfer_next_hop` (the proxy the leg is sent to, the
request URI staying the destination), `transfer_from`, `transfer_p_asserted_identity` and
`transfer_privacy`. Without them the leg goes to the destination's own host and presents the
caller's From."""

TRANSFER_RING_TIMEOUT_SECONDS = int(os.environ.get("BOT_TRANSFER_RING_TIMEOUT", "") or 30)
"""How long the destination rings in `bridge` mode before the transfer counts as not answered.

Keep it below BOT_TRANSFER_VERDICT_TIMEOUT, so the dial's own verdict arrives before the bot gives
up waiting for one."""

BRIDGED_STREAM_GRACE_SECONDS = 10.0
"""How long the bot waits, after a `bridge` transfer, for siphon to close its media stream.

siphon closes it when it joins the caller to the destination. Closing it from this side first
would read, to the routing script, as the bot leaving a caller it still has; so the bot waits, and
ends its pipeline itself only if the stream is somehow still open after this long."""

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


def is_switched_on(value: str) -> bool:
    """Read an on/off variable the way `BOT_TTS_FOLLOW_CALLER` is read: on only when it says so."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


CALLER_LOOKUP_URL = os.environ.get("BOT_CALLER_LOOKUP_URL", "").strip()
"""The URL the caller_lookup tool fetches, with `{caller}` where the caller's number goes. Empty
disables the tool.

A template rather than a base URL, because the service it points at is the deployment's own and so
is its path: `https://accounts.example.invalid/customers/{caller}`. No default, not even a port: a
guessed address is a tool that answers "not available" on every call while looking configured."""

CALLER_LOOKUP_PLACEHOLDER = "{caller}"
"""Where the number goes in `BOT_CALLER_LOOKUP_URL`. A URL without it is refused at startup: it
would look up the same thing for every caller, which is somebody else's details for all but one."""

DEFAULT_CALLER_LOOKUP_DESCRIPTION = (
    "Look up details about the caller's account, for questions about their own account."
)
"""What the model is told the tool does, when the deployment does not say."""

CALLER_LOOKUP_DESCRIPTION = (
    os.environ.get("BOT_CALLER_LOOKUP_DESCRIPTION", "").strip() or DEFAULT_CALLER_LOOKUP_DESCRIPTION
)
"""The tool's description, which is what tells the model when to call it. Worth setting: the model
only knows what the lookup is good for from this sentence."""

CALLER_LOOKUP_TIMEOUT_SECONDS = 1.5
"""How long the caller_lookup request may take. The caller hears silence for all of it, and a
lookup slower than this is better answered with "I cannot see that right now"."""

CALLER_LOOKUP_MAXIMUM_BYTES = 2048
"""The most of the lookup's answer, serialized, that is handed to the model. A larger answer counts
as not available rather than being cut: half a record read aloud is worse than none, and the whole
of it is resent with every turn for the rest of the call."""

CALLER_NUMBER = re.compile(r"[0-9]{6,15}")
"""What a `caller` on the media socket's URL must be: the number's digits, without a plus. ASCII
digits on purpose, since `str.isdigit` also passes other scripts' digits, and the number goes into
a URL."""


def refuse_a_caller_lookup_url_without_the_caller(url: str) -> None:
    """Stop a `BOT_CALLER_LOOKUP_URL` that has nowhere to put the caller's number.

    Raises:
        ValueError: If the URL is set and does not contain `{caller}`.

    """
    if url and CALLER_LOOKUP_PLACEHOLDER not in url:
        raise ValueError(
            f"BOT_CALLER_LOOKUP_URL must contain {CALLER_LOOKUP_PLACEHOLDER} where the caller's "
            "number goes, for example https://accounts.example.invalid/customers/{caller}"
        )


WEATHER_ENABLED = is_switched_on(os.environ.get("BOT_WEATHER", ""))
"""Whether the model gets the weather tool. Off unless `BOT_WEATHER=on`, like every other capability
here: with nothing set the bot is built as before and makes no request to anyone it was not told
about."""

WEATHER_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
"""open-meteo's place search and forecast. Neither takes a key. The forecast's default units are
degrees Celsius and km/h, which is what the tool reports."""

WEATHER_TIMEOUT_SECONDS = 3.0
"""The whole weather lookup, both requests together, is capped at this."""

WEATHER_PLACE_MAXIMUM_LENGTH = 100
"""A place name longer than this is cut, since the model, and through it the caller, chooses it."""

WEATHER_CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}
"""The WMO weather interpretation codes open-meteo reports, as a phrase that can be said aloud. A
code missing here is left out of the result rather than guessed at."""

CALLER_LOOKUP_PROMPT = """

When the caller asks about their own account, call the caller_lookup tool, which looks up the
number they are calling from. If it says the details are not available, tell them you cannot see
that right now."""
"""Appended only when the tool is configured, for the reason HANGUP_PROMPT gives."""

WEATHER_PROMPT = """

When the caller asks about the weather, call the weather tool with the place they mean. If they do
not name one and you know which country they are in, use its capital."""
"""Appended only when the tool is enabled."""

PBX_URL = os.environ.get("BOT_PBX_URL", "").strip()
"""siphon-pbx's address, when this bot answers calls a siphon-pbx flow hands it. Empty: no brief.

A flow's voice block has the engine dial the bot with `&session=<token>`; the bot then fetches the
block's brief from here before it speaks, and reports how the conversation went back here, so the
flow leaves the block on the report instead of waiting out its deadline. Set with
`SIPHON_PBX_BOT_SECRET`, the secret the bot was given in siphon-pbx, or not at all."""

PBX_SECRET_VARIABLE = "SIPHON_PBX_BOT_SECRET"
"""The environment variable holding the bot's siphon-pbx secret. Read once, at startup."""

PBX_STREAM_GRACE_SECONDS = 10.0
"""How long after a report the bot waits for the PBX to take the call on before it ends its leg.

A report the PBX took moves the call out of the block, and the engine closes the bot's stream as it
does. The bot leaves that to the PBX rather than closing first, for the reason
BRIDGED_STREAM_GRACE_SECONDS gives, and ends its pipeline only if the stream outlives this."""

SESSION_PROMPT = """

This call was handed to you with the following instructions. Where they differ from what you were
told above, follow these:
{instructions}"""
"""The block's instructions, appended per call: what this conversation is for."""

CHOOSE_OPTION_PROMPT = """

Once the caller has made clear which of these they want, call the choose_option tool with it, then
tell them in a few words that you are passing them on: the call moves on once your words have
finished playing. Only choose an option the caller asked for."""
"""Appended only when the brief has options. Tool first, words after, as HANGUP_PROMPT explains."""


CALL_BACK_URI = os.environ.get("BOT_CALL_BACK_URI", "").strip()
"""Where a callback rings, as a SIP URI with `{number}` in it. Empty: no call_back tool.

`{number}` is the caller's number as the network asserted it (P-Asserted-Identity), `+` and all:
`sip:{number}@trunk.example.invalid;user=phone`. Never a number the model names, nor the media
socket's `caller` parameter, which whoever dials the bot can set: a callback costs money, and a
model that could choose where it rings could be talked into ringing anyone."""

CALL_BACK_PLACEHOLDER = "{number}"

CALL_BACK_MAXIMUM_DELAY_MINUTES = int(os.environ.get("BOT_CALL_BACK_MAX_DELAY", "") or 60)
"""The furthest ahead, in minutes, the model may arrange a callback.

Callbacks are held by this process and are lost if it stops; each one lost is logged as an error.
A cap keeps what a restart can lose small."""

CALL_BACK_FROM = os.environ.get("BOT_CALL_BACK_FROM", "").strip() or None
"""The From URI a callback presents. Unset: siphon's default for an originate."""

CALL_BACK_RING_TIMEOUT_SECONDS = 30
"""How long a callback rings before siphon gives up on it."""

CALL_BACK_REASON_MAXIMUM_LENGTH = 200
"""What the callback is about, cut to this: it becomes the placed call's opening turn."""

CALL_BACK_SECONDS_PER_MINUTE = 60.0
"""Seconds in one of the tool's minutes. A constant so the tests can shorten a minute."""

ASSERTED_NUMBER = re.compile(r"\+?[0-9]{6,15}")
"""A number a callback may ring: E.164 digits, with the `+` an assertion usually carries."""

CALL_BACK_PROMPT = """

When the caller asks to be called back, call the call_back tool with how many minutes from now and
what the call should be about, then tell them when to expect it. It rings the number they are
calling from; if it says that number is not available, tell them you cannot call them back."""
"""Appended only when the tool is configured, for the reason HANGUP_PROMPT gives."""


@dataclass(frozen=True)
class CallBackSettings:
    """Where callbacks ring, how the engine reaches the bot for them, and how far ahead they go."""

    uri_template: str
    ws_uri: str
    from_uri: str | None
    profile: str
    maximum_delay_minutes: int


def call_back_settings_from_arguments(arguments: argparse.Namespace) -> CallBackSettings | None:
    """Read the callback settings, or None when `BOT_CALL_BACK_URI` is not set.

    Raises:
        ValueError: If callbacks are configured without what placing one needs.

    """
    if not CALL_BACK_URI:
        return None
    if CALL_BACK_PLACEHOLDER not in CALL_BACK_URI:
        raise ValueError(
            f"BOT_CALL_BACK_URI has no {CALL_BACK_PLACEHOLDER}, so every callback would ring the "
            f"same number: put {CALL_BACK_PLACEHOLDER} where the caller's number goes"
        )
    if not arguments.control_url:
        raise ValueError(
            "BOT_CALL_BACK_URI places callbacks through the control plane: set --control-url"
        )
    if not arguments.call_ws_uri:
        raise ValueError(
            "BOT_CALL_BACK_URI needs the address the engine reaches this bot on, with its "
            "{call_id}: set --call-ws-uri or BOT_CALL_WS_URI"
        )
    if CALL_BACK_MAXIMUM_DELAY_MINUTES < 0:
        raise ValueError(
            "BOT_CALL_BACK_MAX_DELAY must be 0 or more minutes, not "
            f"{CALL_BACK_MAXIMUM_DELAY_MINUTES}"
        )
    return CallBackSettings(
        uri_template=CALL_BACK_URI,
        ws_uri=arguments.call_ws_uri,
        from_uri=CALL_BACK_FROM,
        profile=arguments.call_profile,
        maximum_delay_minutes=CALL_BACK_MAXIMUM_DELAY_MINUTES,
    )


def _identities(value: str) -> list[str]:
    """Split a P-Asserted-Identity value at the commas between identities, not those quoted."""
    identities: list[str] = []
    current: list[str] = []
    quoted = False
    bracketed = False
    for character in value:
        if character == '"' and not bracketed:
            quoted = not quoted
        elif character == "<" and not quoted:
            bracketed = True
        elif character == ">" and not quoted:
            bracketed = False
        elif character == "," and not quoted and not bracketed:
            identities.append("".join(current))
            current = []
            continue
        current.append(character)
    identities.append("".join(current))
    return identities


def _number_in(identity: str) -> str | None:
    """Return the number a `tel:` or `sip:` identity names, or None if it names none."""
    opening = identity.find("<")
    if opening != -1:
        closing = identity.find(">", opening)
        if closing == -1:
            return None
        uri = identity[opening + 1 : closing]
    else:
        uri = identity
    uri = uri.strip()
    scheme, _, rest = uri.partition(":")
    scheme = scheme.lower()
    if scheme == "tel":
        number = rest.split(";", 1)[0]
        # RFC 3966 visual separators carry nothing.
        number = re.sub(r"[-.()]", "", number)
    elif scheme in ("sip", "sips"):
        user, at, _host = rest.partition("@")
        if not at:
            return None
        number = user.split(";", 1)[0]
    else:
        return None
    return number if ASSERTED_NUMBER.fullmatch(number) else None


def asserted_number(payload: Any) -> str | None:
    """Return the caller's number as the network asserted it on a handed-over call, or None.

    Read from the `StasisStart` payload's INVITE headers (`[name, value]` pairs, as siphon's
    `build_stasis_payload` gives them): the first P-Asserted-Identity naming a number. None when
    there is none, when the caller withheld their identity (`Privacy: id`), or when the payload is
    not that shape. Never raises.
    """
    invite = payload.get("invite") if isinstance(payload, Mapping) else None
    headers = invite.get("headers") if isinstance(invite, Mapping) else None
    if not isinstance(headers, list):
        return None
    identities: list[str] = []
    for pair in headers:
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            continue
        name, value = pair
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        name = name.strip().lower()
        if name == "privacy" and "id" in (
            token.strip().lower() for token in re.split(r"[;,]", value)
        ):
            return None
        if name == "p-asserted-identity":
            identities.extend(_identities(value))
    for identity in identities:
        number = _number_in(identity)
        if number is not None:
            return number
    return None


CONFIGURATIONS_DIRECTORY = os.environ.get("BOT_CONFIGURATIONS_DIR", "").strip()
"""A directory of `<key>.yaml` files, one per configuration a call may name. Empty: none.

One bot answers for many organisations or services this way. The key comes from whoever handed the
call over, never from the media socket's URL: siphon-pbx's brief names it as the organisation's
`external_ref`, and a siphon-sip routing script as the `bot_configuration` handover var. A call that
names none, or one with no file, or a file that is wrong, gets the bot's own configuration, the
last logged as an error. Files are read per call, so a change needs no restart."""

CONFIGURATION_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")
"""A key that names a file in the directory and nothing outside it."""

CONFIGURATION_FIELDS = (
    "system_prompt",
    "system_prompt_file",
    "transfer_target",
    "caller_lookup_url",
    "caller_lookup_description",
)
"""What a configuration file may set. Anything else in one is refused, not ignored."""

HANDOVER_CONFIGURATION_VARIABLE = "bot_configuration"
"""The handover var a siphon-sip routing script names a call's configuration with."""

HANDOVER_WAIT_SECONDS = 3.0
"""How long a dial-in waits for its call's handover to name a configuration.

siphon-sip dials the bot's stream before it hands the call to the control plane (the handover is
carried out once the routing script returns), so the bot accepts first and then waits, briefly: a
call that is never handed over is served with the bot's own configuration after this."""


@dataclass(frozen=True)
class CallConfiguration:
    """What one call is served with: the bot's own settings, or a configuration file's over them."""

    key: str | None
    system_prompt: str
    transfer_target: str
    caller_lookup_url: str
    caller_lookup_description: str


def default_configuration() -> CallConfiguration:
    """Return the bot's own configuration, from its environment."""
    return CallConfiguration(
        key=None,
        system_prompt=resolve_system_prompt(os.environ),
        transfer_target=TRANSFER_TARGET,
        caller_lookup_url=CALLER_LOOKUP_URL,
        caller_lookup_description=CALLER_LOOKUP_DESCRIPTION,
    )


def _read_configuration(directory: Path, key: str) -> CallConfiguration:
    """Read `<key>.yaml` over the bot's own configuration.

    Raises:
        ValueError: Naming what is wrong with the file.

    """
    path = directory / f"{key}.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{path.name} cannot be read: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"{path.name} is not YAML: {error}") from error
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} is not a mapping of settings")
    unknown = sorted(str(name) for name in document if name not in CONFIGURATION_FIELDS)
    if unknown:
        raise ValueError(f"{path.name} sets what a configuration cannot: {', '.join(unknown)}")
    for name, value in document.items():
        if not isinstance(value, str):
            raise ValueError(f"{path.name}: {name} is not text")
    if "system_prompt" in document and "system_prompt_file" in document:
        raise ValueError(f"{path.name} sets both system_prompt and system_prompt_file")
    configuration = replace(default_configuration(), key=key)
    if "system_prompt_file" in document:
        prompt_path = (directory / document["system_prompt_file"]).resolve()
        if prompt_path.parent != directory.resolve():
            raise ValueError(f"{path.name}: system_prompt_file must be a file beside it")
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"{path.name}: system_prompt_file cannot be read: {error}") from error
        configuration = replace(configuration, system_prompt=prompt)
    if "system_prompt" in document:
        configuration = replace(configuration, system_prompt=document["system_prompt"])
    if not configuration.system_prompt.strip():
        raise ValueError(f"{path.name}: the system prompt is empty")
    configuration = replace(configuration, system_prompt=configuration.system_prompt.strip())
    if "transfer_target" in document:
        configuration = replace(configuration, transfer_target=document["transfer_target"].strip())
    if "caller_lookup_url" in document:
        url = document["caller_lookup_url"].strip()
        try:
            refuse_a_caller_lookup_url_without_the_caller(url)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        configuration = replace(configuration, caller_lookup_url=url)
    if "caller_lookup_description" in document:
        description = document["caller_lookup_description"].strip()
        configuration = replace(
            configuration,
            caller_lookup_description=description or DEFAULT_CALLER_LOOKUP_DESCRIPTION,
        )
    return configuration


def load_configuration(directory: str, key: str | None) -> CallConfiguration:
    """Return the configuration `key` names in `directory`, or the bot's own. Never raises."""
    if not directory or key is None:
        return default_configuration()
    if CONFIGURATION_KEY.fullmatch(key) is None:
        logger.warning("ignoring a configuration key that is not a plain name")
        return default_configuration()
    if not (Path(directory) / f"{key}.yaml").is_file():
        logger.warning(f"no configuration {key}; serving with the bot's own")
        return default_configuration()
    try:
        configuration = _read_configuration(Path(directory), key)
    except ValueError as error:
        logger.error(f"configuration {key} not used, serving with the bot's own: {error}")
        return default_configuration()
    logger.info(f"serving with configuration {key}")
    return configuration


def check_configurations(directory: str) -> None:
    """Refuse to start with a configuration that every call naming it would fall back from.

    Raises:
        ValueError: Naming the directory if it is not one, or every file that is wrong.

    """
    if not directory:
        return
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"BOT_CONFIGURATIONS_DIR {directory} is not a directory")
    problems: list[str] = []
    for file in sorted(path.glob("*.yaml")):
        if CONFIGURATION_KEY.fullmatch(file.stem) is None:
            problems.append(f"{file.name}: not a name a call can give")
            continue
        try:
            _read_configuration(path, file.stem)
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("BOT_CONFIGURATIONS_DIR: " + "; ".join(problems))


@dataclass(frozen=True)
class CallFacts:
    """Whose call it is, from whoever handed it over. Every field None where nobody said."""

    configuration_key: str | None = None
    # Digits only, as `caller_lookup` takes them; never from the media socket's URL.
    caller: str | None = None
    organisation: ContextOrganisation | None = None


def _lookup_digits(number: str | None) -> str | None:
    """Return a number as the caller lookup takes it, or None if it is not one."""
    if number is None:
        return None
    digits = number.removeprefix("+")
    return digits if CALLER_NUMBER.fullmatch(digits) else None


def call_facts_from_context(context: CallContext | None) -> CallFacts:
    """Read a siphon-pbx brief's `context`: the organisation's reference names the configuration."""
    if context is None:
        return CallFacts()
    caller = context.caller
    return CallFacts(
        configuration_key=context.organisation.external_ref,
        caller=_lookup_digits(caller.number) if caller is not None else None,
        organisation=context.organisation,
    )


def call_facts_from_handover(payload: Any) -> CallFacts:
    """Read a siphon-sip handover: its `bot_configuration` var, and the asserted caller."""
    if not isinstance(payload, Mapping):
        return CallFacts()
    variables = payload.get("vars")
    key = variables.get(HANDOVER_CONFIGURATION_VARIABLE) if isinstance(variables, Mapping) else None
    return CallFacts(
        configuration_key=key if isinstance(key, str) else None,
        caller=_lookup_digits(asserted_number(payload)),
    )


def organisation_prompt(facts: CallFacts, now: datetime | None = None) -> str:
    """Tell the model whose line it is answering, and the time there. Empty when nobody said.

    Only what the operator set: the organisation's name and time zone. Nothing the caller controls,
    a display name least of all, goes into the system prompt.
    """
    organisation = facts.organisation
    if organisation is None:
        return ""
    prompt = f"\n\nThis call is for {organisation.name}."
    try:
        zone = ZoneInfo(organisation.time_zone)
    except (ZoneInfoNotFoundError, ValueError):
        return prompt
    local = (now or datetime.now(zone)).astimezone(zone)
    return prompt + f" It is {local:%A %d %B %Y, %H:%M} there ({organisation.time_zone})."


async def resolve_call(
    control: ControlPlane | None, query: Mapping[str, str], session: VoiceSession | None
) -> tuple[CallConfiguration, CallFacts]:
    """Find whose call this is and the configuration it names, before its pipeline is built.

    A briefed call has it in its brief. A handed-over one has it in the handover, which arrives
    after the dial-in, so that is waited for, `HANDOVER_WAIT_SECONDS` at most, and only when there
    is a configuration directory to choose from.
    """
    if session is not None:
        facts = call_facts_from_context(session.brief.context)
    elif control is not None and CONFIGURATIONS_DIRECTORY and query.get("call"):
        handed = await control.handed_over(query["call"], HANDOVER_WAIT_SECONDS)
        if handed is None:
            logger.warning(
                f"no handover for this call within {HANDOVER_WAIT_SECONDS:g}s; serving with the "
                "bot's own configuration"
            )
            facts = CallFacts()
        else:
            facts = call_facts_from_handover(getattr(handed, "payload", None))
    else:
        facts = CallFacts()
    return load_configuration(CONFIGURATIONS_DIRECTORY, facts.configuration_key), facts


def build_voice_session_client(url: str, secret: str) -> VoiceSessionClient | None:
    """Build the client for siphon-pbx's voice sessions, or None when neither half is configured.

    Raises:
        ValueError: If only one of the address and the secret is set, or the address is not http(s).

    """
    if not url and not secret:
        return None
    if not url or not secret:
        raise ValueError(f"BOT_PBX_URL and {PBX_SECRET_VARIABLE} go together: set both, or neither")
    try:
        return VoiceSessionClient(url, secret)
    except ValueError as error:
        raise ValueError(f"BOT_PBX_URL: {error}") from error


async def open_voice_session(
    client: VoiceSessionClient | None, query: Mapping[str, str]
) -> VoiceSession | None:
    """Fetch the brief for the PBX session the dial-in URL names, or None if there is none to use.

    A call that comes without a session, or whose brief cannot be had, is still answered, with the
    bot's own configuration: a caller on a line the bot will not speak on hears nothing at all.
    """
    if SESSION_QUERY_PARAMETER not in query:
        return None
    if client is None:
        logger.warning(
            "the engine dialled in for a siphon-pbx session but BOT_PBX_URL is not set: answering "
            "without the brief, and the flow will wait out its deadline for a report"
        )
        return None
    token = session_token_from(query)
    if token is None:
        logger.warning("ignoring a session parameter that is not a plain token")
        return None
    brief = await client.fetch_brief(token)
    if brief is None:
        logger.error(
            "no brief for this siphon-pbx session: answering with the bot's own configuration, "
            "and nothing can be reported"
        )
        return None
    logger.info(
        f"briefed for call {brief.call_id}: "
        + (
            f"options {', '.join(option.option for option in brief.options)}"
            if brief.options
            else "an open conversation"
        )
        + f", the flow waits {brief.deadline_secs}s for a report"
    )
    return VoiceSession(client, token, brief)


def speaking_language_prompt(languages: SpeakingLanguages) -> str:
    """Tell the model which language it speaks in. Empty when none is configured.

    The voice and the words have to agree. A voice set to Dutch reading an English reply is the
    accent problem the other way round, so the model is told which language the voice is in, and,
    when it can switch, that the switch is a tool call: the voice changes with the tool, not with
    the words, and a model that just starts answering in another language is heard in the old voice.

    Everything the bot says is the model's own reply, the greeting included, so this is the only
    text that needs to carry the language. The instructions themselves stay in English, which every
    model here follows whatever language it is asked to answer in.
    """
    if languages.starting is None:
        return ""
    prompt = (
        "\n\nSpeak the language whose BCP-47 code is "
        f"{languages.starting} on this call, starting with your greeting."
    )
    if not languages.switchable:
        return prompt
    prompt += (
        f" You can also speak {', '.join(languages.available)} (BCP-47 codes). When the caller asks"
        " you to speak one of them, call the set_speaking_language tool with it straight away and"
        " answer in that language from then on: your voice changes with the tool, so do not change"
        " language without calling it. If they ask for a language that is not in that list, tell"
        " them in the language you are speaking that you cannot."
    )
    if languages.follow_caller:
        prompt += (
            " If the caller simply starts speaking one of those languages, answer in theirs;"
            " your voice follows them on its own."
        )
    return prompt


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


BRIDGE_FINAL_EVENTS = frozenset({"ChannelBridged", "DialFailed"})
"""The events that end a `bridge` transfer: the caller joined to the destination, or nobody was."""

BRIDGE_FAILURE_STAGES: dict[str, str] = {
    "rejected": "refused",
    "timeout": "timeout",
    "ring timeout": "timeout",
    "unsent": "not_sent",
    "bridge_failed": "refused",
    "caller_hangup": "call_ended",
}
"""A `DialFailed` cause, as the stage of a REFER's verdict that means the same to the caller."""


@dataclass(frozen=True)
class TransferVerdict:
    """What became of a transfer the bot asked for, read the way the bot has to act on it.

    `ending` is the decision, and there are three because there are three things to do. A
    `completed` transfer and an `ended` call both leave the bot with nobody on its leg, so it ends.
    A `failed` one leaves the caller with the bot, which has to carry on with them.

    `ended` is the one to read twice. `TransferFailed` with stage `call_ended` means siphon tore the
    call down while the transfer was outstanding, and a phone does that on purpose when it BYEs its
    old leg as soon as it has sent the INVITE to the destination, which TS 24.629 allows. That new
    call may well connect. Either way the caller has left *this* call, so there is nobody to tell
    that the transfer did not work, and nothing to recover. The refusals (`rejected`, `refused`,
    `unauthorized`, `no_outcome`) arrive on a call that is still up, which is what tells them apart.
    """

    ending: Literal["completed", "failed", "ended"]
    stage: str | None = None
    status: int | None = None
    reason: str | None = None

    @classmethod
    def from_event(cls, kind: str, outcome: Mapping[str, Any]) -> TransferVerdict:
        """Read a final transfer event's kind and payload into the decision it calls for."""
        stage = outcome.get("stage")
        # The wire calls it `code`; the SDK's docstring calls it `status`. Either is the SIP status
        # the verdict rests on, and neither decides the verdict: the kind does.
        status = outcome.get("code", outcome.get("status"))
        ending: Literal["completed", "failed", "ended"]
        if kind == "TransferCompleted":
            ending = "completed"
        elif stage == "call_ended":
            ending = "ended"
        else:
            ending = "failed"
        return cls(ending=ending, stage=stage, status=status, reason=outcome.get("reason"))

    @classmethod
    def from_bridge_event(cls, kind: str, outcome: Mapping[str, Any]) -> TransferVerdict:
        """Read the end of a `bridge` transfer's dial into the same decision.

        `ChannelBridged` is the caller joined to the destination. `DialFailed` names its `cause`;
        the ones that mean the same as a refused REFER are given that verdict's stage, so the model
        is told the same thing for the same outcome. A caller who hung up while the destination
        rang has left the call, as `call_ended` has.
        """
        if kind == "ChannelBridged":
            return cls(ending="completed", stage="bridged")
        cause = str(outcome.get("cause") or "")
        stage = BRIDGE_FAILURE_STAGES.get(cause, cause or None)
        return cls(
            ending="ended" if stage == "call_ended" else "failed",
            stage=stage,
            status=outcome.get("code"),
            reason=outcome.get("reason"),
        )

    @property
    def leaves_the_call(self) -> bool:
        """Whether the caller is gone from the bot's leg, so the bot should end it."""
        return self.ending != "failed"


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
        self._transfer_verdicts: dict[str, asyncio.Future[TransferVerdict]] = {}
        # The calls this bot places, until the engine dials in for each. One registry per bot,
        # because the dial-in arrives on the media server, not on the control connection.
        self.placed_calls = PlacedCalls()
        self._client: Any | None = None
        self._connected = asyncio.Event()
        self._forgetting: set[asyncio.Task[None]] = set()
        # Whoever waits for a call to be over: a callback, which must not ring during the call.
        self._endings: dict[str, list[asyncio.Future[None]]] = {}
        # Whoever waits for a call to be handed over: a dial-in that came first.
        self._arrivals: dict[str, list[asyncio.Future[Any]]] = {}
        # Set when callbacks are configured; the call_back tool exists only then.
        self.call_backs: CallBacks | None = None

    def attach_client(self, client: Any) -> None:
        """Make `client` the connection calls are placed on: the live one, once it is up."""
        self._client = client
        self._connected.set()

    def detach_client(self) -> None:
        """Forget the connection: it has dropped, and the next one is not up yet."""
        self._client = None
        self._connected.clear()

    def expect_transfer_verdict(self, sip_call_id: str) -> asyncio.Future[TransferVerdict]:
        """Return what will hold the verdict on the next transfer of this call.

        Asked for *before* the REFER goes out. A phone that refuses one answers within a round trip,
        and a verdict that lands before anyone listens for it would be lost. An earlier transfer's
        waiter is cancelled: only the latest REFER on a call is still somebody's question.
        """
        previous = self._transfer_verdicts.pop(sip_call_id, None)
        if previous is not None:
            previous.cancel()
        verdict: asyncio.Future[TransferVerdict] = asyncio.get_running_loop().create_future()
        self._transfer_verdicts[sip_call_id] = verdict
        return verdict

    def _deliver_transfer_verdict(self, sip_call_id: str, verdict: TransferVerdict) -> None:
        """Hand a verdict to whoever is waiting for one on this call, if anyone is."""
        waiting = self._transfer_verdicts.pop(sip_call_id, None)
        if waiting is not None and not waiting.done():
            waiting.set_result(verdict)

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
        for arrival in self._arrivals.pop(sip_call_id, []):
            if not arrival.done():
                arrival.set_result(call)
        logger.info(f"control channel attached to call {sip_call_id}")
        try:
            # Draining events is what keeps the handle alive for the call's life; it is also where
            # a transfer's verdict lands (see `_watch_event`).
            while (event := await call.next_event()) is not None:
                self._watch_event(sip_call_id, event)
        finally:
            self._forget(sip_call_id)

    def _watch_event(self, sip_call_id: str | None, event: Mapping[str, Any]) -> None:
        """Read one control event on a call for a transfer's verdict.

        `transfer()` resolves as soon as siphon has *sent* the REFER -- RFC 3515 §2.4.4 delivers
        the outcome afterwards, on the implicit subscription. The tool waits on this verdict before
        it lets go of the caller, because a phone that refuses the REFER is still on the line with
        the bot.
        """
        kind: Any = event.get("kind")
        if sip_call_id is None:
            return
        if kind in BRIDGE_FINAL_EVENTS:
            # A `bridge` transfer's dial reports on the caller's own channel; the bot dials for
            # nothing else there (a call it places has a channel of its own), so these are always
            # the transfer's.
            verdict = TransferVerdict.from_bridge_event(kind, event.get("payload") or {})
        elif self._transfer_final(kind):
            verdict = TransferVerdict.from_event(kind, self._transfer_outcome(event) or {})
        else:
            return
        detail = f"stage={verdict.stage} status={verdict.status} reason={verdict.reason}"
        # The kind is the verdict. The status is the NOTIFY's code when siphon has one, and a
        # completed transfer has arrived without it: read as the verdict, the missing code logged a
        # caller who was already hearing the destination as a failed transfer. The two failed lines
        # share their wording up to the `--`, which the conversation harness parses: `call_ended`
        # is still not a completion it can count.
        if verdict.ending == "completed":
            logger.info(f"transfer completed on call {sip_call_id}: {detail}")
        else:
            logger.warning(
                f"transfer FAILED on call {sip_call_id}: {detail} "
                + (
                    "-- the call ended with it, the caller may be on their way there"
                    if verdict.ending == "ended"
                    else "-- keeping the caller"
                )
            )
        self._deliver_transfer_verdict(sip_call_id, verdict)

    async def handed_over(self, sip_call_id: str, timeout_seconds: float) -> Any | None:
        """Return the call once it has been handed over, or None if it is not within the timeout."""
        call = self._calls.get(sip_call_id)
        if call is not None:
            return call
        arrival: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._arrivals.setdefault(sip_call_id, []).append(arrival)
        try:
            return await asyncio.wait_for(arrival, timeout_seconds)
        except TimeoutError:
            return None
        finally:
            waiting = self._arrivals.get(sip_call_id)
            if waiting is not None and arrival in waiting:
                waiting.remove(arrival)
                if not waiting:
                    del self._arrivals[sip_call_id]

    async def wait_until_over(self, sip_call_id: str) -> None:
        """Return once the call is no longer one this control plane holds."""
        if sip_call_id not in self._calls:
            return
        ending: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._endings.setdefault(sip_call_id, []).append(ending)
        await ending

    def _forget(self, sip_call_id: str) -> None:
        """Drop a call that has ended, and answer whoever still waits on its transfer or end."""
        self._calls.pop(sip_call_id, None)
        for ending in self._endings.pop(sip_call_id, []):
            if not ending.done():
                ending.set_result(None)
        # siphon flushes `call_ended` before the stream closes, so this is the fallback for a
        # stream that closed without one: the call is gone all the same.
        self._deliver_transfer_verdict(
            sip_call_id, TransferVerdict(ending="ended", stage="call_ended")
        )
        logger.info(f"control channel detached from call {sip_call_id}")

    async def place(self, request: CallRequest) -> PlacedCall:
        """Place `request`'s call, joined like a handed-over one so the tools can act on it.

        siphon sends a placed call no `StasisStart`, so it never reaches `_handle_call`; it is
        registered here instead, on the same SIP Call-ID, which the engine's `start` carries too.

        Raises:
            TimeoutError: If the control plane is not up within `CONTROL_CONNECT_TIMEOUT_SECONDS`.
            Exception: Whatever the control client raises when siphon refuses the originate.

        """
        await asyncio.wait_for(self._connected.wait(), CONTROL_CONNECT_TIMEOUT_SECONDS)
        client = self._client
        if client is None:
            raise ConnectionError("the control connection dropped before the call was placed")
        joined: dict[str, str] = {}
        placed = await self.placed_calls.place(
            client,
            to=request.to,
            ws_uri=request.ws_uri,
            from_uri=request.from_uri,
            profile=request.profile,
            ring_timeout=request.ring_timeout,
            context=request.context,
            destination_label=request.destination_label,
            # The placed call's events are read by `PlacedCall`, the stream's only reader; the
            # transfer verdicts among them still have to land here.
            on_event=lambda event: self._watch_event(joined.get("sip_call_id"), event),
        )
        sip_call_id = placed.sip_call_id
        if sip_call_id is None:
            logger.warning(
                f"placed call {placed.channel} has no SIP Call-ID, so the tools cannot reach it"
            )
            return placed
        joined["sip_call_id"] = sip_call_id
        self._calls[sip_call_id] = placed.call
        logger.info(f"control channel attached to placed call {sip_call_id}")
        forgetting = asyncio.create_task(self._forget_when_over(placed, sip_call_id))
        self._forgetting.add(forgetting)
        forgetting.add_done_callback(self._forgetting.discard)
        return placed

    async def _forget_when_over(self, placed: PlacedCall, sip_call_id: str) -> None:
        await placed.ended()
        self._forget(sip_call_id)

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
                # Connected before it is offered for placing calls, so `--call` waits for a live
                # connection rather than racing the first one.
                await client.connect()
                self.attach_client(client)
                await client.run()
            except Exception as error:  # any connect failure here is worth retrying
                self.detach_client()
                logger.warning(f"control plane unreachable ({error}); retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, CONTROL_RETRY_MAXIMUM_SECONDS)
            else:
                # `run()` returned rather than raised: the client was shut down deliberately.
                self.detach_client()
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


class TransferHold(FrameProcessor):
    """Keep the caller's words from the model while a transfer is being decided.

    Between the REFER and its verdict the bot has nothing to say: it has told the caller they are
    being put through, and their phone is busy with the REFER and the destination. A reply in that
    time talks over the transfer, and anything can start one -- a "hello?" while the destination
    rings, a cough, or pipecat's own recovery prompt for a turn with nothing recognised in it.

    So no turn starts. Muting the voice instead was the other choice and the worse one: the model
    would still answer, and its unheard reply would sit in the context as something the caller was
    told. The recognizer still gets the audio, since a streaming one can drop a connection fed
    nothing for a minute; only what it makes of the audio stops here. What the caller says during
    the hold is lost rather than queued, which is right for words said into a transfer.
    """

    HELD_FRAMES = (
        TranscriptionFrame,
        InterimTranscriptionFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    """What opens or feeds a caller turn: the engine's speech edges and the recognizer's words."""

    def __init__(self, hold_audio: bool = False) -> None:
        """Start open: a call is not on hold until a transfer puts it there.

        `hold_audio` holds the caller's audio as well, for a speech-to-speech model: there is no
        recognizer in front of it, so the audio is what it hears and would answer.
        """
        super().__init__()
        self._holding = False
        self._held_frames: tuple[type[Frame], ...] = self.HELD_FRAMES + (
            (InputAudioRawFrame,) if hold_audio else ()
        )

    @property
    def holds_audio(self) -> bool:
        """Whether the caller's audio is held too, not only what a recognizer made of it."""
        return InputAudioRawFrame in self._held_frames

    @property
    def holding(self) -> bool:
        """Whether the caller is being kept from the model right now."""
        return self._holding

    def engage(self) -> None:
        """Stop the caller's turns, from the REFER until its verdict."""
        self._holding = True

    def release(self) -> None:
        """Let the caller be heard again."""
        self._holding = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass everything except a caller's speech while the hold is on."""
        await super().process_frame(frame, direction)
        if self._holding and isinstance(frame, self._held_frames):
            return
        await self.push_frame(frame, direction)


@dataclass(frozen=True)
class CallRequest:
    """The one call `--call` places: where, from whom, about what, and how long it may ring."""

    to: str
    ws_uri: str
    from_uri: str | None
    profile: str
    ring_timeout: int
    context: Mapping[str, str]
    # What the logs call the destination, when the number is not the operator's to see in full.
    destination_label: str | None = None


def call_request_from_arguments(arguments: argparse.Namespace) -> CallRequest | None:
    """Read `--call` and its options, or None when no call was asked for.

    Raises:
        ValueError: If a call was asked for without what placing it needs.

    """
    if not arguments.call:
        return None
    if not arguments.control_url:
        raise ValueError("--call places its call through the control plane: set --control-url")
    if not arguments.call_ws_uri:
        raise ValueError(
            "--call needs the address the engine reaches this bot on, with its {call_id}: set "
            "--call-ws-uri or BOT_CALL_WS_URI (ws://<this host>:<port>/stream?call={call_id})"
        )
    if arguments.call_timeout < 1:
        raise ValueError(
            f"--call-timeout must be at least 1 second, not {arguments.call_timeout}: siphon reads "
            "0 as a call that rings forever"
        )
    context = {"reason": arguments.call_context} if arguments.call_context else {}
    return CallRequest(
        to=arguments.call,
        ws_uri=arguments.call_ws_uri,
        from_uri=arguments.call_from,
        profile=arguments.call_profile,
        ring_timeout=arguments.call_timeout,
        context=context,
    )


async def place_one_call(control: ControlPlane, request: CallRequest) -> int:
    """Place `request`'s call, see it through, and return the exit status that says how it went.

    0 for a call that was answered and then ended normally, 1 for everything else: refused,
    rejected, unanswered, or answered without the bot ever getting it.
    """
    logger.info(f"placing a call to {request.to}")
    try:
        placed = await control.place(request)
    except Exception as error:
        logger.error(f"placing the call to {request.to} failed: {error}")
        return 1
    answered = await placed.answered()
    if answered.kind != "answered":
        logger.error(f"the call to {request.to} was not answered: {answered}")
        return 1
    logger.info(f"the call to {request.to} was answered")
    ended = await placed.ended()
    if ended.kind != "ended":
        logger.error(f"the call to {request.to} ended badly: {ended}")
        return 1
    logger.info(f"the call to {request.to} is over: {ended}")
    return 0


class CallBacks:
    """The callbacks arranged and not yet placed, at most one per number.

    Each waits its minutes, then for the call it was arranged on to be over, so that 0 minutes is
    "right after this call" and never during it, and then rings through `ControlPlane.place` like
    `--call` does. They live in this process: stopping it loses them, and each lost is logged.
    """

    def __init__(self, control: ControlPlane, settings: CallBackSettings) -> None:
        """Hold callbacks for `control`, placed as `settings` says."""
        self.settings = settings
        self._control = control
        self._pending: dict[str, tuple[asyncio.Task[None], int]] = {}

    def pending(self) -> list[str]:
        """List the numbers a callback is arranged for and not yet placed."""
        return list(self._pending)

    def arrange(self, number: str, minutes: int, reason: str, after_call: str | None) -> bool:
        """Arrange a callback to `number`. False if one is already arranged for it."""
        if number in self._pending:
            return False
        task = asyncio.create_task(self._ring_back(number, minutes, reason, after_call))
        self._pending[number] = (task, minutes)
        logger.info(f"callback to {mask_number(number)} arranged, in {minutes} min")
        return True

    async def _ring_back(
        self, number: str, minutes: int, reason: str, after_call: str | None
    ) -> None:
        label = f"callback to {mask_number(number)}"
        try:
            await asyncio.sleep(minutes * CALL_BACK_SECONDS_PER_MINUTE)
            if after_call is not None:
                await self._control.wait_until_over(after_call)
            request = CallRequest(
                to=self.settings.uri_template.replace(CALL_BACK_PLACEHOLDER, number),
                ws_uri=self.settings.ws_uri,
                from_uri=self.settings.from_uri,
                profile=self.settings.profile,
                ring_timeout=CALL_BACK_RING_TIMEOUT_SECONDS,
                context={"reason": reason} if reason else {},
                destination_label=label,
            )
            logger.info(f"placing the {label}")
            try:
                await self._control.place(request)
            except Exception as error:
                # The type only: what siphon says about a refused originate can name the target.
                logger.error(f"the {label} failed: {type(error).__name__}")
        finally:
            self._pending.pop(number, None)

    def cancel_all(self) -> None:
        """Drop every callback not yet placed, logging each: the bot is stopping."""
        for number, (task, minutes) in list(self._pending.items()):
            if not task.done():
                task.cancel()
                logger.error(
                    f"callback to {mask_number(number)} arranged for {minutes} min lost: the bot "
                    "is stopping"
                )
        self._pending.clear()


async def call_back(params: FunctionCallParams) -> None:
    """Arrange a callback to the caller's asserted number, `minutes` from now, about `reason`.

    The number is never an argument, for the reason `transfer_call`'s destination is not: a number
    the model could name is one a caller could talk it into ringing.
    """
    resources = params.app_resources
    if not isinstance(resources, CallResources) or resources.control.call_backs is None:
        await params.result_callback({"error": "a callback cannot be arranged from here"})
        return
    call_backs = resources.control.call_backs
    maximum = call_backs.settings.maximum_delay_minutes
    minutes = params.arguments.get("minutes")
    if not isinstance(minutes, int) or isinstance(minutes, bool) or not 0 <= minutes <= maximum:
        await params.result_callback(
            {"error": f"minutes must be a whole number from 0 to {maximum}"}
        )
        return
    raw_reason = params.arguments.get("reason")
    reason = (
        raw_reason.strip()[:CALL_BACK_REASON_MAXIMUM_LENGTH] if isinstance(raw_reason, str) else ""
    )

    sip_call_id = resources.serializer.call_id
    control_call = resources.control.call_for(sip_call_id)
    if control_call is None:
        await params.result_callback(_not_available("this call cannot be called back"))
        return
    number = asserted_number(getattr(control_call, "payload", None))
    if number is None:
        await params.result_callback(
            _not_available("the caller's number is withheld or was not passed on")
        )
        return
    if not call_backs.arrange(number, minutes, reason, after_call=sip_call_id):
        await params.result_callback({"error": "a callback to this caller is already arranged"})
        return
    await params.result_callback({"scheduled": True, "in_minutes": minutes})


@dataclass
class CallResources:
    """What a tool handler needs to act on the call it is running inside."""

    serializer: SiphonFrameSerializer
    speech: BotSpeechMonitor
    control: ControlPlane
    # Where `transfer_call` sends the caller: the call's configuration's, the bot's own by default.
    transfer_target: str = field(default_factory=lambda: TRANSFER_TARGET)
    # Set after the worker exists, which is after this object is handed to it. A transfer is an
    # ending for this bot even though it is not an ending for the caller, and ending the pipeline is
    # how the media socket gets closed.
    worker: PipelineWorker | None = None
    # In the pipeline whenever this object is, so the tool can stop the caller's words reaching the
    # model while a transfer is being decided.
    transfer_hold: TransferHold = field(default_factory=TransferHold)
    # From the tool call until its verdict, or until the wait for one runs out. A second request in
    # that time is refused rather than sent, because a second REFER would only race the first.
    transfer_pending: bool = False
    # Watches a verdict that arrives after the wait gave up on it, in case it is a completion.
    late_transfer_verdict: asyncio.Task[None] | None = None
    # After a `bridge` transfer: ends the pipeline if siphon has not closed the stream in time.
    bridged_stream_watch: asyncio.Task[None] | None = None


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


@dataclass
class BriefedCall:
    """A call siphon-pbx handed over with a brief, and the tools that tell it how the call went.

    Each way the bot finishes reports once: `choose_option` the caller's choice, `end_call` that the
    conversation is over, `fail` that the bot could not serve it. A report the PBX took moves the
    call on, so the bot then only waits for its stream to close; one it did not take has nobody
    waiting on it, so the bot ends its leg straight away.
    """

    session: VoiceSession
    speech: BotSpeechMonitor
    # Set after the worker exists, as on `CallResources`.
    worker: PipelineWorker | None = None
    # After a report the PBX took: ends the pipeline if the stream outlives the grace.
    stream_watch: asyncio.Task[None] | None = None

    async def choose_option(self, params: FunctionCallParams) -> None:
        """Say the line passing the caller on, wait for it to play, then report the option."""
        names = [option.option for option in self.session.brief.options]
        option = params.arguments.get("option")
        if self.session.reported is not None:
            await params.result_callback({"error": "how this call goes has already been decided"})
            return
        if option not in names:
            # The schema's enum says the same, but a model is not bound by it.
            await params.result_callback(
                {"error": f"not one of the options; choose one of: {', '.join(names)}"}
            )
            return
        await params.result_callback({"chosen": option})
        if not await self.speech.wait_until_finished():
            logger.warning("the line passing the caller on did not finish within the cap")
        await self._report_and_let_go(Chose(str(option)))

    async def end_call(self, params: FunctionCallParams) -> None:
        """Say goodbye, wait for it to play, then report the conversation finished.

        Not a hangup: what happens to the call next is the flow's to decide, and it may well have
        more for the caller than this bot did.
        """
        if self.session.reported is not None:
            await params.result_callback({"error": "how this call goes has already been decided"})
            return
        await params.result_callback({"ending": True})
        if not await self.speech.wait_until_finished():
            logger.warning("farewell did not finish within the cap, reporting anyway")
        await self._report_and_let_go(Finished())

    async def fail(self, reason: str) -> None:
        """Report that the bot cannot serve the call, and end its leg now: it has nothing to say."""
        await self.session.report(Failed(reason))
        if self.worker is not None:
            await self.worker.queue_frame(EndWorkerFrame())

    def stream_closed(self) -> None:
        """Stand down the grace watch: the engine closed the stream, so there is nothing to end."""
        if self.stream_watch is not None:
            self.stream_watch.cancel()

    async def _report_and_let_go(self, report: Report) -> None:
        if self.worker is None:
            return
        worker = self.worker
        if not await self.session.report(report):
            await worker.queue_frame(EndWorkerFrame())
            return

        async def end_if_still_open() -> None:
            await asyncio.sleep(PBX_STREAM_GRACE_SECONDS)
            logger.warning(
                f"media stream still open {PBX_STREAM_GRACE_SECONDS:g}s after reporting to the "
                "PBX, ending the pipeline"
            )
            await worker.queue_frame(EndWorkerFrame())

        self.stream_watch = asyncio.create_task(end_if_still_open())


TRANSFER_FAILED_NOTE = (
    "[The transfer did not go through: {why}. The caller is still on the line with you. Tell them,"
    " in the language you have been speaking with them, and carry on helping them.]"
)
"""What the model is told when a transfer fails, so that it, not this file, says so to the caller.

Bracketed and third person for the reason KICKOFF is: it lands in the context as a user turn, which
every provider here takes mid-call, and a first-person line would be answered literally. The words
the caller hears are the model's own, in whatever language the call is in."""

TRANSFER_FAILURE_EXPLANATIONS: dict[str, str] = {
    "rejected": "the caller's phone refused to be transferred",
    "unauthorized": "the caller's phone refused to be transferred",
    "refused": "the person it was meant to reach could not be reached",
    "timeout": "no answer came back in time",
    "not_sent": "it could not be started",
}
"""Why, per verdict stage, in the words the note gives the model. Unlisted stages, `no_outcome`
among them, read as an outcome nobody confirmed."""


async def _inform_the_model_the_transfer_failed(resources: CallResources, why: str) -> None:
    """Put the failure in front of the model and have it answer, since nothing else will."""
    if resources.worker is None:
        return
    note = TRANSFER_FAILED_NOTE.format(why=why)
    await resources.worker.queue_frame(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": note}], run_llm=True)
    )


async def _end_on_a_late_completion(
    resources: CallResources, verdict: asyncio.Future[TransferVerdict]
) -> None:
    """End the bot's leg if a transfer it stopped waiting for turns out to have worked after all."""
    try:
        outcome = await verdict
    except asyncio.CancelledError:
        # A later transfer took over the question. Anything else cancelling this task is not ours
        # to swallow.
        if verdict.cancelled():
            return
        raise
    if outcome.leaves_the_call:
        await _let_go_of_the_call(resources, outcome)


async def _let_go_of_the_call(resources: CallResources, verdict: TransferVerdict) -> None:
    """End the bot's leg once the caller has left it, by transfer or by hanging up.

    After a `bridge` transfer siphon closes the media stream itself, which ends the pipeline. The
    bot waits for that rather than closing first: a stream the bot closes reads, to the routing
    script on the other side, as the bot walking away from a caller it still has (see
    BRIDGED_STREAM_GRACE_SECONDS). Every other way out ends the pipeline now.
    """
    if resources.worker is None:
        return
    if verdict.stage != "bridged":
        await resources.worker.queue_frame(EndWorkerFrame())
        return
    worker = resources.worker

    async def end_if_still_open() -> None:
        await asyncio.sleep(BRIDGED_STREAM_GRACE_SECONDS)
        logger.warning(
            f"media stream still open {BRIDGED_STREAM_GRACE_SECONDS:g}s after the caller was "
            f"bridged to {resources.transfer_target}, ending the pipeline"
        )
        await worker.queue_frame(EndWorkerFrame())

    resources.bridged_stream_watch = asyncio.create_task(end_if_still_open())


def _bridge_dial_arguments(control_call: Any, transfer_target: str) -> dict[str, Any]:
    """Build the `dial` that rings the destination for the caller and bridges the two.

    The next hop and the identity come from the call's vars, which the routing script that handed
    the call over sets; the bot names none of its own. Read from the handover's payload rather than
    with `get_var`, so the transfer costs no round trip before the dial.
    """
    payload = getattr(control_call, "payload", None)
    handed = payload.get("vars") if isinstance(payload, Mapping) else None
    handed = handed if isinstance(handed, Mapping) else {}
    target: dict[str, Any] = {"uri": transfer_target}
    if handed.get("transfer_next_hop"):
        target["next_hop"] = str(handed["transfer_next_hop"])
    arguments: dict[str, Any] = {
        "on_answer": "bridge",
        "targets": [target],
        "timeout": TRANSFER_RING_TIMEOUT_SECONDS,
    }
    for variable, argument in (
        ("transfer_from", "from"),
        ("transfer_p_asserted_identity", "p_asserted_identity"),
        ("transfer_privacy", "privacy"),
    ):
        if handed.get(variable):
            arguments[argument] = str(handed[variable])
    return arguments


async def _start_the_transfer(control_call: Any, transfer_target: str) -> None:
    """Send the transfer the configured mode asks for. Raises the SDK's error when it is refused."""
    if TRANSFER_MODE == "bridge":
        await control_call.command("dial", _bridge_dial_arguments(control_call, transfer_target))
        logger.info(f"ringing {transfer_target} to bridge the caller to it")
        return
    await control_call.transfer(transfer_target)
    logger.info(f"asked the caller's phone to transfer to {transfer_target}")


async def _transfer_and_wait_for_the_verdict(
    resources: CallResources, control_call: Any
) -> TransferVerdict | None:
    """Transfer the caller, and return the verdict if they have left the bot's leg by it.

    `None` is a caller who is still here. The caller's words are held from the transfer until the
    verdict (see `TransferHold`), and a failed transfer is put to the model, because the caller who
    was told they were being put through is still there and nothing else on the line will speak to
    them.
    """
    sip_call_id = resources.serializer.call_id or ""
    verdict = resources.control.expect_transfer_verdict(sip_call_id)
    resources.transfer_hold.engage()
    try:
        await _start_the_transfer(control_call, resources.transfer_target)
    except resources.control.error_class as error:
        # The caller hanging up first is the ordinary case, not a fault. If they have, the note
        # below is answered to nobody and the call's own teardown ends the pipeline.
        logger.info(f"transfer rejected ({error}); the caller most likely hung up first")
        verdict.cancel()
        await _inform_the_model_the_transfer_failed(
            resources, TRANSFER_FAILURE_EXPLANATIONS["not_sent"]
        )
        return None

    try:
        # Shielded so the wait running out does not cancel the verdict itself: a late one can still
        # say the caller has gone.
        outcome = await asyncio.wait_for(asyncio.shield(verdict), TRANSFER_VERDICT_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.warning(
            f"no transfer verdict within {TRANSFER_VERDICT_TIMEOUT_SECONDS:g}s, keeping the caller"
        )
        resources.late_transfer_verdict = asyncio.create_task(
            _end_on_a_late_completion(resources, verdict)
        )
        await _inform_the_model_the_transfer_failed(
            resources, TRANSFER_FAILURE_EXPLANATIONS["timeout"]
        )
        return None

    if outcome.leaves_the_call:
        return outcome
    await _inform_the_model_the_transfer_failed(
        resources,
        TRANSFER_FAILURE_EXPLANATIONS.get(outcome.stage or "", "it was never confirmed"),
    )
    return None


async def transfer_call(params: FunctionCallParams) -> None:
    """Say the handover line, wait for it to play, transfer the caller, and wait for the verdict.

    Takes no arguments, exactly like `end_call` and for the same reason: it acts on the call the
    bot is already on and sends it to the one configured destination. A model-chosen destination
    would be a caller-chosen destination.

    The bot lets go of the caller only once the transfer has happened, or the call has gone. Many
    phones refuse a REFER from the network (`405`, `403`), and ending the bot's leg as soon as the
    REFER was sent left exactly those callers on a silent line until the node gave up on the call.
    """
    resources = params.app_resources
    if not isinstance(resources, CallResources) or not resources.transfer_target:
        await params.result_callback({"error": "this call cannot be transferred from here"})
        return

    control_call = resources.control.call_for(resources.serializer.call_id)
    if control_call is None:
        await params.result_callback({"error": "this call cannot be transferred from here"})
        return

    if resources.transfer_pending:
        await params.result_callback(
            {"error": "a transfer is already in progress; wait for it to go through or fail"}
        )
        return

    resources.transfer_pending = True
    try:
        # Answer before the caller leaves: once they are transferred there is nobody to hear a
        # result. As in `end_call`, the result is what gets the handover line said, and the wait
        # below listens for that reply.
        await params.result_callback({"transferring": True})

        if not await resources.speech.wait_until_finished():
            logger.warning("handover line did not finish within the cap, transferring anyway")

        left = await _transfer_and_wait_for_the_verdict(resources, control_call)
        if left is None:
            return
    finally:
        resources.transfer_pending = False
        resources.transfer_hold.release()

    # The transfer happened or the call went with it, so this bot's leg is finished even though the
    # call may not be. Leaving the pipeline running holds the media socket open, and the leg then
    # sits there until the engine's media timeout reaps it -- half a minute of a call nobody is on.
    await _let_go_of_the_call(resources, left)


class SpeakingLanguageController(FrameProcessor):
    """Change the language the bot speaks mid-call: when the model is asked, or after the caller.

    One per call, and only when the deployment configured more than one language. It holds the
    call's current language, serves the `set_speaking_language` tool, and sits after the echo guard
    to hear the language the recognizer reports for each stretch of the caller's speech, so the
    bot's own echo never counts as the caller changing language.

    A switch is a `TTSUpdateSettingsFrame`, pipecat's runtime settings change, carrying the new
    language and that language's voice. Cartesia, which locks voice and language per context,
    flushes and opens a fresh one, so the change takes effect at the next sentence. A recognizer
    pinned to one language gets an `STTUpdateSettingsFrame` too, or it would go on transcribing the
    caller in the old one; one on `multi` hears every language already and is left alone.
    """

    def __init__(
        self,
        languages: SpeakingLanguages,
        *,
        provider: str,
        default_voice: str,
        recognizer_language: str,
        consistent_turns: int = FOLLOW_CALLER_CONSISTENT_TURNS,
    ) -> None:
        """Start the call in `languages.starting`, speaking through `provider`."""
        super().__init__()
        self.languages = languages
        self._provider = provider
        self._default_voice = default_voice
        self._recognizer_is_pinned = recognizer_language.lower() != "multi"
        self._consistent_turns = consistent_turns
        self.current = languages.starting or languages.available[0]
        # Follow-the-caller state: the language heard in the latest run of stretches that differs
        # from the current one, how many stretches in a row it has been heard, and whether the
        # stretch in progress has already been counted.
        self._candidate: str | None = None
        self._candidate_turns = 0
        self._stretch_counted = False
        # An explicit request is the caller saying which language they want. Following them after
        # that would undo it: an English speaker who asks for Dutch keeps speaking English.
        self._requested = False

    def _switch_frames(
        self, language: str
    ) -> tuple[TTSUpdateSettingsFrame, STTUpdateSettingsFrame | None]:
        """Build the settings change for the voice, and a pinned recognizer, to speak `language`."""
        voice_settings = speaking_settings(
            self._provider, language, self.languages, self._default_voice
        )
        if "instructions" in voice_settings:
            # Only OpenAI's voice takes instructions, so only its own settings class carries them.
            from pipecat.services.openai.tts import OpenAITTSService

            voice_delta: TTSSettings = OpenAITTSService.Settings(**voice_settings)
        else:
            voice_delta = TTSSettings(**voice_settings)
        recognizer_frame = (
            STTUpdateSettingsFrame(delta=STTSettings(language=as_pipecat_language(language)))
            if self._recognizer_is_pinned
            else None
        )
        return TTSUpdateSettingsFrame(delta=voice_delta), recognizer_frame

    async def set_speaking_language(self, params: FunctionCallParams) -> None:
        """Serve the tool: switch to the language the model names, if it is a configured one.

        The frames go out through the model's own service, ahead of the reply it gives once this
        returns: downstream into the voice, which is next in the pipeline, so the reply is spoken in
        the new language, and upstream to the recognizer.
        """
        requested = str(params.arguments.get("language", ""))
        language = self.languages.configured_language_for(requested) if requested else None
        if language is None:
            await params.result_callback(
                {
                    "error": f"cannot speak {requested!r} on this call",
                    "available": list(self.languages.available),
                }
            )
            return
        self._requested = True
        if language != self.current:
            voice_frame, recognizer_frame = self._switch_frames(language)
            await params.llm.push_frame(voice_frame)
            if recognizer_frame is not None:
                await params.llm.push_frame(recognizer_frame, FrameDirection.UPSTREAM)
            logger.info(f"speaking language {self.current} -> {language}, as the caller asked")
            self.current = language
        await params.result_callback({"speaking_language": language})

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Count the caller's language per stretch of speech, and pass every frame through."""
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._stretch_counted = False
        elif (
            isinstance(frame, TranscriptionFrame)
            and self.languages.follow_caller
            and not self._requested
            and not self._stretch_counted
            and frame.language
        ):
            heard = self.languages.configured_language_for(str(frame.language))
            # A language nobody configured, or none reported, says nothing about which configured
            # one the caller is speaking, so it neither counts nor breaks a run.
            if heard is not None:
                self._stretch_counted = True
                await self._count(heard)
        await self.push_frame(frame, direction)

    async def _count(self, heard: str) -> None:
        """Count one stretch heard in `heard`, and follow it once it has been heard consistently."""
        if heard == self.current:
            self._candidate, self._candidate_turns = None, 0
            return
        if heard == self._candidate:
            self._candidate_turns += 1
        else:
            self._candidate, self._candidate_turns = heard, 1
        if self._candidate_turns < self._consistent_turns:
            return
        voice_frame, recognizer_frame = self._switch_frames(heard)
        # Ahead of the transcript, so the reply to it is already spoken in the new voice.
        await self.push_frame(voice_frame)
        if recognizer_frame is not None:
            await self.push_frame(recognizer_frame, FrameDirection.UPSTREAM)
        logger.info(
            f"speaking language {self.current} -> {heard}, following the caller "
            f"after {self._candidate_turns} turns in it"
        )
        self.current = heard
        self._candidate, self._candidate_turns = None, 0


def parse_caller(value: str | None) -> str | None:
    """Read the caller's number from the media socket's `caller` parameter, or None for no number.

    Anything but 6 to 15 ASCII digits counts as no number at all rather than as a number to clean
    up: the tool puts it in a URL, and a value that needed cleaning is not one to trust.
    """
    if value is None or CALLER_NUMBER.fullmatch(value) is None:
        return None
    return value


def mask_number(number: str) -> str:
    """Hide all but the last two digits of a phone number, for a log line."""
    return "*" * (len(number) - 2) + number[-2:]


def _not_available(reason: str) -> dict[str, Any]:
    """Build the result a lookup tool gives when it has nothing to say, with why, for the model."""
    return {"available": False, "reason": reason}


def _text(value: object) -> str | None:
    """Read a non-empty string field, or None for anything else."""
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float | None:
    """Read a numeric field, or None for anything else, a bool included (to Python, an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _is_scalar(value: object) -> bool:
    """Whether a JSON value is a string, a finite number or a boolean."""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (str, int, bool))


def _scalars_only(value: object) -> object | None:
    """Keep a scalar, or a list or object of scalars with everything else in it dropped."""
    if _is_scalar(value):
        return value
    if isinstance(value, list):
        return [item for item in value if _is_scalar(item)]
    if isinstance(value, dict):
        return {
            key: item for key, item in value.items() if isinstance(key, str) and _is_scalar(item)
        }
    return None


def pass_through_caller_lookup(body: object) -> dict[str, Any]:
    """Hand the model the lookup's answer as it came, minus anything deeper than one level.

    No field is interpreted: what the record holds is the deployment's business, and the description
    is where the model learns what it means. Nulls and anything nested further are dropped, and an
    answer that is still over `CALLER_LOOKUP_MAXIMUM_BYTES` is not available at all.
    """
    if not isinstance(body, dict):
        return _not_available("the answer could not be read")
    details = {
        key: kept
        for key, value in body.items()
        if isinstance(key, str) and (kept := _scalars_only(value)) is not None
    }
    if len(json.dumps(details, ensure_ascii=False).encode()) > CALLER_LOOKUP_MAXIMUM_BYTES:
        return _not_available("the answer was too long to use")
    return {"available": True, "details": details}


class CallerLookup:
    """Serve the caller_lookup tool for one call: the details the deployment keeps on the caller.

    One per call, holding the number the media socket was opened with. The tool takes no arguments
    for the reason `end_call` takes none: a number the model could name is a number the caller can
    talk it into naming, and then anyone's details are one question away.
    """

    def __init__(self, url_template: str, caller: str | None) -> None:
        """Look `caller` up at `url_template`, with `{caller}` replaced. No number, no lookup."""
        self._url_template = url_template
        self._caller = caller

    async def caller_lookup(self, params: FunctionCallParams) -> None:
        """Serve the tool: hand the model what the lookup found, or that nothing could be seen."""
        await params.result_callback(await self.look_up())

    async def look_up(self) -> dict[str, Any]:
        """Fetch the caller's details, within the timeout."""
        if self._caller is None:
            return _not_available("the caller's number is not known on this call")
        masked = mask_number(self._caller)
        url = self._url_template.replace(CALLER_LOOKUP_PLACEHOLDER, self._caller)
        timeout = aiohttp.ClientTimeout(total=CALLER_LOOKUP_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as reply:
                if reply.status == 404:
                    logger.info(f"caller lookup: {masked} is not known")
                    return _not_available("the caller is not known")
                if not 200 <= reply.status < 300:
                    logger.warning(f"caller lookup: HTTP {reply.status} for {masked}")
                    return _not_available("the lookup failed")
                body = await reply.json(content_type=None)
        except TimeoutError:
            logger.warning(
                f"caller lookup: no answer for {masked} within {CALLER_LOOKUP_TIMEOUT_SECONDS:g}s"
            )
            return _not_available("the lookup took too long")
        except (aiohttp.ClientError, ValueError) as error:
            # The type and not the message: aiohttp puts the URL in its messages, and the URL has
            # the whole number in it.
            logger.warning(f"caller lookup: lookup for {masked} failed ({type(error).__name__})")
            return _not_available("the lookup failed")
        result = pass_through_caller_lookup(body)
        logger.info(f"caller lookup for {masked}: {'found' if result['available'] else 'unusable'}")
        return result


def _first_place(found: object) -> tuple[str, float, float] | None:
    """Read the name, latitude and longitude of the first place found, or None for no usable one."""
    results = found.get("results") if isinstance(found, dict) else None
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    first = results[0]
    name = _text(first.get("name"))
    latitude, longitude = _number(first.get("latitude")), _number(first.get("longitude"))
    if name is None or latitude is None or longitude is None:
        return None
    country = _text(first.get("country"))
    return (f"{name}, {country}" if country else name), latitude, longitude


def interpret_forecast(place: str, forecast: object) -> dict[str, Any]:
    """Turn open-meteo's current conditions into what the model says about them."""
    current = forecast.get("current") if isinstance(forecast, dict) else None
    if not isinstance(current, dict):
        return _not_available("the forecast could not be read")
    temperature = _number(current.get("temperature_2m"))
    if temperature is None:
        return _not_available("the forecast has no temperature")
    code = current.get("weather_code")
    wind = _number(current.get("wind_speed_10m"))
    result: dict[str, Any] = {
        "available": True,
        "place": place,
        "temperature_celsius": round(temperature),
        "conditions": WEATHER_CONDITIONS.get(code) if isinstance(code, int) else None,
        "wind_kilometres_per_hour": None if wind is None else round(wind),
        "observed_at": _text(current.get("time")),
    }
    return {key: value for key, value in result.items() if value is not None}


async def look_up_weather(place: str) -> dict[str, Any]:
    """Find `place` and its current weather on open-meteo, both within one timeout."""
    place = place.strip()[:WEATHER_PLACE_MAXIMUM_LENGTH]
    if not place:
        return _not_available("no place was named")
    try:
        async with asyncio.timeout(WEATHER_TIMEOUT_SECONDS), aiohttp.ClientSession() as session:
            async with session.get(
                WEATHER_GEOCODING_URL, params={"name": place, "count": "1"}
            ) as reply:
                if reply.status != 200:
                    logger.warning(f"weather: place search answered HTTP {reply.status}")
                    return _not_available("the weather service is not answering")
                found = _first_place(await reply.json(content_type=None))
            if found is None:
                return _not_available(f"no place called {place} was found")
            name, latitude, longitude = found
            async with session.get(
                WEATHER_FORECAST_URL,
                params={
                    "latitude": str(latitude),
                    "longitude": str(longitude),
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                },
            ) as reply:
                if reply.status != 200:
                    logger.warning(f"weather: forecast answered HTTP {reply.status}")
                    return _not_available("the weather service is not answering")
                forecast = await reply.json(content_type=None)
    except TimeoutError:
        logger.warning(f"weather: no answer within {WEATHER_TIMEOUT_SECONDS:g}s")
        return _not_available("the weather service took too long")
    except (aiohttp.ClientError, ValueError) as error:
        logger.warning(f"weather: lookup failed ({type(error).__name__}: {error})")
        return _not_available("the weather service is not answering")
    return interpret_forecast(name, forecast)


async def report_weather(params: FunctionCallParams) -> None:
    """Serve the weather tool: the current weather in the place the model names."""
    place = params.arguments.get("place")
    await params.result_callback(await look_up_weather(place if isinstance(place, str) else ""))


def build_speaking_language_controller() -> SpeakingLanguageController | None:
    """Build the call's language switch, or None when there is only one language to speak."""
    languages = configured_speaking_languages()
    if not languages.switchable:
        return None
    return SpeakingLanguageController(
        languages,
        provider=TTS_PROVIDER,
        default_voice=default_tts_voice(),
        recognizer_language=STT_LANGUAGE,
    )


def build_tool_schemas(
    with_control: bool,
    speaking: SpeakingLanguageController | None,
    *,
    caller_lookup: CallerLookup | None = None,
    weather: bool = False,
    briefed: BriefedCall | None = None,
    call_back_minutes: int | None = None,
    transfer_target: str | None = None,
    caller_lookup_description: str | None = None,
) -> list[FunctionSchema]:
    """List the model's tools: hangup and transfer, the PBX's choices, the language switch, lookups.

    Each only when it can act. A tool the model is offered and cannot use is a promise to the caller
    that is never kept, which is the same reason the prompts for them are appended conditionally.
    On a call siphon-pbx briefed, `end_call` reports the conversation finished rather than hanging
    up, since the flow decides what happens next; transfer still needs the control plane.
    `call_back_minutes`, the furthest ahead a callback may go, offers them; they need it too.
    `transfer_target` and `caller_lookup_description` are the call's configuration's, the bot's
    own when not given.
    """
    if transfer_target is None:
        transfer_target = TRANSFER_TARGET
    if caller_lookup_description is None:
        caller_lookup_description = CALLER_LOOKUP_DESCRIPTION
    tools: list[FunctionSchema] = []
    if with_control or briefed is not None:
        tools.append(
            FunctionSchema(
                name="end_call",
                description=(
                    "Hang up the phone call. Call it as soon as the caller is done, "
                    "before saying goodbye, and say the goodbye after it returns: the "
                    "call ends once the goodbye has finished playing."
                ),
                properties={},
                required=[],
                handler=briefed.end_call if briefed is not None else end_call,
            )
        )
    if briefed is not None and briefed.session.brief.options:
        options = briefed.session.brief.options
        tools.append(
            FunctionSchema(
                name="choose_option",
                description=(
                    "Pass the caller on to what they asked for. Call it first, then tell them "
                    "you are passing them on; the call moves on once your words have finished. "
                    "The options: "
                    + "; ".join(
                        f"{option.option} ({option.means})" if option.means else option.option
                        for option in options
                    )
                    + "."
                ),
                properties={
                    "option": {
                        "type": "string",
                        # The brief's list and nothing else; the PBX routes anything else out of
                        # the block's no-match port, which is not what the caller asked for.
                        "enum": [option.option for option in options],
                        "description": "The option the caller asked for.",
                    }
                },
                required=["option"],
                handler=briefed.choose_option,
            )
        )
    if with_control and transfer_target:
        tools.append(
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
        )
    if with_control and call_back_minutes is not None:
        tools.append(
            FunctionSchema(
                name="call_back",
                description=(
                    "Arrange for the caller to be called back on the number they are calling "
                    "from. Call it first, then tell them when to expect the call."
                ),
                properties={
                    "minutes": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": call_back_minutes,
                        "description": "Minutes from now; 0 is as soon as this call is over.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "What the callback is about, in a few words.",
                    },
                },
                required=["minutes", "reason"],
                handler=call_back,
            )
        )
    if speaking is not None:
        tools.append(
            FunctionSchema(
                name="set_speaking_language",
                description=(
                    "Change the language your voice speaks, for the rest of the call. Call it as "
                    "soon as the caller asks you to speak another language, then answer in it."
                ),
                properties={
                    "language": {
                        "type": "string",
                        # The deployment's list, and nothing else: a language the model could name
                        # freely is one the configured voice may not speak natively.
                        "enum": list(speaking.languages.available),
                        "description": "The BCP-47 code of the language to speak.",
                    }
                },
                required=["language"],
                handler=speaking.set_speaking_language,
            )
        )
    if caller_lookup is not None:
        tools.append(
            FunctionSchema(
                name="caller_lookup",
                description=caller_lookup_description,
                # None, on purpose: see `CallerLookup`.
                properties={},
                required=[],
                handler=caller_lookup.caller_lookup,
            )
        )
    if weather:
        tools.append(
            FunctionSchema(
                name="weather",
                description="Get the current weather in a place: temperature, conditions, wind.",
                properties={
                    "place": {
                        "type": "string",
                        "description": "A town or city, optionally with its country.",
                    }
                },
                required=["place"],
                handler=report_weather,
            )
        )
    return tools


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
    return base_language(language) == "en"


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
            # consumes. Except for a speech-to-speech model, which detects the caller's speech in
            # the audio itself: the engine's edges would be a second detector, interrupting it on
            # its own and disagreeing with it about where a turn starts.
            speech_frames="none" if speech_to_speech() else "vad",
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


REQUIRED_ENVIRONMENT: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("llm", "anthropic"): (("ANTHROPIC_API_KEY", "Claude API key"),),
    ("llm", "google"): (("GOOGLE_API_KEY", "Gemini API key"),),
    ("llm", "openai"): (("OPENAI_API_KEY", "OpenAI API key"),),
    # Nothing: the model server's address has a default and takes no key.
    ("llm", "local"): (),
    ("stt", "deepgram"): (("DEEPGRAM_API_KEY", "speech-to-text key"),),
    # Nothing here: Google's speech APIs take application default credentials, which need not be
    # a variable at all. `refuse_missing_google_cloud_credentials` checks them instead.
    ("stt", "google"): (),
    ("stt", "openai"): (("OPENAI_API_KEY", "OpenAI API key"),),
    # Nothing: the model downloads itself.
    ("stt", "local"): (),
    ("tts", "cartesia"): (
        ("CARTESIA_API_KEY", "text-to-speech key"),
        ("CARTESIA_VOICE_ID", "voice to speak with"),
    ),
    ("tts", "google"): (),
    ("tts", "openai"): (("OPENAI_API_KEY", "OpenAI API key"),),
    ("tts", "local"): (),
    ("llm", "gemini-live"): (("GOOGLE_API_KEY", "Gemini API key"),),
    # Nothing: the one model fills these roles, under the key above.
    ("stt", "gemini-live"): (),
    ("tts", "gemini-live"): (),
}
"""What each provider cannot start without, in each role, and what the value is for.

Keyed per role *and* provider because the same provider asks for different things depending on the
job -- Gemini takes an API key while Google's speech APIs take a service account -- and because the
one table is then the only place a key is named. The builders read their purpose text out of it
rather than repeating the pair, so the two cannot drift apart. Google's speech APIs list nothing:
their credential is an identity, not a value, and need not be a variable at all."""

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


GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
"""The scope pipecat's Google recognizer and voice ask application default credentials for."""


def refuse_missing_google_cloud_credentials(providers: Mapping[str, str]) -> None:
    """Stop a selection using Google's speech APIs when no Google identity can be found.

    Those APIs are Cloud services and authenticate as a service account, not with an API key, and
    pipecat passes no credentials of its own, so the client falls back to application default
    credentials. That is `GOOGLE_APPLICATION_CREDENTIALS` when it is set, and otherwise, on Google
    Cloud, the metadata server, with no file at all. Demanding the variable would force a long-lived
    key onto a machine that already has an identity, so this asks the library instead. Gemini is a
    different product with a different credential and is not checked here.

    Raises:
        ValueError: If no credentials resolve, or the recognizer is selected and they name no
            project, which pipecat's recognizer needs and its voice does not.

    """
    roles = [role for role in ("stt", "tts") if providers.get(role) == "google"]
    if not roles:
        return
    import google.auth
    from google.auth.exceptions import GoogleAuthError

    try:
        _credentials, project = google.auth.default(scopes=[GOOGLE_CLOUD_SCOPE])
    # The base class, as pipecat catches: a metadata server that answers badly raises a transport
    # error rather than `DefaultCredentialsError`, and must stop the bot the same way.
    except GoogleAuthError as error:
        raise ValueError(
            f"{_provider_summary(providers)} found no Google Cloud credentials for "
            f"{' and '.join(f'{role}=google' for role in roles)}: set "
            "GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file, or run on Google Cloud "
            "where the metadata server provides the instance's service account. The account needs "
            f"the Speech-to-Text and Text-to-Speech APIs. ({error})"
        ) from error
    if "stt" in roles and not project:
        raise ValueError(
            f"{_provider_summary(providers)} found Google Cloud credentials naming no project, and "
            "stt=google needs one: set GOOGLE_CLOUD_PROJECT and run again."
        )


def _provider_summary(providers: Mapping[str, str]) -> str:
    """Name the selection in an error, as the preset when all three agree with one."""
    chosen = (providers.get("llm", ""), providers.get("stt", ""), providers.get("tts", ""))
    for name, preset in BACKEND_PRESETS.items():
        if chosen == preset:
            return f"the {name} backend"
    return "llm={}, stt={}, tts={}".format(*chosen)


def refuse_a_split_speech_to_speech(providers: Mapping[str, str]) -> None:
    """Stop a speech-to-speech provider chosen for some roles and not the others.

    Its model hears, answers and speaks in one, so a recognizer or a voice beside it would be built
    and billed and never hear or say a word, and a speech-to-speech recognizer behind another model
    has no transcript to give it.

    Raises:
        ValueError: Naming the selection and the preset that selects the provider whole.

    """
    chosen = {providers.get(role, "") for role in ("llm", "stt", "tts")}
    for provider in SPEECH_TO_SPEECH_PROVIDERS:
        if provider in chosen and chosen != {provider}:
            raise ValueError(
                f"{_provider_summary(providers)} splits {provider} across roles, but it hears, "
                f"answers and speaks in one model: set BOT_BACKEND={provider} and leave "
                "BOT_LLM_PROVIDER, BOT_STT_PROVIDER and BOT_TTS_PROVIDER unset"
            )


def refuse_an_unknown_backend(backend: str) -> None:
    """Stop a preset nobody defined, naming the variable and what it accepts.

    Raises:
        ValueError: If `backend` is not one of `BACKENDS`.

    """
    if backend not in BACKEND_PRESETS:
        raise ValueError(f"unknown BOT_BACKEND {backend!r}: expected one of {', '.join(BACKENDS)}")


def refuse_an_unknown_transfer_mode(mode: str) -> None:
    """Stop a transfer mode nobody defined, naming the variable and what it accepts.

    Raises:
        ValueError: If `mode` is not one of `TRANSFER_MODES`.

    """
    if mode not in TRANSFER_MODES:
        raise ValueError(
            f"unknown BOT_TRANSFER_MODE {mode!r}: expected one of {', '.join(TRANSFER_MODES)}"
        )


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
        # In the language and voice the first call starts in, so the one exercised is the one used.
        languages = configured_speaking_languages()
        starting = languages.starting
        pipecat_language = as_pipecat_language(starting) if starting else Language.EN
        language = (
            kokoro.language_to_kokoro_language(pipecat_language)
            if isinstance(pipecat_language, Language)
            else pipecat_language
        )
        voice = speaking_voice("local", starting, languages, TTS_VOICE) if starting else TTS_VOICE
        for synthesizer in kokoro.Kokoro.built():
            synthesizer.create("Ready.", voice=voice, speed=1.0, lang=language)

    logger.info(
        f"local models loaded and exercised in {time.monotonic() - started:.1f}s "
        f"({', '.join(roles)})"
    )


def build_llm_service(
    with_hangup: bool,
    brief: SessionBrief | None = None,
    call_back: bool = False,
    configuration: CallConfiguration | None = None,
    facts: CallFacts | None = None,
    tools: list[FunctionSchema] | None = None,
) -> LLMService[Any]:
    """Build the LLM service for the selected backend, tuned for a conversation in real time.

    `with_hangup` is a control plane: hangup, and transfer when there is a destination. `brief`
    is siphon-pbx's for this call, whose instructions and options the model is told, and whose
    `end_call` reports rather than hangs up. `call_back` says the call_back tool is offered.
    `configuration` is the call's, the bot's own when not given, and `facts` say whose line it is.
    `tools` are given to a speech-to-speech model at construction; the other models read theirs
    from the context, and ignore these.
    """
    if configuration is None:
        configuration = default_configuration()
    system_instruction = (
        configuration.system_prompt
        + organisation_prompt(facts or CallFacts())
        + (HANGUP_PROMPT if with_hangup or brief is not None else "")
        + (TRANSFER_PROMPT if with_hangup and configuration.transfer_target else "")
        + (CALL_BACK_PROMPT if with_hangup and call_back else "")
        + speaking_language_prompt(configured_speaking_languages())
        + (CALLER_LOOKUP_PROMPT if configuration.caller_lookup_url else "")
        + (WEATHER_PROMPT if WEATHER_ENABLED else "")
        + (SESSION_PROMPT.format(instructions=brief.instructions) if brief is not None else "")
        + (CHOOSE_OPTION_PROMPT if brief is not None and brief.options else "")
    )
    builders: dict[str, Callable[[str], LLMService[Any]]] = {
        "anthropic": _build_anthropic_llm_service,
        "google": _build_google_llm_service,
        "openai": _build_openai_llm_service,
        "local": _build_local_llm_service,
        "gemini-live": lambda instruction: _build_gemini_live_service(instruction, tools or []),
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


def gemini_live_language() -> str:
    """Name the language Gemini Live is told it hears and speaks, as the locale it wants.

    The configured speaking language, else the recognizer's. pipecat 1.12 sends a `Language` it is
    given as the enum's own value, `nl` rather than the `nl-NL` Gemini expects, so the conversion
    happens here rather than being left to whichever pipecat is installed.
    """
    from pipecat.services.google.gemini_live.llm import language_to_gemini_language

    code = configured_speaking_languages().starting or STT_LANGUAGE
    language = as_pipecat_language(code)
    return language_to_gemini_language(language) if isinstance(language, Language) else code


def _build_gemini_live_service(
    system_instruction: str, tools: list[FunctionSchema]
) -> LLMService[Any]:
    """Build Gemini's native-audio service, which hears, answers and speaks in one model.

    Its own voice activity detection decides the caller's turns and interruptions; nothing here
    configures it, so a turn ends when Gemini's defaults say so. The tools and the prompt go to the
    constructor rather than the context: tools arriving on the context make it reconnect, which is
    a second session set up while the caller waits for the greeting.
    """
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    return GeminiLiveLLMService(
        api_key=_provider_key("llm", "gemini-live", "GOOGLE_API_KEY"),
        tools=ToolsSchema(standard_tools=tools) if tools else None,
        settings=GeminiLiveLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_instruction,
            voice=TTS_VOICE or DEFAULT_TTS_VOICE["gemini-live"],
            language=gemini_live_language(),
            max_tokens=MAXIMUM_RESPONSE_TOKENS,
            **_sampling_settings("gemini-live"),
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

    # No credentials passed: pipecat falls back to application default credentials, which startup
    # has already checked resolve, to a project as well, so this cannot fail with the library's own
    # `No valid credentials provided.`
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


def _starting_speaking_settings(provider: str, voice: str) -> dict[str, Any]:
    """Give the language and voice a synthesizer starts the call in, or nothing when none is set.

    Nothing rather than the language each service defaults to, so a deployment that sets no
    language builds every voice exactly as it did before languages could be configured.
    """
    languages = resolve_speaking_languages(os.environ, provider, voice)
    return speaking_settings(provider, languages.starting, languages, voice)


def _build_cartesia_tts_service() -> TTSService:
    """Build Cartesia's voice."""
    from pipecat.services.cartesia.tts import CartesiaTTSService

    # The only provider whose voice cannot have a default: a Cartesia voice belongs to an account.
    # `BOT_TTS_VOICE` still wins when it is set, so the two ways of saying it agree.
    voice = TTS_VOICE or _provider_key("tts", "cartesia", "CARTESIA_VOICE_ID")
    return CartesiaTTSService(
        api_key=_provider_key("tts", "cartesia", "CARTESIA_API_KEY"),
        # Without a language Cartesia speaks English, whatever the text: Dutch replies came out
        # in an English accent. The language, and that language's voice, only when one is set.
        settings=CartesiaTTSService.Settings(
            **{"voice": voice, "model": TTS_MODEL, **_starting_speaking_settings("cartesia", voice)}
        ),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def _build_google_tts_service() -> TTSService:
    """Build Google's voice."""
    from pipecat.services.google.tts import GoogleTTSService

    # As with the recognizer: application default credentials, checked at startup.
    # The Chirp3 voices carry their own model, so `model` stays unset unless a deployment names
    # one. `NOT_GIVEN` rather than `None`: pipecat reads None as a value to send.
    from pipecat.utils.types import NOT_GIVEN

    return GoogleTTSService(
        settings=GoogleTTSService.Settings(
            **{
                "voice": TTS_VOICE,
                "model": TTS_MODEL or NOT_GIVEN,
                **_starting_speaking_settings("google", TTS_VOICE),
            }
        ),
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
        settings=OpenAITTSService.Settings(
            **{
                "voice": TTS_VOICE,
                "model": TTS_MODEL,
                **_starting_speaking_settings("openai", TTS_VOICE),
            }
        ),
        sample_rate=OPENAI_TTS_SAMPLE_RATE,
    )


def _build_local_tts_service() -> TTSService:
    """Build Kokoro, on this host."""
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService(
        settings=KokoroTTSService.Settings(
            **{"voice": TTS_VOICE, **_starting_speaking_settings("local", TTS_VOICE)}
        ),
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
    if speech_to_speech():
        # The model decides the turn from the audio, and the serializer passes on no speech edges
        # for a local strategy to act on. Named rather than left unset, because unset is pipecat's
        # default stop strategy, which loads a turn model this call has no use for.
        return LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy()],
            ),
        )
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
    caller: str | None = None,
    placed: PlacedCall | None = None,
    session: VoiceSession | None = None,
    configuration: CallConfiguration | None = None,
    facts: CallFacts | None = None,
) -> PipelineWorker:
    """Assemble one call: transport, recognizer, model, voice, and the turn machinery.

    Everything here is per call. The services each open their own vendor connection, which is the
    honest cost of concurrency: N simultaneous calls are N recognizer sockets and N synthesizer
    sockets, not one shared pool. That is how pipecat is built, and it is why the ceiling is the
    vendors' rate limits rather than anything in this file.

    `observer_factory`, when given, builds an observer that watches this call's pipeline without
    being part of it -- how a test harness reads what the bot heard and decided.

    `caller` is the caller's number from the media socket's URL, already checked by `parse_caller`,
    or None. Only the caller_lookup tool reads it. `placed` is the call this bot placed, when the
    engine is dialling in for one; the bot opens it as the caller. `session` is the siphon-pbx
    session the call was handed over in, already briefed, which the bot reports its outcome to.
    `configuration` is what the call is served with (the bot's own when not given), and `facts`
    whose call it is: a caller number there, from whoever handed the call over, is the one the
    caller lookup uses rather than the URL's.
    """
    if configuration is None:
        configuration = default_configuration()
    facts = facts or CallFacts()
    if facts.caller is not None:
        caller = facts.caller
    serializer = build_serializer()
    transport = build_transport(websocket, serializer)
    # One model that hears and speaks, or a recognizer and a voice around a model that reads text.
    one_model = speech_to_speech()
    speech_to_text = None if one_model else build_speech_to_text()
    call_backs = control.call_backs if control is not None else None
    text_to_speech = None if one_model else build_text_to_speech()
    speech_monitor = BotSpeechMonitor()
    # Not for one model: it drops transcripts whatever their direction, and there the transcripts
    # are the model's own record of what the caller said, travelling upstream. The engine's echo
    # canceller is what keeps the bot from hearing itself.
    echo_guard = None if one_model else EchoGuard(speech_monitor, ECHO_GUARD_TAIL_SECONDS)
    briefed = BriefedCall(session=session, speech=speech_monitor) if session is not None else None

    # The conversation itself. The system prompt lives on the service (and is cached there), so
    # the context carries only the call.
    context = LLMContext()
    # None unless the deployment configured a second language: with one, the voice is built
    # speaking it and there is nothing to switch, so the pipeline stays as it was.
    speaking_language = build_speaking_language_controller()
    tools = build_tool_schemas(
        control is not None,
        speaking_language,
        caller_lookup=(
            CallerLookup(configuration.caller_lookup_url, caller)
            if configuration.caller_lookup_url
            else None
        ),
        weather=WEATHER_ENABLED,
        briefed=briefed,
        call_back_minutes=(
            call_backs.settings.maximum_delay_minutes if call_backs is not None else None
        ),
        transfer_target=configuration.transfer_target,
        caller_lookup_description=configuration.caller_lookup_description,
    )
    llm = build_llm_service(
        with_hangup=control is not None,
        brief=session.brief if session is not None else None,
        call_back=call_backs is not None,
        configuration=configuration,
        facts=facts,
        tools=tools,
    )
    if tools and not one_model:
        context.set_tools(ToolsSchema(standard_tools=tools))
    # The aggregator decides the caller's turn, and nothing else in the pipeline does: see
    # `build_user_aggregator_params` for what a second decider in front of it cost. Its strategies
    # are explicit and differ by backend: the cloud path ends a turn on the engine's endpoint plus
    # the transcript, the local path also holds a sentence the caller stopped in the middle of.
    aggregators = LLMContextAggregatorPair(context, user_params=build_user_aggregator_params())

    resources = (
        CallResources(
            serializer=serializer,
            speech=speech_monitor,
            control=control,
            transfer_target=configuration.transfer_target,
            transfer_hold=TransferHold(hold_audio=one_model),
        )
        if control is not None
        else None
    )

    worker = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                *([speech_to_text] if speech_to_text is not None else []),
                # After the recognizer, which keeps getting audio, and ahead of the turn logic: see
                # `TransferHold`. Only with a control plane, since only then is there a transfer.
                *([resources.transfer_hold] if resources is not None else []),
                # Between the recognizer and the turn logic on purpose: the echo is dropped before
                # anything can treat it as the caller starting a turn.
                *([echo_guard] if echo_guard is not None else []),
                # After the echo guard, so the bot's own echo never counts as the caller changing
                # language, and ahead of the turn, so a switch reaches the voice before the reply.
                *([speaking_language] if speaking_language is not None else []),
                aggregators.user(),
                llm,
                *([text_to_speech] if text_to_speech is not None else []),
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
    if briefed is not None:
        briefed.worker = worker

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

        if briefed is not None:
            # The PBX's call, so the PBX's to end: told the bot failed, the flow takes the block's
            # failure path, which a hangup from here would deny it.
            await briefed.fail(
                f"a stage of the bot failed ({getattr(processor, 'name', 'unknown stage')})"
            )
            return

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
        logger.info(
            "engine connected, greeting the caller"
            if placed is None
            else f"engine connected for placed call {placed.channel}, greeting the callee"
        )
        speech_monitor.reset()
        # Seeding also *clears* the previous call's transcript, which matters now that the
        # server outlives a call: `set_messages` replaces, so call two does not inherit call one.
        kickoff = kickoff_for(placed.context if placed is not None else None)
        context.set_messages([{"role": "user", "content": kickoff}])
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: object, _client: object) -> None:
        logger.info("engine disconnected, call over")
        if briefed is not None:
            briefed.stream_closed()

    return worker


async def serve_call(
    websocket: WebSocket,
    control: ControlPlane | None,
    observer_factory: ObserverFactory | None = None,
    caller: str | None = None,
    placed: PlacedCall | None = None,
    session: VoiceSession | None = None,
) -> None:
    """Answer one call, on its own pipeline, and tear it down when the engine hangs up."""
    await websocket.accept()
    # After the accept: siphon-sip hands a call over only once the stream it dials is up.
    configuration, facts = (
        await resolve_call(control, websocket.query_params, session)
        if placed is None
        else (default_configuration(), CallFacts())
    )
    worker = build_worker(
        websocket,
        control,
        observer_factory,
        caller,
        placed,
        session,
        configuration=configuration,
        facts=facts,
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)
    # Returns when the pipeline ends, which is when the engine closes the socket. Everything the
    # call owned -- the recognizer and synthesizer connections, the context, the turn state -- goes
    # with it.
    await runner.run()


def build_app(
    control: ControlPlane | None,
    observer_factory: ObserverFactory | None = None,
    voice_sessions: VoiceSessionClient | None = None,
) -> FastAPI:
    """Build the HTTP surface: one WebSocket route the engine dials, once per call.

    With `voice_sessions`, a dial-in naming a siphon-pbx session is briefed before it is served.
    """
    app = FastAPI()

    @app.websocket("/{path:path}")
    async def media(websocket: WebSocket) -> None:
        # Any path. The engine's `ws_uri` carries a `{call_id}` query and whatever path the
        # deployment chose, and refusing an unexpected one would be a confusing way to fail a call
        # that is otherwise perfectly formed. The call is identified by the `start` envelope, not
        # by the URL -- except a call this bot placed, whose URL names it (`placed=`), because the
        # bot speaks first and has to know whom it rang before `start` arrives.
        #
        # The other thing read from the URL is the caller's number, a `caller` parameter that
        # whoever dials this socket may add. This example does not verify the URL, so whoever can
        # reach this port can claim any number: keep it on a network only the dialling side
        # reaches, as the docstring's addresses already assume.
        raw_caller = websocket.query_params.get("caller")
        caller = parse_caller(raw_caller)
        if caller is not None:
            logger.info(f"call from {mask_number(caller)}")
        elif raw_caller is not None:
            logger.warning("ignoring a caller parameter that is not 6 to 15 digits")
        placed = control.placed_calls.claim(websocket.query_params) if control is not None else None
        # Before the socket is accepted, so the brief is in hand before the bot can say anything.
        session = await open_voice_session(voice_sessions, websocket.query_params)
        try:
            await serve_call(websocket, control, observer_factory, caller, placed, session=session)
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
    call: CallRequest | None = None,
) -> int:
    """Answer calls until the process is stopped, each watched by `observer_factory`'s observer.

    With `call`, place that one call as well and stop once it is over, returning the exit status
    that says how it went. Without, return 0 when the server is stopped.
    """
    # Before anything listens. A selection that cannot run should stop the bot here, naming what is
    # missing, rather than at the first call, where the caller hears it as a dead line.
    providers = selected_providers()
    try:
        refuse_an_unknown_backend(BACKEND)
        refuse_an_unknown_transfer_mode(TRANSFER_MODE)
        missing = missing_environment(providers, os.environ)
        refuse_a_split_speech_to_speech(providers)
        resolve_system_prompt(os.environ)
        refuse_a_caller_lookup_url_without_the_caller(CALLER_LOOKUP_URL)
        check_configurations(CONFIGURATIONS_DIRECTORY)
        speaking_languages = configured_speaking_languages()
        voice_sessions = build_voice_session_client(
            PBX_URL, os.environ.get(PBX_SECRET_VARIABLE, "").strip()
        )
    except ValueError as error:
        sys.exit(str(error))
    if missing:
        purposes = dict(required_environment(providers))
        sys.exit(
            f"{_provider_summary(providers)} needs "
            + ", ".join(f"{name} (the {purposes[name]})" for name in missing)
            + "; set them and run again."
        )
    try:
        refuse_missing_google_cloud_credentials(providers)
    except ValueError as error:
        sys.exit(str(error))
    if local_roles():
        preload_local_models()

    # Said once, at startup, rather than per call. Without it, "why didn't it transfer" has two
    # indistinguishable answers -- the tool was never registered, or the model chose not to call
    # it -- and only one of them is a bug. Every segment names its provider, because with the three
    # roles chosen separately the model alone no longer says what is running.
    logger.info(
        f"models: llm={LLM_PROVIDER}/{LLM_MODEL}"
        + (f" at {LLM_BASE_URL}" if LLM_BASE_URL else "")
        + (
            f" speech-to-speech in {gemini_live_language()}"
            f" with voice {TTS_VOICE or DEFAULT_TTS_VOICE[LLM_PROVIDER]}"
            if speech_to_speech()
            else f" stt={STT_PROVIDER}/{STT_MODEL}/{STT_LANGUAGE}"
            + (f" on {WHISPER_DEVICE}" if STT_PROVIDER == "local" else "")
            + f" tts={TTS_PROVIDER}/{TTS_MODEL or TTS_VOICE or 'account voice'}"
        )
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
        + (
            f", set_speaking_language -> {', '.join(speaking_languages.available)}"
            if speaking_languages.switchable
            else ""
        )
        + (f", caller_lookup -> {CALLER_LOOKUP_URL}" if CALLER_LOOKUP_URL else "")
        + (", weather" if WEATHER_ENABLED else "")
        + (
            f", call_back -> {control.call_backs.settings.uri_template} within "
            f"{control.call_backs.settings.maximum_delay_minutes} min"
            if control is not None and control.call_backs is not None
            else ""
        )
    )
    # Which language the voice speaks, for the same reason: an accent is not visible in a log.
    if speaking_languages.starting is not None:
        logger.info(
            f"speaking: {speaking_languages.starting}"
            + (
                f" of {', '.join(speaking_languages.available)}"
                if speaking_languages.switchable
                else ""
            )
            + "; voices "
            + ", ".join(
                f"{language}="
                + (
                    speaking_voice(TTS_PROVIDER, language, speaking_languages, default_tts_voice())
                    or "account voice"
                )
                for language in speaking_languages.available
            )
            + (", following the caller" if speaking_languages.follow_caller else "")
        )
    if speaking_languages.follow_caller and STT_LANGUAGE.lower() != "multi":
        logger.warning(
            f"BOT_TTS_FOLLOW_CALLER is on, but the recognizer is pinned to {STT_LANGUAGE} and "
            "reports nothing else, so the voice will only change when the caller asks for it. "
            "Follow the caller with a recognizer on multi (deepgram)."
        )
    if (
        not speech_to_speech()
        and speaking_languages.starting is not None
        and STT_LANGUAGE.lower() != "multi"
        and base_language(STT_LANGUAGE) != base_language(speaking_languages.starting)
    ):
        # Warned rather than changed: the recognizer's language is BOT_STT_LANGUAGE's, and the
        # local recognizer's default model is English-only, so moving it here would stop the bot.
        logger.warning(
            f"the bot speaks {speaking_languages.starting} but the recognizer is pinned to "
            f"{STT_LANGUAGE}: set BOT_STT_LANGUAGE to match, or to multi on deepgram, or callers "
            "answering in that language are transcribed as the other one"
        )
    # The turn-taking posture, for the same reason: "it interrupted itself" and "it would not let me
    # interrupt" are both tuning, and neither is visible from a transcript.
    logger.info(
        "turn-taking: "
        + (
            f"{LLM_PROVIDER}'s own speech detection; the engine's speech edges are ignored"
            if speech_to_speech()
            else (
                f"echo guard {ECHO_GUARD_TAIL_SECONDS:g}s tail"
                if ECHO_GUARD_TAIL_SECONDS > 0
                else "echo guard off (the engine's canceller vetoes its own echo from 0.5.0)"
            )
            + f", interrupt after {MIN_INTERRUPT_WORDS} word(s)"
        )
    )
    if voice_sessions is not None:
        logger.info(f"siphon-pbx sessions: briefed by and reporting to {PBX_URL}")
    if CONFIGURATIONS_DIRECTORY:
        keys = sorted(file.stem for file in Path(CONFIGURATIONS_DIRECTORY).glob("*.yaml"))
        logger.info(
            f"configurations from {CONFIGURATIONS_DIRECTORY}: {', '.join(keys) or 'none yet'}"
        )
    logger.info(f"listening on ws://{host}:{port}/ -- point the engine's ws_uri here")

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(control, observer_factory, voice_sessions),
            host=host,
            port=port,
            log_level="warning",
            # uvicorn's own access log would add a line per call to a log that already says more.
            access_log=False,
        )
    )

    if control is None:
        await server.serve()
        return 0

    # The media path must survive an absent control plane: the engine and this bot restart
    # independently, so "not listening yet" is a routine race and not a reason to exit.
    control_task = asyncio.create_task(control.run_forever())
    # Placed while the server listens, since the engine dials the bot when the callee answers; and
    # once the call is over, so is the run.
    call_task = asyncio.create_task(place_one_call(control, call)) if call is not None else None
    if call_task is not None:
        call_task.add_done_callback(lambda _task: setattr(server, "should_exit", True))
    try:
        await server.serve()
    finally:
        if control.call_backs is not None:
            control.call_backs.cancel_all()
        for task in (call_task, control_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    if call_task is None or call_task.cancelled():
        return 0
    return call_task.result()


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the bind address, the control-plane options, and the call to place, if any."""
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
    parser.add_argument(
        "--call",
        metavar="SIP_URI",
        help="place one call to this URI, greet first when it is answered, and exit when it ends",
    )
    parser.add_argument(
        "--call-ws-uri",
        default=os.environ.get("BOT_CALL_WS_URI") or None,
        help="this bot's address as the engine reaches it, with {call_id} (env BOT_CALL_WS_URI)",
    )
    parser.add_argument("--call-from", help="the From URI the call is placed with")
    parser.add_argument(
        "--call-context",
        help="what the call is about, for the bot to open with (env BOT_CALL_CONTEXT)",
        default=os.environ.get("BOT_CALL_CONTEXT") or None,
    )
    parser.add_argument(
        "--call-profile",
        default=os.environ.get("BOT_CALL_PROFILE") or "agent",
        help="the siphon media profile the placed leg is anchored on (default agent)",
    )
    parser.add_argument(
        "--call-timeout",
        type=int,
        default=30,
        help="seconds the call may ring before siphon gives up on it (default 30)",
    )
    return parser.parse_args(argv)


def run(arguments: argparse.Namespace, observer_factory: ObserverFactory | None = None) -> None:
    """Run the bot with the parsed `arguments` until the process is stopped.

    `observer_factory` watches each call without this file being edited: the conversation harness
    passes one that writes what every call's pipeline heard, decided and said.
    """
    try:
        call = call_request_from_arguments(arguments)
        call_back_settings = call_back_settings_from_arguments(arguments)
    except ValueError as error:
        sys.exit(str(error))
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
    if control_plane is not None and call_back_settings is not None:
        control_plane.call_backs = CallBacks(control_plane, call_back_settings)
    try:
        status = asyncio.run(
            main(arguments.host, arguments.port, control_plane, observer_factory, call)
        )
    except KeyboardInterrupt:
        sys.exit(0)
    sys.exit(status)


if __name__ == "__main__":
    run(parse_arguments())
