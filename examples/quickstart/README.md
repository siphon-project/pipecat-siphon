# Quickstart: a real phone number talking to Claude

Call a number your SIP provider gave you and have Claude answer, with the bot able to hang up
when the conversation is done.

Three processes and one config file. Nothing here is provider-specific — registration is
RFC 3261, so any ITSP that issues SIP credentials works the same way.

```
your phone -> provider -> siphon-sip -> siphon-rtp --(media WebSocket)--> agent_bot.py
                              |                                          STT -> Claude -> TTS
                              +---------(control WebSocket)-------------> hangup
```

## What you need

- **SIP credentials** from any provider that terminates calls to a registered account: an
  address-of-record, a registrar host, a username and a password.
- **A reachable address.** The provider has to be able to send you an INVITE. On a VPS that is
  its public IP; behind a home NAT you need a port forward for 5060 and the RTP range, and
  `advertised_address` set to the public address.
- **Three API keys**: Anthropic, Deepgram, Cartesia.
- **siphon-sip** and **siphon-rtp**, and Python 3.11+.

## 1. Install

```sh
pip install pipecat-siphon "pipecat-ai[anthropic,deepgram,cartesia]" siphon-control
```

`siphon-control` is the bot's control-plane client. It is a Rust extension: there are wheels for
common interpreters, and on anything else pip builds it, which needs a Rust toolchain. Leave it
out if you do not want the bot to be able to hang up — everything else still works.

## 2. Set the environment

```sh
# The provider's account details, exactly as they issued them.
export SIP_AOR="sip:12345@sip.provider.example"
export SIP_REGISTRAR="sip:sip.provider.example"
export SIP_USERNAME="12345"
export SIP_PASSWORD="..."
export SIP_PROVIDER_HOST="sip.provider.example"

# The networks the provider sends calls from. They publish these; the default matches nothing.
export SIP_PROVIDER_NETWORKS="203.0.113.0/24"

# This host, as the provider can reach it.
export PUBLIC_ADDRESS="203.0.113.10"

# Shared between siphon.yaml and the bot. Any random string.
export AGENT_APP_TOKEN="$(openssl rand -hex 16)"
export SIPHON_CONTROL_TOKEN="$AGENT_APP_TOKEN"

# The AI vendors.
export ANTHROPIC_API_KEY="..."
export DEEPGRAM_API_KEY="..."
export CARTESIA_API_KEY="..."
export CARTESIA_VOICE_ID="..."
```

No credential is written into the tree: `siphon.yaml` reads all of these through `${VAR}`
expansion.

## 3. Run the three processes

```sh
# 1. the media engine
siphon-rtp --control 127.0.0.1:8080 --relay-bind-ip "$PUBLIC_ADDRESS"

# 2. the bot
python examples/agent_bot.py --host 127.0.0.1 --port 9001 \
    --control-url ws://127.0.0.1:9092/control/ws

# 3. the SIP stack, which registers on startup
siphon -c examples/quickstart/siphon.yaml
```

Order does not matter. The bot retries its control connection with backoff, so starting it before
siphon is a routine race and not a failure.

## 4. Check it before you dial

```sh
python examples/probe.py ws://127.0.0.1:9001/stream
```

This speaks the engine's half of the media wire and reports whether the bot answers with audio.
It needs no phone, no SIP stack and no engine, so it separates "the AI is broken" from "the media
path is broken" in a couple of seconds:

```
start sent: stream probe-1, 16000 Hz, 20 ms ptime
first audio frame after 840 ms
3.0 s elapsed: 118 audio frames in, 2360 ms of audio, 1 control message
```

Audio coming back means the whole STT → Claude → TTS path works and the greeting fires. Nothing
coming back means the problem is above the media path, and the bot's own log says which vendor.

Then check the registration succeeded — siphon logs it at startup — and call your number.

## Things that go wrong

**The registration succeeds and calls never arrive.** Almost always `advertised_address`: the
provider is sending INVITEs to whatever this node put in its Contact. A private address there
registers perfectly and receives nothing.

**The call connects and the bot hears silence.** That is `received_from`. A caller behind NAT
advertises an unroutable address in its SDP, the engine's ingress gate defaults to an exact match
on it, and every packet from the handset is dropped — with clean signalling from end to end. The
profile in `siphon.yaml` sets it; the engine's built-in `voice_ai` profile does not.

**The bot interrupts itself.** Echo of its own voice coming back through the handset is being read
as the caller speaking. Both halves of the fix are in the profile: `echo_cancellation` (a speech
detector cannot reject echo — echo of speech is speech) and `ws_vad_engine: neural` with
`ws_vad_min_speech_ms`.

**The bot answers but you hear nothing.** Check `ws_sample_rate` in the profile equals
`PIPELINE_SAMPLE_RATE` in `agent_bot.py`, and that it is 8000 or 16000 — the canceller and the
noise suppressor exist only at those two rates.

**Calls stop being answered after a few minutes idle.** You are on an older copy of the bot.
Pipecat's `PipelineWorker` cancels itself and its runner after 300 s of idle by default, which
stops the WebSocket server listening; `agent_bot.py` sets `idle_timeout_secs=None`.
