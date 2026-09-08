# Quickstart: a real phone number talking to Claude

Call a number your SIP provider gave you and have Claude answer, with the bot able to hang up
when the conversation is done.

Nothing here is provider-specific. Registration is RFC 3261, so any ITSP that issues SIP
credentials is driven by the same handful of values.

```
your phone -> provider -> siphon-sip -> siphon-rtp --(media WebSocket)--> agent_bot.py
                              |                                          STT -> Claude -> TTS
                              +---------(control WebSocket)-------------> hangup
```

## Lift off

Docker, a SIP account, and three API keys. That is the whole list.

```sh
git clone https://github.com/siphon-project/pipecat-siphon
cd pipecat-siphon/examples/quickstart
cp .env.example .env
${EDITOR:-vi} .env
docker compose up --build
```

`.env` opens with one block marked "fill these in": the three vendor keys plus a Cartesia voice,
the SIP credentials your provider issued, the address they can reach this host on, and one random
string for the control plane (`openssl rand -hex 16`). Everything after that block already has a
working default.

Three containers come up: the media engine, the bot, and the SIP stack. siphon logs the
registration at startup, and once it succeeds, call your number.

The single most common mistake is `PUBLIC_ADDRESS`. It has to be the address the provider can
actually reach, because it is what goes into this node's Contact. A private address there
registers perfectly and receives nothing at all.

### Check it before you dial

```sh
docker compose run --rm --no-deps --entrypoint python3 bot examples/probe.py \
    ws://127.0.0.1:9001/stream
```

This speaks the engine's half of the media wire and reports whether the bot answers with audio. It
needs no phone, no SIP stack and no engine, so it separates "the AI is broken" from "the media path
is broken" in a couple of seconds:

```
start sent: stream probe-1, 16000 Hz, 20 ms ptime
first audio frame after 840 ms
3.0 s elapsed: 118 audio frames in, 2360 ms of audio, 1 control message
```

Audio coming back means the whole STT -> Claude -> TTS path works and the greeting fires. Nothing
coming back means the problem is above the media path, and the bot's own log names the vendor.

### Why it runs on the host network

Not laziness. A registered account only works if the provider can reach the address this node puts
in its Contact, and the media only works if it can reach the RTP range the engine binds. Under a
bridge network the SIP side survives with `advertised_address`, but the media range would need
every port in `RTP_PORT_MIN..RTP_PORT_MAX` published, and publishing a thousand UDP ports costs a
userland proxy process per port.

The cost is a shared port space. Every port is therefore an environment variable: if something
already owns 5060, move this stack rather than fighting it.

### Changing the bot

The persona, the transfer destination and all three model choices are read from the environment,
so the common edits are a line in `.env` and a restart of one container:

```sh
echo 'BOT_SYSTEM_PROMPT=You are the front desk of a bike shop. Keep it to a sentence.' >> .env
docker compose up -d bot
```

`agent_bot.py` is copied into the image rather than bind-mounted, so an edit to the Python itself
needs `docker compose up --build bot`.

## Or run the three processes by hand

Useful when you are editing the bot, or when you already run siphon-sip and siphon-rtp as
services. You need Python 3.11+ and both siphon binaries on `PATH`.

### 1. Install

```sh
pip install pipecat-siphon \
    "pipecat-ai[anthropic,deepgram,cartesia,websocket]" \
    uvicorn \
    siphon-control
```

`fastapi` (through pipecat's `websocket` extra) and `uvicorn` are what the bot serves the per-call
media socket on. It answers one WebSocket per call rather than holding a single connection, which
is what lets one process take more than one call at a time.

`siphon-control` is the bot's control-plane client, and the only part of this that is fussy about
your interpreter. It is a Rust extension whose published wheel is currently cp314 only, so on
Python 3.14 it installs in a second and on anything older pip builds it from source, which needs a
Rust toolchain and a C linker. Two ways out if you are not on 3.14: install a toolchain, or leave
`siphon-control` off entirely and lose only the bot's ability to hang up and transfer. The
container path sidesteps this by running 3.14.

### 2. Set the environment

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

# The routing script, which siphon.yaml otherwise looks for at the container path.
export ROUTE_SCRIPT="examples/quickstart/route.py"

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

### 3. Run the three processes

```sh
# 1. the media engine
siphon-rtp --control 127.0.0.1:8080 \
    --relay-bind-ip 0.0.0.0 \
    --advertise-ip "$PUBLIC_ADDRESS" \
    --port-min 30000 --port-max 30999

# 2. the bot
python examples/agent_bot.py --host 127.0.0.1 --port 9001 \
    --control-url ws://127.0.0.1:9092/control/ws

# 3. the SIP stack, which registers on startup
siphon -c examples/quickstart/siphon.yaml
```

`--relay-bind-ip` is where the media sockets actually bind and `--advertise-ip` is what goes into
the SDP. They are the same value only on a host with no NAT in front of it; keeping them separate
is what makes the NATed case work at all.

Order does not matter. The bot retries its control connection with backoff, so starting it before
siphon is a routine race and not a failure.

### 4. Check it before you dial

```sh
python examples/probe.py ws://127.0.0.1:9001/stream
```

Same probe as above, reading the same way.

## Things that go wrong

**The registration succeeds and calls never arrive.** Almost always `advertised_address`: the
provider is sending INVITEs to whatever this node put in its Contact. A private address there
registers perfectly and receives nothing.

**The call connects and the bot hears silence.** That is `received_from`. A caller behind NAT
advertises an unroutable address in its SDP, the engine's ingress gate defaults to an exact match
on it, and every packet from the handset is dropped -- with clean signalling from end to end. The
profile in `siphon.yaml` sets it; the engine's built-in `voice_ai` profile does not.

**The bot interrupts itself.** Echo of its own voice coming back through the handset is being read
as the caller speaking. Both halves of the fix are in the profile: `echo_cancellation` (a speech
detector cannot reject echo -- echo of speech is speech) and `ws_vad_engine: neural` with
`ws_vad_min_speech_ms`. If it still happens with both set, run the engine at
`ENGINE_LOG=info,siphon_rtp::media=debug` and read its delay report: a weak-lock warning means the
canceller is aligned on the wrong offset, and `echo_long_tail` in the profile is the answer rather
than a wider search.

**The bot answers but you hear nothing.** Check `ws_sample_rate` in the profile equals
`BOT_SAMPLE_RATE` (16000 by default), and that it is 8000 or 16000 -- the canceller and the noise
suppressor exist only at those two rates.

**Calls stop being answered after a few minutes idle.** You are on an older copy of the bot.
Pipecat's `PipelineWorker` cancels itself and its runner after 300 s of idle by default, which
stops the WebSocket server listening; `agent_bot.py` sets `idle_timeout_secs=None`.

**The second concurrent call is refused.** You are on an older copy of the bot too. The
single-client WebSocket transport holds one connection for the life of the process, which for a
phone number is a ceiling of one; the bot now serves a WebSocket per call.
