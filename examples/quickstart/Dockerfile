# The quickstart bot, containerised.
#
# Build context is the repository root, so the serializer is installed from source rather than
# from PyPI: what runs is the working tree, not the last release.

# ── siphon-control, built from its sdist ─────────────────────────────────────────────────────────
# 3.13 is not a free choice, and neither is this stage. `siphon-control` publishes wheels only for
# CPython 3.14 (its own runtime target is free-threaded 3.14t, where there is no abi3), while
# kokoro-onnx -- which the local backend needs -- declares support for nothing newer than 3.13. No
# single interpreter has a wheel for both, so the control-plane client is compiled here from the
# sdist it publishes alongside those wheels.
#
# Temporary. Once siphon-control ships wheels for 3.13 this stage goes, and the install below is a
# plain `siphon-control` again. Debian's own rustc and cargo are recent enough for the crate, and
# its only native dependency is `ring`, which needs nothing past a C compiler.
FROM python:3.13-slim AS siphon-control

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        rustc \
        cargo \
    && rm -rf /var/lib/apt/lists/*

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels siphon-control==0.5.0

# ── The bot ──────────────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

# Which models the image carries. `cloud` installs the three vendor clients; `local` installs
# faster-whisper and Kokoro instead, plus the two CUDA libraries faster-whisper loads on a GPU. The
# LLM is in neither: on the local path it is the separate `llm` container in the compose file.
ARG BOT_BACKEND=cloud

# The pipecat extras to install, when the two presets above are not the mix you want. The bot picks
# its provider per role at runtime (`BOT_LLM_PROVIDER`, `BOT_STT_PROVIDER`, `BOT_TTS_PROVIDER`), and
# an image can only answer for the libraries it carries, so a mixed selection needs its own list:
#
#   --build-arg BOT_EXTRAS=google,whisper,websocket    # Gemini's model, a local recognizer
#
# One extra per provider: anthropic, google, openai, deepgram, cartesia, whisper, kokoro. `websocket`
# is not a provider and is always needed -- it carries the fastapi the media socket is served on.
ARG BOT_EXTRAS=

WORKDIR /app

# The package first and on its own, so the (slow) pipecat dependency resolution is a cached layer
# that survives every edit to the example.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY --from=siphon-control /wheels /wheels

# `siphon-control` is the control-plane client that lets the bot hang up. See the stage above for
# why it is a local wheel. Dropping it only costs the hangup — everything else works.
#
# The `websocket` extra carries fastapi and `uvicorn` is the server it runs on: the media socket is
# served per call, so the process needs a real HTTP server that keeps accepting while calls are in
# flight. pipecat's single-client WebSocket transport can only hold one call at a time, which for a
# phone number is a hard ceiling of one. These are the same two lines as the `dev` extra in
# pyproject.toml, and they should stay the same two.
#
# The CUDA runtime arrives as NVIDIA's own pip wheels rather than an nvidia/cuda base image:
# ctranslate2 needs only cuBLAS and cuDNN 9 from CUDA, not a toolkit. The driver is in neither
# image. The NVIDIA container runtime injects it at start.
RUN if [ -n "$BOT_EXTRAS" ]; then extras="$BOT_EXTRAS"; \
    else \
        case "$BOT_BACKEND" in \
            cloud) extras="anthropic,deepgram,cartesia,websocket" ;; \
            local) extras="whisper,kokoro,websocket" ;; \
            *) echo "unknown BOT_BACKEND: $BOT_BACKEND (cloud, local)" >&2; exit 2 ;; \
        esac \
    fi \
    # Keyed on the recognizer actually installed rather than on the preset, so a mixed image that
    # carries faster-whisper still gets the libraries it loads on a GPU.
    && case "$extras" in \
        *whisper*) cuda="nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*" ;; \
        *) cuda="" ;; \
    esac \
    && pip install --no-cache-dir . \
        "pipecat-ai[$extras]" \
        "uvicorn>=0.32,<1" \
        /wheels/siphon_control-*.whl \
        $cuda \
    && rm -rf /wheels

# Where those wheels put cuBLAS and cuDNN. ctranslate2 opens them through the dynamic loader rather
# than through Python, so without this path the local path's startup warm-up fails on libcudnn.
# Inert on the cloud path, where the directories do not exist.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib

COPY examples ./examples

ENV PYTHONUNBUFFERED=1

# Unprivileged: this talks to three vendors and answers a socket, and needs nothing from root. The
# cache directory is created here, owned by the bot, because a named volume mounted over it takes
# its ownership from the image -- left to Docker it is root's, and the first model download fails.
RUN useradd --system --create-home --uid 10001 bot \
    && mkdir -p /home/bot/.cache \
    && chown bot /home/bot/.cache
USER bot

# The media WebSocket the engine dials.
EXPOSE 9001

ENTRYPOINT ["python3", "examples/agent_bot.py"]
