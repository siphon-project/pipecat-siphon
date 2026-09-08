# The quickstart bot, containerised.
#
# Build context is the repository root, so the serializer is installed from source rather than
# from PyPI: what runs is the working tree, not the last release.

# 3.14 is not a free choice. `siphon-control` is a Rust extension and the only wheel it publishes
# is cp314: on 3.13 pip falls back to a source build, and `slim` has no linker, so the image fails
# to build with a wall of `linker \`cc\` not found`. Moving this back to match the CI matrix is the
# obvious-looking edit and it breaks the build. Add a Rust toolchain first, or leave it here.
FROM python:3.14-slim

WORKDIR /app

# The package first and on its own, so the (slow) pipecat dependency resolution is a cached layer
# that survives every edit to the example.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# `siphon-control` is the control-plane client that lets the bot hang up. See the interpreter note
# above for why it pins the base image. Dropping it only costs the hangup — everything else works.
#
# The `websocket` extra carries fastapi and `uvicorn` is the server it runs on: the media socket is
# served per call, so the process needs a real HTTP server that keeps accepting while calls are in
# flight. pipecat's single-client WebSocket transport can only hold one call at a time, which for a
# phone number is a hard ceiling of one. These are the same two lines as the `dev` extra in
# pyproject.toml, and they should stay the same two.
RUN pip install --no-cache-dir . \
        "pipecat-ai[anthropic,deepgram,cartesia,websocket]" \
        "uvicorn>=0.32,<1" \
        siphon-control

COPY examples ./examples

ENV PYTHONUNBUFFERED=1

# Unprivileged: this talks to three vendors and answers a socket, and needs nothing from root.
RUN useradd --system --create-home --uid 10001 bot
USER bot

# The media WebSocket the engine dials.
EXPOSE 9001

ENTRYPOINT ["python3", "examples/agent_bot.py"]
