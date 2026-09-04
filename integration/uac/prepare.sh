#!/bin/sh
# Gate the run on every component actually being up, then generate the fixtures.
#
# Run inside the harness network (the UAC image, entrypoint overridden) so the probes go over
# the same network the call will. Each probe is a real readiness signal -- an HTTP 200, an
# accepted TCP connection -- with a bounded timeout, never a sleep. The bot probes are the
# important ones: the engine dials the WebSocket from inside the answer command and tears the
# call down if the dial is refused, so a bot that is not listening yet does not produce a
# late-connecting call, it produces a failed one.
set -eu

ARTIFACTS="${ARTIFACTS_DIR:-/harness/artifacts}"

echo "== waiting for the stack =="
python3 -m tools.wait_for --timeout 120 http "${ENGINE_READY_URL:?}"
python3 -m tools.wait_for --timeout 60 port "${CONTROL_TAP_HOST:?}" "${CONTROL_TAP_PORT:?}"
python3 -m tools.wait_for --timeout 120 http "${PROXY_READY_URL:?}"
python3 -m tools.wait_for --timeout 120 port "${ROUNDTRIP_BOT_HOST:?}" "${BOT_PORT:?}"
python3 -m tools.wait_for --timeout 120 port "${TURNTAKING_BOT_HOST:?}" "${BOT_PORT:?}"
python3 -m tools.wait_for --timeout 120 port "${WIDEBAND_BOT_HOST:?}" "${BOT_PORT:?}"

echo "== checking the fixtures carry what the analyser looks for =="
python3 -m tools.make_fixtures --self-test

echo "== generating the fixtures =="
python3 -m tools.make_fixtures "${ARTIFACTS}"

if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${ARTIFACTS}" 2>/dev/null || true
fi
