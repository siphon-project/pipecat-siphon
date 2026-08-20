#!/bin/sh
# Run one scenario end to end from the caller's side: capture, call, wait, analyse.
#
# The capture is taken on this container's own interface, so it contains exactly the packets the
# caller sent and received -- no bridge-wide capture to filter, no question about which side of
# a NAT it was taken on.
#
# Nothing here sleeps for a fixed period. Each step waits on the thing it actually needs (the
# capture file opening, the bot recording the disconnect, the engine's session gauge dropping)
# with a bounded timeout, so the run is as fast as the machine allows and fails with a message
# saying what never happened.
set -eu

SCENARIO="${1:?usage: run_scenario.sh <roundtrip|turntaking>}"
ARTIFACTS="${ARTIFACTS_DIR:-/harness/artifacts}"
PROXY="${PROXY_ADDRESS:?PROXY_ADDRESS is required}"
LOCAL="${UAC_ADDRESS:?UAC_ADDRESS is required}"
METRICS="${ENGINE_METRICS_URL:?ENGINE_METRICS_URL is required}"
MEDIA_PORT="${MEDIA_PORT:-6000}"
CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-eth0}"

# The capture and the fixture are different files with different owners. tshark drops
# privileges before it creates the capture file, so it cannot write into the bind-mounted
# artifacts directory (which belongs to the host user); it captures to container-local storage
# and the file is copied out when the capture stops, including on failure.
CAPTURE="/tmp/${SCENARIO}-capture.pcap"
KEPT_CAPTURE="${ARTIFACTS}/${SCENARIO}-capture.pcap"
TRACE="${ARTIFACTS}/${SCENARIO}-bot.jsonl"

mkdir -p "${ARTIFACTS}"
rm -f "${CAPTURE}" "${KEPT_CAPTURE}"

capture_pid=""
stop_capture() {
    if [ -n "${capture_pid}" ] && kill -0 "${capture_pid}" 2>/dev/null; then
        # SIGTERM lets tshark flush and close the file rather than leaving it truncated, which
        # matters most in exactly the case we care about: a failed run.
        kill -TERM "${capture_pid}" 2>/dev/null || true
        wait "${capture_pid}" 2>/dev/null || true
    fi
    if [ -f "${CAPTURE}" ]; then
        cp -f "${CAPTURE}" "${KEPT_CAPTURE}"
    fi
}
trap stop_capture EXIT INT TERM

echo "== ${SCENARIO}: capturing on ${CAPTURE_INTERFACE} =="
tshark -i "${CAPTURE_INTERFACE}" -f "udp" -w "${CAPTURE}" \
    >"${ARTIFACTS}/${SCENARIO}-tshark.log" 2>&1 &
capture_pid=$!

# A pcap file header is 24 bytes and is written as soon as the capture is live. Waiting for it
# is waiting for the capture to actually be running, which a sleep would only guess at.
python3 -m tools.wait_for --timeout 30 file "${CAPTURE}" --min-bytes 24

echo "== ${SCENARIO}: placing the call =="
set +e
sipp "${PROXY}:5060" \
    -sf "/harness/scenarios/${SCENARIO}_uac.xml" \
    -i "${LOCAL}" \
    -mp "${MEDIA_PORT}" \
    -m 1 -l 1 \
    -timeout 60 -timeout_error \
    -trace_err -error_file "${ARTIFACTS}/${SCENARIO}-sipp-errors.log" \
    -trace_msg -message_file "${ARTIFACTS}/${SCENARIO}-sipp-messages.log" \
    -trace_screen -screen_file "${ARTIFACTS}/${SCENARIO}-sipp-screen.log" \
    -nostdin
sipp_status=$?
set -e
echo "sipp exited ${sipp_status}"

# The bot records the disconnect when the engine drops the socket, which is the last thing that
# happens in a torn-down call. Waiting for it is how we know the trace is complete before
# reading it, without a race and without a sleep.
python3 -m tools.wait_for --timeout 30 trace "${TRACE}" engine_disconnected || true

# The engine deletes the session asynchronously after the BYE. Give it a bounded window to reach
# zero before the analyser reads the gauge, so a slow delete reads as slow rather than as a leak.
python3 -m tools.wait_for --timeout 30 metric "${METRICS}" siphon_rtp_sessions --equals 0 || true

stop_capture
capture_pid=""

echo "== ${SCENARIO}: analysing =="
set +e
python3 -m tools.analyse "${SCENARIO}" --artifacts "${ARTIFACTS}" --metrics-url "${METRICS}"
analysis_status=$?
set -e

# Artifacts are written by root inside the container; hand them back so the host can read them.
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${ARTIFACTS}" 2>/dev/null || true
fi

if [ "${sipp_status}" -ne 0 ]; then
    echo "FAIL: sipp exited ${sipp_status} (see ${SCENARIO}-sipp-errors.log)"
    exit "${sipp_status}"
fi
exit "${analysis_status}"
