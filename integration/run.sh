#!/usr/bin/env bash
#
# One command: bring the four-way stack up, run both scenarios against it, tear it down.
#
#   ./run.sh                        # build if needed, run both scenarios, tear down
#   ./run.sh --scenario roundtrip   # one scenario, for iterating
#   ./run.sh --keep-up              # leave the stack running afterwards
#   ./run.sh --no-build             # reuse the images exactly as they are
#
# siphon-sip and siphon-rtp are built from the checkouts beside this repository. Point
# SIPHON_SIP_PATH / SIPHON_RTP_PATH somewhere else if yours are not there.
#
# Everything a failed run needs is left in ./artifacts: the captures, both bot traces, the
# control-plane transcript, SIPp's message and error logs, and every container's log.
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE=(docker compose -f docker-compose.yaml)
SERVICES=(engine control-tap proxy bot-echo bot-speaker)
ARTIFACTS="$(pwd)/artifacts"

scenarios=(roundtrip turntaking)
build=1
keep_up=0
clean=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario) scenarios=("$2"); shift 2 ;;
        --no-build) build=0; shift ;;
        --keep-up) keep_up=1; shift ;;
        --no-clean) clean=0; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

collect_logs() {
    for service in "${SERVICES[@]}"; do
        "${COMPOSE[@]}" logs --no-color --timestamps "$service" \
            > "${ARTIFACTS}/${service}.log" 2>&1 || true
    done
    chmod -R u+rw "${ARTIFACTS}" 2>/dev/null || true
}

teardown() {
    collect_logs
    if [[ "${keep_up}" -eq 0 ]]; then
        "${COMPOSE[@]}" down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    else
        echo
        echo "stack left up (--keep-up). Tear it down with:"
        echo "  docker compose -f $(pwd)/docker-compose.yaml down --remove-orphans"
    fi
}

mkdir -p "${ARTIFACTS}"
if [[ "${clean}" -eq 1 ]]; then
    # A run that was killed before its containers could hand the artifacts back leaves
    # root-owned files behind. Say so rather than dying on a permission error five lines into a
    # script whose job is to be diagnosable.
    find "${ARTIFACTS}" -mindepth 1 -not -name .gitignore -delete 2>/dev/null || true
    leftovers="$(find "${ARTIFACTS}" -mindepth 1 -not -name .gitignore | wc -l)"
    if [[ "${leftovers}" -gt 0 ]]; then
        echo "warning: ${leftovers} artifact(s) from an earlier run could not be removed;"
        echo "         they are probably root-owned. sudo rm -rf ${ARTIFACTS}/* to clear them."
    fi
fi

# A stack left behind by an interrupted run would keep the static addresses and fail the
# bring-up below with a confusing conflict, so start from a known-clean state either way.
"${COMPOSE[@]}" down --remove-orphans --timeout 5 >/dev/null 2>&1 || true

if [[ "${build}" -eq 1 ]]; then
    echo "== building (cached layers make this quick after the first run) =="
    # The Rust builds are the long pole and a fresh checkout building jemalloc-sys at full
    # parallelism is a known flake, so cap the job count rather than letting cargo pick.
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}" "${COMPOSE[@]}" build "${SERVICES[@]}" uac
fi

trap teardown EXIT

echo "== starting the stack =="
"${COMPOSE[@]}" up -d "${SERVICES[@]}"

"${COMPOSE[@]}" run --rm --entrypoint /harness/prepare.sh \
    -e ENGINE_READY_URL=http://172.28.7.10:9091/readyz \
    -e PROXY_READY_URL=http://172.28.7.20:8081/admin/ready \
    -e CONTROL_TAP_HOST=172.28.7.15 -e CONTROL_TAP_PORT=8080 \
    -e ROUNDTRIP_BOT_HOST=172.28.7.31 -e TURNTAKING_BOT_HOST=172.28.7.32 \
    -e BOT_PORT=9001 \
    uac

status=0
declare -a results=()
for scenario in "${scenarios[@]}"; do
    echo
    echo "======================================================================"
    echo " scenario: ${scenario}"
    echo "======================================================================"
    if "${COMPOSE[@]}" run --rm uac "${scenario}"; then
        results+=("PASS ${scenario}")
    else
        results+=("FAIL ${scenario}")
        status=1
    fi
done

echo
echo "======================================================================"
for result in "${results[@]}"; do
    echo " ${result}"
done
echo " artifacts: ${ARTIFACTS}"
echo "======================================================================"
exit "${status}"
