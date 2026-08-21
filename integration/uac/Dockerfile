# The caller: SIPp to place the call and play the fixture, tshark to capture what comes back,
# and Python to generate the fixtures and analyse the result.
#
# One image for all four so the capture is taken inside the caller's own network namespace --
# what this container's interface saw is, by definition, what the caller received.
FROM python:3.13-slim

# wireshark-common asks whether non-root users may capture. This container runs as root, so the
# answer is no and the question must not be asked interactively or the build hangs.
RUN echo 'wireshark-common wireshark-common/install-setuid boolean false' | debconf-set-selections \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sip-tester \
        tshark \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# G.711 lives in audioop, which left the standard library in 3.13. Same package pipecat pulls in
# on the bot side, so both ends of the harness encode and decode with the same implementation --
# and neither of them wrote it.
RUN pip install --no-cache-dir "audioop-lts~=0.2.1"

WORKDIR /harness
COPY integration/tools ./tools
COPY integration/scenarios ./scenarios
COPY integration/uac/run_scenario.sh integration/uac/prepare.sh ./
RUN chmod +x ./run_scenario.sh ./prepare.sh

ENV PYTHONPATH=/harness
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/harness/run_scenario.sh"]
