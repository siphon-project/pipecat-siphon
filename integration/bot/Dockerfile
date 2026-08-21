# The pipecat bot the engine dials into.
#
# Build context is the repository root, so the package under test is installed from source
# rather than from PyPI: the harness has to exercise the working tree, not the last release.
FROM python:3.13-slim

WORKDIR /harness

# Installed first and on its own so the (slow) pipecat dependency resolution is a cached layer
# that survives every edit to the harness itself.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY integration/tools ./tools
COPY integration/bot ./bot

ENV PYTHONPATH=/harness
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/harness/bot/harness_bot.py"]
