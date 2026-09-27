"""The agent example, run with the conversation harness watching its pipeline.

The bot under test is `examples/agent_bot.py` as it ships, in the quickstart image. This adds one
observer per call, through the example's own hook, and the observer writes what the pipeline
heard, decided and said to standard output for the harness to read back.
"""

from __future__ import annotations

import agent_bot
from pipecat_siphon import SiphonFrameSerializer
from tools.conversation.call_events import CallEventObserver


def observe(serializer: SiphonFrameSerializer) -> CallEventObserver:
    """Build one call's observer, labelling its events with the Call-ID the serializer learns."""
    return CallEventObserver(call_id=lambda: serializer.call_id)


if __name__ == "__main__":
    agent_bot.run(agent_bot.parse_arguments(), observer_factory=observe)
