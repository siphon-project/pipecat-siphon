"""siphon-sip call script for the conversation harness: hand every call to the agent bot.

This runs inside siphon's embedded interpreter against the `siphon` module that only exists in
that process, so, like the quickstart script it mirrors, it is excluded from mypy.

It is the quickstart's handover without the provider gate. The harness network is a private bridge
with one caller on it, and what is under test is the bot and the call flow, so the routing stays
the one call a deployment makes.
"""

import os

from siphon import b2bua, log

BOT_WS_URI = os.environ["BOT_WS_URI"]
"""Where the media engine dials for the call's audio. `{call_id}` expands to the SIP Call-ID."""

CONTROL_APP = os.environ.get("CONTROL_APP", "agent-app")
"""Must match `control.apps[].name` in siphon.yaml and the bot's `--control-app`."""


@b2bua.on_invite
async def route(call):
    """Answer, anchor the media to the bot's socket, and hand the call to its control app."""
    call.handover(
        CONTROL_APP,
        answer=True,
        profile="agent",
        ws_uri=BOT_WS_URI,
        # A bot whose control connection drops hangs the caller up, and the run fails on it,
        # rather than leaving a call nobody controls.
        on_lost="hangup",
    )
    log.info(f"[{call.call_id}] handed to {CONTROL_APP}, audio bridged to {BOT_WS_URI}")


@b2bua.on_bye
async def on_bye(call, initiator):
    """Log which side hung up, for the proxy log a failed run leaves behind."""
    log.info(f"[{call.call_id}] ended by {initiator.side}")
