"""siphon-sip call script for the voice-AI quickstart: hand every inbound call to the bot.

This runs inside siphon's embedded interpreter against the `siphon` module that only exists in
that process, so it is not importable or type-checkable from this repo -- the same reason the
integration harness's script is excluded from mypy.

`call.handover` is answer-first: siphon sends the 200, the media engine anchors the leg and dials
the bot's WebSocket, and the already-connected call is handed to the control app. The bot then
owns it and can hang up, transfer or hold. Everything the media side needs is on the `agent`
profile in siphon.yaml, because `handover` takes a profile and a `ws_uri` and no media knobs.

If you are running the bot *without* `--control-url`, replace the handover with::

    answer_sdp = await rtpengine.answer_local(call, profile="agent", ws_uri=BOT_WS_URI)
    if answer_sdp is None:
        return                      # no encodable codec; auto-rejected 488
    call.answer(200, "OK", body=answer_sdp, content_type="application/sdp")

which bridges the audio identically. There is simply no control channel, so no hangup.
"""

import asyncio
import os

from siphon import b2bua, log

BOT_WS_URI = os.environ.get("BOT_WS_URI", "ws://127.0.0.1:9001/stream?call={call_id}")
"""Where the media engine dials for this call's audio.

`{call_id}` expands to the **SIP** Call-ID, which is also what arrives on the media socket in the
engine's `start` envelope. That is what lets the bot join its media session to its control channel
-- on `sip_call_id`, not on the control frame's `call_id`, which is siphon's internal UUID.
"""

CONTROL_APP = os.environ.get("CONTROL_APP", "agent-app")
"""Must match the `control.apps[].name` in siphon.yaml and the bot's `--control-app`."""


# No OPTIONS handler here. From siphon-sip 1.8.4 the stack answers an unclaimed OPTIONS itself,
# with an `Allow` built from the methods it actually supports — better than the hardcoded 200 this
# script used to send, and it applies to every deployment rather than the ones that remembered.
# `server.auto_options: false` turns it off for a node that would rather not confirm its own
# existence to a probe. Every other unhandled method now gets 405 rather than 500.


RING_SECONDS = float(os.environ.get("BOT_RING_SECONDS") or 0)
"""How long to ring before answering, in seconds. 0 answers immediately.

A line that picks up on the first ring sounds like a machine, which is a poor start for a bot that
is trying not to. The `180` goes on the wire the moment `progress()` is called -- it is not queued
with the actions that are, so this really is alerting rather than dead air -- and a provisional
stops the caller's INVITE retransmissions, so the wait costs nothing on the signalling side.

Awaited rather than slept: this handler is async, and blocking it would pin one of the script
executor's workers for the whole ring on every inbound call.
"""


@b2bua.on_invite
async def route(call):
    """Answer inbound provider calls into the bot; refuse everything else."""
    # Source-IP membership, so an INVITE from anywhere but the provider never reaches the bot.
    # On UDP treat this as a direction hint rather than authentication.
    if not call.from_gateway("provider"):
        log.warn(f"[{call.call_id}] INVITE from outside the provider group, refused")
        call.reject(403, "Forbidden")
        return

    if RING_SECONDS > 0:
        # Alerting only: a bare 180 with no SDP, so no early media path is opened and nothing is
        # anchored yet. The answer below is what anchors the media, exactly as before.
        call.progress(180, "Ringing")
        log.info(f"[{call.call_id}] ringing for {RING_SECONDS:g}s before answering")
        await asyncio.sleep(RING_SECONDS)

    call.handover(
        CONTROL_APP,
        answer=True,
        profile="agent",
        ws_uri=BOT_WS_URI,
        # If the bot's control connection drops mid-call, hang the caller up rather than leave
        # them on a leg nothing is listening to.
        on_lost="hangup",
    )
    log.info(f"[{call.call_id}] handed to {CONTROL_APP}, audio bridged to {BOT_WS_URI}")


@b2bua.on_bye
async def on_bye(call, initiator):
    """Log which side hung up: the caller, or the bot through the control plane."""
    log.info(f"[{call.call_id}] ended by {initiator.side}")


@b2bua.on_cancel
async def on_cancel(call):
    """Log a caller that gave up before the answer, which no other hook covers."""
    log.warn(f"[{call.call_id}] cancelled before answer")
