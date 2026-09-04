"""siphon-sip call script for the integration harness: answer, and bridge the leg to a bot.

`rtpengine.answer_local` is the single-leg answer verb. The engine picks a codec out of the
caller's offer, synthesises the RFC 3264 answer itself, and dials the WebSocket bot — all inside
that one control command. If the dial fails the engine tears the half-built call down and the
command errors, so reaching `call.answer` below already means the socket is up. There is no B
leg and there never will be one; the bot is the far end of the call.

The scenario is selected by the request URI's user part, so one proxy serves all three. Each maps
to its own media profile (they differ in whether the engine's VAD and barge-in are on, and in
whether the WebSocket wire rate is negotiated away from the codec's) and its own bot, whose
address comes from the environment because under compose the engine's loopback is not the bot's.
"""

import os

from siphon import b2bua, log, proxy, rtpengine

# name -> (media profile, bot WebSocket URI). Both URIs are environment-driven: the compose file
# is the only place addresses are assigned.
SCENARIOS = {
    "roundtrip": (
        "roundtrip",
        os.environ.get("ROUNDTRIP_WS_URI", "ws://127.0.0.1:9001/stream?call={call_id}"),
    ),
    "turntaking": (
        "turntaking",
        os.environ.get("TURNTAKING_WS_URI", "ws://127.0.0.1:9002/stream?call={call_id}"),
    ),
    "wideband": (
        "wideband",
        os.environ.get("WIDEBAND_WS_URI", "ws://127.0.0.1:9003/stream?call={call_id}"),
    ),
}


@proxy.on_request("OPTIONS")
def health(request):
    """Answer the readiness probe so compose can gate on the SIP stack being up."""
    request.reply(200, "OK")


@b2bua.on_invite
async def answer_into_the_bot(call):
    """Answer the call locally and bridge its audio to the scenario's bot."""
    scenario_name = call.ruri.user if call.ruri else None
    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        log.warn(f"[{call.call_id}] no scenario named {scenario_name!r}")
        call.reject(404, "Not Found")
        return

    profile, ws_uri = scenario
    try:
        answer_sdp = await rtpengine.answer_local(call, profile=profile, ws_uri=ws_uri)
    except RuntimeError as error:
        # The engine refused: an unreachable bot, a codec it cannot encode, a control timeout.
        # Say so loudly and fail the call rather than answering into a bridge that is not there.
        log.error(f"[{call.call_id}] answer_local failed for {profile}: {error}")
        call.reject(503, "Service Unavailable")
        return

    if answer_sdp is None:
        log.warn(f"[{call.call_id}] no encodable codec in the offer, rejected 488")
        return

    call.answer(200, "OK", body=answer_sdp, content_type="application/sdp")
    log.info(f"[{call.call_id}] answered {profile}, audio bridged to {ws_uri}")


@b2bua.on_bye
async def on_bye(call, initiator):
    """Log which side hung up. The dispatcher deletes the media session by itself."""
    log.info(f"[{call.call_id}] ended by {initiator.side}")


@b2bua.on_cancel
async def on_cancel(call):
    """Log a caller that gave up before the answer, which no other hook covers."""
    log.warn(f"[{call.call_id}] cancelled before answer")


@rtpengine.on_media_timeout
def on_media_timeout(call_id, from_tag):
    """Log the engine reaping a call whose RTP went dead. Never expected in a passing run."""
    log.warn(f"[{call_id}] engine reaped the call on media timeout (tag {from_tag})")
