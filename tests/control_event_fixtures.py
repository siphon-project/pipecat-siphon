"""Control-plane events for a placed call, as siphon-sip publishes them.

Transcribed from siphon-sip's own source rather than from `outbound.py`'s mapping, so a test that
reads them checks our reading of the server and not our reading of ourselves. Each is the dict the
Python control SDK's `Call.next_event()` returns: `{"kind": <event>, "payload": <payload>}`.

- `ChannelStateChange` ringing / answered: `src/dispatcher/b2bua/originate.rs`, the provisional
  (`"state": phase, "code", "early_media"`) and the 2xx (`"state": "answered", "code", "sdp"`).
- `StasisEnd`: `src/control/registry/events.rs` `on_call_terminated_with_cause`, which puts
  `code` and `response` beside `reason` only when the teardown has them. The reasons are the
  callers' -- `"rejected"` from `originate.rs` on a final non-2xx, `"ring timeout"` from
  `timeouts.rs` through `originate_cancel.rs` (which always reports 487), `"media_failed"` from
  `originate.rs` when the anchor or the WebSocket dial fails on the 2xx, `"bye"` from `bye.rs`.
"""

from typing import Any

RINGING: dict[str, Any] = {
    "kind": "ChannelStateChange",
    "payload": {"state": "ringing", "code": 180, "early_media": False},
}

ANSWERED: dict[str, Any] = {
    "kind": "ChannelStateChange",
    "payload": {"state": "answered", "code": 200, "sdp": "v=0\r\n"},
}

BUSY: dict[str, Any] = {
    "kind": "StasisEnd",
    "payload": {"reason": "rejected", "code": 486, "response": "Busy Here"},
}

RING_TIMEOUT: dict[str, Any] = {
    "kind": "StasisEnd",
    "payload": {"reason": "ring timeout", "code": 487, "response": "Request Terminated"},
}

MEDIA_FAILED: dict[str, Any] = {
    "kind": "StasisEnd",
    "payload": {"reason": "media_failed", "code": 200, "response": "ws dial refused"},
}

FAR_END_HUNG_UP: dict[str, Any] = {"kind": "StasisEnd", "payload": {"reason": "bye"}}
