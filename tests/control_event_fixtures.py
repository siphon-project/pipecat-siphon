"""Control-plane events for a placed call, as siphon-sip publishes them.

Transcribed from siphon-sip's own source rather than from `outbound.py`'s mapping, so a test that
reads them checks our reading of the server and not our reading of ourselves. Each is the dict the
Python control SDK's `Call.next_event()` returns: `{"kind": <event>, "payload": <payload>}`.
The `StasisStart` payloads further down are a handed-over call's `payload`, in the shape
`build_stasis_payload` (`src/dispatcher/b2bua/handover.rs`) gives it.

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


def _handed_over(*headers: tuple[str, str]) -> dict[str, Any]:
    """Build a `StasisStart` payload, the shape `build_stasis_payload` gives a handed-over call.

    `src/dispatcher/b2bua/handover.rs`: the INVITE's headers as `[name, value]` pairs, names in the
    casing they arrived in and one pair per value, beside the source, transport and handover vars.
    """
    return {
        "invite": {
            "method": "INVITE",
            "ruri": "sip:agent@192.0.2.10",
            "headers": [
                ["Via", "SIP/2.0/UDP 192.0.2.20:5060;branch=z9hG4bK776asdhds"],
                ["From", '"Caller" <sip:+15550100@203.0.113.5>;tag=1928301774'],
                ["To", "<sip:agent@192.0.2.10>"],
                ["Call-ID", "a84b4c76e66710@pc33.example.invalid"],
                *[[name, value] for name, value in headers],
            ],
            "body": {"content_type": "application/sdp", "text": "v=0\r\n"},
        },
        "source_ip": "192.0.2.20",
        "source_port": 5060,
        "transport": "udp",
        "vars": {},
    }


ASSERTED_SIP: dict[str, Any] = _handed_over(
    ("P-Asserted-Identity", "<sip:+15550123@trunk.example.invalid;user=phone>")
)
"""A trunk's usual assertion: a SIP URI whose user part is the E.164 number."""

ASSERTED_TEL: dict[str, Any] = _handed_over(
    ("P-Asserted-Identity", '"Test Caller" <tel:+1-555-0124>')
)
"""A `tel:` URI with RFC 3966 visual separators and a display name."""

ASSERTED_TWICE: dict[str, Any] = _handed_over(
    ("P-Asserted-Identity", "<sip:alice@example.invalid>"),
    ("P-Asserted-Identity", "<tel:+15550125>"),
)
"""RFC 3325 allows one SIP and one tel identity, as separate header lines here."""

ASSERTED_IN_ONE_LINE: dict[str, Any] = _handed_over(
    ("P-Asserted-Identity", '"Doe, J" <sip:alice@example.invalid>, <tel:+15550126>')
)
"""The same two identities as one comma-separated value, a comma in the quoted display name."""

ASSERTED_LOWER_CASE: dict[str, Any] = _handed_over(
    ("p-asserted-identity", "<sip:+15550127@trunk.example.invalid>")
)
"""Header names are case-insensitive, and siphon passes on the casing it received."""

ASSERTED_BUT_WITHHELD: dict[str, Any] = _handed_over(
    ("P-Asserted-Identity", "<sip:+15550128@trunk.example.invalid;user=phone>"),
    ("Privacy", "id"),
)
"""The caller withheld their identity: the network asserts it, the bot must not act on it."""

NOT_ASSERTED: dict[str, Any] = _handed_over()
"""No P-Asserted-Identity at all: only the caller's own, unverified From."""
