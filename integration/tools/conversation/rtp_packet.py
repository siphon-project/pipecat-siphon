"""RTP (RFC 3550) for the scripted caller: the packets it sends, and the ones it reads back.

The caller sends only the fixed header, so that is all `pack_rtp` writes. What it reads comes from
the engine, and `parse_rtp` accepts the full header grammar -- contributing sources, a header
extension and padding -- because a parser that only understood what this caller writes would
misread a packet the day the engine sent a different, perfectly valid one.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

VERSION = 2
FIXED_HEADER_BYTES = 12

_FIXED_HEADER = struct.Struct("!BBHII")
_EXTENSION_HEADER = struct.Struct("!HH")


@dataclass(frozen=True)
class RtpHeader:
    """The RTP header fields the harness uses."""

    payload_type: int
    marker: bool
    sequence: int
    timestamp: int
    ssrc: int


def pack_rtp(
    *,
    payload_type: int,
    marker: bool,
    sequence: int,
    timestamp: int,
    ssrc: int,
    payload: bytes,
) -> bytes:
    """Build one RTP packet with a fixed header and no padding, CSRCs or extension.

    Raises:
        ValueError: When a field does not fit its width in the header.

    """
    _require_width("payload_type", payload_type, 0x7F)
    _require_width("sequence", sequence, 0xFFFF)
    _require_width("timestamp", timestamp, 0xFFFF_FFFF)
    _require_width("ssrc", ssrc, 0xFFFF_FFFF)
    first = VERSION << 6
    second = (0x80 if marker else 0) | payload_type
    return _FIXED_HEADER.pack(first, second, sequence, timestamp, ssrc) + payload


def parse_rtp(datagram: bytes) -> tuple[RtpHeader, bytes] | None:
    """Split a datagram into its header and payload, or None if it is not a well-formed packet."""
    if len(datagram) < FIXED_HEADER_BYTES:
        return None
    first, second, sequence, timestamp, ssrc = _FIXED_HEADER.unpack_from(datagram)
    if first >> 6 != VERSION:
        return None

    offset = FIXED_HEADER_BYTES + 4 * (first & 0x0F)
    if len(datagram) < offset:
        return None
    if first & 0x10:
        if len(datagram) < offset + _EXTENSION_HEADER.size:
            return None
        _, words = _EXTENSION_HEADER.unpack_from(datagram, offset)
        offset += _EXTENSION_HEADER.size + 4 * words
        if len(datagram) < offset:
            return None

    end = len(datagram)
    if first & 0x20:
        padding = datagram[-1] if end > offset else 0
        if padding == 0 or offset + padding > end:
            return None
        end -= padding

    header = RtpHeader(
        payload_type=second & 0x7F,
        marker=bool(second & 0x80),
        sequence=sequence,
        timestamp=timestamp,
        ssrc=ssrc,
    )
    return header, datagram[offset:end]


class RtpSender:
    """One outgoing RTP stream: a fixed SSRC and payload type, with sequence and timestamp."""

    def __init__(
        self,
        *,
        ssrc: int,
        payload_type: int,
        samples_per_frame: int,
        first_sequence: int = 0,
        first_timestamp: int = 0,
    ) -> None:
        """Start the stream at `first_sequence` and `first_timestamp`."""
        self._ssrc = ssrc
        self._payload_type = payload_type
        self._samples_per_frame = samples_per_frame
        self._sequence = first_sequence
        self._timestamp = first_timestamp

    def next_packet(self, payload: bytes, *, marker: bool = False) -> bytes:
        """Build the stream's next packet carrying `payload`; `marker` starts a talkspurt."""
        datagram = pack_rtp(
            payload_type=self._payload_type,
            marker=marker,
            sequence=self._sequence,
            timestamp=self._timestamp,
            ssrc=self._ssrc,
            payload=payload,
        )
        self._sequence = (self._sequence + 1) & 0xFFFF
        self._timestamp = (self._timestamp + self._samples_per_frame) & 0xFFFF_FFFF
        return datagram


def _require_width(name: str, value: int, maximum: int) -> None:
    if not 0 <= value <= maximum:
        raise ValueError(f"{name}={value} does not fit its field (0..{maximum})")
