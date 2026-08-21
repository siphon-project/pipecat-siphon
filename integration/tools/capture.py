"""Read RTP out of a capture using tshark, so the decoder that reads it is not ours.

Wireshark is a genuinely independent implementation: it did not learn the RTP header layout
from the same code that wrote it, and it sets its RTP conversations up from the SDP the two
peers actually exchanged rather than from anything the harness told it. If tshark says a packet
is RTP with payload type 8, sequence *n* and a 160-byte payload, that is a third party agreeing
with the engine about the wire -- which a parser written next to the assertion could not give.

Only the payload *bytes* come back here. Decoding A-law and measuring the spectrum happens in
:mod:`tools.signals`, because tshark has no way to tell us what the audio sounds like.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["RtpPacket", "TsharkError", "read_rtp"]

_FIELDS = (
    "frame.number",
    "frame.time_relative",
    "ip.src",
    "ip.dst",
    "udp.srcport",
    "udp.dstport",
    "rtp.p_type",
    "rtp.seq",
    "rtp.timestamp",
    "rtp.ssrc",
    "rtp.payload",
)


class TsharkError(RuntimeError):
    """Raised when tshark is missing or exits non-zero."""


@dataclass(frozen=True, slots=True)
class RtpPacket:
    """One RTP packet as tshark dissected it."""

    number: int
    time: float
    source_address: str
    destination_address: str
    source_port: int
    destination_port: int
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes


def _parse_row(row: str) -> RtpPacket | None:
    """Turn one tshark field row into a packet, or ``None`` if it is not a usable RTP row."""
    parts = row.split("\t")
    if len(parts) != len(_FIELDS):
        return None
    (
        number,
        time_relative,
        source_address,
        destination_address,
        source_port,
        destination_port,
        payload_type,
        sequence,
        timestamp,
        ssrc,
        payload,
    ) = parts
    if not payload_type or not sequence or not payload:
        return None
    try:
        return RtpPacket(
            number=int(number),
            time=float(time_relative),
            source_address=source_address,
            destination_address=destination_address,
            source_port=int(source_port),
            destination_port=int(destination_port),
            payload_type=int(payload_type),
            sequence=int(sequence),
            timestamp=int(timestamp) if timestamp else 0,
            ssrc=int(ssrc, 16) if ssrc.startswith("0x") else int(ssrc),
            payload=bytes.fromhex(payload.replace(":", "")),
        )
    except ValueError:
        return None


def read_rtp(path: str) -> list[RtpPacket]:
    """Return every RTP packet in ``path``, in capture order.

    Raises:
        TsharkError: If tshark is not installed or fails to read the capture.

    """
    executable = shutil.which("tshark")
    if executable is None:
        raise TsharkError("tshark is not installed; the capture cannot be verified")

    command = [
        executable,
        "-r",
        path,
        "-Y",
        "rtp",
        # The SDP in the capture already sets the conversations up. The heuristic is a fallback
        # for a run where the signalling was captured on a different interface than the media.
        "--enable-heuristic",
        "rtp_udp",
        "-T",
        "fields",
        "-E",
        "separator=/t",
        "-E",
        "occurrence=f",
    ]
    for field in _FIELDS:
        command += ["-e", field]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise TsharkError(
            f"tshark failed on {path} (exit {completed.returncode}): {completed.stderr.strip()}"
        )

    packets = []
    for row in completed.stdout.splitlines():
        packet = _parse_row(row)
        if packet is not None:
            packets.append(packet)
    return packets
