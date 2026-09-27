"""SIP messages out of the caller's capture, dissected by tshark rather than by this harness.

The capture is taken on the caller's own interface, so every message in it is one the caller
actually sent or received, stamped on the same clock as the RTP beside it. tshark is an
independent SIP parser: when it says the proxy sent a REFER naming a target, that is a third party
agreeing about the wire, which a parser written next to the assertion could not be.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from tools.capture import TsharkError

SIP_FIELDS = (
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "sip.Method",
    "sip.Status-Code",
    "sip.CSeq.method",
    "sip.Call-ID",
    "sip.Refer-To",
    "sip.to.tag",
    "sdp.connection_info.address",
    "sdp.media.port",
)
"""The tshark fields read for each SIP message, in column order."""


@dataclass(frozen=True)
class SipMessage:
    """One SIP request or response seen on the caller's interface."""

    time: float
    source: str
    destination: str
    method: str | None
    """The request method; None for a response."""
    status: int | None
    """The response status code; None for a request."""
    cseq_method: str
    call_id: str
    refer_to: str | None
    """The Refer-To URI, without brackets, display name or header parameters."""
    to_tag: str | None
    sdp_address: str | None
    sdp_port: int | None

    @property
    def is_request(self) -> bool:
        """Whether this is a request rather than a response."""
        return self.method is not None


def read_sip(path: str) -> list[SipMessage]:
    """Return every SIP message in the capture at `path`, in capture order.

    Raises:
        TsharkError: If tshark is not installed or fails to read the capture.

    """
    executable = shutil.which("tshark")
    if executable is None:
        raise TsharkError("tshark is not installed; the capture cannot be read")
    command = [
        executable,
        "-r",
        path,
        "-Y",
        "sip",
        "-T",
        "fields",
        "-E",
        "separator=/t",
        "-E",
        "occurrence=f",
    ]
    for field in SIP_FIELDS:
        command += ["-e", field]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise TsharkError(
            f"tshark failed on {path} (exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return parse_sip_rows(completed.stdout)


def parse_sip_rows(text: str) -> list[SipMessage]:
    """Parse tshark field rows, one per line in `SIP_FIELDS` order, skipping unusable rows."""
    messages: list[SipMessage] = []
    for row in text.splitlines():
        message = _parse_row(row)
        if message is not None:
            messages.append(message)
    return messages


def refer_to_uri(value: str) -> str:
    """Take the URI out of a Refer-To header value."""
    text = value.strip()
    opening = text.find("<")
    if opening >= 0:
        closing = text.find(">", opening + 1)
        if closing > opening:
            return text[opening + 1 : closing].strip()
    return text.split(";", 1)[0].strip()


def _parse_row(row: str) -> SipMessage | None:
    parts = row.split("\t")
    if len(parts) != len(SIP_FIELDS):
        return None
    (
        time,
        source,
        destination,
        method,
        status,
        cseq_method,
        call_id,
        refer_to,
        to_tag,
        sdp_address,
        sdp_port,
    ) = (part.strip() for part in parts)
    if not call_id or not cseq_method or not (method or status):
        return None
    # A message can carry more than one SDP media line; the first is the audio this harness offers.
    sdp_address = _first(sdp_address)
    sdp_port = _first(sdp_port)
    try:
        parsed_time = float(time)
        parsed_status = int(status) if status else None
        parsed_port = int(sdp_port) if sdp_port else None
    except ValueError:
        return None
    return SipMessage(
        time=parsed_time,
        source=source,
        destination=destination,
        method=method or None,
        status=parsed_status,
        cseq_method=cseq_method,
        call_id=call_id,
        refer_to=refer_to_uri(refer_to) if refer_to else None,
        to_tag=to_tag or None,
        sdp_address=sdp_address or None,
        sdp_port=parsed_port,
    )


def _first(value: str) -> str:
    """Keep the first of the comma-joined values tshark gives a repeated field."""
    return value.split(",", 1)[0].strip()
