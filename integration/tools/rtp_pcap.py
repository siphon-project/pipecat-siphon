"""Write the RTP capture files SIPp's ``play_pcap_audio`` streams into a call.

SIPp reads a pcap, strips the link/network/transport headers off each packet and re-sends the
UDP payload from its own media socket, pacing on the capture's own timestamps. So the file only
has to be *shaped* like a capture; the addresses in it are placeholders SIPp overwrites. The
layout below is byte-for-byte the shape of the ``g711a.pcap`` SIPp ships (zeroed MACs, loopback
addresses, port 6000 both ways, zero UDP checksum), because that is the shape its parser is
known to accept.

Hand-rolled rather than pulled from a library: the file is ninety bytes of header per packet and
a dependency here would have to be installed into the UAC image for no gain.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable

__all__ = ["RTP_PAYLOAD_TYPE_PCMA", "write_rtp_pcap"]

PCAP_MAGIC_MICROSECONDS = 0xA1B2C3D4
LINKTYPE_ETHERNET = 1
ETHERTYPE_IPV4 = 0x0800
IP_PROTOCOL_UDP = 17

RTP_PAYLOAD_TYPE_PCMA = 8
"""G.711 A-law, RFC 3551 §6. Static payload type, so the fixture needs no ``a=rtpmap`` to be
understood by a dissector reading the capture back."""

_SOURCE_ADDRESS = "127.0.0.1"
_DESTINATION_ADDRESS = "127.0.0.2"
_MEDIA_PORT = 6000
_SSRC = 0x12345678


def _internet_checksum(data: bytes) -> int:
    """Return the RFC 1071 one's-complement checksum of ``data``."""
    if len(data) % 2:
        data += b"\x00"
    total: int = sum(struct.unpack(f"!{len(data) // 2}H", data))
    total = (total >> 16) + (total & 0xFFFF)
    total += total >> 16
    return (~total) & 0xFFFF


def _ipv4_header(payload_length: int, identification: int) -> bytes:
    """Return a 20-byte IPv4 header with a correct checksum."""
    source = bytes(int(part) for part in _SOURCE_ADDRESS.split("."))
    destination = bytes(int(part) for part in _DESTINATION_ADDRESS.split("."))
    total_length = 20 + payload_length
    header = (
        struct.pack("!BBHHHBBH", 0x45, 0, total_length, identification, 0, 64, IP_PROTOCOL_UDP, 0)
        + source
        + destination
    )
    return header[:10] + struct.pack("!H", _internet_checksum(header)) + header[12:]


def _udp_datagram(payload: bytes) -> bytes:
    """Return a UDP datagram with the checksum left at zero, as the reference capture does."""
    return struct.pack("!HHHH", _MEDIA_PORT, _MEDIA_PORT, 8 + len(payload), 0) + payload


def _rtp_packet(sequence: int, timestamp: int, payload: bytes) -> bytes:
    """Return an RFC 3550 §5.1 fixed header (V=2, no padding/extension/CSRC) plus payload."""
    return struct.pack("!BBHII", 0x80, RTP_PAYLOAD_TYPE_PCMA, sequence, timestamp, _SSRC) + payload


def write_rtp_pcap(
    path: str,
    payloads: Iterable[bytes],
    *,
    ptime_ms: int,
    samples_per_frame: int,
) -> int:
    """Write ``payloads`` as one RTP-over-UDP-over-IPv4-over-Ethernet capture.

    Args:
        path: File to write.
        payloads: Encoded media payloads, one per packet, in order.
        ptime_ms: Spacing between capture timestamps, which is what SIPp paces on.
        samples_per_frame: RTP timestamp increment per packet (RFC 3550 §5.1: the media clock,
            not the packet count).

    Returns:
        The number of packets written.

    """
    ethernet_header = bytes(12) + struct.pack("!H", ETHERTYPE_IPV4)
    count = 0
    with open(path, "wb") as handle:
        handle.write(
            struct.pack("!IHHiIII", PCAP_MAGIC_MICROSECONDS, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET)
        )
        for index, payload in enumerate(payloads):
            packet = _rtp_packet(
                sequence=index & 0xFFFF,
                timestamp=(index * samples_per_frame) & 0xFFFFFFFF,
                payload=payload,
            )
            datagram = _udp_datagram(packet)
            frame = ethernet_header + _ipv4_header(len(datagram), index & 0xFFFF) + datagram
            microseconds = index * ptime_ms * 1000
            handle.write(
                struct.pack(
                    "!IIII",
                    microseconds // 1_000_000,
                    microseconds % 1_000_000,
                    len(frame),
                    len(frame),
                )
            )
            handle.write(frame)
            count += 1
    return count
