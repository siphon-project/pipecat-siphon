"""The caller's RTP, checked against hand-written header bytes rather than its own parser."""

from __future__ import annotations

import pytest

from tools.conversation.rtp_packet import RtpHeader, RtpSender, pack_rtp, parse_rtp


class TestPack:
    def test_a_plain_alaw_frame(self) -> None:
        datagram = pack_rtp(
            payload_type=8,
            marker=False,
            sequence=1,
            timestamp=160,
            ssrc=0x01020304,
            payload=b"\x55\xd5",
        )

        assert datagram == bytes.fromhex("80080001000000a001020304") + b"\x55\xd5"

    def test_the_marker_bit(self) -> None:
        datagram = pack_rtp(
            payload_type=8, marker=True, sequence=0, timestamp=0, ssrc=0, payload=b""
        )

        assert datagram[:2] == bytes.fromhex("8088")

    @pytest.mark.parametrize(
        ("field", "value"),
        [("payload_type", 128), ("sequence", 65536), ("timestamp", 2**32), ("ssrc", -1)],
    )
    def test_a_field_that_does_not_fit_is_refused(self, field: str, value: int) -> None:
        fields = {"payload_type": 8, "sequence": 0, "timestamp": 0, "ssrc": 0}
        fields[field] = value

        with pytest.raises(ValueError, match=field):
            pack_rtp(marker=False, payload=b"", **fields)


class TestSender:
    def test_sequence_and_timestamp_advance_and_wrap(self) -> None:
        sender = RtpSender(
            ssrc=7,
            payload_type=8,
            samples_per_frame=160,
            first_sequence=65535,
            first_timestamp=2**32 - 160,
        )

        first = parse_rtp(sender.next_packet(b"\x00" * 160))
        second = parse_rtp(sender.next_packet(b"\x00" * 160))

        assert first is not None and second is not None
        assert (first[0].sequence, first[0].timestamp) == (65535, 2**32 - 160)
        assert (second[0].sequence, second[0].timestamp) == (0, 0)

    def test_the_first_frame_of_a_talkspurt_carries_the_marker(self) -> None:
        sender = RtpSender(ssrc=7, payload_type=8, samples_per_frame=160)

        plain = parse_rtp(sender.next_packet(b"\x00"))
        marked = parse_rtp(sender.next_packet(b"\x00", marker=True))

        assert plain is not None and marked is not None
        assert (plain[0].marker, marked[0].marker) == (False, True)


class TestParse:
    def test_the_fields_come_back(self) -> None:
        parsed = parse_rtp(bytes.fromhex("8088fffe0000a0000a0b0c0d") + b"payload")

        assert parsed == (
            RtpHeader(
                payload_type=8, marker=True, sequence=0xFFFE, timestamp=0xA000, ssrc=0x0A0B0C0D
            ),
            b"payload",
        )

    def test_contributing_sources_and_an_extension_are_skipped(self) -> None:
        header = bytes.fromhex("91080001000000a001020304")  # V=2, X=1, CC=1
        contributing_source = bytes.fromhex("11223344")
        extension = bytes.fromhex("beef0001") + bytes.fromhex("deadbeef")  # one 32-bit word

        parsed = parse_rtp(header + contributing_source + extension + b"audio")

        assert parsed is not None
        assert parsed[1] == b"audio"

    def test_padding_is_removed(self) -> None:
        header = bytes.fromhex("a0080001000000a001020304")  # V=2, P=1
        parsed = parse_rtp(header + b"audio" + b"\x00\x00\x03")

        assert parsed is not None
        assert parsed[1] == b"audio"

    @pytest.mark.parametrize(
        "datagram",
        [
            b"",
            bytes.fromhex("8008"),
            bytes.fromhex("40080001000000a001020304"),  # version 1
            bytes.fromhex("82080001000000a001020304") + b"\x00\x00\x00\x00",  # CC=2, one CSRC
            bytes.fromhex("90080001000000a001020304") + bytes.fromhex("beef0004"),  # short ext.
            bytes.fromhex("a0080001000000a001020304") + b"\x09",  # padding past the payload
        ],
        ids=[
            "empty",
            "shorter than a header",
            "wrong version",
            "missing contributing source",
            "truncated extension",
            "padding longer than the packet",
        ],
    )
    def test_a_malformed_datagram_is_none(self, datagram: bytes) -> None:
        assert parse_rtp(datagram) is None
