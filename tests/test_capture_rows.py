"""Reading tshark's field rows: one packet per row, and both clocks on every packet.

The RTP reader and the SIP reader are pointed at the same capture and their results have to be
comparable, because a check like "the BYE went out after the bot fell quiet" reads the time of a SIP
message against the time of an audio packet. tshark's relative time is relative to the first frame
*of that read*, so two reads of the same file agree only by luck. The epoch is absolute, and both
readers now carry it.
"""

from __future__ import annotations

from tools.capture import _parse_row

# The fields in the order `_FIELDS` asks tshark for them. A row is what `-T fields` prints: the
# values, tab separated, one line per frame.
ROW = "\t".join(
    (
        "7",  # frame.number
        "1.234000",  # frame.time_relative
        "1764240000.500000",  # frame.time_epoch
        "203.0.113.10",  # ip.src
        "203.0.113.40",  # ip.dst
        "30000",  # udp.srcport
        "7078",  # udp.dstport
        "8",  # rtp.p_type
        "4242",  # rtp.seq
        "160000",  # rtp.timestamp
        "0x1a2b3c4d",  # rtp.ssrc
        "d5:d5:d5",  # rtp.payload
    )
)


class TestOneRow:
    def test_a_row_reads_into_a_packet(self) -> None:
        packet = _parse_row(ROW)

        assert packet is not None
        assert packet.number == 7
        assert packet.source_address == "203.0.113.10"
        assert packet.destination_port == 7078
        assert packet.payload_type == 8
        assert packet.sequence == 4242
        assert packet.timestamp == 160000
        assert packet.ssrc == 0x1A2B3C4D
        assert packet.payload == b"\xd5\xd5\xd5"

    def test_both_clocks_are_kept(self) -> None:
        """Keep both, because they answer different questions.

        The relative one is what the existing analysis measures spans with. The epoch is what
        lines an audio packet up against a SIP message read from the same capture.
        """
        packet = _parse_row(ROW)

        assert packet is not None
        assert packet.time == 1.234
        assert packet.epoch == 1764240000.5

    def test_a_row_of_the_wrong_width_is_not_a_packet(self) -> None:
        """Rows arrive per frame, including the frames that carry no RTP at all."""
        assert _parse_row("7\t1.234000") is None

    def test_a_row_without_a_payload_is_not_a_packet(self) -> None:
        """A comfort-noise or padding frame dissects as RTP with nothing to read back."""
        assert _parse_row(ROW.rsplit("\t", 1)[0] + "\t") is None

    def test_a_row_whose_numbers_are_not_numbers_is_not_a_packet(self) -> None:
        """Malformed input is dropped rather than raised: a capture is untrusted input."""
        assert _parse_row(ROW.replace("\t4242\t", "\tnot-a-sequence\t")) is None
