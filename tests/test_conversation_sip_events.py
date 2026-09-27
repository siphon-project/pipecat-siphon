"""SIP messages as tshark dissected them from the caller's capture."""

from __future__ import annotations

import pytest

from tools.conversation.sip_events import SIP_FIELDS, SipMessage, parse_sip_rows, refer_to_uri

CALL = "conversation-goodbye-1-1@172.28.8.40"


def _row(
    *,
    time: str = "1789466337.120000000",
    source: str = "172.28.8.40",
    destination: str = "172.28.8.20",
    method: str = "",
    status: str = "",
    cseq_method: str = "INVITE",
    call_id: str = CALL,
    refer_to: str = "",
    to_tag: str = "",
    sdp_address: str = "",
    sdp_port: str = "",
) -> str:
    values = [
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
    ]
    assert len(values) == len(SIP_FIELDS)
    return "\t".join(values)


class TestParseSipRows:
    def test_a_call_that_was_transferred(self) -> None:
        text = "\n".join(
            [
                _row(method="INVITE", sdp_address="172.28.8.40", sdp_port="7078"),
                _row(
                    time="1789466337.180000000",
                    source="172.28.8.20",
                    destination="172.28.8.40",
                    status="200",
                    to_tag="agent-tag",
                    sdp_address="172.28.8.10",
                    sdp_port="30002",
                ),
                _row(
                    time="1789466350.500000000",
                    source="172.28.8.20",
                    destination="172.28.8.40",
                    method="REFER",
                    cseq_method="REFER",
                    refer_to="<sip:manager@harness.example>",
                    to_tag="caller-tag",
                ),
            ]
        )

        invite, answer, refer = parse_sip_rows(text)

        assert invite == SipMessage(
            time=1789466337.12,
            source="172.28.8.40",
            destination="172.28.8.20",
            method="INVITE",
            status=None,
            cseq_method="INVITE",
            call_id=CALL,
            refer_to=None,
            to_tag=None,
            sdp_address="172.28.8.40",
            sdp_port=7078,
        )
        assert (answer.method, answer.status, answer.sdp_address, answer.sdp_port) == (
            None,
            200,
            "172.28.8.10",
            30002,
        )
        assert (refer.method, refer.refer_to, refer.to_tag) == (
            "REFER",
            "sip:manager@harness.example",
            "caller-tag",
        )
        assert refer.is_request and not answer.is_request

    def test_a_repeated_field_keeps_its_first_value(self) -> None:
        text = _row(status="200", sdp_address="172.28.8.10,172.28.8.11", sdp_port="30002,30004")

        (message,) = parse_sip_rows(text)

        assert (message.sdp_address, message.sdp_port) == ("172.28.8.10", 30002)

    @pytest.mark.parametrize(
        "row",
        [
            "\t".join(["1789466337.1", "172.28.8.40"]),
            _row(time="yesterday", method="BYE"),
            _row(),
            _row(method="BYE", call_id=""),
            _row(status="two hundred"),
        ],
        ids=[
            "wrong column count",
            "time not a number",
            "neither request nor response",
            "no call id",
            "status not a number",
        ],
    )
    def test_an_unusable_row_is_skipped(self, row: str) -> None:
        assert parse_sip_rows(row) == []


class TestReferToUri:
    @pytest.mark.parametrize(
        ("value", "uri"),
        [
            ("<sip:manager@harness.example>", "sip:manager@harness.example"),
            ("sip:manager@harness.example;method=INVITE", "sip:manager@harness.example"),
            ('"Manager" <sip:manager@harness.example>;x=1', "sip:manager@harness.example"),
            ("  sip:manager@harness.example  ", "sip:manager@harness.example"),
        ],
    )
    def test_the_uri_is_taken_out_of_the_header_value(self, value: str, uri: str) -> None:
        assert refer_to_uri(value) == uri
