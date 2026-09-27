"""Render the conversation caller's SIPp scenario for one call.

The template's own placeholders are written percent-brace, because SIPp already owns square
brackets (`[call_id]`) and uses a dollar sign for its variables (`[$refer_to]`); nothing in a SIP
message or a SIPp scenario starts with a percent sign followed by a brace.

Values that land inside a SIPp `regexp` are escaped for POSIX extended regular expressions, which
SIPp compiles them as. Python's `re.escape` is the wrong tool: it also escapes ordinary characters
such as `@` and `:`, and POSIX leaves a backslash before an ordinary character undefined.
"""

from __future__ import annotations

import string
import xml.etree.ElementTree as ElementTree

ANY_SIP_URI_REGEXP = r"sip:[^>;\r\n]*"
"""The Refer-To check when the scenario expects no transfer: any REFER is answered, and whether
the bot should have sent one is the evaluator's decision, made from the capture."""

_ERE_METACHARACTERS = frozenset(".[]()*+?{}|^$\\")


class _PercentTemplate(string.Template):
    delimiter = "%"


def ere_escape(literal: str) -> str:
    """Escape `literal` so a POSIX extended regular expression matches exactly it."""
    return "".join(
        f"\\{character}" if character in _ERE_METACHARACTERS else character for character in literal
    )


def render_scenario(
    template_text: str,
    *,
    outcome_timeout_ms: int,
    refer_to: str | None,
    engine_address: str,
    caller_media_port: int,
) -> str:
    """Fill the template for one call.

    Args:
        template_text: The committed scenario template.
        outcome_timeout_ms: How long to wait for the bot's BYE or REFER before hanging up.
        refer_to: The transfer target the REFER must name, or None to accept any.
        engine_address: The address the 200 OK's `c=` line must carry.
        caller_media_port: The RTP port the offer advertises for the caller's media process.

    Raises:
        ValueError: When a placeholder has no value, or the result is not well-formed XML.

    """
    values = {
        "outcome_timeout_ms": str(outcome_timeout_ms),
        "refer_to_regexp": ANY_SIP_URI_REGEXP if refer_to is None else ere_escape(refer_to),
        "engine_address_regexp": ere_escape(engine_address),
        "caller_media_port": str(caller_media_port),
    }
    try:
        rendered = _PercentTemplate(template_text).substitute(values)
    except KeyError as missing:
        raise ValueError(f"the template's placeholder {missing.args[0]} has no value") from None
    except ValueError as malformed:
        raise ValueError(f"the template has a malformed placeholder: {malformed}") from None
    try:
        ElementTree.fromstring(rendered)
    except ElementTree.ParseError as error:
        raise ValueError(f"the rendered scenario is not well-formed XML: {error}") from None
    return rendered
