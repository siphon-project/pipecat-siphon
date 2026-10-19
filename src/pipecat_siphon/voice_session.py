"""A siphon-pbx voice session: the brief a bot fetches before it speaks, and the report it sends.

When a siphon-pbx flow hands a call to a voice bot, it mints a session for that call and has the
engine dial the bot at the bot's address with `?call=<Call-ID>&session=<token>`. The bot then
fetches the session's brief, `GET {pbx}/voice/sessions/{token}`, which says what the bot is for
(the block's instructions), what it may report (the block's options), and how long the flow waits.
When the conversation has a result it reports it, `POST {pbx}/voice/sessions/{token}/report`, so
the flow leaves the block on the report instead of waiting out the deadline.

Both routes take the bot's secret as a bearer, refuse anything carrying an `Origin` header (that is
a browser), and answer every failure with the same bare 404. The shapes here are transcribed from
siphon-pbx's own serde derives: `SessionBrief` (`siphon-pbx-core/src/voice_sessions.rs`), flattened
into the route's `ServedBrief`, and the route's `Report` (`siphon-pbx-api/src/routes/voice.rs`).

The PBX is untrusted input like the media socket: a brief that is not the documented shape reads as
no brief, and nothing here raises on what the PBX sends.
"""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp
from loguru import logger

QUERY_PARAMETER = "session"
"""The dial-in URL's query parameter carrying the session token."""

MAX_BRIEF_BYTES = 64 * 1024
"""The largest brief read. A real one is instructions and a short menu, a few KB at most."""

MAX_REPORTED_OPTION = 200
"""The PBX cuts a reported option to this many characters (`MAX_REPORTED_OPTION` in the route)."""

MAX_REPORTED_REASON = 400
"""The PBX cuts a failure reason to this many characters (`MAX_REPORTED_REASON` in the route)."""

DEFAULT_TIMEOUT_SECONDS = 2.0
"""How long one request to the PBX may take. The brief is fetched before the bot speaks."""

_DEADLINE_MAXIMUM = 65535
"""`deadline_secs` is a `u16` on the PBX side."""

_TOKEN = re.compile(r"[0-9A-Za-z_-]{16,256}")
"""A token as the PBX mints one (`random_hex`), with room to spare but nothing that walks a path."""


@dataclass(frozen=True)
class BriefOption:
    """One outcome a bot may report: the word to send, and what it covers (empty where obvious)."""

    option: str
    means: str = ""


@dataclass(frozen=True)
class SessionBrief:
    """What a bot is told about the call it has been handed.

    `options` is advisory: the PBX checks a report against the block's own list and routes anything
    else out of the block's no-match port. Empty for an open conversation.
    """

    call_id: str
    instructions: str
    options: tuple[BriefOption, ...]
    deadline_secs: int


@dataclass(frozen=True)
class Chose:
    """The caller asked for this option."""

    option: str


@dataclass(frozen=True)
class Finished:
    """The conversation is over and there is nothing to choose."""


@dataclass(frozen=True)
class Failed:
    """The bot cannot serve this call."""

    reason: str = ""


Report = Chose | Finished | Failed


def _option_from(value: Any) -> BriefOption | None:
    if not isinstance(value, Mapping):
        return None
    option = value.get("option")
    means = value.get("means", "")
    if not isinstance(option, str) or not option or not isinstance(means, str):
        return None
    return BriefOption(option=option, means=means)


def parse_brief(raw: bytes | str) -> SessionBrief | None:
    """Read a brief as the PBX serves it, or None if it is not one. Never raises.

    Fields this does not know (a PBX-hosted bot's `runtime`, anything added later) are ignored;
    a known field of the wrong shape makes the whole brief unreadable rather than half-read.
    """
    if len(raw) > MAX_BRIEF_BYTES:
        return None
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    call_id = document.get("call_id")
    instructions = document.get("instructions")
    raw_options = document.get("options")
    deadline = document.get("deadline_secs")
    if not isinstance(call_id, str) or not isinstance(instructions, str):
        return None
    if not isinstance(raw_options, list):
        return None
    if (
        not isinstance(deadline, int)
        or isinstance(deadline, bool)
        or not 0 <= deadline <= _DEADLINE_MAXIMUM
    ):
        return None
    options: list[BriefOption] = []
    for raw_option in raw_options:
        option = _option_from(raw_option)
        if option is None:
            return None
        options.append(option)
    return SessionBrief(
        call_id=call_id,
        instructions=instructions,
        options=tuple(options),
        deadline_secs=deadline,
    )


def encode_report(report: Report) -> bytes:
    """Encode `report` as the route's `#[serde(tag = "outcome")]` `Report` reads it.

    The option and reason are cut where the PBX would cut them, so what is logged here is what the
    PBX acts on.
    """
    document: dict[str, str]
    match report:
        case Chose(option=option):
            document = {"outcome": "chose", "option": option[:MAX_REPORTED_OPTION]}
        case Finished():
            document = {"outcome": "finished"}
        case Failed(reason=reason):
            document = {"outcome": "failed", "reason": reason[:MAX_REPORTED_REASON]}
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()


def session_token_from(query: Mapping[str, str]) -> str | None:
    """Return the dial-in query's session token, or None if there is none or it is not plain."""
    token = query.get(QUERY_PARAMETER)
    if token is None or _TOKEN.fullmatch(token) is None:
        return None
    return token


def _describe(report: Report) -> str:
    match report:
        case Chose(option=option):
            return f"chose {option[:MAX_REPORTED_OPTION]!r}"
        case Finished():
            return "finished"
        case Failed(reason=reason):
            return f"failed ({reason[:MAX_REPORTED_REASON]})"


async def _read_at_most(reply: aiohttp.ClientResponse, limit: int) -> bytes:
    """Read the body, stopping one chunk past `limit`, which `parse_brief` then refuses."""
    body = bytearray()
    async for chunk in reply.content.iter_chunked(64 * 1024):
        body += chunk
        if len(body) > limit:
            break
    return bytes(body)


class VoiceSessionClient:
    """Fetches briefs from, and reports to, one siphon-pbx as one voice bot.

    Neither the secret nor a token is ever logged: the token alone opens the call's brief to
    whoever holds the secret, and logs travel further than either should.
    """

    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Talk to the PBX at `base_url` (scheme and host, and a path prefix if it has one).

        Raises:
            ValueError: If `base_url` is not http(s) with a host, or `secret` is empty.

        """
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("the PBX URL must be http:// or https:// with a host")
        if not secret:
            raise ValueError("the bot's secret is empty")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {secret}"}
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    def _session_url(self, token: str) -> str:
        return f"{self._base_url}/voice/sessions/{quote(token, safe='')}"

    async def fetch_brief(self, token: str) -> SessionBrief | None:
        """Fetch the session's brief: None if the PBX refused, timed out, or sent something else."""
        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as session,
                session.get(
                    self._session_url(token), headers=self._headers, allow_redirects=False
                ) as reply,
            ):
                if reply.status != 200:
                    # The PBX says nothing more than 404 for any failure, on purpose.
                    logger.warning(f"the PBX refused the session's brief: HTTP {reply.status}")
                    return None
                body = await _read_at_most(reply, MAX_BRIEF_BYTES)
        except TimeoutError:
            logger.warning("the PBX did not answer for the session's brief in time")
            return None
        except aiohttp.ClientError as error:
            # Only the type: aiohttp's messages carry the URL, and the URL carries the token.
            logger.warning(
                f"the PBX could not be reached for the session's brief: {type(error).__name__}"
            )
            return None
        brief = parse_brief(body)
        if brief is None:
            logger.warning("the PBX answered the session's brief with something that is not one")
        return brief

    async def report(self, token: str, report: Report) -> bool:
        """Send `report`. True only when the PBX took it for a call that was waiting for one."""
        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as session,
                session.post(
                    f"{self._session_url(token)}/report",
                    data=encode_report(report),
                    headers={**self._headers, "Content-Type": "application/json"},
                    allow_redirects=False,
                ) as reply,
            ):
                status = reply.status
        except TimeoutError:
            logger.warning(f"the PBX did not answer the report ({_describe(report)}) in time")
            return False
        except aiohttp.ClientError as error:
            logger.warning(
                f"the PBX could not be reached to report ({_describe(report)}): "
                f"{type(error).__name__}"
            )
            return False
        if status != 204:
            # A call that already left the block, or ended, has nobody waiting for a report.
            logger.info(f"the PBX did not take the report ({_describe(report)}): HTTP {status}")
            return False
        logger.info(f"reported to the PBX: {_describe(report)}")
        return True


class VoiceSession:
    """One call's session: its brief, and the one report the PBX will act on.

    The PBX's block waits for a single report and then leaves, so the first report sent is the one
    that counts; any later one is logged and dropped rather than sent, including after a first that
    the PBX did not take, since a second could not make the first land.
    """

    def __init__(self, client: VoiceSessionClient, token: str, brief: SessionBrief) -> None:
        """Hold the session `token` names, already briefed with `brief`."""
        self.brief = brief
        self._client = client
        self._token = token
        self._reported: Report | None = None
        self._lock = asyncio.Lock()

    @property
    def reported(self) -> Report | None:
        """The report sent for this call, if one has been."""
        return self._reported

    async def report(self, report: Report) -> bool:
        """Send `report` if nothing has been reported yet. True only when the PBX took it."""
        async with self._lock:
            if self._reported is not None:
                logger.info(
                    f"already reported {_describe(self._reported)}; dropping {_describe(report)}"
                )
                return False
            self._reported = report
        return await self._client.report(self._token, report)
