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
from typing import Any, Literal
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

TRANSFER_TIMEOUT_SECONDS = 160.0
"""How long a transfer request may take. siphon-pbx holds it until the verdict, for at most
`TRANSFER_ANSWER_DEADLINE` (150 s: a cold transfer rings for up to 120 s, plus 30). A client that
gave up first would tell the bot a transfer failed that then went through."""

MAX_TRANSFER_SUMMARY = 1_000
"""The PBX cuts a transfer's summary to this many characters (`MAX_TRANSFER_SUMMARY`)."""

_DEADLINE_MAXIMUM = 65535
"""`deadline_secs` is a `u16` on the PBX side."""

_TOKEN = re.compile(r"[0-9A-Za-z_-]{16,256}(?:\.[0-9A-Za-z][0-9A-Za-z._-]{0,62})?")
"""A token as the PBX mints one (`random_hex`), with room to spare but nothing that walks a path.

A clustered siphon-pbx appends `.` and the id of the node holding the session, and routes on it, so
the suffix is kept: ASCII letters, digits, `.`, `_` and `-`, starting with a letter or digit, at
most 63 characters (`NodeId`, `NODE_ID_MAX`). Matched as a suffix rather than by adding `.` to the
whole class, so the token can never be a `.` or `..` path segment. A single node appends nothing."""


@dataclass(frozen=True)
class BriefOption:
    """One outcome a bot may report: the word to send, and what it covers (empty where obvious)."""

    option: str
    means: str = ""


CONTEXT_VERSION = 1
"""The `context` version this reads. Any other counts as no context at all, as siphon-pbx asks."""


@dataclass(frozen=True)
class ContextOrganisation:
    """The organisation whose call it is. `external_ref` is the operator's own key for it."""

    id: str
    name: str
    time_zone: str
    external_ref: str | None = None
    locale: str | None = None


@dataclass(frozen=True)
class ContextCalled:
    """The number dialled: the request URI as received, and E.164 where the PBX could make it.

    `national` is that number as it is dialled in the organisation's own country (trunk prefix and
    national number), sent only when it is in that country. Absent from a PBX that predates it.
    """

    uri: str
    number: str | None = None
    national: str | None = None


@dataclass(frozen=True)
class ContextCaller:
    """Who is calling. A withheld caller (`privacy` "id") carries neither number nor name.

    `national` is the number as a caller in the organisation's own country knows it, when they are.
    """

    privacy: Literal["none", "id"]
    number: str | None = None
    national: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class ContextTrunk:
    """The organisation's own trunk the call came over, when exactly one matched its source."""

    id: str
    name: str


TransferMode = Literal["cold", "warm", "refer"]
"""How a bot may hand a caller to a destination (siphon-pbx's `TransferMode`, `snake_case`)."""

_TRANSFER_MODES: tuple[TransferMode, ...] = ("cold", "warm", "refer")

_DESTINATION_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
"""A destination name as siphon-pbx accepts one (`check_name`, `MAX_NAME_CHARS`). Checked here as
well, since a name ends up in the model's tool definition."""

MAX_DESTINATION_DESCRIPTION = 500
"""The longest description siphon-pbx accepts (`MAX_DESCRIPTION_CHARS`)."""


@dataclass(frozen=True)
class TransferDestination:
    """Where a bot may hand the caller on: a name to choose, what it is for, and how.

    Never where it leads; the PBX keeps that.
    """

    name: str
    description: str
    modes: tuple[TransferMode, ...]


TransferOutcome = Literal[
    "completed", "declined", "no_answer", "busy", "refused", "failed", "not_allowed"
]
"""How a transfer request ended. The first six are siphon-pbx's verdicts (`TransferVerdict`); the
caller is with the destination only after `completed`, and still with the bot after the rest.
`not_allowed` is a `403`: the PBX placed nothing, and its message says why."""

_TRANSFER_VERDICTS: tuple[TransferOutcome, ...] = (
    "completed",
    "declined",
    "no_answer",
    "busy",
    "refused",
    "failed",
)


@dataclass(frozen=True)
class TransferResult:
    """What came of a transfer request: the outcome, the PBX's reason, and its message on a refusal.

    `reason` is a verdict's own explanation, or a refusal's code (`warm_not_yet_supported`).
    """

    outcome: TransferOutcome
    reason: str | None = None
    message: str | None = None

    @property
    def completed(self) -> bool:
        """Whether the caller has been put through, and is no longer with the bot."""
        return self.outcome == "completed"


def _transfer_failed(reason: str) -> TransferResult:
    return TransferResult(outcome="failed", reason=reason)


def _transfer_result_from(status: int, body: bytes) -> TransferResult:
    """Read the transfer route's answer: a verdict, a refusal, or else a failed transfer."""
    try:
        document = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        document = None
    if not isinstance(document, dict):
        document = {}
    reason = document.get("reason")
    reason = reason if isinstance(reason, str) else None
    if status == 200:
        outcome = document.get("outcome")
        for verdict in _TRANSFER_VERDICTS:
            if outcome == verdict:
                return TransferResult(outcome=verdict, reason=reason)
        return _transfer_failed("the PBX answered with no verdict this bot knows")
    message = document.get("message")
    if status == 403 and reason is not None:
        return TransferResult(
            outcome="not_allowed",
            reason=reason,
            message=message if isinstance(message, str) else None,
        )
    if status == 400:
        # Ours to fix, not the caller's: the request was built here.
        logger.error(f"the PBX could not read the transfer request: {reason} ({message})")
        return _transfer_failed("the PBX could not read the request")
    return _transfer_failed(f"the PBX answered HTTP {status}")


def encode_transfer(
    *, destination: str | None, number: str | None, mode: TransferMode, summary: str
) -> bytes:
    """Encode a transfer request as the route's `TransferBody` reads it.

    One of destination and number, the mode, and the summary cut where the PBX would cut it.
    """
    document: dict[str, str] = (
        {"destination": destination} if destination is not None else {"number": number or ""}
    )
    document["mode"] = mode
    document["summary"] = summary[:MAX_TRANSFER_SUMMARY]
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()


@dataclass(frozen=True)
class CallContext:
    """Whose call it is, as much as the bot's `context` setting in siphon-pbx lets it be told.

    `organisation` alone for a bot set to `organisation`; the rest too, where known, for `full`.
    """

    organisation: ContextOrganisation
    called: ContextCalled | None = None
    caller: ContextCaller | None = None
    direction: Literal["inbound", "outbound"] | None = None
    trunk: ContextTrunk | None = None
    # The organisation's transfer destinations, narrowed to the voice block's, and whether any
    # number may be named instead. Empty and False from a PBX that predates them.
    destinations: tuple[TransferDestination, ...] = ()
    any_number: bool = False


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
    # None for a PBX that sends none, a bot set to `none`, or a context this cannot read.
    context: CallContext | None = None


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


class _Malformed(Exception):
    """A `context` field of the wrong shape: the context is dropped, the brief is kept."""


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise _Malformed(key)
    return value


def _optional_text(document: Mapping[str, Any], key: str) -> str | None:
    value = document.get(key)
    if value is not None and not isinstance(value, str):
        raise _Malformed(key)
    return value


def _part(document: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = document.get(key)
    if value is not None and not isinstance(value, Mapping):
        raise _Malformed(key)
    return value


def _caller_from(document: Mapping[str, Any]) -> ContextCaller:
    number = _optional_text(document, "number")
    national = _optional_text(document, "national")
    display_name = _optional_text(document, "display_name")
    if document.get("privacy") != "none":
        # Withheld, or a privacy this does not know: read as withheld either way, and whatever came
        # with it is not kept. The PBX never undoes a caller's withholding, and neither does this.
        return ContextCaller(privacy="id")
    return ContextCaller(
        privacy="none", number=number, national=national, display_name=display_name
    )


def _destination_from(value: Any) -> TransferDestination | None:
    """Read one destination, or None, logged, for one siphon-pbx would never have sent.

    A mode this does not know is passed over, so a newer PBX's destinations keep the modes this one
    can use; a destination left with none is dropped.
    """
    if not isinstance(value, Mapping):
        logger.warning("ignoring a transfer destination that is not an object")
        return None
    name = value.get("name")
    description = value.get("description")
    modes = value.get("modes")
    if not isinstance(name, str) or _DESTINATION_NAME.fullmatch(name) is None:
        logger.warning("ignoring a transfer destination whose name siphon-pbx would refuse")
        return None
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_DESTINATION_DESCRIPTION
    ):
        logger.warning(f"ignoring transfer destination {name!r}: its description is not one")
        return None
    if not isinstance(modes, list):
        logger.warning(f"ignoring transfer destination {name!r}: its modes are not a list")
        return None
    known = tuple(mode for mode in _TRANSFER_MODES if mode in modes)
    if not known:
        logger.warning(f"ignoring transfer destination {name!r}: no mode this bot knows")
        return None
    return TransferDestination(name=name, description=description, modes=known)


def _destinations_from(document: Mapping[str, Any]) -> tuple[TransferDestination, ...]:
    value = document.get("destinations")
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _Malformed("destinations")
    return tuple(
        destination
        for destination in (_destination_from(entry) for entry in value)
        if destination is not None
    )


def parse_context(value: Any) -> CallContext | None:
    """Read a brief's `context`: None if there is none, of another version, or it is malformed.

    Never raises. A malformed context drops the context and nothing else: the call is still served,
    with the bot's own configuration.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        logger.warning("the brief's context is not an object; serving without it")
        return None
    version = value.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != CONTEXT_VERSION:
        logger.info(
            f"the brief's context is version {version!r}, not {CONTEXT_VERSION}; ignoring it"
        )
        return None
    try:
        organisation = _part(value, "organisation")
        if organisation is None:
            raise _Malformed("organisation")
        called = _part(value, "called")
        caller = _part(value, "caller")
        trunk = _part(value, "trunk")
        direction = value.get("direction")
        any_number = value.get("any_number", False)
        if not isinstance(any_number, bool):
            raise _Malformed("any_number")
        return CallContext(
            organisation=ContextOrganisation(
                id=_text(organisation, "id"),
                name=_text(organisation, "name"),
                time_zone=_text(organisation, "time_zone"),
                external_ref=_optional_text(organisation, "external_ref"),
                locale=_optional_text(organisation, "locale"),
            ),
            called=(
                ContextCalled(
                    uri=_text(called, "uri"),
                    number=_optional_text(called, "number"),
                    national=_optional_text(called, "national"),
                )
                if called is not None
                else None
            ),
            caller=_caller_from(caller) if caller is not None else None,
            direction=direction if direction in ("inbound", "outbound") else None,
            trunk=(
                ContextTrunk(id=_text(trunk, "id"), name=_text(trunk, "name"))
                if trunk is not None
                else None
            ),
            destinations=_destinations_from(value),
            any_number=any_number,
        )
    except _Malformed as malformed:
        logger.warning(f"the brief's context has a malformed {malformed}; serving without it")
        return None


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
        context=parse_context(document.get("context")),
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
        transfer_timeout_seconds: float = TRANSFER_TIMEOUT_SECONDS,
    ) -> None:
        """Talk to the PBX at `base_url` (scheme and host, and a path prefix if it has one).

        `timeout_seconds` bounds a brief and a report; a transfer, which the PBX holds until its
        verdict, has `transfer_timeout_seconds`.

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
        self._transfer_timeout = aiohttp.ClientTimeout(
            total=transfer_timeout_seconds, connect=timeout_seconds
        )

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

    async def transfer(
        self,
        token: str,
        *,
        destination: str | None = None,
        number: str | None = None,
        mode: TransferMode,
        summary: str = "",
    ) -> TransferResult:
        """Ask the PBX to transfer the caller, and wait for the verdict. Never raises.

        Exactly one of `destination` (a name the brief listed) and `number` (only when the brief
        says any number may be named). Every way this can go wrong reads as a `failed` result,
        so the bot always has something to tell the caller, who is still with it.
        """
        if (destination is None) == (number is None):
            logger.error("a transfer names exactly one of a destination and a number")
            return _transfer_failed("the request named neither or both of destination and number")
        body = encode_transfer(destination=destination, number=number, mode=mode, summary=summary)
        target = f"destination {destination!r}" if destination is not None else "a number"
        try:
            async with (
                aiohttp.ClientSession(timeout=self._transfer_timeout) as session,
                session.post(
                    f"{self._session_url(token)}/transfer",
                    data=body,
                    headers={**self._headers, "Content-Type": "application/json"},
                    allow_redirects=False,
                ) as reply,
            ):
                status = reply.status
                answer = await _read_at_most(reply, MAX_BRIEF_BYTES)
        except TimeoutError:
            logger.warning(f"the PBX gave no verdict on the {mode} transfer to {target} in time")
            return _transfer_failed("no verdict came back in time")
        except aiohttp.ClientError as error:
            logger.warning(
                f"the PBX could not be reached to transfer to {target}: {type(error).__name__}"
            )
            return _transfer_failed("the PBX could not be reached")
        result = _transfer_result_from(status, answer)
        logger.info(
            f"{mode} transfer to {target}: {result.outcome}"
            + (f" ({result.reason})" if result.reason else "")
        )
        return result


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
        self._transferred = False
        self._lock = asyncio.Lock()

    @property
    def reported(self) -> Report | None:
        """The report sent for this call, if one has been."""
        return self._reported

    @property
    def transferred(self) -> bool:
        """Whether the caller was transferred, which ends the session: nothing is reported after."""
        return self._transferred

    async def report(self, report: Report) -> bool:
        """Send `report` if nothing has been reported yet. True only when the PBX took it."""
        async with self._lock:
            if self._transferred:
                logger.info(f"the caller was transferred; dropping {_describe(report)}")
                return False
            if self._reported is not None:
                logger.info(
                    f"already reported {_describe(self._reported)}; dropping {_describe(report)}"
                )
                return False
            self._reported = report
        return await self._client.report(self._token, report)

    async def transfer(
        self,
        *,
        destination: str | None = None,
        number: str | None = None,
        mode: TransferMode,
        summary: str = "",
    ) -> TransferResult:
        """Ask for the caller to be transferred, unless the session has already ended.

        Held under the same lock as a report, so a report and a transfer never race: one sent while
        the other waits for its answer would act on a call the first may already have moved on.
        A completed transfer ends the session on siphon-pbx's `transferred` port.
        """
        async with self._lock:
            if self._transferred or self._reported is not None:
                logger.info("this call's session has already ended; not asking for a transfer")
                return _transfer_failed("this call has already been passed on")
            result = await self._client.transfer(
                self._token, destination=destination, number=number, mode=mode, summary=summary
            )
            if result.completed:
                self._transferred = True
            return result
