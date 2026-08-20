"""The siphon-rtp media WebSocket control protocol.

siphon-rtp bridges a call leg's audio to a WebSocket server. It dials out as the WebSocket
*client*, so the application in this package is the *server*. Two frame kinds share the socket:

* **binary** frames carry raw audio in the negotiated :class:`MediaFormat`,
* **text** frames carry a ``{"type": <snake_case tag>, "data": {...}}`` control envelope whose
  ``data`` object uses camelCase field names.

This module is a faithful, dependency-free transcription of the engine-side definition in
``crates/siphon-rtp-media/src/bridge/protocol.rs``. Field order, casing and the
omit-when-absent rules match the Rust ``serde`` derives exactly, so :func:`encode_control`
reproduces the engine's own bytes for a given message.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Final, TypeVar

__all__ = [
    "ClearData",
    "ControlMessage",
    "Direction",
    "DtmfData",
    "Encoding",
    "Endianness",
    "ErrorData",
    "EventData",
    "MarkData",
    "MediaFormat",
    "PlaySource",
    "PlayStartData",
    "PlayStopData",
    "RenegotiateData",
    "SiphonProtocolError",
    "SpeechData",
    "SpeechStartedData",
    "SpeechStoppedData",
    "StartData",
    "StopData",
    "encode_control",
    "parse_control",
]


_EnumT = TypeVar("_EnumT", bound=StrEnum)


class SiphonProtocolError(ValueError):
    """Raised when a control frame cannot be understood.

    The WebSocket carries untrusted input, so every parse failure surfaces as this single
    exception type rather than a ``KeyError``/``TypeError`` leaking out of the parser.
    """


class Encoding(StrEnum):
    """Wire audio encoding for binary frames.

    Serialized uppercase, matching ``#[serde(rename_all = "UPPERCASE")]`` on the Rust enum.
    """

    L16 = "L16"
    """16-bit linear PCM."""

    PCMU = "PCMU"
    """G.711 mu-law."""

    PCMA = "PCMA"
    """G.711 A-law."""


class Endianness(StrEnum):
    """Byte order of L16 samples on the wire.

    The wire default is little-endian. RTP L16 (RFC 3551) is big-endian; the engine
    byte-swaps at the RTP boundary so the WebSocket side sees host-order samples.
    """

    LITTLE = "little"
    BIG = "big"


class Direction(StrEnum):
    """Stream direction relative to the engine."""

    SEND = "send"
    """Engine to server only (uplink). This is what a tee announces."""

    RECV = "recv"
    """Server to engine only (downlink playout)."""

    DUPLEX = "duplex"
    """Bidirectional. This is what a takeover bridge announces."""


class PlaySource(StrEnum):
    """Where ``play_start`` audio comes from."""

    INLINE = "inline"
    """Decode the base64 ``audioData`` field."""

    BINARY = "binary"
    """Audio arrives as subsequent binary frames until ``play_stop``."""


def _require(data: Mapping[str, Any], key: str, tag: str) -> Any:
    try:
        return data[key]
    except (KeyError, TypeError) as exc:
        raise SiphonProtocolError(f"{tag}: missing required field {key!r}") from exc


def _wrong_type(tag: str, key: str, wanted: str, value: Any) -> SiphonProtocolError:
    return SiphonProtocolError(f"{tag}: field {key!r} must be {wanted}, got {type(value).__name__}")


def _as_str(value: Any, key: str, tag: str) -> str:
    if not isinstance(value, str):
        raise _wrong_type(tag, key, "a string", value)
    return value


def _as_int(value: Any, key: str, tag: str) -> int:
    # bool is an int subclass in Python; the wire never carries a boolean here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong_type(tag, key, "an integer", value)
    return value


def _as_bool(value: Any, key: str, tag: str) -> bool:
    if not isinstance(value, bool):
        raise _wrong_type(tag, key, "a boolean", value)
    return value


def _as_optional_str(data: Mapping[str, Any], key: str, tag: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    return _as_str(value, key, tag)


def _as_enum(enum_type: type[_EnumT], value: Any, key: str, tag: str) -> _EnumT:
    if not isinstance(value, str):
        raise _wrong_type(tag, key, "a string", value)
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(member.value for member in enum_type))
        raise SiphonProtocolError(
            f"{tag}: field {key!r} must be one of [{allowed}], got {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class MediaFormat:
    """The negotiated binary audio format announced in ``start``.

    ``sample_rate`` is the negotiated *wire* rate and is authoritative in both directions for
    the whole life of the stream. A controller can select it independently of the call's codec
    rate, so never assume 8000: frame against this number.
    """

    encoding: Encoding = Encoding.L16
    sample_rate: int = 8000
    channels: int = 1
    bit_depth: int = 16
    endianness: Endianness = Endianness.LITTLE
    ptime: int = 20

    @classmethod
    def telephony_default(cls) -> MediaFormat:
        """Return the engine's default: L16, 8 kHz, mono, little-endian, 20 ms."""
        return cls()

    @property
    def frame_bytes(self) -> int:
        """Return the byte count of one binary audio frame at this format.

        Mirrors ``MediaFormat::frame_bytes`` on the engine side, integer truncation included.
        """
        samples = (self.sample_rate // 1000) * self.ptime * self.channels
        return samples * (self.bit_depth // 8)

    def to_wire(self) -> dict[str, Any]:
        """Return the camelCase mapping the engine serializes for this format."""
        return {
            "encoding": self.encoding.value,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "bitDepth": self.bit_depth,
            "endianness": self.endianness.value,
            "ptime": self.ptime,
        }

    @classmethod
    def from_wire(cls, data: Any, tag: str = "media") -> MediaFormat:
        """Parse a ``media`` object, raising :class:`SiphonProtocolError` on anything unexpected."""
        if not isinstance(data, Mapping):
            raise SiphonProtocolError(f"{tag}: must be an object, got {type(data).__name__}")
        return cls(
            encoding=_as_enum(Encoding, _require(data, "encoding", tag), "encoding", tag),
            sample_rate=_as_int(_require(data, "sampleRate", tag), "sampleRate", tag),
            channels=_as_int(_require(data, "channels", tag), "channels", tag),
            bit_depth=_as_int(_require(data, "bitDepth", tag), "bitDepth", tag),
            endianness=_as_enum(Endianness, _require(data, "endianness", tag), "endianness", tag),
            ptime=_as_int(_require(data, "ptime", tag), "ptime", tag),
        )


@dataclass(frozen=True, slots=True)
class StartData:
    """``start`` -- engine to server, the first text frame, announcing the leg and audio format."""

    TAG: ClassVar[str] = "start"

    stream_id: str
    call_id: str
    direction: Direction
    media: MediaFormat
    tracks: tuple[str, ...] = ()
    metadata: Any | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        wire: dict[str, Any] = {
            "streamId": self.stream_id,
            "callId": self.call_id,
            "direction": self.direction.value,
            "media": self.media.to_wire(),
        }
        if self.tracks:
            wire["tracks"] = list(self.tracks)
        if self.metadata is not None:
            wire["metadata"] = self.metadata
        return wire

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> StartData:
        """Parse the ``data`` object of a ``start`` message."""
        tag = cls.TAG
        raw_tracks = data.get("tracks", [])
        if not isinstance(raw_tracks, list):
            raise SiphonProtocolError(f"{tag}: field 'tracks' must be an array")
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            call_id=_as_str(_require(data, "callId", tag), "callId", tag),
            direction=_as_enum(Direction, _require(data, "direction", tag), "direction", tag),
            media=MediaFormat.from_wire(_require(data, "media", tag), f"{tag}.media"),
            tracks=tuple(_as_str(track, "tracks", tag) for track in raw_tracks),
            metadata=data.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class RenegotiateData:
    """``media_renegotiate`` -- mid-stream audio format change."""

    TAG: ClassVar[str] = "media_renegotiate"

    stream_id: str
    media: MediaFormat

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {"streamId": self.stream_id, "media": self.media.to_wire()}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> RenegotiateData:
        """Parse the ``data`` object of a ``media_renegotiate`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            media=MediaFormat.from_wire(_require(data, "media", tag), f"{tag}.media"),
        )


@dataclass(frozen=True, slots=True)
class PlayStartData:
    """``play_start`` -- server to engine, request playback into the call.

    The engine's v1 takeover bridge rejects an inline (base64) ``play_start`` with an ``error``
    message; downlink audio needs no ``play_start`` at all, binary frames are enough.
    """

    TAG: ClassVar[str] = "play_start"

    stream_id: str
    play_id: str
    source: PlaySource
    audio_data_type: str | None = None
    encoding: Encoding | None = None
    sample_rate: int | None = None
    audio_data: str | None = None
    interruptible: bool = True
    mark_name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        wire: dict[str, Any] = {
            "streamId": self.stream_id,
            "playId": self.play_id,
            "source": self.source.value,
        }
        if self.audio_data_type is not None:
            wire["audioDataType"] = self.audio_data_type
        if self.encoding is not None:
            wire["encoding"] = self.encoding.value
        if self.sample_rate is not None:
            wire["sampleRate"] = self.sample_rate
        if self.audio_data is not None:
            wire["audioData"] = self.audio_data
        wire["interruptible"] = self.interruptible
        if self.mark_name is not None:
            wire["markName"] = self.mark_name
        return wire

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> PlayStartData:
        """Parse the ``data`` object of a ``play_start`` message."""
        tag = cls.TAG
        raw_encoding = data.get("encoding")
        raw_sample_rate = data.get("sampleRate")
        raw_interruptible = data.get("interruptible", True)
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            play_id=_as_str(_require(data, "playId", tag), "playId", tag),
            source=_as_enum(PlaySource, _require(data, "source", tag), "source", tag),
            audio_data_type=_as_optional_str(data, "audioDataType", tag),
            encoding=(
                None if raw_encoding is None else _as_enum(Encoding, raw_encoding, "encoding", tag)
            ),
            sample_rate=(
                None if raw_sample_rate is None else _as_int(raw_sample_rate, "sampleRate", tag)
            ),
            audio_data=_as_optional_str(data, "audioData", tag),
            interruptible=_as_bool(raw_interruptible, "interruptible", tag),
            mark_name=_as_optional_str(data, "markName", tag),
        )


@dataclass(frozen=True, slots=True)
class PlayStopData:
    """``play_stop`` -- end of a ``source: binary`` playback segment."""

    TAG: ClassVar[str] = "play_stop"

    stream_id: str
    play_id: str

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {"streamId": self.stream_id, "playId": self.play_id}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> PlayStopData:
        """Parse the ``data`` object of a ``play_stop`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            play_id=_as_str(_require(data, "playId", tag), "playId", tag),
        )


@dataclass(frozen=True, slots=True)
class ClearData:
    """``clear`` -- server to engine, barge-in flush of buffered playout.

    It takes effect within one tick and the engine answers with a ``mark`` named ``cleared``.
    """

    TAG: ClassVar[str] = "clear"

    stream_id: str
    play_id: str | None = None
    reason: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        wire: dict[str, Any] = {"streamId": self.stream_id}
        if self.play_id is not None:
            wire["playId"] = self.play_id
        if self.reason is not None:
            wire["reason"] = self.reason
        return wire

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> ClearData:
        """Parse the ``data`` object of a ``clear`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            play_id=_as_optional_str(data, "playId", tag),
            reason=_as_optional_str(data, "reason", tag),
        )


@dataclass(frozen=True, slots=True)
class MarkData:
    """``mark`` -- engine to server, a playout boundary was rendered (or skipped on clear)."""

    TAG: ClassVar[str] = "mark"

    stream_id: str
    name: str
    play_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        wire: dict[str, Any] = {"streamId": self.stream_id}
        if self.play_id is not None:
            wire["playId"] = self.play_id
        wire["name"] = self.name
        return wire

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> MarkData:
        """Parse the ``data`` object of a ``mark`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            play_id=_as_optional_str(data, "playId", tag),
            name=_as_str(_require(data, "name", tag), "name", tag),
        )


@dataclass(frozen=True, slots=True)
class DtmfData:
    """``dtmf`` -- engine to server, a detected telephone-event digit (RFC 4733)."""

    TAG: ClassVar[str] = "dtmf"

    stream_id: str
    digit: str
    track: str
    duration_ms: int

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {
            "streamId": self.stream_id,
            "digit": self.digit,
            "track": self.track,
            "durationMs": self.duration_ms,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> DtmfData:
        """Parse the ``data`` object of a ``dtmf`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            digit=_as_str(_require(data, "digit", tag), "digit", tag),
            track=_as_str(_require(data, "track", tag), "track", tag),
            duration_ms=_as_int(_require(data, "durationMs", tag), "durationMs", tag),
        )


@dataclass(frozen=True, slots=True)
class SpeechData:
    """``speech_started`` / ``speech_stopped`` -- engine-side local VAD turn boundaries.

    ``speech_started`` marks the start of caller speech (turn start / barge-in, so the server
    should stop generating); ``speech_stopped`` marks the turn *endpoint*, past the VAD
    hangover, at which the server may commit ASR and run the agent.
    """

    TAG: ClassVar[str] = "speech_started"

    stream_id: str

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {"streamId": self.stream_id}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> SpeechData:
        """Parse the ``data`` object of a ``speech_started``/``speech_stopped`` message."""
        tag = cls.TAG
        return cls(stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag))


@dataclass(frozen=True, slots=True)
class SpeechStartedData(SpeechData):
    """``speech_started`` -- the caller began speaking."""

    TAG: ClassVar[str] = "speech_started"


@dataclass(frozen=True, slots=True)
class SpeechStoppedData(SpeechData):
    """``speech_stopped`` -- the caller's turn ended past the VAD hangover."""

    TAG: ClassVar[str] = "speech_stopped"


@dataclass(frozen=True, slots=True)
class StopData:
    """``stop`` -- graceful close, in either direction."""

    TAG: ClassVar[str] = "stop"

    stream_id: str
    reason: str

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {"streamId": self.stream_id, "reason": self.reason}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> StopData:
        """Parse the ``data`` object of a ``stop`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            reason=_as_str(_require(data, "reason", tag), "reason", tag),
        )


@dataclass(frozen=True, slots=True)
class ErrorData:
    """``error`` -- failure report. ``fatal`` means the socket closes after this."""

    TAG: ClassVar[str] = "error"

    stream_id: str
    code: str
    message: str
    fatal: bool

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {
            "streamId": self.stream_id,
            "code": self.code,
            "message": self.message,
            "fatal": self.fatal,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> ErrorData:
        """Parse the ``data`` object of an ``error`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            code=_as_str(_require(data, "code", tag), "code", tag),
            message=_as_str(_require(data, "message", tag), "message", tag),
            fatal=_as_bool(_require(data, "fatal", tag), "fatal", tag),
        )


@dataclass(frozen=True, slots=True)
class EventData:
    """``event`` -- opaque application event passthrough, in either direction."""

    TAG: ClassVar[str] = "event"

    stream_id: str
    name: str
    payload: Any = None

    def to_wire(self) -> dict[str, Any]:
        """Return the ``data`` object for this message."""
        return {"streamId": self.stream_id, "name": self.name, "payload": self.payload}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> EventData:
        """Parse the ``data`` object of an ``event`` message."""
        tag = cls.TAG
        return cls(
            stream_id=_as_str(_require(data, "streamId", tag), "streamId", tag),
            name=_as_str(_require(data, "name", tag), "name", tag),
            payload=_require(data, "payload", tag),
        )


ControlMessage = (
    StartData
    | RenegotiateData
    | PlayStartData
    | PlayStopData
    | ClearData
    | MarkData
    | DtmfData
    | SpeechStartedData
    | SpeechStoppedData
    | StopData
    | ErrorData
    | EventData
)
"""Union of every control message the envelope can carry."""


_PARSERS: Final[dict[str, Any]] = {
    StartData.TAG: StartData.from_wire,
    RenegotiateData.TAG: RenegotiateData.from_wire,
    PlayStartData.TAG: PlayStartData.from_wire,
    PlayStopData.TAG: PlayStopData.from_wire,
    ClearData.TAG: ClearData.from_wire,
    MarkData.TAG: MarkData.from_wire,
    DtmfData.TAG: DtmfData.from_wire,
    SpeechStartedData.TAG: SpeechStartedData.from_wire,
    SpeechStoppedData.TAG: SpeechStoppedData.from_wire,
    StopData.TAG: StopData.from_wire,
    ErrorData.TAG: ErrorData.from_wire,
    EventData.TAG: EventData.from_wire,
}


def encode_control(message: ControlMessage) -> str:
    """Serialize a control message to a JSON text-frame body.

    The output is byte-identical to what the engine's ``serde_json`` emits for the same
    message: compact separators, camelCase keys, declaration order, absent optionals omitted.
    """
    envelope = {"type": message.TAG, "data": message.to_wire()}
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)


def parse_control(text: str | bytes) -> ControlMessage:
    """Parse a JSON text-frame body into a control message.

    Raises:
        SiphonProtocolError: If the body is not JSON, is not a ``type``/``data`` envelope,
            carries an unknown ``type``, or a field is missing or of the wrong type.

    """
    try:
        envelope = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SiphonProtocolError(f"control frame is not valid JSON: {exc}") from exc

    if not isinstance(envelope, Mapping):
        raise SiphonProtocolError(
            f"control frame must be a JSON object, got {type(envelope).__name__}"
        )

    tag = envelope.get("type")
    if not isinstance(tag, str):
        raise SiphonProtocolError("control frame is missing a string 'type' tag")

    parser = _PARSERS.get(tag)
    if parser is None:
        raise SiphonProtocolError(f"unknown control message type {tag!r}")

    data = envelope.get("data")
    if not isinstance(data, Mapping):
        raise SiphonProtocolError(f"{tag}: 'data' must be an object, got {type(data).__name__}")

    parsed: ControlMessage = parser(data)
    return parsed
