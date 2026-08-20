"""Pipecat frame serializer for the siphon-rtp media WebSocket protocol.

siphon-rtp keeps its own native wire: binary L16 audio frames alongside a small
``{"type", "data"}`` JSON control envelope. This package is the adapter that lets a pipecat
pipeline speak it, in the same shape as the serializers pipecat ships in tree.

The engine dials out as the WebSocket client, so a bot built on this is a WebSocket *server*.
"""

from pipecat_siphon.protocol import (
    ClearData,
    ControlMessage,
    Direction,
    DtmfData,
    Encoding,
    Endianness,
    ErrorData,
    EventData,
    MarkData,
    MediaFormat,
    PlaySource,
    PlayStartData,
    PlayStopData,
    RenegotiateData,
    SiphonProtocolError,
    SpeechData,
    SpeechStartedData,
    SpeechStoppedData,
    StartData,
    StopData,
    encode_control,
    parse_control,
)
from pipecat_siphon.serializer import (
    SiphonFrameSerializer,
    SpeechFrameMode,
    StereoInputMode,
)

__version__ = "0.1.0"

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
    "SiphonFrameSerializer",
    "SiphonProtocolError",
    "SpeechData",
    "SpeechFrameMode",
    "SpeechStartedData",
    "SpeechStoppedData",
    "StartData",
    "StereoInputMode",
    "StopData",
    "__version__",
    "encode_control",
    "parse_control",
]
