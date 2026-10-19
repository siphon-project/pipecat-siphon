"""Pipecat frame serializer for the siphon-rtp media WebSocket protocol.

siphon-rtp keeps its own native wire: binary L16 audio frames alongside a small
``{"type", "data"}`` JSON control envelope. This package is the adapter that lets a pipecat
pipeline speak it, in the same shape as the serializers pipecat ships in tree.

The engine dials out as the WebSocket client, so a bot built on this is a WebSocket *server*.
"""

from pipecat_siphon.outbound import (
    CallOutcome,
    ControlCall,
    PlacedCall,
    PlacedCalls,
    PlacingClient,
    outcome_from_event,
)
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
from pipecat_siphon.voice_session import (
    BriefOption,
    Chose,
    Failed,
    Finished,
    Report,
    SessionBrief,
    VoiceSession,
    VoiceSessionClient,
    encode_report,
    parse_brief,
    session_token_from,
)

__version__ = "0.3.0"

__all__ = [
    "BriefOption",
    "CallOutcome",
    "Chose",
    "ClearData",
    "ControlCall",
    "ControlMessage",
    "Direction",
    "DtmfData",
    "Encoding",
    "Endianness",
    "ErrorData",
    "EventData",
    "Failed",
    "Finished",
    "MarkData",
    "MediaFormat",
    "PlacedCall",
    "PlacedCalls",
    "PlacingClient",
    "PlaySource",
    "PlayStartData",
    "PlayStopData",
    "RenegotiateData",
    "Report",
    "SessionBrief",
    "SiphonFrameSerializer",
    "SiphonProtocolError",
    "SpeechData",
    "SpeechFrameMode",
    "SpeechStartedData",
    "SpeechStoppedData",
    "StartData",
    "StereoInputMode",
    "StopData",
    "VoiceSession",
    "VoiceSessionClient",
    "__version__",
    "encode_control",
    "encode_report",
    "outcome_from_event",
    "parse_brief",
    "parse_control",
    "session_token_from",
]
