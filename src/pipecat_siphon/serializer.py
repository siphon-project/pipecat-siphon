"""Pipecat frame serializer for the siphon-rtp media WebSocket protocol."""

from __future__ import annotations

import audioop
import dataclasses
from collections import deque
from typing import Any, Literal

from loguru import logger
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import (
    alaw_to_pcm,
    create_stream_resampler,
    interleave_stereo_audio,
    pcm_to_alaw,
    pcm_to_ulaw,
    ulaw_to_pcm,
)
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    CancelWorkerFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

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
    RenegotiateData,
    SiphonProtocolError,
    SpeechStartedData,
    SpeechStoppedData,
    StartData,
    StopData,
    encode_control,
    parse_control,
)

__all__ = ["SiphonFrameSerializer"]

SpeechFrameMode = Literal["vad", "user", "none"]
"""Which pipecat frames the engine's ``speech_started``/``speech_stopped`` turn into."""

StereoInputMode = Literal["caller", "callee", "mix", "interleaved"]
"""What a 2-channel uplink frame becomes: one leg, a mono mix, or a stereo pipecat frame."""

_SAMPLE_WIDTH = 2
"""Bytes per L16 sample. The wire is 16-bit linear PCM whenever ``encoding`` is L16."""


class SiphonFrameSerializer(FrameSerializer):
    """Serializer for the siphon-rtp media WebSocket protocol.

    siphon-rtp dials out as the WebSocket client, so a pipecat bot using this serializer is the
    *server*: stand up a WebSocket server, point the engine's ``ws_uri`` (takeover) or
    ``ws_tee`` (tee) at it, and hand this serializer to the transport.

    The engine's wire is binary audio plus a small ``type``/``data`` JSON control envelope. No
    base64, one binary frame per ptime.

    Sample rates: the ``start`` envelope's ``sampleRate`` is the negotiated **wire** rate and is
    authoritative in both directions. It is independent of the call's codec rate (a controller
    selects it with ``ws_sample_rate`` / ``ws_tee_sample_rate``), so this serializer resamples
    between it and the pipeline's own rates whenever they differ.

    Example::

        serializer = SiphonFrameSerializer(
            params=SiphonFrameSerializer.InputParams(speech_frames="vad"),
        )
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for :class:`SiphonFrameSerializer`.

        Parameters
        ----------
            sample_rate: Optional override for the pipeline input sample rate. Defaults to the
                ``StartFrame``'s ``audio_in_sample_rate``.
            wire_sample_rate: Wire rate assumed before the engine's ``start`` envelope arrives.
                Once ``start`` is seen, its ``sampleRate`` wins unconditionally.
            wire_ptime: Packetization time in milliseconds assumed before ``start``.
            speech_frames: What ``speech_started``/``speech_stopped`` deserialize to. ``"vad"``
                emits ``VADUserStartedSpeakingFrame``/``VADUserStoppedSpeakingFrame``, which
                pipecat's default ``VADUserTurnStartStrategy`` turns into a user turn *and* an
                interruption. ``"user"`` emits ``UserStartedSpeakingFrame``/
                ``UserStoppedSpeakingFrame`` for ``ExternalUserTurnStartStrategy`` /
                ``ExternalUserTurnStopStrategy`` wiring. ``"none"`` ignores them.
            auto_interrupt: Only meaningful with ``speech_frames="user"``, because
                ``ExternalUserTurnStartStrategy`` never interrupts on its own. When True, an
                ``InterruptionWorkerFrame`` follows the ``UserStartedSpeakingFrame``.
            emit_bot_speaking_frames: Whether an engine ``mark`` (a playout boundary rendered or
                skipped) emits ``BotStoppedSpeakingFrame``.
            stereo_input: How a 2-channel uplink frame is presented. Channel 0 is the caller and
                channel 1 is the callee, so ``"caller"`` keeps the far party out of the ASR.
            clear_reason: ``reason`` sent on the ``clear`` message an interruption produces.
            stop_reason: ``reason`` sent on the ``stop`` message an ``EndFrame``/``CancelFrame``
                produces, when the frame carries no reason of its own.
            event_name: ``name`` used for an ``event`` message built from a transport message
                that does not name itself.

        """

        sample_rate: int | None = None
        wire_sample_rate: int | None = None
        wire_ptime: int | None = None
        speech_frames: SpeechFrameMode = "vad"
        auto_interrupt: bool = True
        emit_bot_speaking_frames: bool = True
        stereo_input: StereoInputMode = "caller"
        clear_reason: str = "barge_in"
        stop_reason: str = "pipeline_ended"
        event_name: str = "pipecat"

    def __init__(
        self,
        stream_id: str | None = None,
        call_id: str | None = None,
        params: InputParams | None = None,
    ):
        """Initialize the serializer.

        Args:
            stream_id: Stream id to address control messages to before the engine's ``start``
                envelope arrives. Normally unnecessary: ``start`` is the first text frame and
                carries the id the engine assigned.
            call_id: SIP/leg correlation id, for the same reason.
            params: Configuration parameters.

        """
        params = params or SiphonFrameSerializer.InputParams()
        super().__init__(params)
        self._params: SiphonFrameSerializer.InputParams = params

        self._stream_id = stream_id
        self._call_id = call_id
        self._direction = Direction.DUPLEX
        self._tracks: tuple[str, ...] = ()

        media = MediaFormat.telephony_default()
        if params.wire_sample_rate is not None:
            media = MediaFormat(
                encoding=media.encoding,
                sample_rate=params.wire_sample_rate,
                channels=media.channels,
                bit_depth=media.bit_depth,
                endianness=media.endianness,
                ptime=params.wire_ptime if params.wire_ptime is not None else media.ptime,
            )
        elif params.wire_ptime is not None:
            media = MediaFormat(
                encoding=media.encoding,
                sample_rate=media.sample_rate,
                channels=media.channels,
                bit_depth=media.bit_depth,
                endianness=media.endianness,
                ptime=params.wire_ptime,
            )
        self._media = media

        self._pipeline_in_sample_rate = 0
        self._pipeline_out_sample_rate = 0

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        # A stereo uplink carries two independent speakers; each channel needs its own
        # streaming resampler so their filter histories never mix.
        self._callee_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        # deserialize() can only hand the transport one frame per WebSocket message, so a
        # message that means two things (a turn start *and* an interruption) parks the tail
        # here. The engine's takeover ticker emits a binary frame every ptime, so the queue
        # drains within one packetization interval.
        self._pending_frames: deque[Frame] = deque()
        self._send_only_warned = False

    @property
    def stream_id(self) -> str | None:
        """Return the stream id the engine assigned, once ``start`` has been seen."""
        return self._stream_id

    @property
    def call_id(self) -> str | None:
        """Return the SIP/leg correlation id, once ``start`` has been seen."""
        return self._call_id

    @property
    def media_format(self) -> MediaFormat:
        """Return the negotiated wire format currently in force."""
        return self._media

    @property
    def wire_sample_rate(self) -> int:
        """Return the authoritative negotiated wire sample rate in Hz."""
        return self._media.sample_rate

    @property
    def direction(self) -> Direction:
        """Return the stream direction the engine announced."""
        return self._direction

    @property
    def tracks(self) -> tuple[str, ...]:
        """Return the informational track labels the engine announced."""
        return self._tracks

    # `FrameSerializer.setup(frame)` already narrows `BaseObject.setup(task_manager)` in pipecat
    # itself; this override matches the serializer signature exactly.
    async def setup(self, frame: StartFrame) -> None:  # type: ignore[override]
        """Record the pipeline's sample rates and reconcile them against the wire rate.

        The engine's ``start`` envelope usually arrives *after* this: the transport must be
        listening before the engine can dial it. So this only captures the pipeline side; the
        wire side is whatever ``start`` (or ``wire_sample_rate``) says, and the two are
        reconciled per frame by the resamplers.

        Args:
            frame: The ``StartFrame`` carrying the pipeline configuration.

        """
        self._pipeline_in_sample_rate = self._params.sample_rate or frame.audio_in_sample_rate
        self._pipeline_out_sample_rate = frame.audio_out_sample_rate
        self._log_rate_reconciliation()

    def _log_rate_reconciliation(self) -> None:
        wire = self._media.sample_rate
        if not self._pipeline_in_sample_rate:
            return
        if self._pipeline_in_sample_rate == wire and self._pipeline_out_sample_rate == wire:
            logger.debug(f"{self}: pipeline and wire both at {wire} Hz, no resampling")
            return
        logger.debug(
            f"{self}: resampling between wire {wire} Hz and pipeline "
            f"(in {self._pipeline_in_sample_rate} Hz, out {self._pipeline_out_sample_rate} Hz)"
        )

    #
    # Pipecat frame -> siphon-rtp wire
    #

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a pipecat frame to a siphon-rtp WebSocket message.

        Audio becomes a binary frame in the negotiated format; everything else becomes a JSON
        control envelope. Unhandled frames return ``None`` and are simply not sent.

        Args:
            frame: The pipecat frame to serialize.

        Returns:
            ``bytes`` for a binary audio frame, ``str`` for a control frame, or ``None``.

        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            reason = getattr(frame, "reason", None) or self._params.stop_reason
            return self._control(StopData(stream_id="", reason=str(reason)))
        if isinstance(frame, InterruptionFrame):
            return self._control(
                ClearData(stream_id="", play_id=None, reason=self._params.clear_reason)
            )
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return self._control(self._event_from_message(frame.message))
        if isinstance(frame, AudioRawFrame):
            return await self._serialize_audio(frame)
        return None

    def _event_from_message(self, message: Any) -> EventData:
        """Build an ``event`` message from a transport message payload."""
        if isinstance(message, dict):
            name = message.get("name")
            if isinstance(name, str) and "payload" in message:
                return EventData(stream_id="", name=name, payload=message["payload"])
        return EventData(stream_id="", name=self._params.event_name, payload=message)

    def _control(self, message: ControlMessage) -> str | None:
        """Stamp the stream id onto a control message and encode it, or drop it if unaddressable."""
        if self._stream_id is None:
            logger.warning(
                f"{self}: dropping a {message.TAG!r} control frame, no stream id yet "
                "(the engine's start envelope has not arrived)"
            )
            return None
        return encode_control(dataclasses.replace(message, stream_id=self._stream_id))

    async def _serialize_audio(self, frame: AudioRawFrame) -> bytes | None:
        """Convert a pipecat audio frame into one binary wire frame."""
        if self._direction is Direction.SEND:
            # A tee is send-only; the engine never injects what a server writes back.
            if not self._send_only_warned:
                self._send_only_warned = True
                logger.warning(
                    f"{self}: dropping outbound audio, the engine announced a send-only stream"
                )
            return None

        pcm = frame.audio
        if not pcm:
            return None

        # Pipecat produces host-order (little-endian) 16-bit PCM. Fold any multi-channel frame
        # down to mono first so the resampler sees a single stream of samples.
        if frame.num_channels == 2:
            pcm = audioop.tomono(pcm, _SAMPLE_WIDTH, 0.5, 0.5)
        elif frame.num_channels != 1:
            logger.warning(
                f"{self}: dropping outbound audio with {frame.num_channels} channels, "
                "only mono and stereo are supported"
            )
            return None

        wire = self._media
        if wire.encoding is Encoding.PCMU:
            payload = await pcm_to_ulaw(
                pcm, frame.sample_rate, wire.sample_rate, self._output_resampler
            )
        elif wire.encoding is Encoding.PCMA:
            payload = await pcm_to_alaw(
                pcm, frame.sample_rate, wire.sample_rate, self._output_resampler
            )
        else:
            payload = await self._output_resampler.resample(
                pcm, frame.sample_rate, wire.sample_rate
            )

        if not payload:
            return None

        if wire.channels == 2:
            # The wire is stereo (caller on 0, callee on 1). A bot speaks with one voice, so the
            # same audio goes to both channels rather than silencing one of them.
            payload = audioop.tostereo(payload, _SAMPLE_WIDTH, 1, 1)

        if wire.encoding is Encoding.L16 and wire.endianness is Endianness.BIG:
            payload = audioop.byteswap(payload, _SAMPLE_WIDTH)

        return bytes(payload)

    #
    # siphon-rtp wire -> pipecat frame
    #

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize one siphon-rtp WebSocket message into a pipecat frame.

        Binary messages become ``InputAudioRawFrame``; text messages are control envelopes.
        Malformed input is logged and dropped: the socket is untrusted, so nothing here raises.

        Args:
            data: One WebSocket message, ``bytes`` for audio or ``str`` for control.

        Returns:
            A pipecat frame, or ``None`` when the message maps to nothing.

        """
        frames = await self._deserialize_frames(data)
        self._pending_frames.extend(frames)
        return self._pending_frames.popleft() if self._pending_frames else None

    async def _deserialize_frames(self, data: str | bytes) -> list[Frame]:
        if isinstance(data, (bytes, bytearray, memoryview)):
            audio = await self._deserialize_audio(bytes(data))
            return [audio] if audio is not None else []

        try:
            message = parse_control(data)
        except SiphonProtocolError as error:
            logger.warning(f"{self}: dropping malformed control frame: {error}")
            return []
        return self._frames_for_control(message)

    def _frames_for_control(self, message: ControlMessage) -> list[Frame]:
        if isinstance(message, StartData):
            return self._handle_start(message)
        if isinstance(message, RenegotiateData):
            return self._handle_renegotiate(message)
        if isinstance(message, SpeechStartedData):
            return self._handle_speech_started()
        if isinstance(message, SpeechStoppedData):
            return self._handle_speech_stopped()
        if isinstance(message, DtmfData):
            return self._handle_dtmf(message)
        if isinstance(message, MarkData):
            return self._handle_mark(message)
        if isinstance(message, StopData):
            logger.info(f"{self}: engine stopped the stream: {message.reason}")
            return [EndWorkerFrame(reason=message.reason)]
        if isinstance(message, ErrorData):
            return self._handle_error(message)
        if isinstance(message, EventData):
            logger.debug(f"{self}: engine event {message.name!r}: {message.payload}")
            return []
        # play_start / play_stop / clear are server-to-engine verbs. The engine never sends
        # them, so seeing one means the peer is not siphon-rtp; ignore it rather than act on it.
        logger.debug(f"{self}: ignoring server-to-engine message {message.TAG!r} from the engine")
        return []

    def _handle_start(self, message: StartData) -> list[Frame]:
        self._stream_id = message.stream_id
        self._call_id = message.call_id
        self._direction = message.direction
        self._tracks = message.tracks
        self._media = message.media
        logger.info(
            f"{self}: stream {message.stream_id} on call {message.call_id}, "
            f"{message.direction.value}, {message.media.encoding.value} "
            f"{message.media.sample_rate} Hz x{message.media.channels} "
            f"{message.media.endianness.value}-endian, ptime {message.media.ptime} ms "
            f"({message.media.frame_bytes} bytes per frame)"
        )
        self._log_rate_reconciliation()
        return []

    def _handle_renegotiate(self, message: RenegotiateData) -> list[Frame]:
        previous = self._media
        self._media = message.media
        logger.info(
            f"{self}: wire format renegotiated from {previous.encoding.value} "
            f"{previous.sample_rate} Hz to {message.media.encoding.value} "
            f"{message.media.sample_rate} Hz"
        )
        self._log_rate_reconciliation()
        return []

    def _handle_speech_started(self) -> list[Frame]:
        mode = self._params.speech_frames
        if mode == "none":
            return []
        if mode == "vad":
            # Pipecat's default VADUserTurnStartStrategy turns this into a user turn start and
            # an interruption, so the engine's VAD replaces a local one without further wiring.
            return [VADUserStartedSpeakingFrame()]
        frames: list[Frame] = [UserStartedSpeakingFrame()]
        if self._params.auto_interrupt:
            # ExternalUserTurnStartStrategy is constructed with enable_interruptions=False, so
            # the barge-in has to be asked for explicitly. InterruptionWorkerFrame reaches the
            # pipeline worker, which broadcasts a real InterruptionFrame across the pipeline.
            frames.append(InterruptionWorkerFrame())
        return frames

    def _handle_speech_stopped(self) -> list[Frame]:
        mode = self._params.speech_frames
        if mode == "none":
            return []
        if mode == "vad":
            return [VADUserStoppedSpeakingFrame()]
        return [UserStoppedSpeakingFrame()]

    def _handle_dtmf(self, message: DtmfData) -> list[Frame]:
        try:
            button = KeypadEntry(message.digit)
        except ValueError:
            # RFC 4733 events 12-15 are the A-D digits, which pipecat's KeypadEntry does not
            # model. Nothing downstream could consume them, so drop rather than invent a frame.
            logger.debug(f"{self}: no pipecat keypad entry for DTMF digit {message.digit!r}")
            return []
        return [InputDTMFFrame(button)]

    def _handle_mark(self, message: MarkData) -> list[Frame]:
        if not self._params.emit_bot_speaking_frames:
            return []
        logger.debug(f"{self}: mark {message.name!r} reached (play {message.play_id})")
        return [BotStoppedSpeakingFrame()]

    def _handle_error(self, message: ErrorData) -> list[Frame]:
        logger.error(
            f"{self}: engine error {message.code}: {message.message} (fatal={message.fatal})"
        )
        if message.fatal:
            # The engine closes the socket after a fatal error, so nothing more will arrive to
            # drain a queued frame. CancelWorkerFrame is the one frame that reliably ends the
            # pipeline from here, and it carries the reason for the logs.
            return [CancelWorkerFrame(reason=f"{message.code}: {message.message}")]
        return [ErrorFrame(error=f"{message.code}: {message.message}", fatal=False)]

    async def _deserialize_audio(self, payload: bytes) -> InputAudioRawFrame | None:
        """Convert one binary wire frame into a pipecat input audio frame."""
        if not payload:
            return None

        wire = self._media
        target_rate = self._pipeline_in_sample_rate or wire.sample_rate

        if wire.encoding is Encoding.PCMU:
            pcm = await ulaw_to_pcm(payload, wire.sample_rate, target_rate, self._input_resampler)
            return self._input_frame(pcm, target_rate, 1)
        if wire.encoding is Encoding.PCMA:
            pcm = await alaw_to_pcm(payload, wire.sample_rate, target_rate, self._input_resampler)
            return self._input_frame(pcm, target_rate, 1)

        if len(payload) % _SAMPLE_WIDTH:
            logger.warning(
                f"{self}: dropping a {len(payload)}-byte L16 frame, not a whole number of samples"
            )
            return None
        if wire.endianness is Endianness.BIG:
            payload = audioop.byteswap(payload, _SAMPLE_WIDTH)

        if wire.channels == 2:
            return await self._deserialize_stereo(payload, target_rate)
        if wire.channels != 1:
            logger.warning(
                f"{self}: dropping audio, the wire announced {wire.channels} channels and only "
                "mono and stereo are supported"
            )
            return None

        pcm = await self._input_resampler.resample(payload, wire.sample_rate, target_rate)
        return self._input_frame(pcm, target_rate, 1)

    async def _deserialize_stereo(
        self, payload: bytes, target_rate: int
    ) -> InputAudioRawFrame | None:
        """Split a 2-channel wire frame (caller on 0, callee on 1) per ``stereo_input``."""
        if len(payload) % (_SAMPLE_WIDTH * 2):
            logger.warning(
                f"{self}: dropping a {len(payload)}-byte stereo frame, not a whole number of "
                "sample pairs"
            )
            return None

        wire_rate = self._media.sample_rate
        mode = self._params.stereo_input
        if mode == "interleaved":
            # Resample each leg on its own and re-interleave; resampling interleaved samples as
            # if they were one stream would smear the two speakers into each other.
            left = await self._input_resampler.resample(
                audioop.tomono(payload, _SAMPLE_WIDTH, 1, 0), wire_rate, target_rate
            )
            right = await self._callee_resampler.resample(
                audioop.tomono(payload, _SAMPLE_WIDTH, 0, 1), wire_rate, target_rate
            )
            if not left or not right:
                return None
            return self._input_frame(interleave_stereo_audio(left, right), target_rate, 2)

        if mode == "callee":
            mono = audioop.tomono(payload, _SAMPLE_WIDTH, 0, 1)
        elif mode == "mix":
            mono = audioop.tomono(payload, _SAMPLE_WIDTH, 0.5, 0.5)
        else:
            mono = audioop.tomono(payload, _SAMPLE_WIDTH, 1, 0)

        pcm = await self._input_resampler.resample(mono, wire_rate, target_rate)
        return self._input_frame(pcm, target_rate, 1)

    def _input_frame(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> InputAudioRawFrame | None:
        if not pcm:
            return None
        return InputAudioRawFrame(
            audio=bytes(pcm), sample_rate=sample_rate, num_channels=num_channels
        )
