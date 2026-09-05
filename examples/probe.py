"""Probe a bot's media WebSocket without a phone, a SIP stack or the engine.

Every way a bot can be broken above the media path -- a model rejecting the request, a pipeline
that cancelled itself, a greeting that never runs -- looks identical from the outside: a socket
that connects, a bot that reports healthy, and silence. This tells the two apart. It speaks the
engine's half of the wire: connect, send `start`, stream silence, and count what comes back.

    python examples/probe.py ws://127.0.0.1:9001/stream

A working bot answers with audio within a second or two, because it greets the caller first::

    start sent: stream probe-1, 16000 Hz, 20 ms ptime
    first audio frame after 840 ms
    ...
    3.0 s elapsed: 118 audio frames in, 2360 ms of audio, 1 control message

A bot that is broken above the media path answers with nothing at all, and that is the whole
diagnosis::

    3.0 s elapsed: 0 audio frames in, 0 ms of audio, 0 control messages
    no audio came back -- the bot accepted the socket and never spoke

Then read the bot's own log: this says where to look, not what is wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time

import websockets

from pipecat_siphon.protocol import (
    Direction,
    Encoding,
    Endianness,
    MediaFormat,
    StartData,
    encode_control,
)

STREAM_ID = "probe-1"
"""Stream id announced in `start`. The bot echoes it on anything it addresses back."""

CALL_ID = "probe@example.invalid"
"""Stands in for the SIP Call-ID the engine would expand into the `ws_uri`.

RFC 2606 reserves `.invalid`, so this can never collide with a real call."""


def build_start(sample_rate: int, ptime_ms: int) -> str:
    """Build the `start` envelope the engine sends as its first text frame.

    Built through the package's own protocol module rather than hand-rolled JSON, so the probe
    cannot drift from the wire the serializer is written against.
    """
    return encode_control(
        StartData(
            stream_id=STREAM_ID,
            call_id=CALL_ID,
            direction=Direction.DUPLEX,
            media=MediaFormat(
                encoding=Encoding.L16,
                sample_rate=sample_rate,
                channels=1,
                bit_depth=16,
                endianness=Endianness.LITTLE,
                ptime=ptime_ms,
            ),
        )
    )


async def stream_silence(
    connection: websockets.ClientConnection, sample_rate: int, ptime_ms: int
) -> None:
    """Send silent uplink frames at the ptime clock, exactly as the engine would.

    Silence rather than speech on purpose: the bot should greet the caller before anyone says
    anything, so this probes the greeting path without needing a recognizer to hear anything.
    """
    samples_per_frame = sample_rate * ptime_ms // 1000
    frame = b"\x00" * (samples_per_frame * 2)
    next_send = time.monotonic()
    while True:
        await connection.send(frame)
        next_send += ptime_ms / 1000
        await asyncio.sleep(max(0.0, next_send - time.monotonic()))


async def probe(uri: str, seconds: float, sample_rate: int, ptime_ms: int) -> int:
    """Run the probe. Returns a process exit code: 0 if audio came back, 1 if none did."""
    try:
        connection = await websockets.connect(uri)
    except OSError as error:
        print(f"could not connect to {uri}: {error}", file=sys.stderr)
        return 1

    audio_frames = 0
    audio_bytes = 0
    control_messages = 0
    first_audio_at: float | None = None

    async with connection:
        await connection.send(build_start(sample_rate, ptime_ms))
        print(f"start sent: stream {STREAM_ID}, {sample_rate} Hz, {ptime_ms} ms ptime")

        started = time.monotonic()
        sender = asyncio.create_task(stream_silence(connection, sample_rate, ptime_ms))
        try:
            while True:
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    message = await asyncio.wait_for(connection.recv(), remaining)
                except TimeoutError:
                    break
                except websockets.ConnectionClosed:
                    print("the bot closed the socket")
                    break

                if isinstance(message, bytes):
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                        latency_ms = (first_audio_at - started) * 1000
                        print(f"first audio frame after {latency_ms:.0f} ms")
                    audio_frames += 1
                    audio_bytes += len(message)
                else:
                    control_messages += 1
                    print(f"control message: {_describe(message)}")
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    elapsed = time.monotonic() - started
    audio_ms = audio_bytes / 2 / sample_rate * 1000
    print(
        f"{elapsed:.1f} s elapsed: {audio_frames} audio frames in, "
        f"{audio_ms:.0f} ms of audio, {control_messages} control messages"
    )

    if audio_frames == 0:
        print("no audio came back -- the bot accepted the socket and never spoke", file=sys.stderr)
        return 1
    return 0


def _describe(message: str) -> str:
    """Summarise a control frame for the log, without trusting it to be well formed."""
    try:
        parsed = json.loads(message)
    except ValueError:
        return f"unparseable text frame ({len(message)} bytes)"
    if not isinstance(parsed, dict):
        return f"unexpected JSON ({type(parsed).__name__})"
    return str(parsed.get("type", "no type field"))


def parse_arguments() -> argparse.Namespace:
    """Parse the target URI and the wire shape to announce."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri", help="the bot's media WebSocket, e.g. ws://127.0.0.1:9001/stream")
    parser.add_argument("--seconds", type=float, default=3.0, help="how long to listen (default 3)")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="wire sample rate to announce, matching the bot's pipeline (default 16000)",
    )
    parser.add_argument(
        "--ptime", type=int, default=20, help="packetization time in ms (default 20)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        sys.exit(
            asyncio.run(
                probe(arguments.uri, arguments.seconds, arguments.sample_rate, arguments.ptime)
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)
