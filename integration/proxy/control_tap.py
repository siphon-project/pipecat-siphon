"""Transparent tap on the engine's JSON control plane, for the transcript.

Neither siphon-sip nor siphon-rtp logs the bodies of the control commands they exchange -- both
log the verb and the outcome and stop there. That is fine in production and useless in a failed
integration run, where the first question is always "what did the proxy actually ask the engine
for". So the harness puts a tap in the middle.

It is a *tap*, not a stand-in. Every byte received on either side is forwarded to the other
side unchanged and in order; the decoder only reads a copy for the transcript, and a body it
cannot parse is logged as opaque bytes rather than dropped or repaired. Nothing here answers a
command, and if this process dies the run fails instead of quietly continuing against a mock.

The wire is a 4-byte big-endian length prefix followed by a JSON body.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from typing import TextIO

LENGTH_PREFIX_BYTES = 4
MAXIMUM_FRAME_BYTES = 1 << 20
"""The engine's own cap. A prefix larger than this means the stream is out of sync."""


class Transcript:
    """Append-only JSON Lines record of the control exchange."""

    def __init__(self, handle: TextIO) -> None:
        """Initialize the transcript writer."""
        self._handle = handle

    def write(self, direction: str, payload: bytes) -> None:
        """Record one framed message, decoded when it can be."""
        record: dict[str, object] = {"direction": direction, "bytes": len(payload)}
        try:
            record["message"] = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            record["opaque"] = payload.hex()
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()


async def _relay(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    direction: str,
    transcript: Transcript,
) -> None:
    """Forward one direction verbatim, decoding a copy into the transcript."""
    buffer = bytearray()
    synchronised = True
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            # Forward first, always. The transcript must never be able to delay or reorder the
            # bytes the two real peers exchange.
            writer.write(chunk)
            await writer.drain()

            if not synchronised:
                continue
            buffer += chunk
            while len(buffer) >= LENGTH_PREFIX_BYTES:
                length = int.from_bytes(buffer[:LENGTH_PREFIX_BYTES], "big")
                if length > MAXIMUM_FRAME_BYTES:
                    transcript.write(f"{direction}-desync", bytes(buffer[:64]))
                    buffer.clear()
                    synchronised = False
                    break
                if len(buffer) < LENGTH_PREFIX_BYTES + length:
                    break
                start = LENGTH_PREFIX_BYTES
                transcript.write(direction, bytes(buffer[start : start + length]))
                del buffer[: start + length]
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        with contextlib.suppress(ConnectionResetError, BrokenPipeError):
            writer.write_eof()


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
    transcript: Transcript,
) -> None:
    """Bridge one accepted control connection to the engine."""
    try:
        engine_reader, engine_writer = await asyncio.open_connection(upstream_host, upstream_port)
    except OSError as error:
        print(f"control tap: cannot reach the engine: {error}", file=sys.stderr, flush=True)
        client_writer.close()
        return

    await asyncio.gather(
        _relay(client_reader, engine_writer, "to-engine", transcript),
        _relay(engine_reader, client_writer, "from-engine", transcript),
    )
    for writer in (client_writer, engine_writer):
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def main(listen_port: int, upstream: str, transcript_path: str) -> None:
    """Listen for control connections and tap each one."""
    host, _, port = upstream.rpartition(":")
    with open(transcript_path, "w", encoding="utf-8") as handle:
        transcript = Transcript(handle)

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _handle(reader, writer, host, int(port), transcript)

        server = await asyncio.start_server(handler, "0.0.0.0", listen_port)
        print(f"control tap: {listen_port} -> {upstream}", flush=True)
        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream", required=True, help="host:port of the real engine")
    parser.add_argument("--transcript", required=True)
    arguments = parser.parse_args()
    try:
        asyncio.run(main(arguments.listen_port, arguments.upstream, arguments.transcript))
    except KeyboardInterrupt:
        sys.exit(0)
