"""Block until something observable is true, or fail with a bounded timeout.

Every wait in this harness goes through here. A ``sleep`` long enough to be reliable on a loaded
machine is also long enough to hide a regression on an idle one, and a ``sleep`` short enough to
be quick is a flake waiting to happen. So nothing here waits for a duration: it waits for a
condition -- a file, a port, a record in the bot's trace, a value on the engine's metrics
endpoint -- and gives up after a stated maximum with a message saying what it was waiting for
and what it last saw.

The poll interval below is not synchronisation. It only decides how often the condition is
re-tested; the condition is what the wait is actually for.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

POLL_SECONDS = 0.2


def _await(description: str, timeout: float, probe: Callable[[], str | None]) -> int:
    """Poll ``probe`` until it reports success, printing the last observation on timeout.

    ``probe`` returns ``None`` while the condition does not hold, or a short success message.
    """
    deadline = time.monotonic() + timeout
    last = "nothing observed yet"
    while time.monotonic() < deadline:
        try:
            outcome = probe()
        except Exception as error:
            last = f"{type(error).__name__}: {error}"
            outcome = None
        if outcome is not None:
            print(f"ready: {description} ({outcome})")
            return 0
        time.sleep(POLL_SECONDS)
    print(f"TIMEOUT after {timeout:g}s waiting for {description}; last saw: {last}")
    return 1


def wait_port(host: str, port: int, timeout: float) -> int:
    """Wait until a TCP connection to ``host:port`` is accepted."""

    def probe() -> str | None:
        with socket.create_connection((host, port), timeout=2):
            return "accepted"

    return _await(f"{host}:{port} to accept connections", timeout, probe)


def wait_http(url: str, status: int, timeout: float) -> int:
    """Wait until an HTTP GET of ``url`` answers with ``status``."""

    def probe() -> str | None:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read(120).decode("utf-8", "replace").strip()
            if response.status == status:
                return f"{response.status} {body!r}"
            raise RuntimeError(f"{response.status} {body!r}")

    return _await(f"{url} to answer {status}", timeout, probe)


def wait_file(path: str, minimum_bytes: int, timeout: float) -> int:
    """Wait until ``path`` exists and holds at least ``minimum_bytes``."""
    target = Path(path)

    def probe() -> str | None:
        if not target.exists():
            return None
        size = target.stat().st_size
        return f"{size} bytes" if size >= minimum_bytes else None

    return _await(f"{path} to reach {minimum_bytes} bytes", timeout, probe)


def wait_trace(path: str, event: str, timeout: float) -> int:
    """Wait until the bot's JSON Lines trace contains a record with ``event``."""
    target = Path(path)

    def probe() -> str | None:
        if not target.exists():
            return None
        seen = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            seen.append(record["event"])
            if record["event"] == event:
                return f"at uplink frame {record.get('uplink_frame')}"
        raise RuntimeError(f"trace has {len(seen)} records, last {seen[-1] if seen else 'none'}")

    return _await(f"{event!r} in {path}", timeout, probe)


def wait_metric(url: str, name: str, equals: float, timeout: float) -> int:
    """Wait until a Prometheus metric on ``url`` reaches ``equals``."""

    def probe() -> str | None:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read().decode("utf-8", "replace")
        for raw in body.splitlines():
            match = re.match(
                r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(-?[\d.eE+]+)$", raw.strip()
            )
            if match and match.group(1) == name:
                value = float(match.group(2))
                if value == equals:
                    return f"{name}={value:g}"
                raise RuntimeError(f"{name}={value:g}")
        raise RuntimeError(f"{name} is not exposed")

    return _await(f"{name} to reach {equals:g} on {url}", timeout, probe)


def main() -> int:
    """Dispatch to one of the waits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    sub = parser.add_subparsers(dest="what", required=True)

    port = sub.add_parser("port", help="wait for a TCP listener")
    port.add_argument("host")
    port.add_argument("port", type=int)

    http = sub.add_parser("http", help="wait for an HTTP endpoint to answer")
    http.add_argument("url")
    http.add_argument("--status", type=int, default=200)

    file_parser = sub.add_parser("file", help="wait for a file to reach a size")
    file_parser.add_argument("path")
    file_parser.add_argument("--min-bytes", type=int, default=1)

    trace = sub.add_parser("trace", help="wait for an event in the bot's trace")
    trace.add_argument("path")
    trace.add_argument("event")

    metric = sub.add_parser("metric", help="wait for a Prometheus metric to reach a value")
    metric.add_argument("url")
    metric.add_argument("name")
    metric.add_argument("--equals", type=float, required=True)

    arguments = parser.parse_args()
    if arguments.what == "port":
        return wait_port(arguments.host, arguments.port, arguments.timeout)
    if arguments.what == "http":
        return wait_http(arguments.url, arguments.status, arguments.timeout)
    if arguments.what == "file":
        return wait_file(arguments.path, arguments.min_bytes, arguments.timeout)
    if arguments.what == "trace":
        return wait_trace(arguments.path, arguments.event, arguments.timeout)
    return wait_metric(arguments.url, arguments.name, arguments.equals, arguments.timeout)


if __name__ == "__main__":
    sys.exit(main())
