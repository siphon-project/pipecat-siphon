"""Make the example scripts importable from the tests.

`examples/` is a flat set of runnable scripts rather than a package -- that is the point of an
example -- so it is not on the import path by default. The tests that exercise the agent
example's logic need it there, the same way pytest already puts `tests/` itself on the path for
`wire_fixtures`. Mirrored by `mypy_path` in pyproject.toml so the type checker resolves it too.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

# The integration harness's tools are a `tools` package under `integration/`, imported by that name
# inside the harness's own containers. The call-scenario harness keeps its pure logic there and
# tests it from here, so the same name has to resolve.
_INTEGRATION = Path(__file__).resolve().parent.parent / "integration"
if str(_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION))

from fake_pbx import FakePbx  # noqa: E402  (tests/ is on the path once pytest has collected this)


@pytest.fixture
async def pbx() -> AsyncIterator[FakePbx]:
    """Serve a fake siphon-pbx on loopback: `BRIEF_MENU`, and every report taken."""
    fake = FakePbx()
    server = TestServer(fake.app(), host="127.0.0.1")
    await server.start_server()
    fake.server = server
    try:
        yield fake
    finally:
        await server.close()
