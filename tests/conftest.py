"""Make the example scripts importable from the tests.

`examples/` is a flat set of runnable scripts rather than a package -- that is the point of an
example -- so it is not on the import path by default. The tests that exercise the agent
example's logic need it there, the same way pytest already puts `tests/` itself on the path for
`wire_fixtures`. Mirrored by `mypy_path` in pyproject.toml so the type checker resolves it too.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
