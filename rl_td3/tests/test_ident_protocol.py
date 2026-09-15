"""Compatibility runner. Implementation: ident/tests/test_ident_protocol.py."""

from pathlib import Path
import runpy

_target = Path(__file__).resolve().parents[2] / "ident" / "tests" / "test_ident_protocol.py"
runpy.run_path(str(_target), run_name="__main__")
