"""Compatibility launcher. Implementation: ident/scripts/run_ident_shadow_diag_pure.py."""

from pathlib import Path
import runpy
import sys

_target = Path(__file__).resolve().parent.parent / "ident" / "scripts" / "run_ident_shadow_diag_pure.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
