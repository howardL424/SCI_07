"""Compatibility launcher. Implementation: ident/scripts/export_ident_params.py."""

from pathlib import Path
import runpy
import sys

_target = Path(__file__).resolve().parent.parent / "ident" / "scripts" / "export_ident_params.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
