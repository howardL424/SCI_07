"""Compatibility shim. Implementation: ident/ident_cfg.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent / "ident" / "ident_cfg.py"
_spec = importlib.util.spec_from_file_location("ident_cfg_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

IDENT_CLI_DEFAULTS = _mod.IDENT_CLI_DEFAULTS
add_ident_args = _mod.add_ident_args
matlab_cfg_from_args = _mod.matlab_cfg_from_args
save_ident_run_config = _mod.save_ident_run_config
