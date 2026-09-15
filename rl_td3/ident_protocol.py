"""Compatibility shim. Implementation: ident/ident_protocol.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent / "ident" / "ident_protocol.py"
_spec = importlib.util.spec_from_file_location("ident_protocol_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

IDENT_INFO_LEN = _mod.IDENT_INFO_LEN
parse_ident_diag = _mod.parse_ident_diag
