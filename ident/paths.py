"""参数辨识专属目录。"""

from __future__ import annotations

from pathlib import Path

IDENT_ROOT = Path(__file__).resolve().parent
GAIL_ROOT = IDENT_ROOT.parent
IDENT_MATLAB = IDENT_ROOT / "matlab"
IDENT_SCRIPTS = IDENT_ROOT / "scripts"
IDENT_TESTS = IDENT_ROOT / "tests"
IDENT_RESULTS = IDENT_ROOT / "results"


def default_results_dir(model_path: str | Path, prefix: str, timestamp: str) -> Path:
    """默认写入 ``ident/results/<checkpoint运行名>/<prefix>_<timestamp>``。"""
    ckpt_run = Path(model_path).resolve().parent.parent.name
    return IDENT_RESULTS / ckpt_run / f"{prefix}_{timestamp}"
