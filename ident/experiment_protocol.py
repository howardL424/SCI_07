"""P0-4 机器可读实验协议加载器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parent / "protocols" / "p0_4_protocol.draft.json"
)


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    """读取并做最低结构校验；不访问协议中记录的任何数据集。"""
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "policies",
        "actor_observation_contract",
        "matlab_python_config",
        "success_criteria",
        "datasets",
        "phase_guards",
        "freeze_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"实验协议缺少字段: {', '.join(missing)}")
    return payload

