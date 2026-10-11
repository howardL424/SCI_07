"""Phase A3 机器可读协议结构测试（不实施数据访问门禁）。"""

from __future__ import annotations

import sys
from pathlib import Path

IDENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(IDENT))

from experiment_protocol import load_protocol  # noqa: E402


def test_protocol_draft_has_required_phase_a_contract() -> None:
    protocol = load_protocol()
    assert protocol["schema_version"] == "p0_4_experiment_protocol_v1"
    assert protocol["status"] == "frozen"
    assert protocol["freeze_state"]["final_test_authorized"] is True
    assert protocol["freeze_state"]["hyperparameters_finally_frozen"] is True
    assert protocol["datasets"]["final_test"]["status"] == "ready_reserved"
    assert protocol["datasets"]["final_test"]["planned_path"] == "samples_clean_test.mat"
    assert protocol["datasets"]["final_test"]["planned_valid_samples"] == 300
    assert protocol["policies"]["shared_interface"]["observation_dimension"] == 13
    assert protocol["policies"]["shared_interface"]["action_dimension"] == 3
    assert protocol["paper_sync"]["known_differences"]
    assert protocol["phase_guards"]["code_access_gate_enabled"] is False
    assert protocol["phase_guards"]["enforcement"] == "none"
    assert protocol["actor_observation_contract"]["p0_4_sources"][0].startswith(
        "D1_KF"
    )
    assert protocol["implementation_status"]["phase_F_started"] is True
    assert protocol["implementation_status"]["phase_F1_freeze"] == "complete"
    assert protocol["diagnostic_protocol"]["info_vector_length"] == 96
    assert protocol["unified_identification_metric"]["burn_in_steps"] == 5
    assert len(protocol["implementation_files_sha256"]) >= 8
    assert protocol["datasets"]["final_test"]["valid_samples"] == 300


if __name__ == "__main__":
    test_protocol_draft_has_required_phase_a_contract()
    print("experiment protocol tests: 1 passed")

