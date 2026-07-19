"""一次性闭环评估脚本：对已训好的 BC checkpoint 做 MC + 专家回放对照。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from train_bc import prepare_bc_data, run_closed_loop  # noqa: E402


def main() -> None:
    args = SimpleNamespace(
        expert=r"E:\matlab\result_0714\expert_lib_1000.mat",
        aligned=str(ROOT / "data" / "expert_transitions_aligned.mat"),
        terminal_cache=str(ROOT / "data" / "expert_terminal_cache.npz"),
        sample_plan=str(ROOT / "data" / "expert_set_99k.npz"),
        budget=99_000,
        seed=0,
        sample_seed=0,
        val_ratio=0.1,
        actor_include_intercepted=False,
        with_ok_mode=False,
        save_dir=str(ROOT / "checkpoints_bc" / "bc_ratio_seed0"),
        smoke=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_episodes=20,
        replay_traj=20,
        action_mapping=DEFAULT_ACTION_MAPPING,
        name="bc_ratio",
    )
    print("[closed_loop] prepare data…", flush=True)
    pack = prepare_bc_data(args)
    ckpt = Path(args.save_dir) / "bc_actor_best.pt"
    print(f"[closed_loop] eval {ckpt}…", flush=True)
    cl = run_closed_loop(args, pack, ckpt)

    report_path = Path(args.save_dir) / "bc_actor_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["closed_loop"] = cl
    j = report["judgment"]
    bc_both = cl["bc_on_expert_ic"].get("both_ok_rate", 0.0)
    ex_both = cl["expert_replay"].get("both_ok_rate", 0.0)
    if ex_both > 0 and bc_both < 0.5 * ex_both:
        j["verdict"] = "fail_closed_loop"
        note = (
            f"闭环联合成功率 {bc_both:.2%} 明显低于专家回放 {ex_both:.2%}，"
            "先做状态/动作消融，不要直接进 TD3。"
        )
        if note not in j["notes"]:
            j["notes"].append(note)
    report["judgment"] = j
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("CLOSED_LOOP", json.dumps(cl, ensure_ascii=False, indent=2), flush=True)
    print("VERDICT", j["verdict"], flush=True)


if __name__ == "__main__":
    main()
