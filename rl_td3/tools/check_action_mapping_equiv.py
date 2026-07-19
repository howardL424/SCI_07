"""
check_action_mapping_equiv.py — 抽检旧/新动作映射下 a_M 与 next_state 等价。

用法:
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/check_action_mapping_equiv.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from action_mapping import (  # noqa: E402
    ACTION_MAPPING_RATIO,
    ACTION_MAPPING_TD3_BOX,
    remap_action_norm,
)
from load_expert_buffer import (  # noqa: E402
    DEFAULT_ALIGNED_PATH,
    DEFAULT_EXPERT_PATH,
    _as_NxD,
    _get_transition_field,
    _iter_trajectories,
    _load_results,
    load_transition_data,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument("--aligned", type=str, default=DEFAULT_ALIGNED_PATH)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = load_transition_data(args.expert, aligned_path=args.aligned)
    src = str(data.get("action_mapping_version", ACTION_MAPPING_TD3_BOX))
    print(f"[equiv] source_action_mapping={src}  n_steps={len(data['actions'])}")
    if src != ACTION_MAPPING_TD3_BOX:
        print(f"[equiv] 源已是 {src}，跳过旧→新抽检（仅做自洽往返）")
        sys.exit(0)

    intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()
    n_steps = np.asarray(data["n_steps"], dtype=np.int32).ravel()
    offsets = np.concatenate([[0], np.cumsum(n_steps)])
    cand = []
    for tid in range(len(n_steps)):
        if intercepted[tid] or n_steps[tid] < 3:
            continue
        cand.append(int(offsets[tid] + n_steps[tid] // 2))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(cand, size=min(args.n, len(cand)), replace=False)

    import matlab
    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.addpath(str(ROOT / "matlab"), nargout=0)
    eng.addpath(str(REPO), nargout=0)
    eng.addpath(str(REPO / "expert_trajectory"), nargout=0)

    results = _load_results(args.expert)
    trajs = _iter_trajectories(results)
    abs_by_src: dict[int, np.ndarray] = {}
    for i, traj in enumerate(trajs):
        sabs = _get_transition_field(traj, "state_abs")
        if sabs is None:
            continue
        abs_by_src[i + 1] = _as_NxD(sabs, 18)

    cfg_old = {"action_mapping_version": ACTION_MAPPING_TD3_BOX}
    cfg_new = {"action_mapping_version": ACTION_MAPPING_RATIO}
    core_old = eng.stage2_core_init(cfg_old, nargout=1)
    core_new = eng.stage2_core_init(cfg_new, nargout=1)

    max_a = 0.0
    max_s = 0.0
    n_ok = 0
    for gi in pick:
        tid = int(np.searchsorted(offsets[1:], int(gi), side="right"))
        local = int(gi - offsets[tid])
        src_i = int(data["source_idx"][tid])
        if src_i not in abs_by_src or local >= len(abs_by_src[src_i]):
            continue
        sabs = abs_by_src[src_i][local].astype(np.float64).ravel()
        Tev = float(data["t_end"][tid])
        t = float(local) * 0.1
        a_old = np.asarray(data["actions"][gi], dtype=np.float64).ravel()
        a_new = remap_action_norm(
            a_old,
            from_version=ACTION_MAPPING_TD3_BOX,
            to_version=ACTION_MAPPING_RATIO,
        ).astype(np.float64).ravel()

        sabs_ml = matlab.double(sabs.tolist())
        a_old_ml = matlab.double(a_old.tolist())
        a_new_ml = matlab.double(a_new.tolist())
        _, s1, _, f1 = eng.stage2_step(core_old, sabs_ml, float(t), float(Tev), a_old_ml, nargout=4)
        _, s2, _, f2 = eng.stage2_step(core_new, sabs_ml, float(t), float(Tev), a_new_ml, nargout=4)

        eng.workspace["f1"] = f1
        eng.workspace["f2"] = f2
        a_m1 = np.array(eng.eval("f1.a_M"), dtype=np.float64).ravel()
        a_m2 = np.array(eng.eval("f2.a_M"), dtype=np.float64).ravel()
        s1a = np.array(s1, dtype=np.float64).ravel()
        s2a = np.array(s2, dtype=np.float64).ravel()
        max_a = max(max_a, float(np.max(np.abs(a_m1 - a_m2))))
        max_s = max(max_s, float(np.max(np.abs(s1a - s2a))))
        n_ok += 1

    eng.quit()
    print(f"[equiv] checked={n_ok}/{len(pick)}")
    print(f"[equiv] max|Δa_M|={max_a:.3e}  max|Δs'|={max_s:.3e}")
    tol_a, tol_s = 1e-4, 1e-5
    if n_ok == 0:
        print("[FAIL] 无有效抽检样本（缺 state_abs？）")
        sys.exit(1)
    if max_a > tol_a or max_s > tol_s:
        print("[FAIL] 旧/新映射不等价")
        sys.exit(1)
    print("[OK] 专家标签旧→新 a_M / next_state 等价")


if __name__ == "__main__":
    main()
