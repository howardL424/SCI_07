"""BC 可学习性诊断：贴边分层误差 + 近邻多解性分析（纯离线）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rl_td3"))

from action_mapping import ACTION_MAPPING_RATIO  # noqa: E402
from bc_actor import load_bc_checkpoint  # noqa: E402
from expert_sampling import bc_arrays_from_plan, load_sample_plan  # noqa: E402
from load_expert_buffer import apply_action_mapping, load_transition_data  # noqa: E402
from obs_norm import normalize_obs  # noqa: E402

P1 = REPO / "P1_E1_BC_WARMSTART_20260927"
OUT = REPO / "P1_E1V_VANILLA_20260928"
ACT_NAMES = ["rhoE_n", "rhoT_n", "gamma_n"]


def load_seed_arrays(seed: int):
    plan = load_sample_plan(P1 / "runs" / f"seed_{seed}" / "bc" / f"bc_sample_plan_seed{seed}.npz")
    data = load_transition_data(
        "E:\\matlab\\result_0714\\expert_lib_1000.mat",
        aligned_path=str(REPO / "rl_td3" / "data" / "expert_transitions_aligned.mat"),
    )
    src_map = str(data.get("action_mapping_version", "td3_box_log_v1"))
    data = apply_action_mapping(data, ACTION_MAPPING_RATIO, source_version=src_map)
    states_n = normalize_obs(np.asarray(data["states"], dtype=np.float32)).astype(np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    s_tr, a_tr, s_va, a_va = bc_arrays_from_plan(states_n, actions, plan)
    return s_tr, a_tr, s_va, a_va


def main() -> None:
    lines = ["# BC 可学习性诊断（离线，无需训练）", ""]
    for seed in (0, 42):
        s_tr, a_tr, s_va, a_va = load_seed_arrays(seed)
        ckpt = P1 / "runs" / f"seed_{seed}" / "bc" / f"bc_seed{seed}_best.pt"
        actor, payload = load_bc_checkpoint(str(ckpt), device="cpu")
        actor.eval()
        with torch.no_grad():
            pred = actor(torch.as_tensor(s_va)).numpy()
        se = ((pred - a_va) ** 2).mean(axis=1)
        ae = np.abs(pred - a_va)
        sat = (np.abs(a_va) >= 0.99).any(axis=1)
        pred_sat = (np.abs(pred) >= 0.99).any(axis=1)
        lines.append(f"## seed {seed}（epoch {payload.get('epoch')}，val n={len(s_va)}）")
        lines.append("")
        lines.append(f"- val MSE={se.mean():.4f}；专家贴边样本占比={sat.mean() * 100:.1f}%；BC预测贴边占比={pred_sat.mean() * 100:.3f}%")
        lines.append(f"- 贴边真值 MAE={ae[sat].mean():.4f}（n={sat.sum()}）vs 非贴边 MAE={ae[~sat].mean():.4f}（n={(~sat).sum()}）")
        lines.append(f"- 贴边真值 MSE={se[sat].mean():.4f} vs 非贴边 MSE={se[~sat].mean():.4f}")
        for j, name in enumerate(ACT_NAMES):
            m_sat = ae[sat][:, j].mean() if sat.sum() else float("nan")
            m_nonsat = ae[~sat][:, j].mean() if (~sat).sum() else float("nan")
            lines.append(f"- {name}：专家贴边率={(np.abs(a_va[:, j]) >= 0.99).mean() * 100:.1f}%，贴边MAE={m_sat:.4f}，非贴边MAE={m_nonsat:.4f}")
        # gamma 符号准确率
        gs = np.sign(a_va[:, 2])
        ps = np.sign(pred[:, 2])
        nz = gs != 0
        lines.append(f"- gamma 符号准确率（真值非零）={(gs[nz] == ps[nz]).mean() * 100:.1f}%")
        lines.append("")
        # 近邻多解性：在 train 中找每个 val 点的 50 近邻（分块）
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(len(s_va), size=min(2000, len(s_va)), replace=False)
        sub = s_va[sub_idx]
        K = 50
        spreads = []
        BS = 500
        for b in range(0, len(sub), BS):
            blk = sub[b : b + BS]
            d2 = ((blk[:, None, :] - s_tr[None, :, :]) ** 2).sum(axis=2)
            nn = np.argpartition(d2, K, axis=1)[:, :K]
            nbr_a = a_tr[nn]  # (B,K,3)
            spreads.append(nbr_a.max(axis=1) - nbr_a.min(axis=1))
        spread = np.concatenate(spreads, axis=0)  # (2000,3)
        multi_gamma = (spread[:, 2] > 1.0).mean() * 100
        multi_any = ((spread > 1.0).any(axis=1)).mean() * 100
        sub_se = se[sub_idx]
        hi = sub_se >= np.median(sub_se)
        lines.append("## 近邻多解性（val抽样2000点，train中50近邻，状态为归一化13D欧氏）")
        lines.append("")
        lines.append(f"- 近邻gamma极差>1.0（半量程）的val点占比={multi_gamma:.1f}%")
        lines.append(f"- 任一动作维极差>1.0的占比={multi_any:.1f}%")
        lines.append(f"- 近邻gamma极差均值：高误差半组={spread[hi][:, 2].mean():.3f} vs 低误差半组={spread[~hi][:, 2].mean():.3f}")
        lines.append(f"- 近邻rhoE极差均值：高误差半组={spread[hi][:, 0].mean():.3f} vs 低误差半组={spread[~hi][:, 0].mean():.3f}")
        lines.append("")
    (OUT / "LEARNABILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
