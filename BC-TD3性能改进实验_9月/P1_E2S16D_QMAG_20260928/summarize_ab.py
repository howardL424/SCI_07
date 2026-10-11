"""summarize_ab.py — 13D vs 16D 纯 BC A/B 汇总（离线，无需训练）。

对每 seed×每臂读取 report.json 最优 epoch 指标，并用 best checkpoint +
落盘划分复算：gamma 符号准确率、专家/BC 贴边率、近邻动作歧义（13D vs 16D）。
输出 AB_SUMMARY.md 与 ab_summary.json。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/summarize_ab.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
GAIL_ROOT = EXP.parent.parent
sys.path.insert(0, str(GAIL_ROOT / "rl_td3"))

from bc_actor import load_bc_checkpoint, predict_actions  # noqa: E402
from expert_sampling import bc_arrays_from_plan, load_sample_plan  # noqa: E402
from load_expert_buffer import apply_action_mapping, load_transition_data  # noqa: E402
from obs_norm import normalize_obs  # noqa: E402
from obs_qmag import normalize_obs16  # noqa: E402

EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"
ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")
SEEDS = (0, 42, 21)
BOUND = 0.99


def _gamma_sign_acc(pred: np.ndarray, tgt: np.ndarray, gidx: int = 2) -> float:
    m = np.abs(tgt[:, gidx]) > 1e-6
    if not m.sum():
        return float("nan")
    return float(np.mean(np.sign(pred[m, gidx]) == np.sign(tgt[m, gidx])))


def _nn_ambiguity(s_tr: np.ndarray, s_va: np.ndarray, a_tr: np.ndarray,
                  n_q: int = 2000, k: int = 50, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s_va), size=min(n_q, len(s_va)), replace=False)
    q = s_va[idx].astype(np.float64)
    tr = s_tr.astype(np.float64)
    # 分块算距离，避免一次性 2000×68k×16 爆内存
    grange, anyrange = [], []
    for i in range(0, len(q), 250):
        d2 = ((q[i:i + 250, None, :] - tr[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        aa = a_tr[nn]  # [b,k,3]
        rngs = aa.max(axis=1) - aa.min(axis=1)
        grange.append(rngs[:, 2])
        anyrange.append(rngs.max(axis=1))
    grange = np.concatenate(grange)
    anyrange = np.concatenate(anyrange)
    return {"gamma_range_gt1": float(np.mean(grange > 1.0)),
            "any_range_gt1": float(np.mean(anyrange > 1.0))}


def main() -> None:
    print("[ab] 加载 aligned transition（与各 run 同源）…")
    data = load_transition_data(EXPERT, aligned_path=ALIGNED)
    src_map = str(data.get("action_mapping_version", "td3_box_log_v1"))
    data = apply_action_mapping(data, "ratio_log_box_v1", source_version=src_map)
    states13 = normalize_obs(data["states"]).astype(np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    q_all = np.load(EXP / "q_all_aligned_order.npy")
    device = torch.device("cpu")
    rows = []
    for seed in SEEDS:
        plan_file = EXP / f"runs/ab13_s{seed}/bc_sample_plan_seed{seed}.npz"
        if not plan_file.exists():
            # seed0 配对复用了默认 99k 计划（与 P1 BC 同划分）
            plan_file = GAIL_ROOT / "rl_td3" / "data" / "expert_set_99k.npz"
        plan = load_sample_plan(str(plan_file))
        s13_tr, a_tr, s13_va, a_va = bc_arrays_from_plan(states13, actions, plan)
        for arm, obs_dim in (("13D", 13), ("16D", 16)):
            rep = json.load(open(
                EXP / f"runs/ab{obs_dim}_s{seed}/bc_ab{obs_dim}_s{seed}_report.json",
                encoding="utf-8"))
            if obs_dim == 16:
                sc = np.load(EXP / f"runs/ab16_s{seed}/q_scale_seed{seed}.npz")["q_scale"]
                states16 = normalize_obs16(
                    np.asarray(data["states"]), q_all, np.asarray(sc)).astype(np.float32)
                s_tr, _, s_va, _ = bc_arrays_from_plan(states16, actions, plan)
            else:
                s_tr, s_va = s13_tr, s13_va
            ckpt = EXP / f"runs/ab{obs_dim}_s{seed}/bc_ab{obs_dim}_s{seed}_best.pt"
            actor, _ = load_bc_checkpoint(ckpt, device=device)
            pred = predict_actions(actor, s_va, device=device)
            bv = rep["history"][rep["best_epoch"] - 1]["val"]
            exp_br = {i: float(np.mean(np.abs(a_va[:, i]) >= BOUND))
                      for i in range(3)}
            bc_br = {i: float(np.mean(np.abs(pred[:, i]) >= BOUND))
                     for i in range(3)}
            amb = _nn_ambiguity(s_tr, s_va, a_tr, seed=seed)
            rows.append({
                "seed": seed, "arm": arm,
                "best_epoch": rep["best_epoch"], "val_mse": bv["mse"],
                "val_mae": bv["mae"],
                "mae_rhoE": bv["mae_per_action"]["rhoE_n"],
                "mae_rhoT": bv["mae_per_action"]["rhoT_n"],
                "mae_gamma": bv["mae_per_action"]["gamma_n"],
                "gamma_sign_acc": _gamma_sign_acc(pred, a_va),
                "exp_boundary_any": float(np.mean(np.abs(a_va).max(axis=1) >= BOUND)),
                "bc_boundary_any": float(np.mean(np.abs(pred).max(axis=1) >= BOUND)),
                "exp_gamma_edge": exp_br[2], "bc_gamma_edge": bc_br[2],
                **amb,
            })
            print(f"[ab] seed={seed} {arm}: best_ep={rep['best_epoch']} "
                  f"MSE={bv['mse']:.4f} gMAE={bv['mae_per_action']['gamma_n']:.4f} "
                  f"sign={rows[-1]['gamma_sign_acc']:.3f} "
                  f"nn_gamma_gt1={amb['gamma_range_gt1']:.3f}")
    (EXP / "ab_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 13D vs 16D 纯 BC A/B 汇总", "",
             "| seed | 臂 | best_ep | val MSE | γ MAE | γ符号acc | BC贴边(any) | 近邻γ极差>1 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['arm']} | {r['best_epoch']} | {r['val_mse']:.4f} | "
            f"{r['mae_gamma']:.4f} | {r['gamma_sign_acc']:.3f} | "
            f"{r['bc_boundary_any']*100:.2f}% | {r['gamma_range_gt1']*100:.1f}% |")
    lines += ["", "## 配对差值（16D−13D）", ""]
    for seed in SEEDS:
        a = next(r for r in rows if r["seed"] == seed and r["arm"] == "13D")
        b = next(r for r in rows if r["seed"] == seed and r["arm"] == "16D")
        dmse = (b["val_mse"] - a["val_mse"]) / a["val_mse"] * 100
        lines.append(
            f"- seed {seed}：ΔMSE={dmse:+.1f}%（{a['val_mse']:.4f}→{b['val_mse']:.4f}），"
            f"ΔγMAE={b['mae_gamma']-a['mae_gamma']:+.4f}，"
            f"Δ符号acc={b['gamma_sign_acc']-a['gamma_sign_acc']:+.3f}，"
            f"近邻γ歧义 {a['gamma_range_gt1']*100:.1f}%→{b['gamma_range_gt1']*100:.1f}%")
    mses = [next(r for r in rows if r["seed"] == s and r["arm"] == "13D")["val_mse"]
            for s in SEEDS]
    mses16 = [next(r for r in rows if r["seed"] == s and r["arm"] == "16D")["val_mse"]
              for s in SEEDS]
    lines.append(
        f"- 平均：13D={np.mean(mses):.4f} vs 16D={np.mean(mses16):.4f}，"
        f"相对 {(np.mean(mses16)-np.mean(mses))/np.mean(mses)*100:+.1f}%")
    (EXP / "AB_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("[ab] 已写入 ab_summary.json 与 AB_SUMMARY.md")


if __name__ == "__main__":
    main()
