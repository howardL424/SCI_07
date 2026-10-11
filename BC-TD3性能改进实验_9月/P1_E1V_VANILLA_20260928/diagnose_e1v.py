"""E1V 50k 预筛诊断：两 seed 早期 Actor/Critic 动态 + BC 拟合对照。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P1 = ROOT.parent / "P1_E1_BC_WARMSTART_20260927"

TIMESTEP_RE = re.compile(r"\|\s*total_timesteps\s*\|\s*([\d,]+)")
ACTOR_RE = re.compile(r"\|\s*actor_loss\s*\|\s*([-\d.eE+]+)")
CRITIC_RE = re.compile(r"\|\s*critic_loss\s*\|\s*([-\d.eE+]+)")
REW_RE = re.compile(r"\|\s*ep_rew_mean\s*\|\s*([-\d.eE+]+)")


def parse_train_log(path: Path) -> list[dict]:
    cur: dict = {}
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TIMESTEP_RE.search(line)
        if m:
            cur["total_timesteps"] = int(m.group(1).replace(",", ""))
            continue
        m = ACTOR_RE.search(line)
        if m:
            try:
                cur["actor_loss"] = float(m.group(1))
            except ValueError:
                pass
            continue
        m = CRITIC_RE.search(line)
        if m:
            try:
                cur["critic_loss"] = float(m.group(1))
            except ValueError:
                pass
            continue
        m = REW_RE.search(line)
        if m:
            try:
                cur["ep_rew_mean"] = float(m.group(1))
            except ValueError:
                pass
            continue
        if line.strip() == "---------------------------------" and "total_timesteps" in cur:
            rows.append(dict(cur))
            cur = {}
    if "total_timesteps" in cur:
        rows.append(dict(cur))
    return rows


def bc_best_entry(report_path: Path) -> dict:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    best = int(rep["best_epoch"])
    for h in rep["history"]:
        if int(h["epoch"]) == best:
            return {"best_epoch": best, "best_val_mse": rep["best_val_mse"], **h}
    return {}


def main() -> None:
    out: list[str] = ["# E1V 50k 诊断（早期动态 + BC 对照）", ""]
    for seed in (0, 42):
        rows = parse_train_log(ROOT / "runs" / f"seed_{seed}" / "E1V" / "train.log")
        early = [r for r in rows if r.get("total_timesteps", 0) <= 26_000 and "actor_loss" in r]
        out.append(f"## seed {seed}：0~25k 训练动态（actor_loss = -Q，因 λ=0）")
        out.append("")
        out.append("| total_timesteps | actor_loss(-Q) | critic_loss | ep_rew_mean |")
        out.append("|---|---|---|---|")
        # 抽样打印约 8 行
        idx = np.linspace(0, max(len(early) - 1, 0), min(8, len(early))).astype(int) if early else []
        for i in idx:
            r = early[int(i)]
            out.append(
                f"| {r.get('total_timesteps')} | {r.get('actor_loss')} "
                f"| {r.get('critic_loss')} | {r.get('ep_rew_mean')} |"
            )
        крит = [r["critic_loss"] for r in early if "critic_loss" in r]
        act = [r["actor_loss"] for r in early if "actor_loss" in r]
        out.append("")
        out.append(
            f"- 早期 critic_loss：max={max(крит) if крит else None}，"
            f"末段均值(后1/3)={float(np.mean(крит[len(крит) * 2 // 3:])) if крит else None}"
        )
        out.append(
            f"- 早期 actor_loss(-Q)：首={act[0] if act else None}，"
            f"末={act[-1] if act else None}"
        )
        out.append("")
        bc = bc_best_entry(P1 / "runs" / f"seed_{seed}" / "bc" / f"bc_seed{seed}_report.json")
        val = bc.get("val", {})
        out.append(f"## seed {seed}：BC 最优 checkpoint 对照（epoch {bc.get('best_epoch')}）")
        out.append("")
        out.append(f"- val MSE={bc.get('best_val_mse')}；val MAE={val.get('mae')}；分维 MAE={val.get('mae_per_action')}")
        out.append(f"- 训练集分维 MAE={bc.get('train', {}).get('mae_per_action')}")
        out.append("")
    # 评估奖励分布（若 evaluations.npz 可读）
    for seed in (0, 42):
        p = ROOT / "runs" / f"seed_{seed}" / "E1V" / "eval_logs" / "evaluations.npz"
        try:
            z = np.load(p, allow_pickle=True)
            out.append(f"## seed {seed} evaluations.npz：keys={list(z.keys())}")
            for k in list(z.keys()):
                a = z[k]
                out.append(f"- {k}: shape={a.shape}, dtype={a.dtype}")
            out.append("")
        except Exception as e:  # noqa: BLE001
            out.append(f"## seed {seed} evaluations.npz 读取失败：{e}")
            out.append("")
    (ROOT / "DIAGNOSIS_50k.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
