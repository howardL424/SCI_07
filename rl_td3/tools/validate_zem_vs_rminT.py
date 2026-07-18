"""
零训练量化验证：规避段终点 ZEM_T vs 全段真实终端脱靶量 r_minT。

数据源：expert_lib_1000.mat（warmstart 全段 tau：stage2 式47 + stage3 PN）
输出：统计表 + 散点/直方图到 rl_td3/plots/
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERT_MAT = r"E:\matlab\result_0702\expert_lib_1000.mat"
OUT_DIR = Path(__file__).resolve().parent.parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def target_zem(M: np.ndarray) -> float:
    """复刻 rl_td3/matlab/target_zem.m（与 env 一致）。"""
    posx, posy, posz, thetaM, psiM, VM = [float(x) for x in M]
    r = np.sqrt(posx**2 + posy**2 + posz**2 + 1e-6)
    q_y = np.arcsin(np.clip(posy / r, -1.0, 1.0))
    q_z = -np.arctan2(posz, posx)
    rdot = VM * (
        np.cos(thetaM) * np.cos(q_y) * np.cos(q_z - psiM) + np.sin(thetaM) * np.sin(q_y)
    )
    term = VM * (
        np.cos(thetaM) * np.sin(q_y) * np.cos(q_z - psiM) - np.sin(thetaM) * np.cos(q_y)
    )
    qd_y = -term / r
    cos_qy = np.sqrt(np.cos(q_y) ** 2 + 1e-10)
    qd_z = (-VM * np.cos(thetaM) * np.sin(q_z - psiM)) / (r * cos_qy)
    Q = np.sqrt(qd_y**2 + qd_z**2 * np.cos(q_y) ** 2 + 1e-10)
    if rdot < -1e-3:
        tgo = -r / rdot
    else:
        tgo = r / 1e-3
    Qt = float(np.clip(Q * tgo, -30.0, 30.0))
    zT = np.cosh(Qt) * r + np.sinh(Qt) / Q * rdot
    return abs(float(zT))


def soft_sat_log(r: float, dstar: float = 20.0, kill: float = 6.0) -> float:
    """用户 R2/R3 + (r-d*) 软饱和：r>d* 后不再随距离增长。"""
    if r <= kill:
        return 0.0
    r_eff = min(r, dstar)
    return float(np.log10(max(r_eff - 5.0, 1e-6)))


def R1(r_minT: float) -> float:
    return 80.0 / (5.0 + r_minT) if r_minT <= 25.0 else 0.0


def main() -> None:
    f = h5py.File(EXPERT_MAT, "r")
    tr = f["results"]["trajectories"]
    n = tr["ok_use"].shape[0]

    rows = []
    for i in range(n):
        ok = bool(np.array(f[tr["ok_use"][i, 0]]).squeeze())
        if not ok:
            continue
        tau = f[tr["tau"][i, 0]]
        stage = np.array(tau["stage"]).ravel()
        M = np.array(tau["M"])  # (N, 6) in this file layout
        if M.ndim != 2 or M.shape[1] != 6:
            # fallback if transposed
            M = np.array(tau["M"]).T
        idx2 = np.where(stage == 2)[0]
        if idx2.size == 0:
            continue
        i0, ie = int(idx2[0]), int(idx2[-1])
        zT0 = target_zem(M[i0])
        zTe = target_zem(M[ie])
        r_minT = float(np.array(tau["r_minT"]).squeeze())
        r_min1 = float(np.array(tau["r_min1"]).squeeze())
        r_min2 = float(np.array(tau["r_min2"]).squeeze())
        reached = bool(np.array(tau["reached"]).squeeze())
        intercepted = bool(np.array(tau["intercepted"]).squeeze())

        missT = np.nan
        sb_ref = tr["stageB"][i, 0]
        if sb_ref:
            sb = f[sb_ref]
            if "miss" in sb:
                miss = np.array(sb["miss"]).ravel()
                if miss.size >= 3:
                    missT = float(miss[2])

        # stage2 末 r_MT（距离，非 ZEM）
        r_MT = np.array(tau["r_MT"]).ravel()
        r_MT_end2 = float(r_MT[ie]) if ie < r_MT.size else np.nan

        rows.append(
            dict(
                zT0=zT0,
                zTe=zTe,
                r_minT=r_minT,
                r_min1=r_min1,
                r_min2=r_min2,
                missT=missT,
                r_MT_end2=r_MT_end2,
                reached=reached,
                intercepted=intercepted,
                R1=R1(r_minT),
                R2=soft_sat_log(r_min1),
                R3=soft_sat_log(r_min2),
                R2_raw=float(np.log10(r_min1 - 5)) if r_min1 > 6 else 0.0,
                R3_raw=float(np.log10(r_min2 - 5)) if r_min2 > 6 else 0.0,
            )
        )

    f.close()
    D = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    N = len(rows)
    print(f"ok_use 轨迹数: {N}")

    def pct(x, p):
        return float(np.percentile(x, p))

    def summarize(name, x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        print(
            f"  {name}: n={len(x)}  mean={x.mean():.3g}  med={np.median(x):.3g}  "
            f"p90={pct(x,90):.3g}  p99={pct(x,99):.3g}  max={x.max():.3g}"
        )

    print("\n=== 核心对比：规避段终点 ZEM_T vs 真实 r_minT ===")
    summarize("stage2起点 ZEM_T", D["zT0"])
    summarize("stage2终点 ZEM_T", D["zTe"])
    summarize("全段真实 r_minT", D["r_minT"])
    summarize("stageB.missT(NLP)", D["missT"])
    summarize("stage2终点 r_MT", D["r_MT_end2"])

    # 相关性：ZEM 能否预测真实精度？
    zTe = D["zTe"]
    rT = D["r_minT"]
    # 对极端 ZEM 做 log 相关
    mask = np.isfinite(zTe) & np.isfinite(rT) & (zTe > 0) & (rT > 0)
    corr_lin = float(np.corrcoef(zTe[mask], rT[mask])[0, 1])
    corr_log = float(np.corrcoef(np.log10(zTe[mask]), np.log10(rT[mask]))[0, 1])
    print(f"\n  corr(ZEM_end2, r_minT)      = {corr_lin:.4f}")
    print(f"  corr(log ZEM, log r_minT)  = {corr_log:.4f}")

    # 典型矛盾：ZEM 很大但真实命中很好
    big_zem_ok = (zTe > 100) & (rT <= 10)
    big_zem_ok25 = (zTe > 100) & (rT <= 25)
    huge_zem_ok = (zTe > 1000) & (rT <= 10)
    print(f"\n  ZEM_end2>100 且 r_minT≤10m : {big_zem_ok.sum()}/{N} = {100*big_zem_ok.mean():.1f}%")
    print(f"  ZEM_end2>100 且 r_minT≤25m : {big_zem_ok25.sum()}/{N} = {100*big_zem_ok25.mean():.1f}%")
    print(f"  ZEM_end2>1000 且 r_minT≤10m: {huge_zem_ok.sum()}/{N} = {100*huge_zem_ok.mean():.1f}%")

    # 达标率
    print("\n=== 真实精度达标率（r_minT）===")
    for thr in [5, 10, 25, 50]:
        print(f"  r_minT ≤ {thr:2d}m : {(rT <= thr).mean()*100:5.1f}%")

    print("\n=== 若用 stage2 终点 ZEM 当精度（错误指标）===")
    for thr in [5, 10, 25, 50, 100, 1000]:
        print(f"  ZEM_end2 ≤ {thr:4d}m : {(zTe <= thr).mean()*100:5.1f}%")

    # 用户奖励 R1 分布
    print("\n=== 用户提议 R1=80/(5+r_minT) if ≤25 else 0 ===")
    summarize("R1", D["R1"])
    print(f"  R1>0 比例: {(D['R1']>0).mean()*100:.1f}%")
    print(f"  r_minT=10 → R1={80/(5+10):.3f}")
    print(f"  r_minT=0  → R1={80/5:.3f}")
    print(f"  r_minT=25 → R1={80/(5+25):.3f}")

    # R2/R3：原始 vs 软饱和
    print("\n=== R2/R3：原始 log10(r-5) vs 软饱和到 d*=20 ===")
    summarize("R2_raw (用 r_min1)", D["R2_raw"])
    summarize("R2_sat d*=20", D["R2"])
    summarize("R3_raw", D["R3_raw"])
    summarize("R3_sat", D["R3"])
    # 若用 episode 最小距离当过程奖励的“代表值”（仅示意；真实是逐步）
    R_ep_raw = D["R1"] + D["R2_raw"] + D["R3_raw"]
    R_ep_sat = D["R1"] + D["R2"] + D["R3"]
    summarize("R1+R2raw+R3raw (示意)", R_ep_raw)
    summarize("R1+R2sat+R3sat (示意)", R_ep_sat)
    # 过程项相对终端项的量级
    print(
        f"  中位 R1 / 中位(R2sat+R3sat) = "
        f"{np.median(D['R1'][D['R1']>0]) / np.median(D['R2']+D['R3']):.2f}"
        f"  （注意：真实训练中 R2/R3 是逐步累加，量级会再放大 ~Tev/dt 倍）"
    )

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(np.log10(np.clip(zTe, 1e-3, None)), np.log10(np.clip(rT, 1e-3, None)), s=8, alpha=0.35)
    ax.axhline(np.log10(10), color="g", ls="--", lw=1, label="r_minT=10m")
    ax.axhline(np.log10(25), color="orange", ls="--", lw=1, label="r_minT=25m")
    ax.set_xlabel("log10(ZEM at end of stage2)")
    ax.set_ylabel("log10(true r_minT after stage3 PN)")
    ax.set_title(f"ZEM_end2 vs r_minT  (corr_log={corr_log:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    bins = np.logspace(-2, 5, 50)
    ax.hist(zTe, bins=bins, alpha=0.55, label="ZEM_end2", color="C0")
    ax.hist(rT, bins=bins, alpha=0.55, label="r_minT", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel("meters")
    ax.set_ylabel("count")
    ax.set_title("Distribution: ZEM_end2 vs true r_minT")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(D["R1"], bins=40, color="teal", alpha=0.8)
    ax.axvline(80 / (5 + 10), color="r", ls="--", label="R1@10m")
    ax.set_xlabel("R1")
    ax.set_title("Proposed terminal reward R1 on expert r_minT")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(D["R2_raw"] + D["R3_raw"], bins=40, alpha=0.55, label="R2+R3 raw (at r_min)", color="C0")
    ax.hist(D["R2"] + D["R3"], bins=40, alpha=0.55, label="R2+R3 soft-sat d*=20", color="C1")
    ax.set_xlabel("R2+R3 (episode-min proxy)")
    ax.set_title("Evasion reward: raw vs soft-sat")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Zero-train validation on expert_lib (N={N} ok_use)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "zem_vs_rminT_validation.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved: {out}")

    # 额外：ratio ZEM/r_minT
    ratio = zTe / np.clip(rT, 1e-6, None)
    print("\n=== ZEM_end2 / r_minT 比值（越大=代理越离谱）===")
    summarize("ratio", ratio)
    print(f"  ratio>10  : {(ratio>10).mean()*100:.1f}%")
    print(f"  ratio>100 : {(ratio>100).mean()*100:.1f}%")
    print(f"  ratio>1000: {(ratio>1000).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
