"""H1 共享回放结果的离线归因分析。

回答三个问题：

1. **时变速率**：真值参数每拍漂移多大？把"估计值滞后 m 拍"折算成 NRMSE，
   得到滞后—误差曲线，从而反推 FF-RLS 遗忘因子的可行上限。
2. **分量归因**：11 维里哪几个分量主导了各估计器的统一 NRMSE。
3. **模型有效性**：周期拟冻结模型残差相对 r̈ 本身的量级。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_IDENT = _SCRIPT.parent
_RL = _IDENT.parent / "rl_td3"
sys.path.insert(0, str(_IDENT))
sys.path.insert(0, str(_RL))

from evaluate_td3 import _theta_scaled_nrmse  # noqa: E402

PARAM_NAMES = (
    "A21_D1", "B1_D1", "B2_D1", "C1_D1",
    "A21_D2", "B1_D2", "B2_D2", "C1_D2",
    "A21_T", "B1_T", "B2_T",
)
SCALE_KEYS = ("r0", "a0", "a0", "one") * 2 + ("r0", "a0", "a0")


def _lag_nrmse(true_current, true_causal, lag, *, r0, a0, floor):
    """估计器输出 vartheta_true[k-lag] 时的 NRMSE（lag=0 即当前真值）。"""
    n = len(true_causal)
    if lag >= n:
        return float("nan")
    # true_current[k] = vartheta[k]，故 vartheta[k-lag] = true_current[k-lag]
    est = [true_current[k - lag] for k in range(lag, n)]
    tgt = [true_causal[k] for k in range(lag, n)]
    agg, _, _ = _theta_scaled_nrmse(est, tgt, r0=r0, a0=a0, denominator_floor=floor)
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--lags", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20, 50])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "h1_shared_replay_series.npz", allow_pickle=True)

    cfg = summary["matlab_python_config"]
    r0 = float(cfg["rls_r0"])
    a0 = float(cfg["rls_a0"])
    floor = float(summary["nrmse"]["denominator_floor"])
    scale = np.array([r0, a0, a0, 1.0, r0, a0, a0, 1.0, r0, a0, a0])

    true_cur_all = data["series__theta_true_current"]
    true_cau_all = data["series__theta_true_causal"]
    hat_all = data["series__theta_hat"]
    noisy_all = data["series__theta_analytic_noisy_causal"]
    frozen_all = data["series__frozen_model_residual"]
    n_ep = len(true_cau_all)

    print(f"== H1 归因分析：{run_dir}  ({n_ep} IC) ==\n")

    # ---- 1. 滞后—误差曲线 -------------------------------------------------
    print("1) 真值参数滞后 m 拍后的统一 NRMSE（m=0 即神谕当前真值）")
    print(f"   dt=0.1s；FF-RLS 有效记忆窗 ≈ 1/(1-lambda) = "
          f"{1.0/(1.0-float(cfg['rls_lambda'])):.0f} 拍")
    rows = []
    for lag in args.lags:
        vals = [
            _lag_nrmse(true_cur_all[i], true_cau_all[i], lag,
                       r0=r0, a0=a0, floor=floor)
            for i in range(n_ep)
        ]
        med = float(np.nanmedian(vals))
        rows.append((lag, med))
        print(f"   lag={lag:>3d} 拍 ({0.1*lag:>4.1f} s)  NRMSE_median={med:.4g}")
    print()

    # ---- 2. 逐拍相对漂移 --------------------------------------------------
    print("2) 逐拍真值相对漂移 |theta[k]-theta[k-1]| / max(RMS(theta),floor)（中位数）")
    drift_med = []
    for j in range(11):
        per_ep = []
        for i in range(n_ep):
            tc = np.asarray(true_cau_all[i], dtype=float)[:, j] * scale[j]
            cur = np.asarray(true_cur_all[i], dtype=float)[:, j] * scale[j]
            fin = np.isfinite(tc) & np.isfinite(cur)
            if fin.sum() < 2:
                continue
            denom = max(float(np.sqrt(np.mean(tc[fin] ** 2))), floor)
            per_ep.append(float(np.sqrt(np.mean((cur[fin] - tc[fin]) ** 2)) / denom))
        drift_med.append(float(np.median(per_ep)) if per_ep else float("nan"))
    for j, name in enumerate(PARAM_NAMES):
        print(f"   {name:<8s} {drift_med[j]:>10.4g}")
    print()

    # ---- 3. 分量归因 ------------------------------------------------------
    print("3) 逐参数 NRMSE 中位数与对聚合值的贡献占比")
    print(f"   {'param':<8s}{'noisy':>12s}{'share%':>9s}"
          f"{'ffrls':>12s}{'share%':>9s}{'drift(m=1)':>12s}")
    comp = {
        n: np.stack(data[f"nrmse_components__{n}"]) for n in
        ("analytic_noisy", "sparse_ffrls", "zoh_drift_floor")
    }
    med = {n: np.nanmedian(v, axis=0) for n, v in comp.items()}
    sq = {n: med[n] ** 2 for n in med}
    tot = {n: np.nansum(sq[n]) for n in sq}
    for j, name in enumerate(PARAM_NAMES):
        print(f"   {name:<8s}"
              f"{med['analytic_noisy'][j]:>12.4g}"
              f"{100*sq['analytic_noisy'][j]/tot['analytic_noisy']:>9.1f}"
              f"{med['sparse_ffrls'][j]:>12.4g}"
              f"{100*sq['sparse_ffrls'][j]/tot['sparse_ffrls']:>9.1f}"
              f"{med['zoh_drift_floor'][j]:>12.4g}")
    print()

    # ---- 4. A21 通道是否被软阈值压零 --------------------------------------
    print("4) A21 分量缩放域量级（检查 eta 软阈值与 denominator_floor 是否压零）")
    print(f"   rls_eta={cfg['rls_eta']}  denominator_floor={floor}")
    for j in (0, 4, 8):
        name = PARAM_NAMES[j]
        t_rms, h_rms, h_zero = [], [], []
        for i in range(n_ep):
            tc = np.asarray(true_cau_all[i], dtype=float)[:, j] * scale[j]
            hh = np.asarray(hat_all[i], dtype=float)[:, j] * scale[j]
            fin = np.isfinite(tc) & np.isfinite(hh)
            if not fin.any():
                continue
            t_rms.append(float(np.sqrt(np.mean(tc[fin] ** 2))))
            h_rms.append(float(np.sqrt(np.mean(hh[fin] ** 2))))
            h_zero.append(float(np.mean(np.abs(hh[fin]) < 1e-9)))
        print(f"   {name:<8s} RMS(truth_sc)={np.median(t_rms):.4g}  "
              f"RMS(hat_sc)={np.median(h_rms):.4g}  "
              f"hat==0 占比={100*np.median(h_zero):.1f}%")
    print()

    # ---- 5. 冻结模型残差 --------------------------------------------------
    print("5) 周期拟冻结模型残差 RMS（rddot 域，m/s^2）")
    for ch, label in enumerate(("D1", "D2", "T")):
        res = []
        for i in range(n_ep):
            fr = np.asarray(frozen_all[i], dtype=float)[:, ch]
            fin = np.isfinite(fr)
            if fin.any():
                res.append(float(np.sqrt(np.mean(fr[fin] ** 2))))
        med_all = np.median(res) if res else float("nan")
        # 去掉末 5 拍（交汇前 A21 爆炸段）后的残差
        res_tail_cut = []
        for i in range(n_ep):
            fr = np.asarray(frozen_all[i], dtype=float)[:-5, ch]
            fin = np.isfinite(fr)
            if fin.any():
                res_tail_cut.append(float(np.sqrt(np.mean(fr[fin] ** 2))))
        print(f"   {label}: 全程={med_all:.4g}   去末5拍="
              f"{np.median(res_tail_cut) if res_tail_cut else float('nan'):.4g}")
    print()

    # ---- 6. 逐 IC 最差分量 -----------------------------------------------
    print("6) FF-RLS 逐 IC 聚合 NRMSE 与最差分量")
    for i in range(n_ep):
        c = np.asarray(data["nrmse_components__sparse_ffrls"][i], dtype=float)
        jmax = int(np.nanargmax(c))
        print(f"   IC{int(data['local_idx'][i])}: "
              f"agg={float(data['nrmse__sparse_ffrls'][i]):.4g}  "
              f"worst={PARAM_NAMES[jmax]}={c[jmax]:.4g}")
    print()

    # ---- 7. D2 通道 B 分量时序剖面 ----------------------------------------
    print("7) B2_D2（j=6）缩放域逐拍剖面：估计 vs 真值 vs P_trace(D2)")
    j = 6
    ptr_all = data["series__P_trace"]
    for i in range(n_ep):
        tc = np.asarray(true_cau_all[i], dtype=float)[:, j] * scale[j]
        hh = np.asarray(hat_all[i], dtype=float)[:, j] * scale[j]
        nn = np.asarray(noisy_all[i], dtype=float)[:, j] * scale[j]
        ptr = np.asarray(ptr_all[i], dtype=float)[:, 1]
        n = len(tc)
        marks = sorted({0, 1, 2, 5, 10, n // 4, n // 2, 3 * n // 4,
                        n - 6, n - 3, n - 1} & set(range(n)))
        print(f"   IC{int(data['local_idx'][i])} (steps={n})")
        print(f"     {'k':>5s}{'truth_sc':>12s}{'ffrls_sc':>12s}"
              f"{'noisy_sc':>12s}{'P_trace_D2':>12s}")
        for k in marks:
            print(f"     {k:>5d}{tc[k]:>12.4g}{hh[k]:>12.4g}"
                  f"{nn[k]:>12.4g}{ptr[k]:>12.4g}")
    print()

    # ---- 8. 交汇末段剔除敏感性 --------------------------------------------
    # 诊断用途：定位误差来源，不替代协议中已冻结的全程/去初始化 NRMSE 主指标。
    print("8) 剔除末 K 拍（交汇奇异段）后的统一 NRMSE 中位数")
    print(f"   {'K':>4s}{'analytic_noisy':>18s}{'sparse_ffrls':>16s}"
          f"{'zoh_drift_floor':>18s}")
    est_map = {
        "analytic_noisy": noisy_all,
        "sparse_ffrls": hat_all,
        "zoh_drift_floor": true_cur_all,
    }
    for K in (0, 3, 5, 10, 15):
        cells = []
        for name in ("analytic_noisy", "sparse_ffrls", "zoh_drift_floor"):
            vals = []
            for i in range(n_ep):
                est = np.asarray(est_map[name][i], dtype=float)
                tgt = np.asarray(true_cau_all[i], dtype=float)
                n = len(tgt) - K
                if n <= 5:
                    continue
                agg, _, _ = _theta_scaled_nrmse(
                    list(est[:n]), list(tgt[:n]),
                    r0=r0, a0=a0, denominator_floor=floor,
                )
                vals.append(agg)
            cells.append(float(np.nanmedian(vals)) if vals else float("nan"))
        print(f"   {K:>4d}{cells[0]:>18.4g}{cells[1]:>16.4g}{cells[2]:>18.4g}")
    print()

    # ---- 9. 无技能参照估计器 ----------------------------------------------
    # 恒零估计器在每个分量上的 NRMSE 恒为 RMS(truth)/RMS(truth)=1（当
    # RMS(truth)>=floor）。因此 1.0 是该指标的"无技能"水平：分量值 >1
    # 说明该估计器在这一参数上不如什么都不输出。聚合 NRMSE 因为是比值的
    # RMS，会被单一劣化分量主导，不能单独作为验收判据。
    print("9) 无技能参照：恒零估计器与 warm-start 保持")
    refs: dict[str, list] = {
        "zero_estimator": [],
        "warm_start_hold": [],
        "analytic_noisy": [],
        "sparse_ffrls": [],
    }
    ref_comp: dict[str, list] = {k: [] for k in refs}
    for i in range(n_ep):
        tgt = np.asarray(true_cau_all[i], dtype=float)
        noisy = np.asarray(noisy_all[i], dtype=float)
        finite_rows = np.isfinite(noisy).all(axis=1)
        k0 = int(np.argmax(finite_rows)) if finite_rows.any() else 0
        candidates = {
            "zero_estimator": np.zeros_like(tgt),
            "warm_start_hold": np.tile(noisy[k0], (len(tgt), 1)),
            "analytic_noisy": noisy,
            "sparse_ffrls": np.asarray(hat_all[i], dtype=float),
        }
        for name, est in candidates.items():
            agg, c, _ = _theta_scaled_nrmse(
                list(est), list(tgt), r0=r0, a0=a0, denominator_floor=floor
            )
            refs[name].append(agg)
            ref_comp[name].append(c)
    print(f"   {'estimator':<18s}{'median':>10s}{'p95':>10s}{'max':>10s}"
          f"{'分量优于无技能':>16s}")
    for name in refs:
        v = np.asarray(refs[name], dtype=float)
        c_med = np.nanmedian(np.stack(ref_comp[name]), axis=0)
        n_skill = int(np.sum(c_med < 1.0 - 1e-9))
        print(f"   {name:<18s}{np.nanmedian(v):>10.4g}{np.nanpercentile(v, 95):>10.4g}"
              f"{np.nanmax(v):>10.4g}{f'{n_skill}/11':>16s}")
    print("\n   逐参数 NRMSE 中位数（<1 才算有技能）")
    print("   " + f"{'param':<9s}" + "".join(f"{k:>18s}" for k in refs))
    for j, name in enumerate(PARAM_NAMES):
        cells = "".join(
            f"{np.nanmedian(np.stack(ref_comp[k])[:, j]):>18.4g}" for k in refs
        )
        print(f"   {name:<9s}{cells}")


if __name__ == "__main__":
    main()
