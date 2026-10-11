"""dig_q_ambiguity.py — C项离线深挖：Q为何没降低近邻歧义（不训练、不碰在线）.

C1 近邻重叠+距离分解：同query下13D-NN与16D-NN的Jaccard重叠；16D距离中Q分量占比；Q/13D各维std.
C2 Q-动作信息量：Q与gamma的相关；Q分箱后gamma方差收缩；Q线性探针R^2.
C3 改进定位：同val集上16D vs 13D逐样本SE差分布，与|Q|/|gamma|/贴边的关系.
C4 网络利用度：16D首层Q列权重占比；Q置均值消融后val MSE回退.
C5 贴边精/召回与符号条件分析（回D：过关标准不仅看像专家）.

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/dig_q_ambiguity.py"
输出：dig_q_report.json, DIG_Q_AMBIGUITY.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

EXP = Path(__file__).resolve().parent
GAIL_ROOT = EXP.parent.parent
sys.path.insert(0, str(GAIL_ROOT / "rl_td3"))

from bc_actor import load_bc_checkpoint, predict_actions
from expert_sampling import bc_arrays_from_plan, load_sample_plan
from load_expert_buffer import apply_action_mapping, load_transition_data
from obs_norm import normalize_obs
from obs_qmag import normalize_obs16

EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"
ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")
SEEDS = (0, 42, 21)
BOUND = 0.99
N_Q = 2000
K = 50

def jaccard(a, b):
    inter = len(set(a.tolist()) & set(b.tolist()))
    return inter / (2 * K - inter) if (2*K-inter) else 0.0

def main():
    print("[dig] load aligned (once, ~4min mat73)...", flush=True)
    data = load_transition_data(EXPERT, aligned_path=ALIGNED)
    src_map = str(data.get("action_mapping_version", "td3_box_log_v1"))
    data = apply_action_mapping(data, "ratio_log_box_v1", source_version=src_map)
    phys13 = np.asarray(data["states"], dtype=np.float64)
    states13 = normalize_obs(phys13).astype(np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    q_all = np.load(EXP / "q_all_aligned_order.npy").astype(np.float64)
    device = torch.device("cpu")
    report = {"seeds": {}, "config": {"n_q": N_Q, "k": K, "bound": BOUND}}

    for seed in SEEDS:
        print(f"[dig] seed {seed} ...", flush=True)
        plan_file = EXP / f"runs/ab13_s{seed}/bc_sample_plan_seed{seed}.npz"
        if not plan_file.exists():
            plan_file = GAIL_ROOT / "rl_td3" / "data" / "expert_set_99k.npz"
        plan = load_sample_plan(str(plan_file))
        s13_tr, a_tr, s13_va, a_va = bc_arrays_from_plan(states13, actions, plan)
        sc = np.load(EXP / f"runs/ab16_s{seed}/q_scale_seed{seed}.npz")["q_scale"].astype(np.float64)
        states16 = normalize_obs16(phys13, q_all, sc).astype(np.float32)
        s16_tr, _, s16_va, _ = bc_arrays_from_plan(states16, actions, plan)

        # ---- C1: NN overlap + distance decomposition ----
        rng = np.random.default_rng(seed)
        qidx = rng.choice(len(s13_va), size=min(N_Q, len(s13_va)), replace=False)
        q13 = s13_va[qidx].astype(np.float64)
        q16 = s16_va[qidx].astype(np.float64)
        t13 = s13_tr.astype(np.float64); t16 = s16_tr.astype(np.float64)
        jac, qfrac, g13r, g16r = [], [], [], []
        for i in range(0, len(q13), 250):
            qb13 = q13[i:i+250]; qb16 = q16[i:i+250]
            d13 = ((qb13[:, None, :] - t13[None, :, :]) ** 2).sum(-1)
            d16 = ((qb16[:, None, :] - t16[None, :, :]) ** 2).sum(-1)
            n13 = np.argpartition(d13, K, axis=1)[:, :K]
            n16 = np.argpartition(d16, K, axis=1)[:, :K]
            for r in range(len(qb13)):
                jac.append(jaccard(n13[r], n16[r]))
                # Q分量占比：取16D最近邻的平方距离中后3维占比（中位）
                diff = (qb16[r][None, :] - t16[n16[r]]) ** 2  # [K,16]
                tot = diff.sum(axis=1).clip(min=1e-12)
                qf = (diff[:, 13:16].sum(axis=1) / tot)
                qfrac.append(float(np.median(qf)))
                gr13 = float(a_tr[n13[r]][:, 2].max() - a_tr[n13[r]][:, 2].min())
                gr16 = float(a_tr[n16[r]][:, 2].max() - a_tr[n16[r]][:, 2].min())
                g13r.append(gr13); g16r.append(gr16)
        g13r = np.array(g13r); g16r = np.array(g16r)
        c1 = {
            "jaccard_mean": float(np.mean(jac)),
            "jaccard_p50": float(np.median(jac)),
            "jaccard_frac_gt05": float(np.mean(np.array(jac) > 0.5)),
            "q_dist_frac_median": float(np.median(qfrac)),
            "q_dist_frac_p90": float(np.quantile(qfrac, 0.9)),
            "gamma_range_gt1_13": float(np.mean(g13r > 1.0)),
            "gamma_range_gt1_16": float(np.mean(g16r > 1.0)),
            "gamma_range_mean_13": float(g13r.mean()),
            "gamma_range_mean_16": float(g16r.mean()),
            "s13_std_mean": float(s13_tr.std(axis=0).mean()),
            "s16_q_std": [float(v) for v in s16_tr[:, 13:16].std(axis=0)],
            "s16_q_mean": [float(v) for v in s16_tr[:, 13:16].mean(axis=0)],
            "s13_perdim_std": [float(v) for v in s13_tr.std(axis=0)],
            "q_scale": [float(v) for v in sc],
        }

        # ---- C2: Q-action info ----
        # 用val集：Q物理 vs gamma
        q_va_phys = q_all[np.asarray(plan.val_actor_step_indices)] if hasattr(plan, "val_actor_step_indices") else None
        # fallback: 用a_va索引反查——plan.val_actor_step_indices即val行号
        try:
            vidx = np.asarray(plan.val_actor_step_indices)
            qv = q_all[vidx]
        except Exception:
            qv = None
        g = a_va[:, 2].astype(np.float64)
        c2 = {}
        if qv is not None:
            for j, nm in enumerate(["Q1", "Q2", "QT"]):
                x = qv[:, j]
                # Pearson (log1p变换也试，因重尾)
                def pear(u, v):
                    u = u - u.mean(); v = v - v.mean()
                    d = np.sqrt((u**2).sum() * (v**2).sum())
                    return float((u * v).sum() / d) if d > 0 else 0.0
                c2[f"pear_{nm}_gamma"] = pear(x, g)
                c2[f"pear_log1p{nm}_gamma"] = pear(np.log1p(np.abs(x)) * np.sign(x), g)
                c2[f"pear_{nm}_absgamma"] = pear(x, np.abs(g))
            # Q分箱方差收缩：按Q2三分位
            q2 = qv[:, 1]
            qs = np.quantile(q2, [1/3, 2/3])
            bins = np.digitize(q2, qs)
            overall_var = float(g.var())
            within = float(np.mean([g[bins == b].var() for b in range(3)]))
            c2["gamma_var_overall"] = overall_var
            c2["gamma_var_within_Q2tertile_mean"] = within
            c2["var_shrink_Q2"] = float(1 - within / overall_var) if overall_var > 0 else 0.0
            # QT三分位同样
            qt = qv[:, 2]
            qs_t = np.quantile(qt, [1/3, 2/3])
            binst = np.digitize(qt, qs_t)
            within_t = float(np.mean([g[binst == b].var() for b in range(3)]))
            c2["gamma_var_within_QTtertile_mean"] = within_t
            c2["var_shrink_QT"] = float(1 - within_t / overall_var) if overall_var > 0 else 0.0
            # 线性探针 R^2: [13D, Q] vs 仅13D 预测gamma（最小二乘，val内拟合仅作信息量上界参考）
            for nm2, S in (("s13", s13_va.astype(np.float64)), ("s16", s16_va.astype(np.float64))):
                A = np.concatenate([S, np.ones((len(S), 1))], axis=1)
                coef, *_ = np.linalg.lstsq(A, g, rcond=None)
                pred = A @ coef
                ss = 1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum()
                c2[f"linprobe_R2_{nm2}_gamma"] = float(ss)

        # ---- C3/C4/C5: preds ----
        a13, _ = load_bc_checkpoint(EXP / f"runs/ab13_s{seed}/bc_ab13_s{seed}_best.pt", device=device)
        a16, _ = load_bc_checkpoint(EXP / f"runs/ab16_s{seed}/bc_ab16_s{seed}_best.pt", device=device)
        p13 = predict_actions(a13, s13_va, device=device).astype(np.float64)
        p16 = predict_actions(a16, s16_va, device=device).astype(np.float64)
        se13 = ((p13 - a_va.astype(np.float64)) ** 2).mean(axis=1)
        se16 = ((p16 - a_va.astype(np.float64)) ** 2).mean(axis=1)
        dse = se13 - se16  # >0 表示16D更好
        c3 = {
            "dse_mean": float(dse.mean()),
            "dse_p10": float(np.quantile(dse, 0.1)),
            "dse_p50": float(np.median(dse)),
            "dse_p90": float(np.quantile(dse, 0.9)),
            "frac_improved": float(np.mean(dse > 0)),
            "frac_improved_big005": float(np.mean(dse > 0.05)),
            "frac_worsened_big005": float(np.mean(dse < -0.05)),
        }
        # 改进与|Q_norm|/|gamma|/贴边的关系
        qn = s16_va[:, 13:16].astype(np.float64)
        qabs = np.abs(qn).max(axis=1)
        def grp_corr(u, v):
            u = u - u.mean(); v = v - v.mean()
            d = np.sqrt((u**2).sum()*(v**2).sum())
            return float((u*v).sum()/d) if d>0 else 0.0
        c3["corr_dse_qabsmax"] = grp_corr(dse, qabs)
        c3["corr_dse_absgamma"] = grp_corr(dse, np.abs(g))
        exp_edge = (np.abs(a_va).max(axis=1) >= BOUND)
        for tag, m in (("edge", exp_edge), ("nonedge", ~exp_edge)):
            c3[f"dse_mean_{tag}"] = float(dse[m].mean()) if m.sum() else 0.0
            c3[f"frac_improved_{tag}"] = float(np.mean(dse[m] > 0)) if m.sum() else 0.0

        # C4 权重 + 消融
        W = a16.mu[0].weight.detach().cpu().numpy()  # [256,16]
        col = np.sqrt((W ** 2).sum(axis=0))
        c4 = {
            "colnorm_13_mean": float(col[:13].mean()),
            "colnorm_Q": [float(v) for v in col[13:16]],
            "colnorm_Q_over_13": [float(v / (col[:13].mean()+1e-12)) for v in col[13:16]],
            "colnorm_13_perdim": [float(v) for v in col[:13]],
        }
        qmean = s16_tr[:, 13:16].mean(axis=0, keepdims=True).astype(np.float32)
        s16_va_abl = s16_va.copy(); s16_va_abl[:, 13:16] = qmean
        p16_abl = predict_actions(a16, s16_va_abl, device=device).astype(np.float64)
        mse16 = float(((p16 - a_va) ** 2).mean())
        mse16_abl = float(((p16_abl - a_va) ** 2).mean())
        mse13 = float(((p13 - a_va) ** 2).mean())
        c4["mse13"] = mse13; c4["mse16"] = mse16; c4["mse16_qmean_abl"] = mse16_abl
        c4["abl_revert_frac"] = float((mse16_abl - mse16) / (mse13 - mse16)) if abs(mse13-mse16)>1e-12 else 0.0

        # C5 贴边 精/召回 + 条件符号
        def pr(exp_e, bc_e):
            tp = int((exp_e & bc_e).sum()); fp = int(((~exp_e) & bc_e).sum()); fn = int((exp_e & (~bc_e)).sum())
            prec = tp/(tp+fp) if (tp+fp) else 0.0
            rec = tp/(tp+fn) if (tp+fn) else 0.0
            return prec, rec
        bc13_e = (np.abs(p13).max(axis=1) >= BOUND); bc16_e = (np.abs(p16).max(axis=1) >= BOUND)
        p13g, r13 = pr(exp_edge, bc13_e); p16g, r16 = pr(exp_edge, bc16_e)
        # 符号准确率分层（mask作用于全量数组，避免已切片后二次索引错位）
        def signacc(p):
            m = np.abs(a_va[:, 2]) > 1e-6
            return float(np.mean(np.sign(p[m, 2]) == np.sign(a_va[m, 2])))

        def signacc_masked(p_full, mask):
            m = mask & (np.abs(a_va[:, 2]) > 1e-6)
            if not m.sum():
                return 0.0
            return float(np.mean(np.sign(p_full[m, 2]) == np.sign(a_va[m, 2])))
        c5 = {
            "exp_edge_rate": float(exp_edge.mean()),
            "bc13_edge_rate": float(bc13_e.mean()), "bc16_edge_rate": float(bc16_e.mean()),
            "prec13": float(p13g), "rec13": float(r13), "prec16": float(p16g), "rec16": float(r16),
            "sign13": signacc(p13), "sign16": signacc(p16),
            "sign13_edge": signacc_masked(p13, exp_edge),
            "sign16_edge": signacc_masked(p16, exp_edge),
            "sign13_nonedge": signacc_masked(p13, ~exp_edge),
            "sign16_nonedge": signacc_masked(p16, ~exp_edge),
            "gmae13_edge": float(np.abs(p13[exp_edge][:,2]-a_va[exp_edge][:,2]).mean()) if exp_edge.sum() else 0.0,
            "gmae16_edge": float(np.abs(p16[exp_edge][:,2]-a_va[exp_edge][:,2]).mean()) if exp_edge.sum() else 0.0,
        }
        report["seeds"][str(seed)] = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}
        print(f"[dig] seed {seed} done: jac={c1['jaccard_mean']:.3f} qfrac={c1['q_dist_frac_median']:.4f} "
              f"R2_13={c2.get('linprobe_R2_s13_gamma',0):.3f} R2_16={c2.get('linprobe_R2_s16_gamma',0):.3f} "
              f"abl_revert={c4['abl_revert_frac']:.2f}", flush=True)

    (EXP / "dig_q_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- MD ----
    L = ["# DIG_Q：Q为何没降低近邻歧义（纯离线深挖）", ""]
    L.append("结论先行（3 seed 一致才算结论，否则记分歧）：")
    for seed in SEEDS:
        r = report["seeds"][str(seed)]
        L.append(f"## seed {seed}")
        L.append(f"- C1 重叠：Jaccard均值={r['c1']['jaccard_mean']:.3f}（中位={r['c1']['jaccard_p50']:.3f}，>0.5占比={r['c1']['jaccard_frac_gt05']*100:.1f}%)；"
                 f"16D距离中Q分量中位={r['c1']['q_dist_frac_median']*100:.2f}%（p90={r['c1']['q_dist_frac_p90']*100:.2f}%）；"
                 f"Q归一化std={['%.3f'%v for v in r['c1']['s16_q_std']]} vs 13D维平均std={r['c1']['s13_std_mean']:.3f}")
        L.append(f"  近邻γ极差>1：13D={r['c1']['gamma_range_gt1_13']*100:.1f}% vs 16D={r['c1']['gamma_range_gt1_16']*100:.1f}%"
                 f"（均值极差 {r['c1']['gamma_range_mean_13']:.3f}→{r['c1']['gamma_range_mean_16']:.3f}）")
        L.append(f"- C2 信息量：pear(Q1,γ)={r['c2'].get('pear_Q1_gamma',0):+.3f}，pear(Q2,γ)={r['c2'].get('pear_Q2_gamma',0):+.3f}，"
                 f"pear(QT,γ)={r['c2'].get('pear_QT_gamma',0):+.3f}；Q2分箱方差收缩={r['c2'].get('var_shrink_Q2',0)*100:.1f}%；"
                 f"QT分箱收缩={r['c2'].get('var_shrink_QT',0)*100:.1f}%；线性探针R² {r['c2'].get('linprobe_R2_s13_gamma',0):.3f}→{r['c2'].get('linprobe_R2_s16_gamma',0):.3f}")
        L.append(f"- C3 改进定位：ΔSE均值={r['c3']['dse_mean']:.4f}（p10={r['c3']['dse_p10']:.4f}/p50={r['c3']['dse_p50']:.4f}/p90={r['c3']['dse_p90']:.4f}）；"
                 f"改善占比={r['c3']['frac_improved']*100:.1f}%（大幅改善>0.05占={r['c3']['frac_improved_big005']*100:.1f}%，大幅恶化={r['c3']['frac_worsened_big005']*100:.1f}%）；"
                 f"corr(ΔSE,|Q|)={r['c3']['corr_dse_qabsmax']:+.3f}，corr(ΔSE,|γ|)={r['c3']['corr_dse_absgamma']:+.3f}；"
                 f"贴边步ΔSE={r['c3']['dse_mean_edge']:.4f} vs 非贴边={r['c3']['dse_mean_nonedge']:.4f}")
        L.append(f"- C4 利用度：首层列范数Q/13D均值比={['%.2f'%v for v in r['c4']['colnorm_Q_over_13']]}；"
                 f"Q置均值消融MSE {r['c4']['mse16']:.4f}→{r['c4']['mse16_qmean_abl']:.4f}（13D={r['c4']['mse13']:.4f}，回退比例={r['c4']['abl_revert_frac']*100:.0f}%）")
        L.append(f"- C5 贴边：专家贴边={r['c5']['exp_edge_rate']*100:.1f}%；BC 13D={r['c5']['bc13_edge_rate']*100:.2f}%→16D={r['c5']['bc16_edge_rate']*100:.2f}%；"
                 f"精/召 13D={r['c5']['prec13']:.2f}/{r['c5']['rec13']:.3f} vs 16D={r['c5']['prec16']:.2f}/{r['c5']['rec16']:.3f}；"
                 f"符号acc 13D={r['c5']['sign13']:.3f}→16D={r['c5']['sign16']:.3f}"
                 f"（贴边子集 {r['c5']['sign13_edge']:.3f}→{r['c5']['sign16_edge']:.3f}，非贴边 {r['c5']['sign13_nonedge']:.3f}→{r['c5']['sign16_nonedge']:.3f}）")
        L.append("")
    L.append("## 综合判读（待跑完回填）")
    (EXP / "DIG_Q_AMBIGUITY.md").write_text("\n".join(L), encoding="utf-8")
    print("[dig] wrote dig_q_report.json + DIG_Q_AMBIGUITY.md", flush=True)

if __name__ == "__main__":
    main()
