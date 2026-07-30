"""Collect stage2 actions from MC / full test-IC and plot histograms.

Examples
--------
# 0711 (旧坐标 + raw obs)，100 轮随机 MC：
  python rl_td3/plots/plot_mc_action_distribution.py \\
    --seed 1 --episodes 100 \\
    --model rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model \\
    --action-mapping td3_box_log_v1 --obs-mode raw

# 260k (比值坐标 + scaled)，全量 test-IC（约 300）：
  python rl_td3/plots/plot_mc_action_distribution.py \\
    --seed 0 --episodes 0 \\
    --model rl_td3/checkpoints/training_260k_20260723_152457/best/best_model \\
    --action-mapping ratio_log_box_v1 --obs-mode scaled \\
    --out-dir rl_td3/out_pictures/260k_action_distribution_test300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from action_mapping import (  # noqa: E402
    ACTION_MAPPING_RATIO,
    ACTION_MAPPING_TD3_BOX,
    action_names,
    action_norm_to_phys,
)
from evasion_env import EvasionEnv, SAMPLES_TEST_FILE  # noqa: E402
from td3_bc import TD3BC  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MC action distribution histograms")
    p.add_argument(
        "--model",
        type=str,
        default=str(
            ROOT
            / "checkpoints"
            / "training_300k_td3bc_0711"
            / "best"
            / "best_model"
        ),
    )
    p.add_argument("--action-mapping", type=str, default=ACTION_MAPPING_TD3_BOX)
    p.add_argument("--obs-mode", choices=["raw", "scaled"], default="raw")
    p.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="随机抽 N 轮；0=测完全部初态池（与 evaluate_td3 全量 test-IC 一致）",
    )
    p.add_argument(
        "--samples",
        type=str,
        default=SAMPLES_TEST_FILE,
        help="初态池 mat（默认 samples_clean_test.mat）",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "out_pictures" / "0711_300k_action_distribution_new",
        help="Directory for PNG/NPZ outputs",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="If set and exists, load actions from this npz and skip MC replay",
    )
    p.add_argument(
        "--title-tag",
        type=str,
        default="",
        help="图标题前缀（默认从 model 路径推断）",
    )
    return p.parse_args()


def _title_tag(args: argparse.Namespace) -> str:
    if args.title_tag:
        return args.title_tag
    p = Path(args.model)
    # .../training_XXX/best/best_model → training_XXX
    if p.name == "best_model" and p.parent.name == "best":
        return p.parent.parent.name
    return p.name


def run_mc(args: argparse.Namespace) -> dict[str, np.ndarray]:
    model_path = args.model
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    model_zip = model_path + ".zip"
    if not Path(model_zip).exists():
        raise SystemExit(f"missing model: {model_zip}")

    np.random.seed(args.seed)
    sweep_all = args.episodes <= 0
    print(f"[mc] model={model_zip}")
    print(f"[mc] mapping={args.action_mapping} obs={args.obs_mode}")
    print(f"[mc] samples={args.samples}")

    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        obs_normalize=(args.obs_mode == "scaled"),
        record_stage3_traj=False,
    )
    try:
        model = TD3BC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    n_plan = env.n_samples if sweep_all else int(args.episodes)
    mode = f"全量 test-IC（{n_plan}）" if sweep_all else f"随机 {n_plan} 轮 seed={args.seed}"
    print(f"[mc] mode={mode}")

    an_l: list[np.ndarray] = []
    g_l: list[float] = []
    wt_l: list[float] = []
    we_l: list[float] = []
    ep_id_l: list[int] = []
    steps_per_ep: list[int] = []
    local_idx_l: list[int] = []
    n_skip = 0
    ep_done = 0

    for ep in range(n_plan):
        ep_seed = args.seed if ep == 0 else None
        try:
            if sweep_all:
                obs, info = env.reset(
                    seed=ep_seed, options={"local_idx": int(ep)}
                )
                local_idx = int(info.get("local_idx", ep))
            else:
                obs, info = (
                    env.reset(seed=ep_seed) if ep_seed is not None else env.reset()
                )
                local_idx = int(info.get("local_idx", -1))
        except RuntimeError as exc:
            n_skip += 1
            print(f"  [skip local_idx={ep}] {exc}")
            continue

        terminated = False
        n_steps = 0
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float64).ravel()
            gamma, wT, wE = action_norm_to_phys(action, version=args.action_mapping)
            g = float(np.asarray(gamma).ravel()[0])
            wt = float(np.asarray(wT).ravel()[0])
            we = float(np.asarray(wE).ravel()[0])
            an_l.append(action.copy())
            g_l.append(g)
            wt_l.append(wt)
            we_l.append(we)
            ep_id_l.append(ep_done)
            local_idx_l.append(local_idx)
            obs, _, terminated, _, _ = env.step(action)
            n_steps += 1
        steps_per_ep.append(n_steps)
        ep_done += 1
        if ep_done % 10 == 0 or ep + 1 == n_plan:
            print(f"  [done {ep_done:3d} | scanned {ep+1:3d}/{n_plan}] steps={n_steps}")

    if n_skip:
        print(f"[mc] warm-start 跳过 {n_skip}/{n_plan}")

    env.close()
    return {
        "action_norm": np.asarray(an_l, dtype=np.float64),
        "gamma": np.asarray(g_l, dtype=np.float64),
        "wT": np.asarray(wt_l, dtype=np.float64),
        "wE": np.asarray(we_l, dtype=np.float64),
        "ep_id": np.asarray(ep_id_l, dtype=np.int32),
        "local_idx": np.asarray(local_idx_l, dtype=np.int32),
        "steps_per_ep": np.asarray(steps_per_ep, dtype=np.int32),
        "seed": np.asarray([args.seed], dtype=np.int32),
        "episodes": np.asarray([ep_done], dtype=np.int32),
        "action_mapping": np.asarray([args.action_mapping]),
        "obs_mode": np.asarray([args.obs_mode]),
        "sweep_all": np.asarray([int(sweep_all)], dtype=np.int32),
    }


def _corner_bar(ax, labels, counts, total, color, title):
    vals = np.asarray(counts, dtype=float)
    bars = ax.bar(labels, vals, color=color, edgecolor="k", linewidth=0.4, width=0.65)
    for bar, v in zip(bars, vals):
        pct = 100.0 * v / max(total, 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(v)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel("Count (steps)")
    ymax = float(vals.max()) if len(vals) else 1.0
    ax.set_ylim(0, ymax * 1.25)
    ax.grid(True, axis="y", alpha=0.3)


def _bound_vs_interior_bar(ax, arr: np.ndarray, color: str, title: str, thr: float = 0.99):
    n = int(arr.size)
    bound = int(np.sum(np.abs(arr) >= thr))
    interior = n - bound
    _corner_bar(
        ax,
        [f"|a|<{thr:g}", f"|a|≥{thr:g}"],
        [interior, bound],
        n,
        color,
        title,
    )


def plot_histograms(
    data: dict[str, np.ndarray],
    out_dir: Path,
    seed: int,
    n_ep: int,
    *,
    action_mapping: str,
    title_tag: str,
) -> list[Path]:
    gamma = data["gamma"]
    wT = data["wT"]
    wE = data["wE"]
    an = data["action_norm"]
    n = int(gamma.size)
    names = action_names(action_mapping)

    g_pos = int(np.sum(gamma > 0))
    g_neg = int(np.sum(gamma < 0))
    g_mid = n - g_pos - g_neg

    log_wT = np.log10(np.clip(wT, 1e-30, None))
    log_wE = np.log10(np.clip(wE, 1e-30, None))
    # 旧权重盒角点（td3_box）；ratio 下大量落在盒外，计入 other
    wt_lo = int(np.sum(np.isclose(log_wT, 2.0, atol=0.05)))
    wt_hi = int(np.sum(np.isclose(log_wT, 5.0, atol=0.05)))
    wt_other = n - wt_lo - wt_hi
    we_lo = int(np.sum(np.isclose(log_wE, 2.0, atol=0.05)))
    we_hi = int(np.sum(np.isclose(log_wE, 4.0, atol=0.05)))
    we_other = n - we_lo - we_hi

    outs: list[Path] = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))

    if action_mapping == ACTION_MAPPING_RATIO:
        # 比值坐标：上排看归一化贴边（坍塌诊断），下排归一化直方
        colors = ["#4c72b0", "#55a868", "#c44e52"]
        for i, (ax, name, color) in enumerate(zip(axes[0], names, colors)):
            _bound_vs_interior_bar(
                ax, an[:, i], color, rf"Bound check ${name}$"
            )
    else:
        g_labels, g_counts = [], []
        if g_neg:
            g_labels.append(r"$\gamma=-3$")
            g_counts.append(g_neg)
        if g_mid:
            g_labels.append("other")
            g_counts.append(g_mid)
        if g_pos:
            g_labels.append(r"$\gamma=+3$")
            g_counts.append(g_pos)
        _corner_bar(axes[0, 0], g_labels, g_counts, n, "#4c72b0", r"Physical $\gamma$")

        wt_labels, wt_counts = [], []
        if wt_lo:
            wt_labels.append(r"$w_T=10^{2}$")
            wt_counts.append(wt_lo)
        if wt_other:
            wt_labels.append("other")
            wt_counts.append(wt_other)
        if wt_hi:
            wt_labels.append(r"$w_T=10^{5}$")
            wt_counts.append(wt_hi)
        _corner_bar(axes[0, 1], wt_labels, wt_counts, n, "#55a868", r"Physical $w_T$")

        we_labels, we_counts = [], []
        if we_lo:
            we_labels.append(r"$w_E=10^{2}$")
            we_counts.append(we_lo)
        if we_other:
            we_labels.append("other")
            we_counts.append(we_other)
        if we_hi:
            we_labels.append(r"$w_E=10^{4}$")
            we_counts.append(we_hi)
        _corner_bar(axes[0, 2], we_labels, we_counts, n, "#c44e52", r"Physical $w_E$")

    bins = np.linspace(-1.05, 1.05, 22)
    specs = [
        (axes[1, 0], an[:, 0], rf"${names[0]}$", "#4c72b0"),
        (axes[1, 1], an[:, 1], rf"${names[1]}$", "#55a868"),
        (axes[1, 2], an[:, 2], rf"${names[2]}$", "#c44e52"),
    ]
    for ax, arr, title, color in specs:
        ax.hist(arr, bins=bins, color=color, edgecolor="k", linewidth=0.3)
        ax.axvline(-0.99, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(0.99, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("normalized action")
        ax.set_xlim(-1.15, 1.15)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1, 0].set_ylabel("Count (steps)")

    top_desc = (
        "Top: |a|≥0.99 bound vs interior"
        if action_mapping == ACTION_MAPPING_RATIO
        else "Top: physical corner counts"
    )
    fig.suptitle(
        f"{title_tag} MC action distribution  "
        f"(seed={seed}, {n_ep} eps, {n} stage2 steps, {action_mapping})\n"
        f"{top_desc}  |  Bottom: normalized histograms (dashed = |a|=0.99)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    p1 = out_dir / f"action_distribution_mc_seed{seed}_{n_ep}ep.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outs.append(p1)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].hist(gamma, bins=21, color="#4c72b0", edgecolor="k", linewidth=0.3)
    axes[0].set_title(r"$\gamma$")
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel("Count (steps)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].hist(log_wT, bins=21, color="#55a868", edgecolor="k", linewidth=0.3)
    axes[1].set_title(r"$\log_{10}(w_T)$")
    axes[1].set_xlabel(r"$\log_{10}(w_T)$")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].hist(log_wE, bins=21, color="#c44e52", edgecolor="k", linewidth=0.3)
    axes[2].set_title(r"$\log_{10}(w_E)$")
    axes[2].set_xlabel(r"$\log_{10}(w_E)$")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{title_tag} MC physical action histograms "
        f"(seed={seed}, {n_ep} eps, {n} steps)",
        fontsize=11,
        y=1.05,
    )
    fig.tight_layout()
    p2 = out_dir / f"action_histogram_mc_seed{seed}_{n_ep}ep_physical.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outs.append(p2)

    summary = {
        "title_tag": title_tag,
        "action_mapping": action_mapping,
        "seed": seed,
        "episodes": n_ep,
        "n_steps": n,
        "gamma_pos_frac": g_pos / n,
        "gamma_neg_frac": g_neg / n,
        "gamma_other_frac": g_mid / n,
        "wT_lo_frac": wt_lo / n,
        "wT_hi_frac": wt_hi / n,
        "wT_other_frac": wt_other / n,
        "wE_lo_frac": we_lo / n,
        "wE_hi_frac": we_hi / n,
        "wE_other_frac": we_other / n,
        "gamma_mean": float(gamma.mean()),
        "wT_mean": float(wT.mean()),
        "wE_mean": float(wE.mean()),
        "bound_frac_dim0": float(np.mean(np.abs(an[:, 0]) >= 0.99)),
        "bound_frac_dim1": float(np.mean(np.abs(an[:, 1]) >= 0.99)),
        "bound_frac_dim2": float(np.mean(np.abs(an[:, 2]) >= 0.99)),
        "bound_frac_any": float(np.mean(np.any(np.abs(an) >= 0.99, axis=1))),
        "bound_frac_all": float(np.mean(np.all(np.abs(an) >= 0.99, axis=1))),
        "action_names": list(names),
    }
    sp = out_dir / f"action_distribution_mc_seed{seed}_{n_ep}ep_summary.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outs.append(sp)
    print(json.dumps(summary, indent=2))
    return outs


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def _mapping_from_data(data: dict[str, np.ndarray], fallback: str) -> str:
    if "action_mapping" in data:
        v = data["action_mapping"]
        return str(v.item() if getattr(v, "shape", ()) == () else v.flat[0])
    return fallback


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = _title_tag(args)

    cache = args.cache
    if cache is None:
        cache = args.out_dir / f"mc_actions_seed{args.seed}_{args.episodes}ep.npz"
        # episodes=0 → 文件名用 all 避免歧义
        if args.episodes <= 0:
            cache = args.out_dir / f"mc_actions_seed{args.seed}_all.npz"

    if cache.exists() and args.cache is not None:
        print(f"[mc] load cache {cache}")
        data = _load_cache(cache)
    elif cache.exists() and args.cache is None:
        print(f"[mc] reuse existing cache {cache} (delete it to re-run MC)")
        data = _load_cache(cache)
    else:
        data = run_mc(args)
        np.savez_compressed(cache, **data)
        print(f"[mc] wrote {cache}")

    mapping = _mapping_from_data(data, args.action_mapping)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 200})
    outs = plot_histograms(
        data,
        args.out_dir,
        int(data["seed"][0]),
        int(data["episodes"][0]),
        action_mapping=mapping,
        title_tag=tag,
    )
    for p in outs:
        print("saved:", p)


if __name__ == "__main__":
    main()
