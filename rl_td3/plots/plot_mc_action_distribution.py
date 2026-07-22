"""Collect all stage2 actions from seed=1 100-ep MC and plot histograms.

0711 best_model must use:
  --action-mapping td3_box_log_v1
  --obs-mode raw
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

from action_mapping import action_norm_to_phys  # noqa: E402
from evasion_env import EvasionEnv  # noqa: E402
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
    p.add_argument("--action-mapping", type=str, default="td3_box_log_v1")
    p.add_argument("--obs-mode", choices=["raw", "scaled"], default="raw")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "out_pictures" / "0711_300k_action_distribution_new",
        help="Directory for PNG/NPZ outputs "
        "(default: rl_td3/out_pictures/0711_300k_action_distribution_new)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="If set and exists, load actions from this npz and skip MC replay",
    )
    return p.parse_args()


def run_mc(args: argparse.Namespace) -> dict[str, np.ndarray]:
    model_path = args.model
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    model_zip = model_path + ".zip"
    if not Path(model_zip).exists():
        raise SystemExit(f"missing model: {model_zip}")

    np.random.seed(args.seed)
    print(f"[mc] model={model_zip}")
    print(f"[mc] mapping={args.action_mapping} obs={args.obs_mode}")
    print(f"[mc] episodes={args.episodes} seed={args.seed}")

    env = EvasionEnv(
        action_mapping=args.action_mapping,
        obs_normalize=(args.obs_mode == "scaled"),
        record_stage3_traj=False,
    )
    try:
        model = TD3BC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    an_l: list[np.ndarray] = []
    g_l: list[float] = []
    wt_l: list[float] = []
    we_l: list[float] = []
    ep_id_l: list[int] = []
    steps_per_ep: list[int] = []

    for ep in range(args.episodes):
        ep_seed = args.seed if ep == 0 else None
        obs, _ = env.reset(seed=ep_seed) if ep_seed is not None else env.reset()
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
            ep_id_l.append(ep)
            obs, _, terminated, _, _ = env.step(action)
            n_steps += 1
        steps_per_ep.append(n_steps)
        if (ep + 1) % 10 == 0:
            print(f"  [ep {ep+1:3d}/{args.episodes}] steps={n_steps}")

    env.close()
    return {
        "action_norm": np.asarray(an_l, dtype=np.float64),
        "gamma": np.asarray(g_l, dtype=np.float64),
        "wT": np.asarray(wt_l, dtype=np.float64),
        "wE": np.asarray(we_l, dtype=np.float64),
        "ep_id": np.asarray(ep_id_l, dtype=np.int32),
        "steps_per_ep": np.asarray(steps_per_ep, dtype=np.int32),
        "seed": np.asarray([args.seed], dtype=np.int32),
        "episodes": np.asarray([args.episodes], dtype=np.int32),
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


def plot_histograms(data: dict[str, np.ndarray], out_dir: Path, seed: int, n_ep: int) -> list[Path]:
    gamma = data["gamma"]
    wT = data["wT"]
    wE = data["wE"]
    an = data["action_norm"]
    n = int(gamma.size)

    # Physical corner counts (log-space bins for weights)
    g_pos = int(np.sum(gamma > 0))
    g_neg = int(np.sum(gamma < 0))
    g_mid = n - g_pos - g_neg

    log_wT = np.log10(np.clip(wT, 1e-30, None))
    log_wE = np.log10(np.clip(wE, 1e-30, None))
    # Expected box corners for td3_box_log_v1: wT in [1e2,1e5], wE in [1e2,1e4]
    wt_lo = int(np.sum(np.isclose(log_wT, 2.0, atol=0.05)))
    wt_hi = int(np.sum(np.isclose(log_wT, 5.0, atol=0.05)))
    wt_other = n - wt_lo - wt_hi
    we_lo = int(np.sum(np.isclose(log_wE, 2.0, atol=0.05)))
    we_hi = int(np.sum(np.isclose(log_wE, 4.0, atol=0.05)))
    we_other = n - we_lo - we_hi

    outs: list[Path] = []

    # Figure 1: physical corner bars + normalized hist
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))

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
        (axes[1, 0], an[:, 0], r"$\gamma_n$", "#4c72b0"),
        (axes[1, 1], an[:, 1], r"$w_{T,n}$", "#55a868"),
        (axes[1, 2], an[:, 2], r"$w_{E,n}$", "#c44e52"),
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

    fig.suptitle(
        f"0711 best_model MC action distribution  (seed={seed}, {n_ep} eps, {n} stage2 steps)\n"
        "Top: physical corner counts  |  Bottom: normalized action histograms (dashed = |a|=0.99)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    p1 = out_dir / f"action_distribution_mc_seed{seed}_{n_ep}ep.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outs.append(p1)

    # Figure 2: log10(w) histograms for extra clarity
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
        f"0711 MC physical action histograms (seed={seed}, {n_ep} eps, {n} steps)",
        fontsize=11,
        y=1.05,
    )
    fig.tight_layout()
    p2 = out_dir / f"action_histogram_mc_seed{seed}_{n_ep}ep_physical.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outs.append(p2)

    summary = {
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
        "bound_frac_any": float(np.mean(np.any(np.abs(an) >= 0.99, axis=1))),
        "bound_frac_all": float(np.mean(np.all(np.abs(an) >= 0.99, axis=1))),
    }
    sp = out_dir / f"action_distribution_mc_seed{seed}_{n_ep}ep_summary.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outs.append(sp)
    print(json.dumps(summary, indent=2))
    return outs


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = args.cache
    if cache is None:
        cache = args.out_dir / f"mc_actions_seed{args.seed}_{args.episodes}ep.npz"

    if cache.exists() and args.cache is not None:
        print(f"[mc] load cache {cache}")
        raw = np.load(cache)
        data = {k: raw[k] for k in raw.files}
    elif cache.exists() and args.cache is None:
        # default: reuse cache if present to avoid accidental long re-run;
        # pass --cache with a new path or delete file to force rerun
        print(f"[mc] reuse existing cache {cache} (delete it to re-run MC)")
        raw = np.load(cache)
        data = {k: raw[k] for k in raw.files}
    else:
        data = run_mc(args)
        np.savez_compressed(cache, **data)
        print(f"[mc] wrote {cache}")

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 200})
    outs = plot_histograms(data, args.out_dir, int(data["seed"][0]), int(data["episodes"][0]))
    for p in outs:
        print("saved:", p)


if __name__ == "__main__":
    main()
