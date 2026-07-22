"""Plot 0711-only eval curve and action distribution bars/histograms.

Outputs:
  - training curve → rl_td3/out_pictures/
  - action bars/histograms → rl_td3/out_pictures/0711_300k_action_distribution_old/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RL_TD3 = Path(__file__).resolve().parents[1]
OUT_DIR = RL_TD3 / "out_pictures"
ACTION_OUT_DIR = OUT_DIR / "0711_300k_action_distribution_old"

EVAL_NPZ = (
    RL_TD3
    / "checkpoints"
    / "training_300k_td3bc_0711"
    / "eval_logs"
    / "evaluations.npz"
)

# Full-traj probe (20 eps, 1106 steps) — from full_traj_action_probe_20ep.log
N_STEPS = 1106
GAMMA_COUNTS = {3.0: 1106}  # 100%
WT_COUNTS = {1e2: 80, 1e5: 1026}  # 7.2% / 92.8%
WE_COUNTS = {1e2: 1098, 1e4: 8}  # 99.3% / 0.7%

# MC eval initial-state action probe (20 samples) — from monte_carlo_eval log
MC_INIT_GAMMA = [3.0] * 20
MC_INIT_WT = [1e5] * 20
MC_INIT_WE = [
    1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e4, 1e2,
    1e4, 1e2, 1e2, 1e4, 1e4, 1e2, 1e2, 1e2, 1e4, 1e2,
]


def plot_training_curve() -> Path:
    ev = np.load(EVAL_NPZ)
    ts = ev["timesteps"].astype(float)
    mean = ev["results"].mean(axis=1)
    std = ev["results"].std(axis=1)
    best_i = int(np.argmax(mean))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = ts / 1000
    ax.plot(x, mean, color="#1f77b4", lw=2.0, label="eval mean (20 eps)")
    ax.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.18, label="+/-1 std")
    ax.scatter(
        [ts[best_i] / 1000],
        [mean[best_i]],
        s=80,
        marker="*",
        color="#1f77b4",
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
        label=f"best @ {int(ts[best_i])} (mean={mean[best_i]:.1f})",
    )
    ax.axvline(150, color="gray", ls="--", lw=1, alpha=0.75, label=r"$\lambda\to 0$ (150k)")
    ax.set_xlabel("Training steps (x1000)")
    ax.set_ylabel("Eval mean reward (20 eps)")
    ax.set_title("0711 TD3+BC training eval curve")
    ax.set_xlim(0, 310)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.92)
    fig.tight_layout()
    out = OUT_DIR / "training_curve_0711.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _pct_bar(ax, labels, counts, total, color, title, ylabel="Count"):
    vals = np.asarray(counts, dtype=float)
    bars = ax.bar(labels, vals, color=color, edgecolor="k", linewidth=0.4, width=0.65)
    for bar, v in zip(bars, vals):
        pct = 100.0 * v / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(v)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.22 if len(vals) else 1)
    ax.grid(True, axis="y", alpha=0.3)


def plot_action_distribution() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))

    # Row 1: full-traj step counts (20 eps, 1106 steps)
    _pct_bar(
        axes[0, 0],
        [r"$\gamma=+3$"],
        [GAMMA_COUNTS[3.0]],
        N_STEPS,
        "#4c72b0",
        r"Full-traj $\gamma$ (1106 steps)",
    )
    _pct_bar(
        axes[0, 1],
        [r"$w_T=10^{2}$", r"$w_T=10^{5}$"],
        [WT_COUNTS[1e2], WT_COUNTS[1e5]],
        N_STEPS,
        "#55a868",
        r"Full-traj $w_T$ (1106 steps)",
    )
    _pct_bar(
        axes[0, 2],
        [r"$w_E=10^{2}$", r"$w_E=10^{4}$"],
        [WE_COUNTS[1e2], WE_COUNTS[1e4]],
        N_STEPS,
        "#c44e52",
        r"Full-traj $w_E$ (1106 steps)",
    )

    # Row 2: MC initial-state probe (20 samples) as bars + soft hist on log10
    mc_g = np.asarray(MC_INIT_GAMMA)
    mc_wt = np.asarray(MC_INIT_WT)
    mc_we = np.asarray(MC_INIT_WE)

    g_pos = (mc_g > 0).sum()
    _pct_bar(
        axes[1, 0],
        [r"$\gamma=+3$"],
        [g_pos],
        len(mc_g),
        "#4c72b0",
        r"MC init probe $\gamma$ (n=20)",
    )

    wt_lo = (np.isclose(mc_wt, 1e2)).sum()
    wt_hi = (np.isclose(mc_wt, 1e5)).sum()
    _pct_bar(
        axes[1, 1],
        [r"$w_T=10^{2}$", r"$w_T=10^{5}$"],
        [wt_lo, wt_hi],
        len(mc_wt),
        "#55a868",
        r"MC init probe $w_T$ (n=20)",
    )

    we_lo = (np.isclose(mc_we, 1e2)).sum()
    we_hi = (np.isclose(mc_we, 1e4)).sum()
    _pct_bar(
        axes[1, 2],
        [r"$w_E=10^{2}$", r"$w_E=10^{4}$"],
        [we_lo, we_hi],
        len(mc_we),
        "#c44e52",
        r"MC init probe $w_E$ (n=20)",
    )

    fig.suptitle(
        "0711 best_model action collapse\n"
        "Top: full-trajectory probe (20 eps)  |  Bottom: MC initial-state probe (20 ICs)\n"
        "All dimensions at action-box corners (no interior values)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = ACTION_OUT_DIR / "action_distribution_0711.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_action_hist_normalized() -> Path:
    """Normalized action a_n in [-1,1]: reconstruct step counts from probe summary."""
    # From probe: gamma_n always +1; wT_n: 7.2% -1, 92.8% +1; wE_n: 99.3% -1, 0.7% +1
    n = N_STEPS
    gamma_n = np.ones(n)
    wt_n = np.concatenate([np.full(80, -1.0), np.full(1026, 1.0)])
    we_n = np.concatenate([np.full(1098, -1.0), np.full(8, 1.0)])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    specs = [
        (axes[0], gamma_n, r"$\gamma_n$", "#4c72b0"),
        (axes[1], wt_n, r"$w_{T,n}$", "#55a868"),
        (axes[2], we_n, r"$w_{E,n}$", "#c44e52"),
    ]
    bins = np.linspace(-1.05, 1.05, 22)
    for ax, arr, title, color in specs:
        ax.hist(arr, bins=bins, color=color, edgecolor="k", linewidth=0.3, density=False)
        ax.axvline(-0.99, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(0.99, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("normalized action")
        ax.set_xlim(-1.15, 1.15)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Count (steps)")
    fig.suptitle(
        "0711 full-traj normalized action histogram (20 eps, 1106 steps; dashed = |a|=0.99)",
        fontsize=11,
        y=1.05,
    )
    fig.tight_layout()
    out = ACTION_OUT_DIR / "action_histogram_0711_normalized.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ACTION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 200,
        }
    )
    p1 = plot_training_curve()
    p2 = plot_action_distribution()
    p3 = plot_action_hist_normalized()
    print("saved:", p1)
    print("saved:", p2)
    print("saved:", p3)


if __name__ == "__main__":
    main()
