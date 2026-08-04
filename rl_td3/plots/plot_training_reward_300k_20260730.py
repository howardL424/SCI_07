"""Fig.7-style training reward curves: TD3 vs BC-TD3 (timesteps).

Data:
  - Mean line: TensorBoard rollout/ep_rew_mean (smoothed)
  - Shade: eval_logs/evaluations.npz mean +/- std (150 eval episodes)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "rl_td3" / "out_pictures"

RUNS = [
    {
        "name": "Reward-TD3",
        "std_name": "Reward_std",
        "ckpt": ROOT / "td3_baseline" / "checkpoints" / "training_300k_20260730_1917",
        "color": "#2ca02c",
        "markerfacecolor": "#2ca02c",
        "markeredgecolor": "#2ca02c",
        "marker": "o",
    },
    {
        "name": "Reward-BC-TD3",
        "std_name": "Reward_std",
        "ckpt": ROOT / "rl_td3" / "checkpoints" / "training_300k_20260730_1917",
        "color": "#d62728",
        "markerfacecolor": "none",
        "markeredgecolor": "#d62728",
        "marker": "o",
    },
]


def load_tb_ep_rew(tb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    events = ea.Scalars("rollout/ep_rew_mean")
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, vals


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    # reflect-pad to keep length
    pad = window // 2
    yp = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def load_eval(eval_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(eval_npz)
    ts = d["timesteps"].astype(float)
    res = d["results"]
    return ts, res.mean(axis=1), res.std(axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    smooth_window = 11
    marker_stride = 25

    for cfg in RUNS:
        ckpt = cfg["ckpt"]
        tb_dir = ckpt / "tb_logs" / "TD3_1"
        eval_npz = ckpt / "eval_logs" / "evaluations.npz"

        tb_x, tb_y = load_tb_ep_rew(tb_dir)
        tb_ys = moving_average(tb_y, smooth_window)
        ev_x, ev_mean, ev_std = load_eval(eval_npz)

        # Shade from evaluation mean +/- std
        ax.fill_between(
            ev_x,
            ev_mean - ev_std,
            ev_mean + ev_std,
            color=cfg["color"],
            alpha=0.18,
            linewidth=0,
            label=cfg["std_name"],
            zorder=1,
        )

        # Dense training mean from TensorBoard
        ax.plot(
            tb_x,
            tb_ys,
            color=cfg["color"],
            lw=1.8,
            solid_capstyle="round",
            zorder=2,
            label="_nolegend_",
        )

        # Markers on a subsample of the TB curve
        mk = slice(0, None, marker_stride)
        ax.plot(
            tb_x[mk],
            tb_ys[mk],
            linestyle="None",
            marker=cfg["marker"],
            markersize=5.5,
            markerfacecolor=cfg["markerfacecolor"],
            markeredgecolor=cfg["markeredgecolor"],
            markeredgewidth=1.2,
            zorder=3,
            label=cfg["name"],
        )

        # Eval mean markers (align shade center)
        ax.plot(
            ev_x,
            ev_mean,
            linestyle="None",
            marker=cfg["marker"],
            markersize=6.5,
            markerfacecolor=cfg["markerfacecolor"],
            markeredgecolor=cfg["markeredgecolor"],
            markeredgewidth=1.3,
            zorder=4,
            label="_nolegend_",
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title("Training reward of TD3 / BC-TD3")
    ax.set_xlim(0, 300_000)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Legend order: std / mean for each series (Fig.7 style)
    handles, labels = ax.get_legend_handles_labels()
    # fill_between and marker plots were added per run as: std, name
    ax.legend(handles, labels, loc="lower right", framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    out = OUT_DIR / "training_reward_td3_vs_bctd3_300k_20260730.png"
    fig.savefig(out, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
