"""Plot eval curves for 0711 vs 0715 TD3+BC runs. Outputs PNGs under rl_td3/out_pictures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RL_TD3 = Path(__file__).resolve().parents[1]
OUT_DIR = RL_TD3 / "out_pictures"

RUNS = [
    {
        "name": "0711 (result_0702)",
        "short": "0711",
        "path": RL_TD3
        / "checkpoints"
        / "training_300k_td3bc_0711"
        / "eval_logs"
        / "evaluations.npz",
        "color": "#1f77b4",
    },
    {
        "name": "0715 (result_0714)",
        "short": "0715",
        "path": RL_TD3
        / "checkpoints"
        / "training_300k_20260715_1008"
        / "eval_logs"
        / "evaluations.npz",
        "color": "#d62728",
    },
]


def load_run(cfg: dict) -> dict:
    ev = np.load(cfg["path"])
    ts = ev["timesteps"].astype(float)
    res = ev["results"]
    mean = res.mean(axis=1)
    std = res.std(axis=1)
    best_i = int(np.argmax(mean))
    return {**cfg, "ts": ts, "mean": mean, "std": std, "best_i": best_i}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = [load_run(r) for r in RUNS]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    # Overlay
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for d in data:
        x = d["ts"] / 1000
        ax.plot(x, d["mean"], color=d["color"], lw=1.8, label=d["name"])
        ax.fill_between(
            x,
            d["mean"] - d["std"],
            d["mean"] + d["std"],
            color=d["color"],
            alpha=0.15,
        )
        bi = d["best_i"]
        ax.scatter(
            [d["ts"][bi] / 1000],
            [d["mean"][bi]],
            color=d["color"],
            s=55,
            zorder=5,
            marker="*",
            edgecolors="k",
            linewidths=0.4,
            label=(
                f"{d['short']} best @ {int(d['ts'][bi]) // 1000}k "
                f"({d['mean'][bi]:.1f})"
            ),
        )

    ax.axvline(150, color="gray", ls="--", lw=1, alpha=0.7, label=r"$\lambda\to 0$ (150k)")
    ax.set_xlabel("Training steps (x1000)")
    ax.set_ylabel("Eval mean reward (20 eps)")
    ax.set_title("TD3+BC eval curve: 0711 vs 0715")
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_xlim(0, 310)
    fig.tight_layout()
    p1 = OUT_DIR / "training_curves_0711_vs_0715.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, d in zip(axes, data):
        x = d["ts"] / 1000
        ax.plot(x, d["mean"], color=d["color"], lw=1.8)
        ax.fill_between(
            x,
            d["mean"] - d["std"],
            d["mean"] + d["std"],
            color=d["color"],
            alpha=0.18,
        )
        bi = d["best_i"]
        ax.scatter(
            [d["ts"][bi] / 1000],
            [d["mean"][bi]],
            color=d["color"],
            s=60,
            zorder=5,
            marker="*",
        )
        ax.axvline(150, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_title(
            f"{d['name']}\n"
            f"best @ {int(d['ts'][bi])} steps, mean={d['mean'][bi]:.1f}"
        )
        ax.set_xlabel("Training steps (x1000)")
        ax.set_xlim(0, 310)
    axes[0].set_ylabel("Eval mean reward (20 eps)")
    fig.suptitle(
        "TD3+BC training eval curves (shaded = +/-1 std over 20 eval eps)",
        y=1.02,
    )
    fig.tight_layout()
    p2 = OUT_DIR / "training_curves_0711_0715_side_by_side.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    print("saved:", p1)
    print("saved:", p2)
    for d in data:
        bi = d["best_i"]
        print(
            f"{d['short']}: best step={int(d['ts'][bi])}, "
            f"mean={d['mean'][bi]:.2f}, final={d['mean'][-1]:.2f}"
        )


if __name__ == "__main__":
    main()
