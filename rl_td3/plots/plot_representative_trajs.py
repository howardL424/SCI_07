"""绘制 MC 代表轨迹的三维轨迹、三分量加速度和动作参数曲线。

默认读取 `rep_trajs_seed1` 的 `selected_trajs.npz` 与 `selected_meta.json`：
每条已选轨迹输出三张图。三维轨迹与惯性加速度拼接 stage2+stage3；
动作参数仍仅 stage2（RL 权重段）。

在仓库根目录运行：
    E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe ^
      rl_td3/plots/plot_representative_trajs.py ^
      --run-dir rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制代表轨迹的轨迹、加速度与动作图")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            ROOT
            / "checkpoints"
            / "training_300k_td3bc_0711"
            / "rep_trajs_seed1"
        ),
        help="含 selected_trajs.npz 与 selected_meta.json 的 MC 落盘目录",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认 {run-dir}/figures",
    )
    return parser.parse_args()


def _meta_summary(meta: dict) -> str:
    state = (
        "both_ok"
        if meta["both_ok"]
        else ("evade-only" if meta["evade_ok"] else "intercepted")
    )
    return (
        f"{meta['tier']} | ep={meta['ep']} | {meta['action_type']}-type | "
        f"{state} | r_minT={meta['r_minT']:.3g} m"
    )


def _optional_array(
    data: np.lib.npyio.NpzFile, key: str, fallback_shape: tuple[int, ...]
) -> np.ndarray:
    if key not in data.files:
        return np.zeros(fallback_shape, dtype=np.float64)
    return np.asarray(data[key], dtype=np.float64)


def _stitch_stage23_positions(
    data: np.lib.npyio.NpzFile, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """拼接 stage2+stage3 位置；返回 (M,D1,D2, n_stage2)。stage3 首点若与 stage2 末点重合则跳过。"""
    m2 = np.asarray(data[f"{prefix}/pos_M"], dtype=np.float64)
    d12 = np.asarray(data[f"{prefix}/pos_D1"], dtype=np.float64)
    d22 = np.asarray(data[f"{prefix}/pos_D2"], dtype=np.float64)
    n2 = int(m2.shape[0])

    m3 = _optional_array(data, f"{prefix}/pos_M_s3", (0, 3))
    d13 = _optional_array(data, f"{prefix}/pos_D1_s3", (0, 3))
    d23 = _optional_array(data, f"{prefix}/pos_D2_s3", (0, 3))
    if m3.shape[0] == 0:
        return m2, d12, d22, n2

    skip = 0
    if n2 > 0 and np.allclose(m3[0], m2[-1], rtol=0.0, atol=1e-6):
        skip = 1
    if skip >= m3.shape[0]:
        return m2, d12, d22, n2
    return (
        np.vstack((m2, m3[skip:])),
        np.vstack((d12, d13[skip:])),
        np.vstack((d22, d23[skip:])),
        n2,
    )


def _stitch_stage23_accel(
    data: np.lib.npyio.NpzFile, prefix: str
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """拼接 stage2+stage3 惯性加速度与时间；返回 (t, accel, t_switch)。"""
    t2 = np.asarray(data[f"{prefix}/t_step"], dtype=np.float64)
    a2 = np.asarray(data[f"{prefix}/a_M_inertial"], dtype=np.float64)
    t_switch = None
    if f"{prefix}/t_switch" in data.files:
        ts = np.asarray(data[f"{prefix}/t_switch"], dtype=np.float64).reshape(-1)
        if ts.size and np.isfinite(ts[0]):
            t_switch = float(ts[0])
    elif t2.size:
        t_switch = float(t2[-1])

    t3 = _optional_array(data, f"{prefix}/t_s3", (0,))
    a3 = _optional_array(data, f"{prefix}/a_M_inertial_s3", (0, 3))
    if t3.size == 0 or a3.shape[0] == 0:
        return t2, a2, t_switch

    n = min(t3.size, a3.shape[0])
    t3 = t3[:n]
    a3 = a3[:n]
    skip = 0
    if t2.size and np.isclose(t3[0], t2[-1], rtol=0.0, atol=1e-9):
        skip = 1
    if skip >= n:
        return t2, a2, t_switch
    return np.concatenate((t2, t3[skip:])), np.vstack((a2, a3[skip:])), t_switch


def _set_equal_3d_aspect(ax: plt.Axes, *positions: np.ndarray) -> None:
    """按多个 [N,3] 位置数组设置 x/z/y 三轴一致的显示比例。"""
    all_pos = np.vstack(positions)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    centers = (mins + maxs) / 2.0
    half_range = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    # 图的坐标顺序为 x, z, y。
    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[2] - half_range, centers[2] + half_range)
    ax.set_zlim(centers[1] - half_range, centers[1] + half_range)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _plot_3d_path(
    ax: plt.Axes,
    xyz: np.ndarray,
    *,
    color: str,
    line_style: str,
    label: str,
    line_width: float = 1.5,
    with_markers: bool = True,
) -> None:
    """按专家使用的 x-z-y 坐标顺序绘路径；with_markers=False 时为平滑折线。"""
    kwargs: dict = {
        "color": color,
        "ls": line_style,
        "lw": line_width,
        "label": label,
    }
    if with_markers:
        kwargs.update({"marker": "o", "ms": 2.7, "markevery": 1})
    ax.plot(xyz[:, 0], xyz[:, 2], xyz[:, 1], **kwargs)


def _add_start_markers(ax: plt.Axes, m: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> None:
    """以不同空心符号显示起点；即使同点也能表达物体确实重合。"""
    ax.scatter(0.0, 0.0, 0.0, marker="^", s=82, color="black", label="Target")
    ax.scatter(
        m[0, 0], m[0, 2], m[0, 1],
        marker="o", s=78, facecolors="none", edgecolors="black", linewidths=1.3,
        label="M start",
    )
    ax.scatter(
        d1[0, 0], d1[0, 2], d1[0, 1],
        marker="s", s=68, facecolors="none", edgecolors="#d62728", linewidths=1.3,
        label="D1 start",
    )
    ax.scatter(
        d2[0, 0], d2[0, 2], d2[0, 1],
        marker="D", s=58, facecolors="none", edgecolors="#1f77b4", linewidths=1.3,
        label="D2 start",
    )


def _decorate_trajectory_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)


def plot_trajectory(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    meta: dict,
    out_path: Path,
) -> None:
    m, d1, d2, n2 = _stitch_stage23_positions(data, prefix)
    has_s3 = m.shape[0] > n2
    # 最近交会只用 stage2，并自动选 min(r_MD1,r_MD2) 更小的那枚防御弹
    md1_s2 = np.linalg.norm(m[:n2] - d1[:n2], axis=1) if n2 > 0 else np.zeros(0)
    md2_s2 = np.linalg.norm(m[:n2] - d2[:n2], axis=1) if n2 > 0 else np.zeros(0)
    min_d1 = float(md1_s2.min()) if md1_s2.size else np.inf
    min_d2 = float(md2_s2.min()) if md2_s2.size else np.inf
    use_d1 = min_d1 <= min_d2
    def_label = "D1" if use_d1 else "D2"
    def_pos = d1 if use_d1 else d2
    def_color = "#d62728" if use_d1 else "#1f77b4"
    def_style = "--" if use_d1 else "-."
    md_s2 = md1_s2 if use_d1 else md2_s2
    min_idx = int(np.argmin(md_s2)) if md_s2.size else 0
    min_dist = float(md_s2[min_idx]) if md_s2.size else float("nan")
    kill_note = (
        f"< 6 m kill → intercepted"
        if min_dist < 6.0
        else f"> 6 m kill radius"
    )

    all_pos = np.vstack((m, d1, d2))
    centered = all_pos - all_pos.mean(axis=0)
    # 排除浮点积分带来的 1e-14 级横向噪声，避免把真实一维共线轨迹误判为三维。
    singular_values = np.linalg.svd(centered, compute_uv=False)
    collinear_tol = max(float(singular_values[0]) * 1e-8, 1e-6)
    is_collinear = int(np.count_nonzero(singular_values > collinear_tol)) <= 1

    if is_collinear:
        fig = plt.figure(figsize=(9.0, 7.3))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=19, azim=132)
    else:
        fig = plt.figure(figsize=(12.2, 6.8))
        ax = fig.add_subplot(121, projection="3d")
        ax.view_init(elev=21, azim=132)

    if has_s3:
        _plot_3d_path(
            ax, m[:n2], color="black", line_style="-", line_width=2.0,
            label="M stage2 (RL)", with_markers=False,
        )
        _plot_3d_path(
            ax, m[n2 - 1 :], color="0.35", line_style="-", line_width=1.8,
            label="M stage3 (PN)", with_markers=False,
        )
        _plot_3d_path(
            ax, d1[:n2], color="#d62728", line_style="--", label="D1 stage2", with_markers=False,
        )
        _plot_3d_path(
            ax, d1[n2 - 1 :], color="#ff9896", line_style="--", label="D1 stage3", with_markers=False,
        )
        _plot_3d_path(
            ax, d2[:n2], color="#1f77b4", line_style="-.", label="D2 stage2", with_markers=False,
        )
        _plot_3d_path(
            ax, d2[n2 - 1 :], color="#aec7e8", line_style="-.", label="D2 stage3", with_markers=False,
        )
        ax.scatter(
            m[n2 - 1, 0], m[n2 - 1, 2], m[n2 - 1, 1],
            marker="x", s=70, color="purple", linewidths=1.6, label="stage2→3", zorder=9,
        )
    else:
        _plot_3d_path(
            ax, m, color="black", line_style="-", line_width=2.0,
            label="Attacker M", with_markers=False,
        )
        _plot_3d_path(
            ax, d1, color="#d62728", line_style="--", label="Defender D1", with_markers=False,
        )
        _plot_3d_path(
            ax, d2, color="#1f77b4", line_style="-.", label="Defender D2", with_markers=False,
        )
    _add_start_markers(ax, m, d1, d2)
    _set_equal_3d_aspect(ax, m, d1, d2, np.zeros((1, 3)))
    scope = "stage2+stage3" if has_s3 else "stage2 only"
    _decorate_trajectory_axes(ax, f"3D trajectories ({scope}) — {_meta_summary(meta)}")

    if is_collinear:
        ax.text2D(
            0.02,
            0.02,
            "All vehicles and target coincide at stage-2 start;\n"
            "all paths are collinear along x (not separable by camera angle).\n"
            "Main view: smooth polylines (no 0.1 s markers).",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.90},
        )
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.scatter(
            m[min_idx, 0], m[min_idx, 2], m[min_idx, 1],
            color="limegreen", edgecolors="black", s=55, zorder=8,
            label=f"M-{def_label} closest (stage2)",
        )
        ax.scatter(
            def_pos[min_idx, 0], def_pos[min_idx, 2], def_pos[min_idx, 1],
            color="gold", edgecolors="black", s=55, zorder=8,
        )
        ax.plot(
            [m[min_idx, 0], def_pos[min_idx, 0]],
            [m[min_idx, 2], def_pos[min_idx, 2]],
            [m[min_idx, 1], def_pos[min_idx, 1]],
            color="limegreen",
            ls=":",
            lw=2.0,
        )
        ax.text2D(
            0.02,
            0.02,
            f"min(M,{def_label})|stage2 = {min_dist:.2f} m ({kill_note})\n"
            f"(other: min(M,{'D2' if use_d1 else 'D1'})="
            f"{(min_d2 if use_d1 else min_d1):.2f} m)\n"
            "Main view: smooth; inset keeps 0.1 s markers.",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.90},
        )
        ax.legend(loc="upper left", fontsize=8)

        zoom = fig.add_subplot(122, projection="3d")
        for path, color, style, label, width in (
            (m[:n2], "black", "-", "Attacker M (stage2)", 2.0),
            (def_pos[:n2], def_color, def_style, f"Defender {def_label} (stage2)", 1.5),
        ):
            _plot_3d_path(
                zoom, path, color=color, line_style=style, line_width=width,
                label=label, with_markers=True,
            )
        zoom.scatter(
            m[min_idx, 0], m[min_idx, 2], m[min_idx, 1],
            color="limegreen", edgecolors="black", s=60,
        )
        zoom.scatter(
            def_pos[min_idx, 0], def_pos[min_idx, 2], def_pos[min_idx, 1],
            color="gold", edgecolors="black", s=60,
        )
        zoom.plot(
            [m[min_idx, 0], def_pos[min_idx, 0]],
            [m[min_idx, 2], def_pos[min_idx, 2]],
            [m[min_idx, 1], def_pos[min_idx, 1]],
            color="limegreen", ls=":", lw=2.0,
        )
        center = 0.5 * (m[min_idx] + def_pos[min_idx])
        half_range = max(50.0, 2.5 * float(min_dist))
        zoom.set_xlim(center[0] - half_range, center[0] + half_range)
        zoom.set_ylim(center[2] - half_range, center[2] + half_range)
        zoom.set_zlim(center[1] - half_range, center[1] + half_range)
        zoom.set_box_aspect((1.0, 1.0, 1.0))
        zoom.view_init(elev=22, azim=132)
        _decorate_trajectory_axes(
            zoom, f"M-{def_label} closest approach (stage2): {min_dist:.2f} m"
        )
        zoom.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_acceleration(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    meta: dict,
    out_path: Path,
) -> None:
    t, accel, t_switch = _stitch_stage23_accel(data, prefix)
    labels = (r"$a_x$", r"$a_y$", r"$a_z$")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    has_s3 = t_switch is not None and t.size > 0 and float(t[-1]) > float(t_switch) + 1e-9

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.1), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, accel[:, i], color=colors[i], lw=1.4, marker="o", ms=2.8)
        ax.axhline(0.0, color="0.45", lw=0.75)
        if t_switch is not None and has_s3:
            ax.axvline(t_switch, color="purple", ls="--", lw=1.0, alpha=0.85)
        ax.set_ylabel(f"{labels[i]} (m/s²)")
        ax.grid(True, alpha=0.32)
    axes[-1].set_xlabel("Time from stage-2 start (s)")
    scope = "stage2+stage3" if has_s3 else "stage2"
    note = "purple dashed = stage2→3" if has_s3 else "no stage3 traj (intercepted or absent)"
    fig.suptitle(
        f"Attacker inertial acceleration ({scope}) — {_meta_summary(meta)}\n"
        f"Markers ≈ 0.1 s samples; {note}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_actions(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    meta: dict,
    out_path: Path,
) -> None:
    t = data[f"{prefix}/t_step"]
    gamma = data[f"{prefix}/gamma"]
    w_t = data[f"{prefix}/wT"]
    w_e = data[f"{prefix}/wE"]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.1), sharex=True)
    axes[0].step(t, gamma, color="#9467bd", lw=1.5, where="post")
    axes[0].plot(t, gamma, color="#9467bd", ls="none", marker="o", ms=3.2)
    axes[0].set_ylabel(r"$\gamma$")
    axes[0].set_ylim(-3.2, 3.2)
    axes[0].axhline(-3.0, color="0.5", ls="--", lw=0.75)
    axes[0].axhline(3.0, color="0.5", ls="--", lw=0.75)

    axes[1].step(t, w_t, color="#2ca02c", lw=1.5, where="post")
    axes[1].plot(t, w_t, color="#2ca02c", ls="none", marker="o", ms=3.2)
    axes[1].set_ylabel(r"$w_T$")
    axes[2].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].step(t, w_e, color="#d62728", lw=1.5, where="post")
    axes[2].plot(t, w_e, color="#d62728", ls="none", marker="o", ms=3.2)
    axes[2].set_ylabel(r"$w_E$")
    axes[2].set_xlabel("Stage-2 time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.32, which="both")
    fig.suptitle(
        f"Policy action parameters (stage2 only) — {_meta_summary(meta)}\n"
        "Markers = control samples at dt = 0.1 s; step lines = zero-order hold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "figures").resolve()
    traj_path = run_dir / "selected_trajs.npz"
    meta_path = run_dir / "selected_meta.json"
    if not traj_path.is_file() or not meta_path.is_file():
        raise SystemExit(
            f"缺少落盘数据：需要 {traj_path.name} 与 {meta_path.name}，目录为 {run_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_list = json.loads(meta_path.read_text(encoding="utf-8"))
    written: list[Path] = []
    with np.load(traj_path) as data:
        for meta in meta_list:
            if meta.get("ep") is None:
                continue
            prefix = f"{meta['tier']}_ep{int(meta['ep']):03d}"
            required = (f"{prefix}/pos_M", f"{prefix}/a_M_inertial", f"{prefix}/gamma")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{prefix} 缺少数据字段：{missing}")

            stem = f"ep{int(meta['ep']):03d}_{meta['tier']}"
            trajectory_path = out_dir / f"{stem}_3d_trajectory.png"
            accel_path = out_dir / f"{stem}_inertial_acceleration_3x1.png"
            action_path = out_dir / f"{stem}_action_parameters_3x1.png"
            plot_trajectory(data, prefix, meta, trajectory_path)
            plot_acceleration(data, prefix, meta, accel_path)
            plot_actions(data, prefix, meta, action_path)
            written.extend((trajectory_path, accel_path, action_path))

    print(f"[plot] wrote {len(written)} figures to {out_dir}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
