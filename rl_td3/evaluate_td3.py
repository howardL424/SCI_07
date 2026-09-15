"""
evaluate_td3.py — 加载训练好的 TD3/TD3BC 模型进行蒙特卡洛评估。

用法：
    # 默认：测完全部 samples_clean_test.mat 有效初态
    python evaluate_td3.py --model rl_td3/checkpoints/.../best/best_model

    # 旧协议：随机抽 N 轮（有放回）
    python evaluate_td3.py --model path/to/model --episodes 100 --seed 0

    # 0711/0715 等旧 checkpoint（raw 观测 + td3_box_log_v1）
    python evaluate_td3.py --model .../best/best_model --obs-mode raw --action-mapping td3_box_log_v1

默认初态池为 samples_clean_test.mat；--episodes 省略或 0 表示测全池。
终端输出默认 tee 到模型所在 training 目录：
    {training_dir}/monte_carlo_eval_{best|final|...}_seed{seed}_{时间戳}.log
逐轮指标（散点图用）同步写入同目录同名 .npz：
    r_MD1_min / r_MD2_min / r_minT （及 evade_ok / strike_ok / both_ok / row_idx / local_idx）
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

sys.path.insert(0, str(Path(__file__).parent))
from action_mapping import DEFAULT_ACTION_MAPPING
from evasion_env import EvasionEnv, KILL_R, SAMPLES_TEST_FILE, STRIKE_OK
from ident_cfg import IDENT_CLI_DEFAULTS, add_ident_args, matlab_cfg_from_args
from td3_bc import TD3BC


class _Tee:
    """同时写入多个流（终端 + 日志文件）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()


@contextmanager
def tee_terminal_log(log_path: Path):
    """将 stdout/stderr 同时写入 log_path，退出时恢复。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    try:
        yield log_path
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


def _normalize_model_path(model: str) -> Path:
    p = Path(model)
    if p.suffix == ".zip":
        p = p.with_suffix("")
    return p


def resolve_mc_log_path(
    model: str,
    seed: int,
    log_file: str | None = None,
) -> Path:
    """默认日志：{training_dir}/monte_carlo_eval_{tag}_seed{seed}_{ts}.log。

    training_dir / tag 推断：
      .../best/best_model  → training_dir=.../ , tag=best
      .../{name}.zip       → training_dir=.../ , tag=final
      .../{name}_N_steps   → training_dir=.../ , tag={name}_N_steps
    """
    if log_file:
        return Path(log_file)

    model_path = _normalize_model_path(model)
    stem = model_path.name
    if model_path.parent.name == "best" and stem == "best_model":
        training_dir = model_path.parent.parent
        tag = "best"
    else:
        training_dir = model_path.parent
        tag = "final" if not stem.endswith("_steps") else stem

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return training_dir / f"monte_carlo_eval_{tag}_seed{seed}_{ts}.log"


def resolve_mc_metrics_path(
    model: str,
    seed: int,
    metrics_file: str | None = None,
    log_file: str | None = None,
    log_path: Path | None = None,
) -> Path:
    """默认与日志同 stem 的 .npz；可用 --metrics-file 覆盖。"""
    if metrics_file:
        return Path(metrics_file)
    base = log_path if log_path is not None else resolve_mc_log_path(
        model, seed, log_file
    )
    return Path(base).with_suffix(".npz")


def save_mc_metrics(results: list[dict], path: Path) -> Path:
    """将每轮规避距离与打击精度落盘，供后续蒙特卡洛散点图使用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "r_MD1_min": np.asarray(
            [r["r_MD1_min"] for r in results], dtype=np.float64
        ),
        "r_MD2_min": np.asarray(
            [r["r_MD2_min"] for r in results], dtype=np.float64
        ),
        "r_minT": np.asarray([r["r_minT"] for r in results], dtype=np.float64),
        "evade_ok": np.asarray([r["evade_ok"] for r in results], dtype=bool),
        "strike_ok": np.asarray([r["strike_ok"] for r in results], dtype=bool),
        "both_ok": np.asarray([r["both_ok"] for r in results], dtype=bool),
        "intercepted": np.asarray(
            [r["intercepted"] for r in results], dtype=bool
        ),
        "row_idx": np.asarray([r["row_idx"] for r in results], dtype=np.int64),
        "local_idx": np.asarray([r["local_idx"] for r in results], dtype=np.int64),
    }
    optional_metrics = (
        "a_energy",
        "a_total_variation",
        "saturation_rate",
        "theta_causal_rmse",
        "theta_causal_tail_rmse",
        "theta_command_rmse",
        "theta_command_tail_rmse",
        "frozen_model_rmse",
        "frozen_model_tail_rmse",
        "a_cmd_rmse",
        "a_cmd_tail_rmse",
        "a_cmd_sat_rmse",
        "a_cmd_sat_tail_rmse",
    )
    for key in optional_metrics:
        if all(key in r for r in results):
            payload[key] = np.asarray([r[key] for r in results], dtype=np.float64)
    np.savez_compressed(path, **payload)
    print(
        f"[evaluate] 逐轮指标已写入 {path}\n"
        f"           keys: r_MD1_min, r_MD2_min, r_minT, "
        f"evade_ok, strike_ok, both_ok, intercepted, row_idx, local_idx"
    )
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TD3 制导律权重训练模型评估")
    p.add_argument("--model", type=str, required=True, help="模型路径（不含 .zip）")
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="随机抽 N 轮（有放回）；0=测完全部初态池（默认）",
    )
    p.add_argument(
        "--samples",
        type=str,
        default=SAMPLES_TEST_FILE,
        help="初态池 mat（默认 samples_clean_test.mat）",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="打印每步信息（调试用）")
    p.add_argument(
        "--obs-mode",
        choices=["scaled", "raw"],
        default="scaled",
        help="scaled=OBS_SCALE（当前默认）；raw=兼容 0711/0715 等旧模型",
    )
    p.add_argument(
        "--action-mapping",
        type=str,
        default=DEFAULT_ACTION_MAPPING,
        help="动作映射版本（旧模型用 td3_box_log_v1）",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "终端输出日志路径（默认写入模型所在 training 目录："
            "monte_carlo_eval_{tag}_seed{seed}_{时间戳}.log）"
        ),
    )
    p.add_argument("--no-log", action="store_true", help="不写入终端输出日志")
    p.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help=(
            "逐轮 r_MD1_min/r_MD2_min/r_minT 的 npz 路径"
            "（默认与日志同 stem 的 .npz）"
        ),
    )
    p.add_argument(
        "--no-metrics",
        action="store_true",
        help="不保存逐轮指标 npz",
    )
    add_ident_args(p)
    return p.parse_args()


def _metric_vector(value, size: int) -> np.ndarray:
    """将可选诊断量规整为固定长度；缺失时返回 NaN。"""
    if value is None:
        return np.full(size, np.nan, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < size:
        return np.full(size, np.nan, dtype=np.float64)
    return arr[:size].copy()


def _normalize_theta_error(value, *, r0: float, a0: float) -> np.ndarray:
    """按辨识诊断口径归一化11维参数误差。"""
    err = _metric_vector(value, 11)
    err[[0, 4, 8]] *= r0 / max(a0, 1e-12)
    err[[3, 7]] /= max(a0, 1e-12)
    return err


def _rows_rmse(rows: list[np.ndarray], *, tail: bool = False) -> float:
    """计算逐拍向量的整体RMSE；tail取后20%且至少5拍。"""
    if not rows:
        return float("nan")
    use = rows
    if tail:
        n_tail = min(len(rows), max(5, int(np.ceil(0.2 * len(rows)))))
        use = rows[-n_tail:]
    arr = np.stack(use)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.nanmean(arr**2)))


def run_episode(
    env: EvasionEnv,
    model: TD3,
    render: bool = False,
    seed: int | None = None,
    local_idx: int | None = None,
):
    """运行一个 episode，返回统计指标。

    - local_idx 非 None：定点测该初态（全量 test 模式）
    - 否则随机抽行；seed 非 None 时传入 env.reset(seed=...)（随机模式首轮）
    warm-start 失败时返回 None（仅定点模式；随机模式由 env 内部重试）。
    """
    try:
        if local_idx is not None:
            obs, info0 = env.reset(
                seed=seed, options={"local_idx": int(local_idx)}
            )
        else:
            obs, info0 = env.reset(seed=seed) if seed is not None else env.reset()
    except RuntimeError as e:
        if local_idx is not None and "warm-start" in str(e):
            return None
        raise

    terminated = False
    total_reward = 0.0
    r_MD1_min = np.inf
    r_MD2_min = np.inf
    r_minT = np.nan
    intercepted = False
    evade_ok = False
    strike_ok = False
    step = 0
    a_prev: np.ndarray | None = None
    a_energy = 0.0
    a_total_variation = 0.0
    saturated_steps = 0
    theta_causal_rows: list[np.ndarray] = []
    theta_command_rows: list[np.ndarray] = []
    frozen_rows: list[np.ndarray] = []
    a_cmd_rows: list[np.ndarray] = []
    a_cmd_sat_rows: list[np.ndarray] = []

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)

        total_reward += reward
        r_MD1_min = min(r_MD1_min, info["r_MD1"])
        r_MD2_min = min(r_MD2_min, info["r_MD2"])
        intercepted = intercepted or info["intercepted"]
        if np.isfinite(info.get("r_minT", np.nan)):
            r_minT = float(info["r_minT"])
            evade_ok = bool(info.get("evade_ok", False))
            strike_ok = bool(info.get("strike_ok", False))
        a_sat = np.asarray(info.get("a_M_sat", [np.nan, np.nan]), dtype=float)
        a_raw = np.asarray(info.get("a_M_raw", a_sat), dtype=float)
        if np.isfinite(a_sat).all():
            a_energy += 0.1 * float(np.dot(a_sat, a_sat))
            if a_prev is not None:
                a_total_variation += float(np.linalg.norm(a_sat - a_prev))
            a_prev = a_sat.copy()
            saturated_steps += int(
                np.isfinite(a_raw).all()
                and np.max(np.abs(a_raw - a_sat)) > 1e-9
            )
        ident_diag = info.get("ident_diag")
        if ident_diag is not None:
            r0 = float(
                env.matlab_cfg.get("rls_r0", IDENT_CLI_DEFAULTS["rls_r0"])
            )
            a0 = float(
                env.matlab_cfg.get("rls_a0", IDENT_CLI_DEFAULTS["rls_a0"])
            )
            theta_causal_rows.append(
                _normalize_theta_error(
                    ident_diag.get("theta_err_causal"), r0=r0, a0=a0
                )
            )
            theta_command_rows.append(
                _normalize_theta_error(
                    ident_diag.get("theta_err_command"), r0=r0, a0=a0
                )
            )
            frozen_rows.append(
                _metric_vector(ident_diag.get("frozen_model_residual"), 3)
            )
            a_cmd_rows.append(_metric_vector(ident_diag.get("a_cmd_err"), 2))
            a_cmd_sat_rows.append(
                _metric_vector(ident_diag.get("a_cmd_err_sat"), 2)
            )
        step += 1

        if render:
            ver = getattr(env, "action_mapping", None)
            kwargs = {"version": ver} if ver else {}
            gamma, wT, wE = EvasionEnv.denorm_action(action, **kwargs)
            print(
                f"  step={step:3d} | r_MD1={info['r_MD1']:7.1f}m "
                f"r_MD2={info['r_MD2']:7.1f}m r_MT={info['r_MT']:8.1f}m "
                f"| γ={gamma:5.2f} wT={wT:.1e} wE={wE:.1e} "
                f"| r={reward:7.3f}"
            )

    if not evade_ok:
        evade_ok = (r_MD1_min >= KILL_R) and (r_MD2_min >= KILL_R)
    if not strike_ok and np.isfinite(r_minT):
        strike_ok = r_minT <= STRIKE_OK

    return {
        "total_reward": total_reward,
        "r_MD1_min": r_MD1_min,
        "r_MD2_min": r_MD2_min,
        "r_minT": r_minT,
        "intercepted": intercepted,
        "evade_ok": evade_ok,
        "strike_ok": strike_ok,
        "both_ok": evade_ok and strike_ok,
        "a_energy": a_energy,
        "a_total_variation": a_total_variation,
        "saturation_rate": saturated_steps / max(step, 1),
        "theta_causal_rmse": _rows_rmse(theta_causal_rows),
        "theta_causal_tail_rmse": _rows_rmse(theta_causal_rows, tail=True),
        "theta_command_rmse": _rows_rmse(theta_command_rows),
        "theta_command_tail_rmse": _rows_rmse(theta_command_rows, tail=True),
        "frozen_model_rmse": _rows_rmse(frozen_rows),
        "frozen_model_tail_rmse": _rows_rmse(frozen_rows, tail=True),
        "a_cmd_rmse": _rows_rmse(a_cmd_rows),
        "a_cmd_tail_rmse": _rows_rmse(a_cmd_rows, tail=True),
        "a_cmd_sat_rmse": _rows_rmse(a_cmd_sat_rows),
        "a_cmd_sat_tail_rmse": _rows_rmse(a_cmd_sat_rows, tail=True),
        "steps": step,
        "row_idx": int(info0.get("row_idx", -1)),
        "local_idx": int(info0.get("local_idx", -1)),
    }


def print_stats(label: str, results: list[dict]):
    n = len(results)
    if n == 0:
        print(f"\n[evaluate] 无有效 episode 可统计（label={label}）")
        return

    evade_ok = [r for r in results if r["evade_ok"]]
    strike_ok = [r for r in results if r["strike_ok"]]
    both_ok = [r for r in results if r["both_ok"]]
    intercepted = [r for r in results if r["intercepted"]]

    d1_arr = np.array([r["r_MD1_min"] for r in results])
    d2_arr = np.array([r["r_MD2_min"] for r in results])
    rT_arr = np.array([r["r_minT"] for r in results], dtype=float)
    rw_arr = np.array([r["total_reward"] for r in results])
    finite_T = np.isfinite(rT_arr)

    print(f"\n{'='*60}")
    print(f"  评估结果：{label}  （{n} 轮）")
    print(f"{'='*60}")
    print(f"  双防均规避成功  (d1,d2≥{KILL_R:.0f}m)       : {len(evade_ok):3d}/{n} = {100*len(evade_ok)/n:.1f}%")
    print(f"  打击成功        (r_minT≤{STRIKE_OK:.0f}m)     : {len(strike_ok):3d}/{n} = {100*len(strike_ok)/n:.1f}%")
    print(f"  规避+打击 全部达标                  : {len(both_ok):3d}/{n} = {100*len(both_ok)/n:.1f}%")
    print(f"  被拦截          (r_MD<{KILL_R:.0f}m)          : {len(intercepted):3d}/{n} = {100*len(intercepted)/n:.1f}%")
    print(f"  最小脱靶距离 D1  mean±std (m)       : {d1_arr.mean():.1f} ± {d1_arr.std():.1f}")
    print(f"  最小脱靶距离 D2  mean±std (m)       : {d2_arr.mean():.1f} ± {d2_arr.std():.1f}")
    if finite_T.any():
        print(
            f"  终端脱靶 r_minT  mean±std (m)       : "
            f"{rT_arr[finite_T].mean():.2f} ± {rT_arr[finite_T].std():.2f}"
            f"  (n={finite_T.sum()})"
        )
    else:
        print("  终端脱靶 r_minT  : (无有效样本，可能全部被拦截)")
    print(f"  累计奖励         mean±std            : {rw_arr.mean():.2f} ± {rw_arr.std():.2f}")


def _run_eval(
    args: argparse.Namespace,
    metrics_path: Path | None = None,
) -> None:
    np.random.seed(args.seed)

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path_zip = model_path + ".zip"
    else:
        model_path_zip = model_path
        model_path = model_path[:-4]

    if not Path(model_path_zip).exists():
        print(f"[evaluate] 错误：找不到模型文件 {model_path_zip}")
        sys.exit(1)

    print(f"[evaluate] 加载模型：{model_path_zip}")
    print(f"[evaluate] obs_mode={args.obs_mode}  action_mapping={args.action_mapping}")
    print(f"[evaluate] 初态池：{args.samples}")
    matlab_cfg = matlab_cfg_from_args(args)
    print(
        f"[evaluate] coeff_mode={matlab_cfg['coeff_mode']}  "
        f"ident_apply_cmd={matlab_cfg['ident_apply_cmd']}  "
        f"rls_mu={matlab_cfg['rls_mu']}  noise_seed={matlab_cfg['noise_seed']}"
    )
    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        obs_normalize=(args.obs_mode == "scaled"),
        matlab_cfg=matlab_cfg,
    )
    try:
        model = TD3BC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    sweep_all = args.episodes <= 0
    if sweep_all:
        n_plan = env.n_samples
        mode_desc = f"全量 test-IC（{n_plan} 个有效初态）"
    else:
        n_plan = args.episodes
        mode_desc = f"随机 {n_plan} 轮（seed={args.seed}）"
        print(f"[evaluate] 固定种子 seed={args.seed}（首轮 reset 注入 env.np_random）")

    print(f"\n[evaluate] 开始评估：{mode_desc}…")
    results = []
    n_skip = 0
    for ep in range(n_plan):
        if sweep_all:
            ep_seed = args.seed if ep == 0 else None
            ep_result = run_episode(
                env,
                model,
                render=(args.render and ep == 0),
                seed=ep_seed,
                local_idx=ep,
            )
            if ep_result is None:
                n_skip += 1
                print(f"  [skip local_idx={ep}] warm-start 无有效规避段")
                continue
        else:
            ep_seed = args.seed if ep == 0 else None
            ep_result = run_episode(
                env, model, render=(args.render and ep == 0), seed=ep_seed
            )

        results.append(ep_result)
        done_n = len(results)
        if done_n % 10 == 0 or ep + 1 == n_plan:
            done_so_far = [r for r in results if r["both_ok"]]
            rT = ep_result["r_minT"]
            rT_str = f"{rT:.2f}m" if np.isfinite(rT) else "nan"
            print(
                f"  [done {done_n:3d} | scanned {ep+1:3d}/{n_plan}] "
                f"双防+打击成功率 {100*len(done_so_far)/done_n:.1f}%  "
                f"r_MD1_min={ep_result['r_MD1_min']:.1f}m  "
                f"r_minT={rT_str}"
            )

    if sweep_all and n_skip:
        print(f"[evaluate] warm-start 跳过 {n_skip}/{n_plan} 个初态")

    label = "TD3 (全量 test-IC)" if sweep_all else "TD3 (随机 MC)"
    print_stats(label, results)

    if metrics_path is not None and results:
        save_mc_metrics(results, metrics_path)
    elif metrics_path is not None and not results:
        print("[evaluate] 无有效 episode，跳过指标落盘")

    # 动作分布探针：取池内前 min(20,n) 个有效初态（定点，可复现）
    n_probe = min(20, env.n_samples)
    print(f"\n[evaluate] 探针前 {n_probe} 个初态，输出策略网络动作分布…")
    gammas, wTs, wEs = [], [], []
    for i in range(n_probe):
        try:
            obs, _ = env.reset(seed=args.seed if i == 0 else None, options={"local_idx": i})
        except RuntimeError:
            print(f"    [skip local_idx={i}] warm-start 失败")
            continue
        action, _ = model.predict(obs, deterministic=True)
        ver = getattr(env, "action_mapping", None)
        kwargs = {"version": ver} if ver else {}
        gamma, wT, wE = EvasionEnv.denorm_action(action, **kwargs)
        gammas.append(gamma)
        wTs.append(wT)
        wEs.append(wE)
        print(f"    γ={gamma:6.3f}  wT={wT:.2e}  wE={wE:.2e}")

    if gammas:
        print(f"\n  γ  均值={np.mean(gammas):.3f}  范围=[{np.min(gammas):.3f}, {np.max(gammas):.3f}]")
        print(f"  wT 均值={np.mean(wTs):.2e}  范围=[{np.min(wTs):.2e}, {np.max(wTs):.2e}]")
        print(f"  wE 均值={np.mean(wEs):.2e}  范围=[{np.min(wEs):.2e}, {np.max(wEs):.2e}]")

    env.close()


def main():
    args = parse_args()
    log_path = resolve_mc_log_path(args.model, args.seed, args.log_file)
    metrics_path = None
    if not args.no_metrics:
        metrics_path = resolve_mc_metrics_path(
            args.model,
            args.seed,
            metrics_file=args.metrics_file,
            log_path=log_path,
        )

    if args.no_log:
        _run_eval(args, metrics_path=metrics_path)
        return

    with tee_terminal_log(log_path):
        print(f"[evaluate] 终端输出同时写入 {log_path}")
        if metrics_path is not None:
            print(f"[evaluate] 逐轮指标将写入 {metrics_path}")
        _run_eval(args, metrics_path=metrics_path)


if __name__ == "__main__":
    main()

