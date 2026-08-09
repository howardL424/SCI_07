"""固定 100 初态 MC：test 池定序置换取子集，多模型共用同一清单。

与旧协议（有放回随机抽）不同：先对 samples_clean_test 有效行做
permutation(order_seed)，取前 n_ic 个 local_idx，无放回、可复现；
多组模型全部按同一清单、同一顺序评估。

用法示例：
  # 只生成并落盘 IC 清单（不跑仿真）
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe ^
    plot_table2_matlab/mc_fixed100/run_mc_fixed100.py --dump-ic-only

    # 单模型
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe ^
    plot_table2_matlab/mc_fixed100/run_mc_fixed100.py ^
    --model bctd3_best=path/to/best/best_model --prefer-td3bc

  # 多模型（约 10 组）：写 models.json 后一次跑完
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe ^
    plot_table2_matlab/mc_fixed100/run_mc_fixed100.py ^
    --models-file plot_table2_matlab/mc_fixed100/models.example.json ^
    --ic-pool plot_table2_matlab/mc_fixed100/results/ic_pool_n100_seed0.json

  # 烟雾：只跑清单前 2 局
  ... --smoke 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_GAIL = Path(__file__).resolve().parents[2]
_RL = _GAIL / "rl_td3"
sys.path.insert(0, str(_RL))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_TEST_FILE  # noqa: E402
from evaluate_td3 import print_stats, run_episode, save_mc_metrics  # noqa: E402
from ident_cfg import add_ident_args, matlab_cfg_from_args  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402
from td3_bc import TD3BC  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR / "results"
DEFAULT_IC_NAME = "ic_pool_n{n}_seed{seed}.json"


def build_ic_pool(
    n_pool: int,
    n_ic: int,
    order_seed: int,
) -> list[int]:
    """定序置换取前 n_ic 个 local_idx（与 FixedICEvalEnv 同一思路）。"""
    n = min(int(n_ic), int(n_pool))
    order = np.random.default_rng(int(order_seed)).permutation(int(n_pool))
    return [int(i) for i in order[:n]]


def ic_pool_path(out_dir: Path, n_ic: int, order_seed: int) -> Path:
    return out_dir / DEFAULT_IC_NAME.format(n=n_ic, seed=order_seed)


def save_ic_pool(
    path: Path,
    local_idx: list[int],
    *,
    samples: str,
    order_seed: int,
    row_idx: list[int] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": str(samples),
        "order_seed": int(order_seed),
        "n_ic": len(local_idx),
        "local_idx": [int(i) for i in local_idx],
        "row_idx": [int(i) for i in row_idx] if row_idx is not None else None,
        "note": (
            "local_idx = 过滤全零行后的库内下标；"
            "多模型 MC 须共用本文件，保证配对可比。"
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_ic_pool(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "local_idx" not in data or not data["local_idx"]:
        raise ValueError(f"IC 清单缺少 local_idx: {path}")
    return data


def parse_model_spec(spec: str) -> dict[str, Any]:
    """解析 --model name=path 或 --model path（自动用目录名作标签）。"""
    parts = spec.split("=", 1)
    name_token = parts[0].strip() if len(parts) == 2 else ""
    looks_like_name = (
        len(parts) == 2
        and name_token
        and ("/" not in name_token)
        and ("\\" not in name_token)
        and (not name_token.endswith(".zip"))
    )
    if looks_like_name:
        name, path = name_token, parts[1].strip()
    else:
        path = spec.strip()
        name = Path(path).name
        if name in ("best_model", "best_model.zip"):
            name = Path(path).parent.parent.name  # training_xxx
    return {"name": name, "model": path, "prefer_td3bc": None}


def load_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if args.models_file:
        raw = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("--models-file 须为 JSON 数组")
        for i, item in enumerate(raw):
            if isinstance(item, str):
                jobs.append(parse_model_spec(item))
            elif isinstance(item, dict):
                if "model" not in item:
                    raise ValueError(f"models[{i}] 缺少 model 字段")
                jobs.append(
                    {
                        "name": str(item.get("name") or f"m{i}"),
                        "model": str(item["model"]),
                        "prefer_td3bc": item.get("prefer_td3bc"),
                        "action_mapping": item.get("action_mapping"),
                        "obs_mode": item.get("obs_mode"),
                    }
                )
            else:
                raise ValueError(f"models[{i}] 类型不支持: {type(item)}")
    for spec in args.model or []:
        jobs.append(parse_model_spec(spec))
    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="固定子集 MC（permutation 取 n_ic，多模型共用清单）"
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="模型：标签=路径（如 bctd3=.../best_model）或仅路径；可重复",
    )
    p.add_argument(
        "--models-file",
        type=str,
        default=None,
        help="JSON 数组：[{name, model, prefer_td3bc?}, ...]",
    )
    p.add_argument("--samples", type=str, default=SAMPLES_TEST_FILE)
    p.add_argument(
        "--n-ic",
        type=int,
        default=100,
        help="固定初态个数（默认 100）",
    )
    p.add_argument(
        "--order-seed",
        type=int,
        default=0,
        help="定序置换种子（默认 0；与 --ic-pool 二选一）",
    )
    p.add_argument(
        "--ic-pool",
        type=str,
        default=None,
        help="已有 IC 清单 JSON；指定则忽略 --n-ic/--order-seed 的重新抽样",
    )
    p.add_argument(
        "--dump-ic-only",
        action="store_true",
        help="只生成/刷新 IC 清单后退出（不加载模型、不启 MATLAB 评估）",
    )
    p.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="只跑清单前 K 局（调试）；0=跑满清单",
    )
    p.add_argument("--seed", type=int, default=0, help="首轮 env.reset 的 RNG 种子")
    p.add_argument("--action-mapping", type=str, default=DEFAULT_ACTION_MAPPING)
    p.add_argument(
        "--obs-mode",
        choices=["scaled", "raw"],
        default="scaled",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT),
        help="结果目录（默认本目录 results/）",
    )
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument(
        "--prefer-td3bc",
        action="store_true",
        help="优先按 TD3BC 加载（也可在 models.json 里逐条写）",
    )
    p.add_argument(
        "--sigma-scale",
        type=float,
        # default=1.0,  # 原默认：1×；与 table2_noisy_ident / mc_noisy_300 原约定一致
        default=30.0,  # 2026-08-06：改为 30×（见 SESSION_SUMMARY_noise_scale_20260806.md）
        help="将全部 σ_* 统一乘此倍率（默认 30；与 table2_noisy_ident / mc_noisy_300 一致）",
    )
    add_ident_args(p)
    return p.parse_args()


def apply_sigma_scale(args: argparse.Namespace) -> None:
    """就地把 CLI 上的 σ_* 乘以 --sigma-scale。"""
    s = float(args.sigma_scale)
    if abs(s - 1.0) < 1e-15:
        return
    for key in (
        "sigma_q",
        "sigma_qdot",
        "sigma_theta",
        "sigma_psi",
        "sigma_r",
        "sigma_rdot",
    ):
        setattr(args, key, float(getattr(args, key)) * s)


def _normalize_model_path(model: str) -> str:
    path = model
    if path.endswith(".zip"):
        path = path[:-4]
    return path


def _load_model(model_path: str, env: EvasionEnv, prefer_td3bc: bool | None):
    path = _normalize_model_path(model_path)
    zip_path = path + ".zip"
    if not Path(zip_path).exists():
        raise FileNotFoundError(f"找不到模型: {zip_path}")
    use_bc = True if prefer_td3bc is None else bool(prefer_td3bc)
    if use_bc:
        try:
            return TD3BC.load(path, env=env)
        except Exception:
            if prefer_td3bc is True:
                raise
    return TD3.load(path, env=env)


def _read_pool_meta(samples_file: str) -> tuple[int, np.ndarray]:
    """轻量读 mat，过滤规则与 EvasionEnv 一致。返回 (n_pool, row_ids)。"""
    import scipy.io

    sc = scipy.io.loadmat(samples_file, squeeze_me=True)
    samples_all = np.asarray(sc["samples_clean"], dtype=np.float64)
    valid = np.linalg.norm(samples_all, axis=1) > 1e-12
    row_ids = np.flatnonzero(valid).astype(np.int64)
    return int(valid.sum()), row_ids


def resolve_ic_list(
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[int], Path, dict[str, Any]]:
    """返回 (local_idx 列表, 清单路径, meta)。不启动 MATLAB。"""
    if args.ic_pool:
        path = Path(args.ic_pool)
        meta = load_ic_pool(path)
        local_idx = [int(i) for i in meta["local_idx"]]
        return local_idx, path, meta

    n_pool, row_ids = _read_pool_meta(args.samples)
    local_idx = build_ic_pool(n_pool, args.n_ic, args.order_seed)
    row_idx = [int(row_ids[i]) for i in local_idx]
    path = ic_pool_path(out_dir, len(local_idx), args.order_seed)
    save_ic_pool(
        path,
        local_idx,
        samples=args.samples,
        order_seed=args.order_seed,
        row_idx=row_idx,
    )
    meta = {
        "samples": args.samples,
        "order_seed": args.order_seed,
        "n_ic": len(local_idx),
        "local_idx": local_idx,
        "row_idx": row_idx,
        "n_pool": n_pool,
    }
    return local_idx, path, meta


def run_one_model(
    job: dict[str, Any],
    local_idx: list[int],
    args: argparse.Namespace,
    out_dir: Path,
    matlab_cfg: dict,
) -> dict[str, Any]:
    name = job["name"]
    model_path = job["model"]
    action_mapping = job.get("action_mapping") or args.action_mapping
    obs_mode = job.get("obs_mode") or args.obs_mode
    prefer = job.get("prefer_td3bc")
    if prefer is None:
        prefer = True if args.prefer_td3bc else None

    print(f"\n===== {name} =====")
    print(f"model: {model_path}")
    print(f"obs_mode={obs_mode}  action_mapping={action_mapping}")

    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=action_mapping,
        obs_normalize=(obs_mode == "scaled"),
        matlab_cfg=dict(matlab_cfg),
    )
    # 核对清单不越界
    bad = [i for i in local_idx if i < 0 or i >= env.n_samples]
    if bad:
        env.close()
        raise IndexError(
            f"IC 清单含越界 local_idx（池大小={env.n_samples}）: {bad[:10]}"
        )

    model = _load_model(model_path, env, prefer)
    plan = local_idx[: args.smoke] if args.smoke > 0 else local_idx

    results = []
    n_skip = 0
    noise_seed0 = int(matlab_cfg.get("noise_seed", 0))
    t0 = datetime.now()
    for k, li in enumerate(plan):
        # 与 mc_noisy_300 对齐：每局 noise_seed = seed0 + local_idx
        cfg_i = dict(matlab_cfg)
        cfg_i["noise_seed"] = noise_seed0 + int(li)
        env.matlab_cfg = cfg_i
        ep = run_episode(
            env,
            model,
            seed=args.seed if k == 0 else None,
            local_idx=int(li),
        )
        if ep is None:
            n_skip += 1
            print(f"  [skip local_idx={li}] warm-start 失败")
            continue
        results.append(ep)
        done = len(results)
        if done % args.progress_every == 0 or k + 1 == len(plan):
            both = sum(1 for r in results if r["both_ok"])
            rT = ep["r_minT"]
            rT_s = f"{rT:.2f}" if np.isfinite(rT) else "nan"
            print(
                f"  [{done:3d}/{len(plan)}] both={100 * both / done:.1f}%  "
                f"local_idx={li}  rMD=[{ep['r_MD1_min']:.1f},{ep['r_MD2_min']:.1f}] "
                f"rT={rT_s}"
            )

    elapsed = (datetime.now() - t0).total_seconds()
    env.close()

    print_stats(f"{name} (fixed-IC)", results)
    if n_skip:
        print(f"[mc] warm-start 跳过 {n_skip}/{len(plan)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = out_dir / f"mc_{name}_N{len(results)}_{ts}.npz"
    if results:
        save_mc_metrics(results, npz_path)

    summary = {
        "name": name,
        "model": model_path,
        "n": len(results),
        "n_plan": len(plan),
        "n_skip": n_skip,
        "elapsed_s": elapsed,
        "rate_evade": float(np.mean([r["evade_ok"] for r in results])) if results else 0.0,
        "rate_strike": float(np.mean([r["strike_ok"] for r in results])) if results else 0.0,
        "rate_both": float(np.mean([r["both_ok"] for r in results])) if results else 0.0,
        "rate_intercepted": float(np.mean([r["intercepted"] for r in results]))
        if results
        else 0.0,
        "metrics_file": str(npz_path) if results else "",
        "obs_mode": obs_mode,
        "action_mapping": action_mapping,
        "coeff_mode": matlab_cfg.get("coeff_mode"),
        "sigma_scale": float(getattr(args, "sigma_scale", 30.0)),
        "noise_seed0": noise_seed0,
    }
    (out_dir / f"summary_{name}_{ts}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[mc] 已保存: {npz_path}")
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_idx, ic_path, meta = resolve_ic_list(args, out_dir)

    if args.dump_ic_only:
        print(f"[mc] 已写入 IC 清单: {ic_path}")
        print(
            f"     n_ic={meta['n_ic']}  order_seed={meta.get('order_seed')}  "
            f"n_pool={meta.get('n_pool', '?')}"
        )
        print(f"     local_idx 前 10: {local_idx[:10]}")
        return

    jobs = load_models(args)
    if not jobs:
        raise SystemExit(
            "请用 --model name=path 和/或 --models-file 指定至少 1 个模型；"
            "或用 --dump-ic-only 只生成清单。"
        )

    # 若用户传入外部 --ic-pool，把副本落到 out_dir 方便归档
    if args.ic_pool:
        archive = out_dir / Path(args.ic_pool).name
        if archive.resolve() != Path(args.ic_pool).resolve():
            save_ic_pool(
                archive,
                local_idx,
                samples=meta.get("samples", args.samples),
                order_seed=int(meta.get("order_seed", args.order_seed)),
                row_idx=meta.get("row_idx"),
            )
            ic_path = archive

    apply_sigma_scale(args)
    matlab_cfg = matlab_cfg_from_args(args, action_mapping=args.action_mapping)
    print(
        f"[mc] samples={args.samples}\n"
        f"     IC 清单={ic_path}  n={len(local_idx)}"
        f"{'  smoke=' + str(args.smoke) if args.smoke else ''}\n"
        f"     models={len(jobs)}  coeff_mode={matlab_cfg.get('coeff_mode')}\n"
        f"     sigma_scale={args.sigma_scale:g}  "
        f"σ_q={matlab_cfg['sigma_q']:.4g}  σ_qdot={matlab_cfg['sigma_qdot']:.4g}\n"
        f"     out={out_dir}"
    )
    print(f"     local_idx 前 10: {local_idx[:10]}")

    summaries = []
    for job in jobs:
        summaries.append(
            run_one_model(job, local_idx, args, out_dir, matlab_cfg)
        )

    print("\n==================== 固定-IC MC 汇总 ====================")
    print(f"{'名称':<28} {'N':>5} {'双规避':>10} {'打击':>10} {'联合':>10}")
    for s in summaries:
        print(
            f"{s['name']:<28} {s['n']:5d} "
            f"{100 * s['rate_evade']:6.1f}%   "
            f"{100 * s['rate_strike']:6.1f}%   "
            f"{100 * s['rate_both']:6.1f}%"
        )
    print("=========================================================")

    batch_path = out_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    batch_path.write_text(
        json.dumps(
            {
                "ic_pool": str(ic_path),
                "n_ic": len(local_idx),
                "models": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[mc] batch 汇总: {batch_path}")


if __name__ == "__main__":
    main()
