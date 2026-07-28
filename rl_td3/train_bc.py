"""
train_bc.py — 独立纯 Actor BC 诊断入口（无 Critic / 无 TD3）。

流程：
  1. 加载专家 transition + 轨迹均衡采样（train/val 按轨划分）
  2. 仅监督 Actor：MSE(π(s), a_E)
  3. 记录 train/val MSE、逐动作 MAE、动作贴边率
  4. 可选闭环：BC 策略 MC + 同一派生环境专家动作回放对照

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/train_bc.py --smoke
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/train_bc.py `
    --sample-plan rl_td3/data/expert_set_99k.npz `
    --save-dir rl_td3/checkpoints_bc/bc_diag_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from action_mapping import (  # noqa: E402
    ACTION_MAPPING_RATIO,
    ACTION_MAPPING_TD3_BOX,
    DEFAULT_ACTION_MAPPING,
    action_names,
    old_box_feasible_from_ratio_action,
)
from bc_actor import (  # noqa: E402
    BOUNDARY_THRESH,
    BcActor,
    compute_bc_metrics,
    evaluate_bc,
    save_bc_checkpoint,
    train_bc_epoch,
)
from expert_sampling import (  # noqa: E402
    bc_arrays_from_plan,
    build_expert_sample_plan,
    load_ok_modes_from_expert,
    load_sample_plan,
)
from load_expert_buffer import (  # noqa: E402
    DATA_DIR,
    DEFAULT_ALIGNED_PATH,
    DEFAULT_EXPERT_PATH,
    DEFAULT_TERMINAL_CACHE,
    _load_terminal_cache,
    apply_action_mapping,
    load_transition_data,
)
from obs_norm import normalize_obs  # noqa: E402
from reward_projection import DEFAULT_REWARD_CFG  # noqa: E402


LOG_DIR = Path(__file__).parent / "logs"

DEFAULTS = dict(
    epochs=80,
    batch_size=256,
    lr=1e-3,
    budget=99_000,
    net_arch=(256, 256),
    val_ratio=0.1,
    boundary_thresh=BOUNDARY_THRESH,
    patience=15,
)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="纯 Actor BC 诊断")
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument("--aligned", type=str, default=DEFAULT_ALIGNED_PATH)
    p.add_argument("--terminal-cache", type=str, default=DEFAULT_TERMINAL_CACHE)
    p.add_argument(
        "--sample-plan",
        type=str,
        default=None,
        help="已有采样计划 npz（默认重建或用 expert_set_99k.npz）",
    )
    p.add_argument("--budget", type=int, default=DEFAULTS["budget"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=DEFAULTS["val_ratio"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument(
        "--actor-include-intercepted",
        action="store_true",
        help="Actor 标签含拦截轨迹（等价于 --actor-eligible all）",
    )
    p.add_argument(
        "--actor-eligible",
        type=str,
        default="both_ok",
        choices=["both_ok", "non_intercepted", "all"],
        help="Actor BC 轨迹资格（默认 both_ok）",
    )
    p.add_argument("--with-ok-mode", action="store_true")
    p.add_argument(
        "--save-dir",
        type=str,
        default="rl_td3/checkpoints_bc",
    )
    p.add_argument("--name", type=str, default="bc_actor")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--closed-loop",
        action="store_true",
        help="训练后做闭环：MC(samples_clean) + 专家动作回放对照",
    )
    p.add_argument("--eval-episodes", type=int, default=50, help="MC 轮数")
    p.add_argument(
        "--replay-traj",
        type=int,
        default=40,
        help="专家动作回放对照的 val 轨条数（未拦截）",
    )
    p.add_argument(
        "--action-mapping",
        type=str,
        default=DEFAULT_ACTION_MAPPING,
        choices=[ACTION_MAPPING_RATIO, ACTION_MAPPING_TD3_BOX],
        help="动作映射版本（默认 ratio_log_box_v1；对照可用 td3_box_log_v1）",
    )
    p.add_argument("--smoke", action="store_true", help="冒烟：少 epoch、小预算")
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 5)
        args.budget = min(args.budget, 4000)
        args.save_dir = str(Path(args.save_dir) / "smoke")
        if args.name in ("bc_actor", "bc_ratio"):
            args.name = "bc_smoke"
        args.eval_episodes = min(args.eval_episodes, 5)
        args.replay_traj = min(args.replay_traj, 5)
        args.patience = min(args.patience, 3)
    if args.sample_seed is None:
        args.sample_seed = args.seed
    return args


def _metrics_row(label: str, m) -> str:
    per = " ".join(f"{k}={v:.4f}" for k, v in m.mae_per_action.items())
    br = " ".join(f"{k}={v*100:.1f}%" for k, v in m.boundary_rate_per_action.items())
    return (
        f"  [{label}] n={m.n}  MSE={m.mse:.6f}  MAE={m.mae:.6f}  "
        f"MAE/act[{per}]  贴边(any)={m.boundary_rate*100:.1f}%  贴边/维[{br}]"
    )


def prepare_bc_data(args: argparse.Namespace) -> dict[str, Any]:
    print(f"[train_bc] 加载 transition：expert={args.expert}")
    data = load_transition_data(args.expert, aligned_path=args.aligned)
    src_map = str(data.get("action_mapping_version", ACTION_MAPPING_TD3_BOX))
    data = apply_action_mapping(data, args.action_mapping, source_version=src_map)
    print(
        f"[train_bc] 动作映射：{data.get('actions_source_version', src_map)} "
        f"→ {data['action_mapping_version']}"
    )
    if data["action_mapping_version"] == ACTION_MAPPING_RATIO:
        feas = old_box_feasible_from_ratio_action(data["actions"])
        print(
            f"[train_bc] 比值→旧权重盒可行域比例："
            f"{float(np.mean(feas))*100:.2f}%（理论应≈100%）"
        )
    cache = _load_terminal_cache(
        args.terminal_cache, args.expert, require_path_match=True, verbose=True
    )
    states_n = normalize_obs(data["states"]).astype(np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    names = action_names(args.action_mapping)

    plan_path = args.sample_plan
    if plan_path is None:
        default_plan = DATA_DIR / "expert_set_99k.npz"
        if default_plan.exists() and not args.smoke:
            plan_path = str(default_plan)
            print(f"[train_bc] 使用已有采样计划：{plan_path}")

    if plan_path and Path(plan_path).exists() and not args.smoke:
        plan = load_sample_plan(plan_path)
        # 旧计划可能缺新字段：用 qualities 补全专家基线
        if "expert_both_ok_rate_val" not in plan.summary and plan.qualities:
            val_q = [plan.qualities[int(t)] for t in plan.val_traj_ids]
            val_non = [q for q in val_q if not q.intercepted]
            plan.summary["expert_both_ok_rate_val"] = (
                float(sum(1 for q in val_q if q.both_ok) / len(val_q)) if val_q else 0.0
            )
            plan.summary["expert_both_ok_rate_val_non_intercept"] = (
                float(sum(1 for q in val_non if q.both_ok) / len(val_non)) if val_non else 0.0
            )
        if len(plan.val_actor_step_indices) == 0:
            raise RuntimeError("采样计划 val_actor 为空，请重建 expert_set 或去掉 --sample-plan")
        print(
            f"[train_bc] 已载入计划：train_actor={len(plan.actor_step_indices)} "
            f"val_actor={len(plan.val_actor_step_indices)} "
            f"val_traj={len(plan.val_traj_ids)}"
        )
    else:
        ok_modes = None
        if args.with_ok_mode:
            print("[train_bc] 加载 ok_mode…")
            ok_modes = load_ok_modes_from_expert(args.expert, data["source_idx"])
        plan = build_expert_sample_plan(
            data,
            args.budget,
            seed=args.sample_seed,
            val_ratio=args.val_ratio,
            actor_exclude_intercepted=not args.actor_include_intercepted,
            actor_eligible=(
                "all" if args.actor_include_intercepted else args.actor_eligible
            ),
            terminal_cache=cache,
            ok_modes=ok_modes,
            kill_radius=float(DEFAULT_REWARD_CFG.kill_radius),
            strike_ok=float(DEFAULT_REWARD_CFG.strike_ok),
        )
        out_plan = Path(args.save_dir) / f"bc_sample_plan_seed{args.sample_seed}.npz"
        plan.save(out_plan)
        print(f"[train_bc] 新建采样计划已保存：{out_plan}")

    s_tr, a_tr, s_va, a_va = bc_arrays_from_plan(states_n, actions, plan)
    print(
        f"[train_bc] BC 数据：train={len(s_tr)}  val={len(s_va)}  "
        f"expert_both_ok_val={plan.summary.get('expert_both_ok_rate_val', float('nan')):.3f}  "
        f"non_int={plan.summary.get('expert_both_ok_rate_val_non_intercept', float('nan')):.3f}"
    )
    if len(s_tr) == 0:
        raise RuntimeError("Actor BC 训练集为空（检查拦截过滤 / 采样预算）")
    if len(s_va) == 0:
        raise RuntimeError("Actor BC 验证集为空（检查 val_ratio / 拦截过滤）")

    return {
        "data": data,
        "plan": plan,
        "s_tr": s_tr,
        "a_tr": a_tr,
        "s_va": s_va,
        "a_va": a_va,
        "expert_actions_all": actions,
        "states_norm_all": states_n,
        "action_mapping": args.action_mapping,
        "action_names": names,
    }


def run_training(args: argparse.Namespace, pack: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    device = torch.device(args.device)
    actor = BcActor(obs_dim=13, act_dim=3, net_arch=DEFAULTS["net_arch"]).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    best_path = save_dir / f"{args.name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_bc_epoch(
            actor,
            opt,
            pack["s_tr"],
            pack["a_tr"],
            batch_size=args.batch,
            device=device,
            rng=rng,
        )
        m_tr = evaluate_bc(
            actor, pack["s_tr"], pack["a_tr"], device=device,
            action_names=pack["action_names"],
        )
        m_va = evaluate_bc(
            actor, pack["s_va"], pack["a_va"], device=device,
            action_names=pack["action_names"],
        )
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train": m_tr.to_dict(),
            "val": m_va.to_dict(),
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}/{args.epochs}] loss={tr_loss:.6f}\n"
            f"{_metrics_row('train', m_tr)}\n{_metrics_row('val  ', m_va)}"
        )

        if m_va.mse + 1e-12 < best_val:
            best_val = m_va.mse
            best_epoch = epoch
            bad = 0
            save_bc_checkpoint(
                best_path,
                actor,
                metrics={"train": m_tr.to_dict(), "val": m_va.to_dict()},
                config={
                    "seed": args.seed,
                    "sample_seed": args.sample_seed,
                    "lr": args.lr,
                    "batch": args.batch,
                    "budget": args.budget,
                    "actor_exclude_intercepted": not args.actor_include_intercepted,
                    "actor_eligible": (
                        "all" if args.actor_include_intercepted else args.actor_eligible
                    ),
                    "expert": args.expert,
                    "action_mapping": args.action_mapping,
                },
                epoch=epoch,
            )
            print(f"  → 新最佳 val MSE={best_val:.6f}，已保存 {best_path}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train_bc] early stop @ epoch {epoch}（patience={args.patience}）")
                break

    final_path = save_dir / f"{args.name}_final.pt"
    save_bc_checkpoint(
        final_path,
        actor,
        metrics=history[-1] if history else {},
        config={"seed": args.seed, "action_mapping": args.action_mapping},
        epoch=history[-1]["epoch"] if history else 0,
    )

    # 专家动作自身贴边率（对照）
    names = pack["action_names"]
    expert_br = {
        names[i]: float(np.mean(np.abs(pack["a_va"][:, i]) >= BOUNDARY_THRESH))
        for i in range(3)
    }
    feas_rate = None
    if args.action_mapping == ACTION_MAPPING_RATIO:
        feas_rate = float(np.mean(old_box_feasible_from_ratio_action(pack["a_va"])))
    report = {
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "best_ckpt": str(best_path),
        "final_ckpt": str(final_path),
        "action_mapping": args.action_mapping,
        "action_names": list(names),
        "history": history,
        "plan_summary": pack["plan"].summary,
        "expert_val_boundary_rate_per_action": expert_br,
        "old_box_feasible_rate_val": feas_rate,
        "judgment": _judgment_gate(
            best_val,
            history[best_epoch - 1]["val"] if best_epoch > 0 else None,
            pack["plan"].summary,
            action_mapping=args.action_mapping,
        ),
    }
    return report


def _judgment_gate(
    best_val_mse: float,
    best_val_metrics: Optional[dict],
    plan_summary: dict,
    *,
    action_mapping: str = DEFAULT_ACTION_MAPPING,
) -> dict[str, Any]:
    """
    软判断门（报告用，不自动进入 TD3）：
      - val MSE 持续偏高 / 贴边远超专家 → 先做状态/动作可辨识性消融
      - 误差可接受且不坍塌 → 可进入 TD3 精调
      - 比值坐标下勿仅凭 MSE 下降下结论；关注 gamma_n MAE 与闭环
    """
    notes: list[str] = []
    # 归一化动作空间 [-1,1]，MSE~0.05 量级仍粗；<0.02 较可用；>0.08 偏高
    if best_val_mse > 0.08:
        verdict = "fail_high_val_error"
        notes.append("验证 MSE 偏高：先做状态/动作可辨识性消融，不要直接调 TD3/奖励/GAIL。")
    elif best_val_mse > 0.03:
        verdict = "marginal"
        notes.append("验证 MSE 中等：建议结合闭环联合成功率再决定；必要时做短程消融。")
    else:
        verdict = "pass_offline"
        notes.append("离线验证误差可接受；若闭环接近专家且动作不坍塌，可进入 TD3 精调。")

    if best_val_metrics:
        br = float(best_val_metrics.get("boundary_rate", 0.0))
        if br > 0.85:
            notes.append(
                f"策略贴边率(any)={br*100:.1f}% 很高，需对照专家贴边率；"
                "若远超专家则视为坍塌风险。"
            )
            if verdict == "pass_offline":
                verdict = "marginal"
        mae_per = best_val_metrics.get("mae_per_action") or {}
        gamma_mae = mae_per.get("gamma_n")
        if gamma_mae is not None and float(gamma_mae) > 0.45:
            notes.append(
                f"gamma_n MAE={float(gamma_mae):.3f} 仍高（比值坐标未改变 γ）；"
                "若闭环无实质改善，进入状态空间消融。"
            )
            if verdict == "pass_offline":
                verdict = "marginal"

    notes.append(f"action_mapping={action_mapping}")
    notes.append(
        "专家标签基线（val 轨 both_ok）："
        f"{plan_summary.get('expert_both_ok_rate_val', float('nan')):.3f}；"
        "未拦截子集："
        f"{plan_summary.get('expert_both_ok_rate_val_non_intercept', float('nan')):.3f}"
    )
    return {"verdict": verdict, "notes": notes}


def _summarize_episode_results(results: list[dict]) -> dict[str, float]:
    n = max(len(results), 1)
    return {
        "n": float(len(results)),
        "both_ok_rate": float(sum(1 for r in results if r.get("both_ok")) / n),
        "evade_ok_rate": float(sum(1 for r in results if r.get("evade_ok")) / n),
        "strike_ok_rate": float(sum(1 for r in results if r.get("strike_ok")) / n),
        "intercept_rate": float(sum(1 for r in results if r.get("intercepted")) / n),
        "mean_reward": float(np.mean([r.get("total_reward", 0.0) for r in results]))
        if results
        else 0.0,
    }


def run_closed_loop(
    args: argparse.Namespace,
    pack: dict[str, Any],
    ckpt_path: Path,
) -> dict[str, Any]:
    from bc_actor import load_bc_checkpoint, predict_actions
    from evasion_env import EvasionEnv, KILL_R, STRIKE_OK
    from evaluate_td3 import print_stats, run_episode

    device = torch.device(args.device)
    actor, _ = load_bc_checkpoint(ckpt_path, device=device)

    class _ActorPolicy:
        """最小 predict 接口，兼容 evaluate_td3.run_episode。"""

        def predict(self, obs, deterministic=True):  # noqa: ARG002
            obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            a = predict_actions(actor, obs, device=device)[0]
            return a, None

    print("[train_bc] 闭环 MC（samples_clean）…")
    amap = getattr(args, "action_mapping", pack.get("action_mapping", DEFAULT_ACTION_MAPPING))
    env = EvasionEnv(action_mapping=amap)
    policy = _ActorPolicy()
    mc_results = []
    for i in range(args.eval_episodes):
        seed = args.seed if i == 0 else None
        mc_results.append(run_episode(env, policy, seed=seed))
    print_stats("BC Actor MC", mc_results)
    mc_sum = _summarize_episode_results(mc_results)

    print("[train_bc] 专家动作回放对照（val 未拦截轨，同一派生环境）…")
    plan = pack["plan"]
    data = pack["data"]
    intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()
    val_ids = [
        int(t)
        for t in plan.val_traj_ids
        if not intercepted[int(t)]
    ][: args.replay_traj]

    expert_results = []
    bc_on_expert_ic = []
    for tid in val_ids:
        payload = _load_traj_rollout_payload(args.expert, args.aligned, data, tid)
        if payload is None:
            continue
        # 专家动作开环回放
        expert_results.append(
            _rollout_from_abs(env, payload, action_fn=lambda _o, j: payload["actions"][j])
        )
        # 同一初态，BC 闭环
        bc_on_expert_ic.append(
            _rollout_from_abs(
                env,
                payload,
                action_fn=lambda o, _j: predict_actions(
                    actor, np.asarray(o, dtype=np.float32).reshape(1, -1), device=device
                )[0],
            )
        )

    env.close()
    print_stats("Expert action replay (val IC)", expert_results)
    print_stats("BC on expert IC", bc_on_expert_ic)
    return {
        "mc_samples_clean": mc_sum,
        "expert_replay": _summarize_episode_results(expert_results),
        "bc_on_expert_ic": _summarize_episode_results(bc_on_expert_ic),
        "n_replay_traj": len(expert_results),
        "kill_r": KILL_R,
        "strike_ok": STRIKE_OK,
    }


def _load_traj_rollout_payload(
    expert_path: str,
    aligned_path: str,  # noqa: ARG001  保留签名，便于日后从 aligned 回退
    data: dict,
    traj_id: int,
) -> Optional[dict[str, Any]]:
    """取轨迹首绝对态 + 全程 action_norm + Tev。"""
    del aligned_path
    offsets = np.concatenate([[0], np.cumsum(np.asarray(data["n_steps"], dtype=np.int64))])
    start, end = int(offsets[traj_id]), int(offsets[traj_id + 1])
    actions = np.asarray(data["actions"][start:end], dtype=np.float32)
    Tev = float(data["t_end"][traj_id])
    if Tev <= 0:
        Tev = float(end - start) * 0.1

    init_abs = _read_init_state_abs(expert_path, data, traj_id)
    if init_abs is None:
        # 回退：无法从库读绝对态时跳过该轨
        return None
    return {
        "state_abs0": init_abs,
        "Tev": Tev,
        "actions": actions,
        "traj_id": traj_id,
        "source_idx": int(data["source_idx"][traj_id]),
    }


_INIT_ABS_CACHE: dict[str, dict[int, np.ndarray]] = {}


def _read_init_state_abs(expert_path: str, data: dict, traj_id: int) -> Optional[np.ndarray]:
    """从 expert_lib 的 transitions.state_abs 读首步绝对态（按库路径缓存）。"""
    from load_expert_buffer import (
        _as_NxD,
        _get_transition_field,
        _iter_trajectories,
        _load_results,
    )

    key = str(Path(expert_path).resolve())
    if key not in _INIT_ABS_CACHE:
        results = _load_results(expert_path)
        trajs = _iter_trajectories(results)
        cache: dict[int, np.ndarray] = {}
        for i, traj in enumerate(trajs):
            sabs = _get_transition_field(traj, "state_abs")
            if sabs is None:
                continue
            sabs = _as_NxD(sabs, 18)
            if len(sabs) == 0:
                continue
            cache[i + 1] = sabs[0].astype(np.float64).ravel()  # source_idx 1-based
        _INIT_ABS_CACHE[key] = cache
        print(f"[train_bc] 已缓存 {len(cache)} 条轨迹的 init state_abs")

    source_idx = int(data["source_idx"][traj_id])
    return _INIT_ABS_CACHE[key].get(source_idx)


def _rollout_from_abs(
    env,
    payload: dict,
    action_fn: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, Any]:
    obs = env.reset_from_abs(payload["state_abs0"], Tev=payload["Tev"])
    total_reward = 0.0
    intercepted = False
    r_MD1_min = np.inf
    r_MD2_min = np.inf
    r_minT = np.nan
    evade_ok = False
    strike_ok = False
    n_act = len(payload["actions"])
    step = 0
    terminated = False
    while not terminated and step < n_act + 5:
        action = np.asarray(action_fn(obs, min(step, n_act - 1)), dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        r_MD1_min = min(r_MD1_min, info["r_MD1"])
        r_MD2_min = min(r_MD2_min, info["r_MD2"])
        intercepted = intercepted or info["intercepted"]
        if np.isfinite(info.get("r_minT", np.nan)):
            r_minT = float(info["r_minT"])
            evade_ok = bool(info.get("evade_ok", False))
            strike_ok = bool(info.get("strike_ok", False))
        step += 1

    from evasion_env import KILL_R, STRIKE_OK

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
        "steps": step,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        print("=" * 60)
        print("  纯 Actor BC 诊断")
        print(f"  seed={args.seed}  sample_seed={args.sample_seed}")
        print(f"  action_mapping={args.action_mapping}")
        print(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch}")
        print(f"  device={args.device}  save_dir={save_dir}")
        print("=" * 60)
        pack = prepare_bc_data(args)
        report = run_training(args, pack, save_dir)
        if args.closed_loop:
            cl = run_closed_loop(args, pack, Path(report["best_ckpt"]))
            report["closed_loop"] = cl
            j = report["judgment"]
            bc_both = cl["bc_on_expert_ic"].get("both_ok_rate", 0.0)
            ex_both = cl["expert_replay"].get("both_ok_rate", 0.0)
            if ex_both > 0 and bc_both < 0.5 * ex_both:
                j["verdict"] = "fail_closed_loop"
                j["notes"].append(
                    f"闭环联合成功率 {bc_both:.2%} 明显低于专家回放 {ex_both:.2%}，"
                    "先做状态/动作消融，不要直接进 TD3。"
                )
            elif j["verdict"] == "pass_offline" and bc_both >= 0.8 * max(ex_both, 1e-6):
                j["verdict"] = "pass_ready_for_td3"
                j["notes"].append("闭环接近专家回放，可通过判断门进入 TD3 精调。")

        report_path = save_dir / f"{args.name}_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[train_bc] 报告已写入 {report_path}")
        print(f"[train_bc] 判断门：{report['judgment']['verdict']}")
        for note in report["judgment"]["notes"]:
            print(f"  - {note}")

    if args.no_log:
        _run()
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{args.name}_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 简易 tee
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
            self.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

        def isatty(self):
            return getattr(self.streams[0], "isatty", lambda: False)()

    with log_path.open("w", encoding="utf-8") as lf:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, lf)  # type: ignore[assignment]
        sys.stderr = _Tee(old_err, lf)  # type: ignore[assignment]
        try:
            print(f"[train_bc] 终端输出同时写入 {log_path}")
            _run()
        finally:
            sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    main()
