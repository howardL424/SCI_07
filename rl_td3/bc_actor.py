"""
bc_actor.py — 纯 Actor 行为克隆（与 SB3 TD3 MlpPolicy Actor 同构）。

结构对齐 SB3：
  mu = Sequential(Linear, ReLU, Linear, ReLU, Linear, Tanh)
默认 net_arch=[256, 256]，便于后续把权重拷入 TD3BC.actor。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from obs_norm import (
    OBS_DIM_16,
    OBS_NORM_VERSION,
    OBS_SCHEMA_REL13_QMAG16,
    OBS_SCHEMA_VERSION,
    OBS16_NORM_VERSION,
)


ACTION_NAMES = ("gamma_n", "wT_n", "wE_n")
BOUNDARY_THRESH = 0.99


class BcActor(nn.Module):
    """确定性 Actor：obs → tanh 动作 ∈ [-1,1]^act_dim。"""

    def __init__(
        self,
        obs_dim: int = 13,
        act_dim: int = 3,
        net_arch: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.net_arch = tuple(int(h) for h in net_arch)

        layers: list[nn.Module] = []
        last = self.obs_dim
        for h in self.net_arch:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, self.act_dim))
        layers.append(nn.Tanh())
        self.mu = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(obs)

    def sb3_state_dict(self) -> dict[str, torch.Tensor]:
        """键名与 SB3 TD3 Actor.mu.* 对齐（不含 features_extractor）。"""
        return {f"mu.{k}" if not k.startswith("mu.") else k: v for k, v in self.state_dict().items()}


@dataclass
class BcMetrics:
    mse: float
    mae: float
    mae_per_action: dict[str, float]
    boundary_rate: float
    boundary_rate_per_action: dict[str, float]
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_bc_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    boundary_thresh: float = BOUNDARY_THRESH,
    action_names: Optional[Sequence[str]] = None,
) -> BcMetrics:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape 不一致: {pred.shape} vs {target.shape}")
    if pred.ndim != 2 or pred.shape[1] < 1:
        raise ValueError("pred/target 须为 [N, act_dim]")

    err = pred - target
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    act_dim = pred.shape[1]
    if action_names is None:
        names = list(ACTION_NAMES[:act_dim]) + [
            f"a{i}" for i in range(len(ACTION_NAMES), act_dim)
        ]
    else:
        names = list(action_names[:act_dim]) + [
            f"a{i}" for i in range(len(action_names), act_dim)
        ]
    mae_per = {names[i]: float(np.mean(np.abs(err[:, i]))) for i in range(act_dim)}
    bound = np.abs(pred) >= boundary_thresh
    br_per = {names[i]: float(bound[:, i].mean()) for i in range(act_dim)}
    return BcMetrics(
        mse=mse,
        mae=mae,
        mae_per_action=mae_per,
        boundary_rate=float(bound.any(axis=1).mean()),
        boundary_rate_per_action=br_per,
        n=int(pred.shape[0]),
    )


@torch.no_grad()
def predict_actions(
    actor: BcActor,
    states: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    actor.eval()
    states = np.asarray(states, dtype=np.float32)
    outs: list[np.ndarray] = []
    for i in range(0, len(states), batch_size):
        batch = torch.as_tensor(states[i : i + batch_size], device=device)
        outs.append(actor(batch).cpu().numpy())
    if not outs:
        return np.zeros((0, actor.act_dim), dtype=np.float32)
    return np.concatenate(outs, axis=0)


def evaluate_bc(
    actor: BcActor,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    device: torch.device,
    boundary_thresh: float = BOUNDARY_THRESH,
    action_names: Optional[Sequence[str]] = None,
) -> BcMetrics:
    pred = predict_actions(actor, states, device=device)
    return compute_bc_metrics(
        pred, actions, boundary_thresh=boundary_thresh, action_names=action_names
    )


def train_bc_epoch(
    actor: BcActor,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    actor.train()
    n = len(states)
    if n == 0:
        return 0.0
    order = rng.permutation(n)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        s = torch.as_tensor(states[idx], device=device)
        a = torch.as_tensor(actions[idx], device=device)
        pred = actor(s)
        loss = F.mse_loss(pred, a)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_bc_checkpoint(
    path: str | Path,
    actor: BcActor,
    *,
    metrics: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
    epoch: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "obs_dim": actor.obs_dim,
        "act_dim": actor.act_dim,
        "net_arch": list(actor.net_arch),
        "state_dict": actor.state_dict(),
        "sb3_mu_state_dict": {
            k: v.cpu() for k, v in actor.state_dict().items()
        },
        "metrics": metrics or {},
        "config": config or {},
    }
    torch.save(payload, path)


def load_bc_checkpoint(
    path: str | Path,
    *,
    device: Optional[torch.device] = None,
) -> tuple[BcActor, dict[str, Any]]:
    path = Path(path)
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    actor = BcActor(
        obs_dim=int(payload.get("obs_dim", 13)),
        act_dim=int(payload.get("act_dim", 3)),
        net_arch=tuple(payload.get("net_arch", (256, 256))),
    )
    actor.load_state_dict(payload["state_dict"])
    actor.to(device)
    actor.eval()
    return actor, payload


def validate_bc_checkpoint(
    payload: dict[str, Any],
    *,
    obs_dim: int,
    act_dim: int,
    net_arch: Sequence[int],
    action_mapping: str,
) -> None:
    """拒绝把维数、观测语义或动作映射不兼容的 BC 权重用于 TD3。"""
    expected = {
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "net_arch": [int(v) for v in net_arch],
    }
    actual = {
        "obs_dim": int(payload.get("obs_dim", -1)),
        "act_dim": int(payload.get("act_dim", -1)),
        "net_arch": [int(v) for v in payload.get("net_arch", ())],
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"BC checkpoint {key} 不兼容: {actual[key]!r} != {value!r}"
            )

    config = payload.get("config") or {}
    # 16D 扩展兼容：obs_dim=16 时允许新 schema/norm 版本，其余仍严格匹配。
    # 13D 默认行为不变；16D checkpoint 与 13D 互斥（由 obs_dim 字段保证）。
    if int(obs_dim) == int(OBS_DIM_16):
        allowed_schema = {OBS_SCHEMA_REL13_QMAG16}
        allowed_norm = {OBS16_NORM_VERSION}
    else:
        allowed_schema = {OBS_SCHEMA_VERSION}
        allowed_norm = {OBS_NORM_VERSION}
    required = {
        "action_mapping": action_mapping,
    }
    for key, value in required.items():
        got = config.get(key)
        if got is None:
            raise ValueError(f"BC checkpoint 缺少兼容性字段 config.{key}")
        if got != value:
            raise ValueError(
                f"BC checkpoint config.{key} 不兼容: {got!r} != {value!r}"
            )
    got_schema = config.get("obs_schema_version")
    if got_schema is None:
        raise ValueError("BC checkpoint 缺少兼容性字段 config.obs_schema_version")
    if got_schema not in allowed_schema:
        raise ValueError(
            f"BC checkpoint config.obs_schema_version 不兼容: {got_schema!r} "
            f"不在 {sorted(allowed_schema)!r}（obs_dim={obs_dim}）"
        )
    got_norm = config.get("obs_norm_version")
    if got_norm is None:
        raise ValueError("BC checkpoint 缺少兼容性字段 config.obs_norm_version")
    if got_norm not in allowed_norm:
        raise ValueError(
            f"BC checkpoint config.obs_norm_version 不兼容: {got_norm!r} "
            f"不在 {sorted(allowed_norm)!r}（obs_dim={obs_dim}）"
        )


def copy_bc_weights_to_td3_actor(td3_actor: nn.Module, bc_actor: BcActor) -> None:
    """
    将 BcActor 权重写入 SB3 TD3 Actor（仅 mu.* Linear 层）。
    要求 net_arch / obs_dim / act_dim 一致。
    """
    src = bc_actor.state_dict()
    dst = td3_actor.state_dict()
    device = next(td3_actor.parameters()).device
    mapped = {}
    for k, v in src.items():
        # BcActor: mu.0.weight … ；SB3 同样是 mu.0.weight
        if k in dst:
            if dst[k].shape != v.shape:
                raise ValueError(f"形状不匹配 {k}: {tuple(v.shape)} vs {tuple(dst[k].shape)}")
            mapped[k] = v.to(device)
        else:
            raise KeyError(f"TD3 actor 缺少键 {k}")
    td3_actor.load_state_dict(mapped, strict=False)


def copy_bc_weights_to_td3_actors(
    td3_actor: nn.Module,
    td3_actor_target: nn.Module,
    bc_actor: BcActor,
) -> None:
    """原位初始化 TD3 Actor 及 Actor target，不替换优化器引用的模块。"""
    copy_bc_weights_to_td3_actor(td3_actor, bc_actor)
    copy_bc_weights_to_td3_actor(td3_actor_target, bc_actor)
