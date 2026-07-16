"""
td3_bc.py — 带衰减式 BC 正则的 TD3（TD3+BC 变体）。

actor_loss = -Q(s, π(s)) + λ(t) * ||π(s_E) - a_E||²
λ(t) = λ0 * max(0, 1 - t / T_bc)

BC 项只在专家 (s,a) 上计算，不限制后期超越专家。
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import TD3
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update


class TD3BC(TD3):
    """TD3 + 衰减行为克隆正则。"""

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        bc_lambda0: float = 1.0,
        bc_decay_steps: int = 150_000,
        **kwargs: Any,
    ):
        self.bc_lambda0 = float(bc_lambda0)
        self.bc_decay_steps = int(bc_decay_steps)
        self._expert_states: Optional[th.Tensor] = None
        self._expert_actions: Optional[th.Tensor] = None
        super().__init__(policy, env, learning_rate=learning_rate, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_expert_states",
            "_expert_actions",
        ]

    def set_expert_dataset(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """注入专家 (s,a)，供 actor BC 项采样。"""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("expert states/actions 须为二维数组")
        if len(states) != len(actions):
            raise ValueError("expert states/actions 长度不一致")
        if len(states) == 0:
            self._expert_states = None
            self._expert_actions = None
            return
        self._expert_states = th.as_tensor(states, device=self.device)
        self._expert_actions = th.as_tensor(actions, device=self.device)

    def bc_lambda(self) -> float:
        if self.bc_lambda0 <= 0 or self.bc_decay_steps <= 0:
            return 0.0
        return self.bc_lambda0 * max(0.0, 1.0 - float(self.num_timesteps) / float(self.bc_decay_steps))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, bc_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()

                lam = self.bc_lambda()
                bc_loss_val = 0.0
                if (
                    lam > 0.0
                    and self._expert_states is not None
                    and self._expert_actions is not None
                    and len(self._expert_states) > 0
                ):
                    n_e = self._expert_states.shape[0]
                    idx = th.randint(0, n_e, (batch_size,), device=self.device)
                    s_e = self._expert_states[idx]
                    a_e = self._expert_actions[idx]
                    bc_loss = F.mse_loss(self.actor(s_e), a_e)
                    actor_loss = actor_loss + lam * bc_loss
                    bc_loss_val = float(bc_loss.item())

                actor_losses.append(actor_loss.item())
                bc_losses.append(bc_loss_val)

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(
                    self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0
                )
                polyak_update(
                    self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_lambda", self.bc_lambda())
        self.logger.record("train/critic_loss", np.mean(critic_losses))
