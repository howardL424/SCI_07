"""
evasion_env.py — Gymnasium 环境：双防御者规避 + 目标打击一体化制导律权重训练
动作空间：[γ_n, wT_n, wE_n] ∈ [-1,1]³（归一化），w 固定 1e4，d*=20m
状态空间：13维时变态势（见 extract_expert_transitions.m）
MATLAB Engine 调用 env_reset.m / env_step.m 执行仿真
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional

import matlab.engine
import numpy as np
import scipy.io
import gymnasium as gym
from gymnasium import spaces

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GAIL_ROOT    = str(Path(__file__).parent.parent.resolve())        # .../GAIL
MATLAB_DIR   = str(Path(__file__).parent / "matlab")             # .../rl_td3/matlab
EXPERT_DIR   = str(Path(GAIL_ROOT) / "expert_trajectory")
SAMPLES_FILE = str(Path(GAIL_ROOT) / "samples_clean.mat")

# ── 常量 ──────────────────────────────────────────────────────────────────────
OBS_DIM   = 13
ACT_DIM   = 3       # [gamma_n, wT_n, wE_n]
W_FIXED   = 1e4
D_STAR    = 20.0    # 期望规避脱靶量 (m)


class EvasionEnv(gym.Env):
    """
    双防御者规避 + 目标打击 Gymnasium 环境。

    每步输出归一化动作 [γ_n, wT_n, wE_n] ∈ [-1,1]³，env_step.m 将其反归一化后
    代入式47 计算制导指令，并推进仿真 dt=0.1s，返回新的 13维 obs 和 reward。

    动作反归一化（在 env_step.m 中执行）：
        gamma = gamma_n * 3                → [-3, 3]
        wT    = 10^(wT_n + 3)             → [1e2, 1e4]
        wE    = 10^(wE_n + 3)             → [1e2, 1e4]
        w1    = W / (1 + exp(-gamma))
        w2    = W * exp(-gamma) / (1 + exp(-gamma))
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        samples_file: str = SAMPLES_FILE,
        matlab_cfg: Optional[dict] = None,
        max_retries: int = 10,
    ):
        super().__init__()

        self.samples_file = samples_file
        self.matlab_cfg   = matlab_cfg or {}
        self.max_retries  = max_retries

        # ── 观测 / 动作空间 ──────────────────────────────────────────────────
        obs_high = np.array([
            2e4,  2e3,  np.pi,  np.pi,   # r_MD1, rdot_MD1, qy_D1M, qz_D1M
            2e4,  2e3,  np.pi,  np.pi,   # r_MD2, rdot_MD2, qy_D2M, qz_D2M
            2e5,  np.pi, np.pi, 1e3,     # r_MT, qy_MT, qz_MT, V_M
            1.0,                          # tgo_norm
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # ── 加载初态样本库 ────────────────────────────────────────────────────
        sc = scipy.io.loadmat(samples_file, squeeze_me=True)
        self._samples: np.ndarray = sc["samples_clean"].astype(np.float64)
        self._n_samples = len(self._samples)

        # ── 启动 MATLAB Engine ────────────────────────────────────────────────
        print("[EvasionEnv] 正在启动 MATLAB Engine，请稍候…")
        self._eng = matlab.engine.start_matlab()
        self._eng.addpath(MATLAB_DIR,  nargout=0)
        self._eng.addpath(GAIL_ROOT,   nargout=0)
        self._eng.addpath(EXPERT_DIR,  nargout=0)
        print("[EvasionEnv] MATLAB Engine 就绪")

        # ── episode 内部状态 ──────────────────────────────────────────────────
        self._state_abs: Optional[np.ndarray] = None  # [18]
        self._t:   float = 0.0
        self._Tev: float = 10.0
        self._r_MT0: float = 1.0

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        for _ in range(self.max_retries):
            row_idx = self.np_random.integers(0, self._n_samples)
            row_ml  = matlab.double(self._samples[row_idx].tolist())

            obs_ml, sabs_ml, Tev_ml, r_MT0_ml, ok_ml = self._eng.env_reset(
                row_ml, self.matlab_cfg, nargout=5
            )
            ok = bool(ok_ml)
            if not ok:
                continue

            obs = np.array(obs_ml, dtype=np.float32).flatten()
            self._state_abs = np.array(sabs_ml, dtype=np.float64).flatten()
            self._t         = 0.0
            self._Tev       = float(Tev_ml)
            self._r_MT0     = float(r_MT0_ml)
            return obs, {"row_idx": int(row_idx)}

        raise RuntimeError(
            f"[EvasionEnv] {self.max_retries} 次采样均未找到有效规避段，"
            "请检查 samples_clean.mat 或 warm-start 配置。"
        )

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        if self._state_abs is None:
            raise RuntimeError("请先调用 env.reset() 初始化 episode。")

        action = np.clip(action, -1.0, 1.0)
        act_ml  = matlab.double(action.tolist())
        sabs_ml = matlab.double(self._state_abs.tolist())

        obs_ml, sabs_next_ml, rew_ml, term_ml, info_ml = self._eng.env_step(
            sabs_ml,
            float(self._t),
            float(self._Tev),
            float(self._r_MT0),
            act_ml,
            self.matlab_cfg,
            nargout=5,
        )

        obs_next = np.array(obs_ml,      dtype=np.float32).flatten()
        info_vec = np.array(info_ml,     dtype=np.float64).flatten()
        reward   = float(rew_ml)
        terminated = bool(term_ml)

        self._state_abs = np.array(sabs_next_ml, dtype=np.float64).flatten()
        self._t += 0.1   # dt = 0.1 s

        info = {
            "r_MD1":       info_vec[0],
            "r_MD2":       info_vec[1],
            "r_MT":        info_vec[2],
            "intercepted": bool(info_vec[3]),
            "reached":     bool(info_vec[4]),
            "t":           self._t,
        }

        return obs_next, reward, terminated, False, info

    # ── close ─────────────────────────────────────────────────────────────────
    def close(self):
        if hasattr(self, "_eng") and self._eng is not None:
            self._eng.quit()
            self._eng = None
        super().close()

    # ── 归一化辅助（供 load_expert_buffer 使用）─────────────────────────────
    @staticmethod
    def norm_action(gamma: float, wT: float, wE: float) -> np.ndarray:
        """将物理权重参数归一化到 [-1,1]³（与 env_step.m 反归一化对应）。"""
        gn  = np.clip(gamma / 3.0, -1.0, 1.0)
        wTn = np.clip(np.log10(np.clip(wT, 1e2, 1e4)) - 3.0, -1.0, 1.0)
        wEn = np.clip(np.log10(np.clip(wE, 1e2, 1e4)) - 3.0, -1.0, 1.0)
        return np.array([gn, wTn, wEn], dtype=np.float32)

    @staticmethod
    def denorm_action(action_norm: np.ndarray) -> tuple[float, float, float]:
        """将归一化动作反归一化回物理值 (gamma, wT, wE)。"""
        gamma = float(action_norm[0]) * 3.0
        wT    = 10 ** (float(action_norm[1]) + 3.0)
        wE    = 10 ** (float(action_norm[2]) + 3.0)
        return gamma, wT, wE
