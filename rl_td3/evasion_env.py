"""
evasion_env.py — Gymnasium 环境：双防御者规避 + 目标打击一体化制导律权重训练
动作空间：[γ_n, wT_n, wE_n] ∈ [-1,1]³（归一化），w 固定 1e4，d*=20m
状态空间：13维时变态势，经 obs_norm 固定逐维物理归一化后输入网络
MATLAB Engine 调用 env_reset.m / env_step.m / stage3_rollout.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matlab.engine
import numpy as np
import scipy.io
import gymnasium as gym
from gymnasium import spaces

from obs_norm import OBS_BOX_HIGH, OBS_DIM, normalize_obs
from reward_projection import (
    DEFAULT_REWARD_CFG,
    project_terminal,
    terminal_R1,
)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GAIL_ROOT    = str(Path(__file__).parent.parent.resolve())
MATLAB_DIR   = str(Path(__file__).parent / "matlab")
EXPERT_DIR   = str(Path(GAIL_ROOT) / "expert_trajectory")
SAMPLES_FILE = str(Path(GAIL_ROOT) / "samples_clean.mat")

# ── 常量（与 reward_projection.DEFAULT_REWARD_CFG 同步，供评估脚本 import）──
ACT_DIM   = 3
W_FIXED   = 1e4
D_STAR    = DEFAULT_REWARD_CFG.dstar
KILL_R    = DEFAULT_REWARD_CFG.kill_radius
STRIKE_OK = DEFAULT_REWARD_CFG.strike_ok
R1_CEIL   = DEFAULT_REWARD_CFG.r1_ceil
BONUS_BOTH = DEFAULT_REWARD_CFG.bonus_both


class EvasionEnv(gym.Env):
    """
    双防御者规避 + 目标打击 Gymnasium 环境。

    stage2：RL 输出权重，式47 制导；过程奖励 0.05*(R2_sat+R3_sat)；拦截 -50。
    stage2 结束（t≥Tev）后：MATLAB 一次跑完 stage3 PN，结算 R1，若双防≥6 且
    r_minT≤10 再 +100。
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

        # 网络输入为归一化观测（obs/OBS_SCALE，不裁剪）；Box 仅定义空间
        obs_bound = np.full(OBS_DIM, OBS_BOX_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_bound, high=obs_bound, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        sc = scipy.io.loadmat(samples_file, squeeze_me=True)
        self._samples: np.ndarray = sc["samples_clean"].astype(np.float64)
        self._n_samples = len(self._samples)

        print("[EvasionEnv] 正在启动 MATLAB Engine，请稍候…")
        self._eng = matlab.engine.start_matlab()
        self._eng.addpath(MATLAB_DIR,  nargout=0)
        self._eng.addpath(GAIL_ROOT,   nargout=0)
        self._eng.addpath(EXPERT_DIR,  nargout=0)
        print("[EvasionEnv] MATLAB Engine 就绪")

        self._state_abs: Optional[np.ndarray] = None
        self._t:   float = 0.0
        self._Tev: float = 10.0
        self._zT0: float = 1.0   # 兼容 env_reset 返回值，奖励不再使用
        self._r_min1: float = np.inf
        self._r_min2: float = np.inf

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

            obs_ml, sabs_ml, Tev_ml, zT0_ml, ok_ml = self._eng.env_reset(
                row_ml, self.matlab_cfg, nargout=5
            )
            ok = bool(ok_ml)
            if not ok:
                continue

            obs_phys = np.array(obs_ml, dtype=np.float32).flatten()
            self._state_abs = np.array(sabs_ml, dtype=np.float64).flatten()
            self._t         = 0.0
            self._Tev       = float(Tev_ml)
            self._zT0       = float(zT0_ml)
            self._r_min1    = np.inf
            self._r_min2    = np.inf
            return normalize_obs(obs_phys), {"row_idx": int(row_idx)}

        raise RuntimeError(
            f"[EvasionEnv] {self.max_retries} 次采样均未找到有效规避段，"
            "请检查 samples_clean.mat 或 warm-start 配置。"
        )

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
            float(self._zT0),
            act_ml,
            self.matlab_cfg,
            nargout=5,
        )

        obs_phys = np.array(obs_ml, dtype=np.float32).flatten()
        info_vec = np.array(info_ml, dtype=np.float64).flatten()
        reward   = float(rew_ml)
        terminated = bool(term_ml)

        self._state_abs = np.array(sabs_next_ml, dtype=np.float64).flatten()
        self._t += 0.1

        r_MD1 = float(info_vec[0])
        r_MD2 = float(info_vec[1])
        r_MT  = float(info_vec[2])
        intercepted = bool(info_vec[3])
        stage2_done = bool(info_vec[4])

        self._r_min1 = min(self._r_min1, r_MD1)
        self._r_min2 = min(self._r_min2, r_MD2)

        info = {
            "r_MD1":       r_MD1,
            "r_MD2":       r_MD2,
            "r_MT":        r_MT,
            "intercepted": intercepted,
            "stage2_done": stage2_done,
            "r_min1":      float(self._r_min1),
            "r_min2":      float(self._r_min2),
            "t":           self._t,
            "r_minT":      np.nan,
            "R1":          0.0,
            "bonus":       0.0,
            "evade_ok":    False,
            "strike_ok":   False,
            "reached":     False,   # 兼容旧评估脚本：打击成功才为 True
            "zT":          r_MT,    # 兼容旧字段名：现为当前 r_MT，非 ZEM
        }

        # 规避段正常结束 → MATLAB 一次 PN 跑完 stage3，结算终端奖励
        if terminated and stage2_done and not intercepted:
            sabs_ml = matlab.double(self._state_abs.tolist())
            r_minT_ml, reached_ml, diverged_ml, _n_ml = self._eng.stage3_rollout(
                sabs_ml, float(self._t), self.matlab_cfg, nargout=4
            )
            r_minT = float(r_minT_ml)
            reached_pn = bool(reached_ml)
            diverged = bool(diverged_ml)

            R1, bonus, both_ok = project_terminal(
                r_minT, self._r_min1, self._r_min2, intercepted=False
            )
            evade_ok = (self._r_min1 >= KILL_R) and (self._r_min2 >= KILL_R)
            strike_ok = r_minT <= STRIKE_OK
            reward = reward + R1 + bonus

            info.update({
                "r_minT":    r_minT,
                "R1":        R1,
                "bonus":     bonus,
                "both_ok":   bool(both_ok),
                "evade_ok":  evade_ok,
                "strike_ok": strike_ok,
                "reached":   strike_ok,   # 评估脚本用 reached 表示打击成功
                "reached_pn": reached_pn,
                "diverged":  diverged,
                "zT":        r_minT,      # 评估打印终端精度时用 r_minT
            })

        return normalize_obs(obs_phys), reward, terminated, False, info

    def close(self):
        if hasattr(self, "_eng") and self._eng is not None:
            self._eng.quit()
            self._eng = None
        super().close()

    @staticmethod
    def norm_action(gamma: float, wT: float, wE: float) -> np.ndarray:
        """将物理权重参数归一化到 [-1,1]³（与 env_step.m 反归一化对应）。"""
        gn  = np.clip(gamma / 3.0, -1.0, 1.0)
        wTn = np.clip((np.log10(np.clip(wT, 1e2, 1e5)) - 3.5) / 1.5, -1.0, 1.0)
        wEn = np.clip(np.log10(np.clip(wE, 1e2, 1e4)) - 3.0, -1.0, 1.0)
        return np.array([gn, wTn, wEn], dtype=np.float32)

    @staticmethod
    def denorm_action(action_norm: np.ndarray) -> tuple[float, float, float]:
        """将归一化动作反归一化回物理值 (gamma, wT, wE)。"""
        gamma = float(action_norm[0]) * 3.0
        wT    = 10 ** (float(action_norm[1]) * 1.5 + 3.5)
        wE    = 10 ** (float(action_norm[2]) + 3.0)
        return gamma, wT, wE
