"""
evasion_env.py — Gymnasium 环境：双防御者规避 + 目标打击一体化制导律权重训练
动作空间：[γ_n, wT_n, wE_n] ∈ [-1,1]³（归一化），w 固定 1e4，d*=20m
状态空间：13维时变态势（见 extract_expert_transitions.m）
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

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GAIL_ROOT    = str(Path(__file__).parent.parent.resolve())
MATLAB_DIR   = str(Path(__file__).parent / "matlab")
EXPERT_DIR   = str(Path(GAIL_ROOT) / "expert_trajectory")
SAMPLES_FILE = str(Path(GAIL_ROOT) / "samples_clean.mat")

# ── 常量 ──────────────────────────────────────────────────────────────────────
OBS_DIM   = 13
ACT_DIM   = 3       # [gamma_n, wT_n, wE_n]
W_FIXED   = 1e4
D_STAR    = 20.0    # 期望规避脱靶量 (m)，过程奖励软饱和上限
KILL_R    = 6.0     # 拦截半径 / 规避成功下限
STRIKE_OK = 10.0    # 打击成功：r_minT ≤ 10m → +100
R1_CEIL   = 25.0    # R1 非零上限：r_minT ≤ 25m
BONUS_BOTH = 100.0  # 规避成功且打击成功额外奖励


def terminal_R1(r_minT: float) -> float:
    """全段后结算：R1 = 80/(5+r_minT) if r_minT≤25 else 0。"""
    if r_minT <= R1_CEIL:
        return 80.0 / (5.0 + r_minT)
    return 0.0


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

        obs_high = np.array([
            2e4,  2e3,  np.pi,  np.pi,
            2e4,  2e3,  np.pi,  np.pi,
            2e5,  np.pi, np.pi, 1e3,
            1.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
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

            obs = np.array(obs_ml, dtype=np.float32).flatten()
            self._state_abs = np.array(sabs_ml, dtype=np.float64).flatten()
            self._t         = 0.0
            self._Tev       = float(Tev_ml)
            self._zT0       = float(zT0_ml)
            self._r_min1    = np.inf
            self._r_min2    = np.inf
            return obs, {"row_idx": int(row_idx)}

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

        obs_next = np.array(obs_ml,      dtype=np.float32).flatten()
        info_vec = np.array(info_ml,     dtype=np.float64).flatten()
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

            R1 = terminal_R1(r_minT)
            evade_ok = (self._r_min1 >= KILL_R) and (self._r_min2 >= KILL_R)
            strike_ok = r_minT <= STRIKE_OK
            bonus = BONUS_BOTH if (evade_ok and strike_ok) else 0.0
            reward = reward + R1 + bonus

            info.update({
                "r_minT":    r_minT,
                "R1":        R1,
                "bonus":     bonus,
                "evade_ok":  evade_ok,
                "strike_ok": strike_ok,
                "reached":   strike_ok,   # 评估脚本用 reached 表示打击成功
                "reached_pn": reached_pn,
                "diverged":  diverged,
                "zT":        r_minT,      # 评估打印终端精度时用 r_minT
            })

        return obs_next, reward, terminated, False, info

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
