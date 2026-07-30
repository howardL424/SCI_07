"""
evasion_env.py — Gymnasium 环境：双防御者规避 + 目标打击一体化制导律权重训练
动作空间：归一化 [-1,1]³（默认 ratio_log_box_v1: [ρE_n, ρT_n, γ_n]），w 固定 1e4，d*=20m
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

from action_mapping import (
    DEFAULT_ACTION_MAPPING,
    W_FIXED as AM_W_FIXED,
    action_norm_to_phys,
    phys_to_action_norm,
)
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
SAMPLES_VAL_FILE = str(Path(GAIL_ROOT) / "samples_clean_val.mat")
SAMPLES_TEST_FILE = str(Path(GAIL_ROOT) / "samples_clean_test.mat")

# ── 常量（与 reward_projection.DEFAULT_REWARD_CFG 同步，供评估脚本 import）──
ACT_DIM   = 3
W_FIXED   = AM_W_FIXED
D_STAR    = DEFAULT_REWARD_CFG.dstar
KILL_R    = DEFAULT_REWARD_CFG.kill_radius
STRIKE_OK = DEFAULT_REWARD_CFG.strike_ok
R1_CEIL   = DEFAULT_REWARD_CFG.r1_ceil
BONUS_BOTH = DEFAULT_REWARD_CFG.bonus_both

# 0711 及更早 checkpoint 训练时的 observation_space（未做 OBS_SCALE）
LEGACY_OBS_HIGH = np.array(
    [
        2.0e4,
        2.0e3,
        np.pi,
        np.pi,
        2.0e4,
        2.0e3,
        np.pi,
        np.pi,
        2.0e5,
        np.pi,
        np.pi,
        1.0e3,
        1.0,
    ],
    dtype=np.float32,
)


def _matlab_stage3_traj_to_dict(traj_ml) -> dict:
    """将 stage3_rollout 第 5 输出 struct/dict 转为 Python dict（行向量时间序列）。"""
    empty = {
        "t": np.zeros(0, dtype=np.float64),
        "pos_M": np.zeros((0, 3), dtype=np.float64),
        "pos_D1": np.zeros((0, 3), dtype=np.float64),
        "pos_D2": np.zeros((0, 3), dtype=np.float64),
        "att_M": np.zeros((0, 3), dtype=np.float64),
        "a_M_body": np.zeros((0, 2), dtype=np.float64),
        "r_MT": np.zeros(0, dtype=np.float64),
        "r_MD1": np.zeros(0, dtype=np.float64),
        "r_MD2": np.zeros(0, dtype=np.float64),
    }
    if traj_ml is None:
        return empty

    def _get(name: str):
        if isinstance(traj_ml, dict):
            return traj_ml[name]
        return getattr(traj_ml, name)

    def _vec1(name: str) -> np.ndarray:
        return np.asarray(_get(name), dtype=np.float64).reshape(-1)

    def _mat_rows(name: str, n_cols: int) -> np.ndarray:
        a = np.asarray(_get(name), dtype=np.float64)
        if a.size == 0:
            return np.zeros((0, n_cols), dtype=np.float64)
        a = np.atleast_2d(a)
        # MATLAB 存为 n_cols×N；若已是 N×n_cols 也兼容
        if a.shape[0] == n_cols:
            return a.T
        if a.shape[1] == n_cols:
            return a
        return a.reshape(-1, n_cols)

    return {
        "t": _vec1("t"),
        "pos_M": _mat_rows("pos_M", 3),
        "pos_D1": _mat_rows("pos_D1", 3),
        "pos_D2": _mat_rows("pos_D2", 3),
        "att_M": _mat_rows("att_M", 3),
        "a_M_body": _mat_rows("a_M", 2),
        "r_MT": _vec1("r_MT"),
        "r_MD1": _vec1("r_MD1"),
        "r_MD2": _vec1("r_MD2"),
    }


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
        action_mapping: str = DEFAULT_ACTION_MAPPING,
        obs_normalize: bool = True,
        record_stage3_traj: bool = False,
    ):
        super().__init__()

        self.samples_file = samples_file
        self.action_mapping = str(action_mapping)
        self.obs_normalize = bool(obs_normalize)
        self.record_stage3_traj = bool(record_stage3_traj)
        self.matlab_cfg = dict(matlab_cfg or {})
        self.matlab_cfg.setdefault("action_mapping_version", self.action_mapping)
        self.max_retries = max_retries

        if self.obs_normalize:
            # 网络输入为归一化观测（obs/OBS_SCALE，不裁剪）；Box 仅定义空间
            obs_bound = np.full(OBS_DIM, OBS_BOX_HIGH, dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-obs_bound, high=obs_bound, dtype=np.float32
            )
        else:
            # 兼容 0711 等旧模型：物理量直接进网络
            self.observation_space = spaces.Box(
                low=-LEGACY_OBS_HIGH, high=LEGACY_OBS_HIGH, dtype=np.float32
            )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        sc = scipy.io.loadmat(samples_file, squeeze_me=True)
        samples_all = np.asarray(sc["samples_clean"], dtype=np.float64)
        if samples_all.ndim != 2:
            raise ValueError(
                f"samples_clean 须为 2D，收到 shape={samples_all.shape}"
            )
        # 过滤全零/近零填充行（否则 reconstruct_geometry 会得到共点初态，
        # warm-start 立刻触发假 stage2，env_reset 仍返回 ok=true）
        valid_mask = np.linalg.norm(samples_all, axis=1) > 1e-12
        self._sample_row_ids = np.flatnonzero(valid_mask).astype(np.int64)
        self._samples = samples_all[valid_mask]
        self._n_samples = int(self._samples.shape[0])
        n_drop = int(samples_all.shape[0] - self._n_samples)
        if self._n_samples == 0:
            raise RuntimeError(
                f"[EvasionEnv] {samples_file} 过滤全零行后无剩余样本"
            )
        print(
            f"[EvasionEnv] {Path(samples_file).name}: "
            f"保留 {self._n_samples}/{samples_all.shape[0]} 行"
            + (f"（已滤除全零 {n_drop} 行）" if n_drop else "")
        )

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
        self._stage2_core = None  # 懒初始化，供 reset_from_abs

    def _format_obs(self, obs_phys: np.ndarray) -> np.ndarray:
        """按 obs_normalize 开关输出网络观测。"""
        x = np.asarray(obs_phys, dtype=np.float32).ravel()
        if self.obs_normalize:
            return normalize_obs(x)
        return x.astype(np.float32)

    @property
    def n_samples(self) -> int:
        """过滤全零后的有效初态数量。"""
        return self._n_samples

    @property
    def sample_row_ids(self) -> np.ndarray:
        """库内下标 → 初态 mat 原始行号的映射（只读副本）。"""
        return self._sample_row_ids.copy()

    def _reset_at_local(self, local_idx: int):
        """
        用过滤后库内下标启动 episode。
        warm-start 失败时返回 None（定点评估勿静默换行）。
        """
        local_idx = int(local_idx)
        if local_idx < 0 or local_idx >= self._n_samples:
            raise IndexError(
                f"local_idx={local_idx} 超出范围 [0, {self._n_samples})"
            )
        row_idx = int(self._sample_row_ids[local_idx])  # 原始 mat 行号
        row_ml = matlab.double(self._samples[local_idx].tolist())
        obs_ml, sabs_ml, Tev_ml, zT0_ml, ok_ml = self._eng.env_reset(
            row_ml, self.matlab_cfg, nargout=5
        )
        if not bool(ok_ml):
            return None

        obs_phys = np.array(obs_ml, dtype=np.float32).flatten()
        self._state_abs = np.array(sabs_ml, dtype=np.float64).flatten()
        self._t = 0.0
        self._Tev = float(Tev_ml)
        self._zT0 = float(zT0_ml)
        self._r_min1 = np.inf
        self._r_min2 = np.inf
        return self._format_obs(obs_phys), {
            "row_idx": row_idx,
            "local_idx": local_idx,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        options = options or {}

        # 定点初态（训后 MC 测全 test 集）：失败不换行
        if "local_idx" in options:
            out = self._reset_at_local(int(options["local_idx"]))
            if out is None:
                raise RuntimeError(
                    f"[EvasionEnv] local_idx={int(options['local_idx'])} "
                    f"warm-start 无有效规避段（file={self.samples_file}）"
                )
            return out

        for _ in range(self.max_retries):
            local_idx = int(self.np_random.integers(0, self._n_samples))
            out = self._reset_at_local(local_idx)
            if out is not None:
                return out

        raise RuntimeError(
            f"[EvasionEnv] {self.max_retries} 次采样均未找到有效规避段，"
            f"请检查 {self.samples_file} 或 warm-start 配置。"
        )

    def reset_from_abs(
        self,
        state_abs: np.ndarray,
        *,
        Tev: float,
        t: float = 0.0,
        zT0: float = 1.0,
    ) -> np.ndarray:
        """
        从给定绝对态启动 episode（BC 诊断：专家动作回放 / 同 IC 对照）。
        返回归一化观测（不经 samples_clean 抽样）。
        """
        self._state_abs = np.asarray(state_abs, dtype=np.float64).ravel()
        if self._state_abs.size != 18:
            raise ValueError(f"state_abs 须为 18 维，收到 {self._state_abs.size}")
        self._t = float(t)
        self._Tev = float(Tev)
        self._zT0 = float(zT0)
        self._r_min1 = np.inf
        self._r_min2 = np.inf

        if self._stage2_core is None:
            self._stage2_core = self._eng.stage2_core_init(self.matlab_cfg, nargout=1)
        sabs_ml = matlab.double(self._state_abs.tolist())
        obs_ml = self._eng.stage2_obs_from_abs(
            self._stage2_core,
            sabs_ml,
            float(self._t),
            float(self._Tev),
            nargout=1,
        )
        obs_phys = np.array(obs_ml, dtype=np.float32).flatten()
        return self._format_obs(obs_phys)

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
        # env_step 扩展：后 4 项为 a_M_sat(y,z) / a_M_raw(y,z)；旧 5 元 info 仍兼容
        if info_vec.size >= 9:
            a_M_sat = np.array([info_vec[5], info_vec[6]], dtype=np.float64)
            a_M_raw = np.array([info_vec[7], info_vec[8]], dtype=np.float64)
        elif info_vec.size >= 7:
            a_M_sat = np.array([info_vec[5], info_vec[6]], dtype=np.float64)
            a_M_raw = a_M_sat.copy()
        else:
            a_M_sat = np.full(2, np.nan, dtype=np.float64)
            a_M_raw = np.full(2, np.nan, dtype=np.float64)

        self._r_min1 = min(self._r_min1, r_MD1)
        self._r_min2 = min(self._r_min2, r_MD2)

        info = {
            "r_MD1":       r_MD1,
            "r_MD2":       r_MD2,
            "r_MT":        r_MT,
            "intercepted": intercepted,
            "stage2_done": stage2_done,
            "a_M_sat":     a_M_sat,
            "a_M_raw":     a_M_raw,
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
            if self.record_stage3_traj:
                r_minT_ml, reached_ml, diverged_ml, n_ml, traj_ml = self._eng.stage3_rollout(
                    sabs_ml, float(self._t), self.matlab_cfg, nargout=5
                )
                info["stage3_traj"] = _matlab_stage3_traj_to_dict(traj_ml)
            else:
                r_minT_ml, reached_ml, diverged_ml, n_ml = self._eng.stage3_rollout(
                    sabs_ml, float(self._t), self.matlab_cfg, nargout=4
                )
            r_minT = float(r_minT_ml)
            reached_pn = bool(reached_ml)
            diverged = bool(diverged_ml)
            n_steps_stage3 = int(n_ml)

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
                "n_steps_stage3": n_steps_stage3,
                "zT":        r_minT,      # 评估打印终端精度时用 r_minT
            })

        return self._format_obs(obs_phys), reward, terminated, False, info

    def close(self):
        if hasattr(self, "_eng") and self._eng is not None:
            self._eng.quit()
            self._eng = None
        super().close()

    @staticmethod
    def norm_action(
        gamma: float,
        wT: float,
        wE: float,
        *,
        version: str = DEFAULT_ACTION_MAPPING,
        w: float = W_FIXED,
    ) -> np.ndarray:
        """将物理权重参数归一化到 [-1,1]³。"""
        an, _ = phys_to_action_norm(gamma, wT, wE, version=version, w=w)
        return np.asarray(an, dtype=np.float32).ravel()

    @staticmethod
    def denorm_action(
        action_norm: np.ndarray,
        *,
        version: str = DEFAULT_ACTION_MAPPING,
        w: float = W_FIXED,
    ) -> tuple[float, float, float]:
        """将归一化动作反归一化回物理值 (gamma, wT, wE)。"""
        gamma, wT, wE = action_norm_to_phys(action_norm, version=version, w=w)
        return float(np.asarray(gamma).ravel()[0]), float(np.asarray(wT).ravel()[0]), float(
            np.asarray(wE).ravel()[0]
        )
