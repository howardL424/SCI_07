"""Independent Gymnasium direct-acceleration environment (no BC dependency)."""
from pathlib import Path
import numpy as np
import scipy.io
import gymnasium as gym
from gymnasium import spaces
from .common import ROOT, PACKAGE, OBS_BOX_HIGH, normalize_obs, project_terminal

class DirectEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, samples_file=None, matlab_cfg=None, max_retries=10, engine=None):
        self.samples_file = str(Path(samples_file or ROOT / 'samples_clean.mat').resolve())
        rows = np.asarray(scipy.io.loadmat(self.samples_file)['samples_clean'], dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 14 or not np.all(np.isfinite(rows)):
            raise ValueError('Expected finite samples_clean[N,14]')
        valid = np.linalg.norm(rows, axis=1) > 1e-12
        self.sample_row_ids = np.flatnonzero(valid)
        self.samples = rows[valid]
        self.n_samples = len(self.samples)
        self.n_zero_rows = int((~valid).sum())
        if not self.n_samples:
            raise ValueError('No valid initial conditions')
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_BOX_HIGH, OBS_BOX_HIGH, (13,), dtype=np.float32)
        self.cfg = dict(matlab_cfg or {})
        # Terminal projection intentionally uses the same frozen main-route
        # reward configuration. Reject inconsistent process-only overrides.
        reward_defaults=dict(dstar=20.,kill_radius=6.,proc_coef=.05,intercept_pen=50.)
        for key,value in reward_defaults.items():
            if self.cfg.get(key,value)!=value:
                raise ValueError(f'This comparison freezes {key}={value}')
        self.max_retries = max_retries
        self.eng = engine
        self.owns_engine = engine is None
        self.core = None
        self.finished = True
        self.n_clipped_total = 0

    def _start(self):
        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab('-nodesktop -nosplash')
        if self.core is None:
            self.eng.addpath(str(PACKAGE / 'matlab'), nargout=0)
            self.core = self.eng.ps_core_init(self.cfg)

    @staticmethod
    def _ml(x):
        import matlab
        return matlab.double(np.asarray(x, dtype=float).reshape(-1, 1).tolist())

    def _init_episode(self, sabs, Tev, t=0.):
        self.sabs = np.asarray(sabs, dtype=float).reshape(18)
        if not np.all(np.isfinite(self.sabs)) or not np.isfinite(Tev) or Tev <= t:
            raise ValueError('Invalid absolute state or evasion window')
        self.Tev, self.t = float(Tev), float(t)
        self.r_min1 = self.r_min2 = np.inf  # Same post-step minima as main route.
        self.n_clipped = 0
        self.energy_int = 0.
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start()
        fixed = (options or {}).get('local_idx')
        if fixed is not None and not 0 <= int(fixed) < self.n_samples:
            raise IndexError(fixed)
        for _ in range(1 if fixed is not None else self.max_retries):
            idx = int(fixed) if fixed is not None else int(self.np_random.integers(self.n_samples))
            obs, sabs, Tev, zT0, ok = self.eng.ps_env_reset(
                self._ml(self.samples[idx]), self.cfg, nargout=5)
            if bool(ok):
                self._init_episode(sabs, Tev)
                self.local_idx = idx
                self.zT0 = float(zT0)
                return normalize_obs(np.asarray(obs).reshape(13)), dict(
                    local_idx=idx, row_idx=int(self.sample_row_ids[idx]), Tev=self.Tev)
        raise RuntimeError('warm-start failed to provide an evasion window')

    def reset_from_abs(self, state_abs, Tev, *, t0=0., seed=None):
        super().reset(seed=seed)
        self._start()
        self._init_episode(state_abs, Tev, t0)
        obs = self.eng.stage2_obs_from_abs(self.core, self._ml(self.sabs), self.t, self.Tev)
        return normalize_obs(np.asarray(obs).reshape(13)), dict(Tev=self.Tev)

    def step(self, action):
        if self.finished:
            raise RuntimeError('Call reset before stepping a finished episode')
        obs, sabs, t, reward, fact = self.eng.ps_env_step(
            self.core, self._ml(self.sabs), self.t, self.Tev, self._ml(action), nargout=5)
        self.sabs = np.asarray(sabs).reshape(18)
        self.t = float(t)
        self.raw_obs = np.asarray(obs).reshape(13)
        self.r_min1 = min(self.r_min1, float(fact['r_MD1']))
        self.r_min2 = min(self.r_min2, float(fact['r_MD2']))
        clipped = int(fact['n_clipped'])
        self.n_clipped += clipped
        self.n_clipped_total += clipped
        executed = np.asarray(fact['a_norm_executed']).reshape(2)
        self.energy_int += float(np.sum(np.asarray(fact['a_M_sat']) ** 2)) * self.cfg.get('dt', .1)
        done, intercepted = bool(fact['done']), bool(fact['intercepted'])
        info = dict(r_MD1=float(fact['r_MD1']), r_MD2=float(fact['r_MD2']),
                    r_MT=float(fact['r_MT']), r_min1=self.r_min1, r_min2=self.r_min2,
                    intercepted=intercepted, stage2_done=bool(fact['stage2_done']),
                    n_clipped=self.n_clipped, a_norm_executed=executed,
                    stage2_energy_int=self.energy_int)
        reward = float(reward)
        if done:
            self.finished = True
            rT, reached, energy = np.inf, False, 0.
            if not intercepted:
                rT, reached, _, _, traj = self.eng.stage3_rollout(
                    self._ml(self.sabs), self.t, dict(dt=.001, return_traj=False), nargout=5)
                rT, reached, energy = float(rT), bool(reached), float(traj['energy_int'])
            R1, bonus, both = project_terminal(rT, self.r_min1, self.r_min2, intercepted=intercepted)
            reward += R1 + bonus
            info.update(r_minT=rT, R1=R1, bonus=bonus, both_ok=bool(both),
                        evade_ok=not intercepted and min(self.r_min1,self.r_min2)>=6,
                        strike_ok=not intercepted and rT<=10, reached=reached,
                        is_success=bool(both), stage3_energy_int=energy,
                        terminal_abs=self.sabs.copy(), t_end=self.t)
        return normalize_obs(self.raw_obs), reward, done, False, info

    def close(self):
        if self.eng is not None and self.owns_engine:
            self.eng.quit()
        self.eng, self.core = None, None
