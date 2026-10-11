"""Trajectory-based discriminator data; success is recomputed from facts."""
from dataclasses import dataclass
import numpy as np
import scipy.io
from .common import META, normalize_obs, fingerprint

def load_library(path):
    data = scipy.io.loadmat(path, simplify_cells=True)['results']
    for key, value in META.items():
        if data['meta'].get(key) != value:
            raise ValueError(f'Expert metadata mismatch: {key}')
    entries = data['trajectories']
    if isinstance(entries, dict):
        entries = [entries]
    return data, list(np.asarray(entries, dtype=object).reshape(-1))

def transition_rows(tr, key, dim):
    array = np.asarray(tr[key], dtype=np.float64)
    return array.reshape(dim, -1).T

def validate_transition(tr):
    state = transition_rows(tr,'state',13)
    action = transition_rows(tr,'a_norm',2)
    ns = transition_rows(tr,'next_state',13)
    done = np.asarray(tr['done'], dtype=bool).reshape(-1)
    if not len(state) or len(action)!=len(state) or len(ns)!=len(state):
        raise ValueError('Inconsistent expert transition lengths')
    if len(done)!=len(state) or done.sum()!=1 or not done[-1]:
        raise ValueError('Expected exactly one terminal transition at the end')
    if not np.all(np.isfinite(np.concatenate((state,action,ns),axis=1))):
        raise ValueError('Nonfinite expert transitions')
    if np.any(np.linalg.norm(action,axis=1)>1+1e-10):
        raise ValueError('Expert acceleration violates 8g')
    abs_rows = transition_rows(tr,'next_state_abs',18)
    if not np.array_equal(abs_rows[-1], np.asarray(tr['terminal_abs']).reshape(18)):
        raise ValueError('terminal_abs is not the executed last state')
    return state, action, ns

@dataclass
class ExpertData:
    train_s: np.ndarray
    train_a: np.ndarray
    val_s: np.ndarray
    val_a: np.ndarray
    train_ids: list
    val_ids: list
    expert_fingerprint: str

def load_expert_data(path, sidecar, seed=0, val_fraction=.2):
    _, entries = load_library(path)
    cache = np.load(sidecar, allow_pickle=False)
    for key,value in META.items():
        if key not in cache.files or str(cache[key])!=value:
            raise ValueError(f'Terminal cache metadata mismatch: {key}')
    fp = fingerprint(path)
    if str(cache['expert_fingerprint']) != fp or float(cache['stage3_dt']) != .001:
        raise ValueError('Terminal cache fingerprint or stage3 dt mismatch')
    accepted = [e for e in entries if e['ok_use']]
    if len(cache['r_minT']) != len(accepted):
        raise ValueError('Terminal cache length does not match accepted trajectories')
    good=[]
    for i,e in enumerate(accepted):
        tr=e['transitions']; s,a,ns=validate_transition(tr)
        intercepted=bool(np.asarray(tr['intercepted']).any())
        # Never trust cached both_ok/r_min1/r_min2 columns.
        if not intercepted and min(ns[:,0].min(),ns[:,4].min())>=6 and cache['r_minT'][i]<=10:
            good.append((e['row_idx'],normalize_obs(s),a.astype(np.float32)))
    if len(good)<2:
        raise ValueError(f'Need at least two both_ok trajectories for train/val; found {len(good)}')
    order=np.random.default_rng(seed).permutation(len(good))
    nv=max(1,min(len(good)-1,round(len(good)*val_fraction)))
    val=[good[i] for i in order[:nv]]; train=[good[i] for i in order[nv:]]
    return ExpertData(np.concatenate([g[1] for g in train]),np.concatenate([g[2] for g in train]),
                      np.concatenate([g[1] for g in val]),np.concatenate([g[2] for g in val]),
                      [int(g[0]) for g in train],[int(g[0]) for g in val],fp)
