"""Box mapping followed by direction-preserving 8g ball projection."""
import numpy as np

AMAX = 8 * 9.81

def project_action(action):
    a = np.asarray(action, dtype=np.float64)
    if a.shape[-1:] != (2,) or not np.all(np.isfinite(a)):
        raise ValueError("Action must have two finite components")
    box = np.clip(a, -1., 1.)
    norm = np.linalg.norm(box, axis=-1, keepdims=True)
    executed = box / np.maximum(norm, 1.)
    count = int(np.count_nonzero(np.any(np.abs(executed - a) > 1e-12, axis=-1)))
    return executed, count

def action_norm_to_phys(action, clip=False, return_count=False):
    a = np.asarray(action, dtype=np.float64)
    count = 0
    if clip:
        a, count = project_action(a)
    result = a * AMAX
    return (result, count) if return_count else result

def phys_to_action_norm(accel, clip=False, return_count=False):
    result = np.asarray(accel, dtype=np.float64) / AMAX
    count = 0
    if clip:
        result, count = project_action(result)
    return (result, count) if return_count else result
