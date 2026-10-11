from pathlib import Path
import hashlib
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
# Only the explicitly shared observation/reward/evaluation modules are imported.
sys.path.append(str(ROOT / "rl_td3"))
from obs_norm import OBS_SCALE, OBS_BOX_HIGH, normalize_obs
from reward_projection import DEFAULT_REWARD_CFG, project_terminal

META = dict(schema_version="ps_schema_v1",
            stage2_dynamics_version="ps_direct_accel_v1",
            action_mapping_version="ps_direct_accel_v1",
            obs_norm_version="obs_scale_0714_v1",
            reward_version=DEFAULT_REWARD_CFG.version)

def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def output_path(path):
    """Enforce branch-only writes, including caller-supplied destinations."""
    path = Path(path).resolve()
    if not path.is_relative_to(PACKAGE):
        raise ValueError(f"Output must be inside {PACKAGE}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
