"""验证 PyTorch 强化学习 + MATLAB Engine 环境"""
import sys


def check(name, fn):
    try:
        result = fn()
        print(f"[OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


print("=" * 50)
print("Python:", sys.executable)
print("Version:", sys.version.split()[0])
print("=" * 50)

ok = True

ok &= check("NumPy", lambda: __import__("numpy").__version__)


def check_torch():
    import torch

    cuda = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if cuda else "CPU"
    return f"{torch.__version__}, CUDA={cuda}, device={dev}"


ok &= check("PyTorch", check_torch)
ok &= check("Gymnasium", lambda: __import__("gymnasium").__version__)


def check_gym_env():
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    env.close()
    return f"CartPole-v1 obs shape={obs.shape}"


ok &= check("Gymnasium env", check_gym_env)


def check_matlab():
    import matlab.engine

    eng = matlab.engine.start_matlab()
    ver = eng.version()
    eng.quit()
    return f"MATLAB {ver}"


ok &= check("MATLAB Engine", check_matlab)

print("=" * 50)
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
