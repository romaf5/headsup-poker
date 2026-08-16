import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_POLICY_PATH = os.path.join(MODELS_DIR, "deepcfr_policy.pth")
DEFAULT_ONNX_PATH = os.path.join(MODELS_DIR, "rl_games_exploiter.onnx")
