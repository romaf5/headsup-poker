import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_POLICY_PATH = os.path.join(MODELS_DIR, "deepcfr_policy.pth")  # DeepCFR average-strategy net (history features, paper net)
DEFAULT_BLUEPRINT_PATH = os.path.join(MODELS_DIR, "blueprint_nlhe.pt")  # tabular Pluribus-style blueprint (play-only counters)
