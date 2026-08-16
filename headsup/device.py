"""Device selection: Apple GPU (MPS) > CUDA > CPU, overridable via HEADSUP_DEVICE."""

import os

import torch


def get_device(preferred: str | None = None) -> torch.device:
    name = preferred or os.environ.get("HEADSUP_DEVICE", "auto")
    if name == "auto":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    if name.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available in this torch build")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
