from __future__ import annotations

import os
import platform
import random
from contextlib import nullcontext

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU mode requested, but CUDA is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_precision(device: torch.device, requested: str = "auto") -> str:
    if device.type != "cuda":
        return "fp32"
    if requested in {"bf16", "fp16", "fp32"}:
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but unsupported by this GPU")
        return requested
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def environment_summary() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3) if cuda else 0.0,
        "cpu_count": os.cpu_count(),
    }

