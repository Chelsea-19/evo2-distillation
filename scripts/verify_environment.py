#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil

import torch

from evo2_distill.training.runtime import environment_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    summary = environment_summary()
    summary["disk_free_gb"] = round(shutil.disk_usage("/").free / 2**30, 3)
    print(json.dumps(summary, indent=2))
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("GPU mode requested but CUDA is unavailable. In Colab choose Runtime > Change runtime type > GPU.")


if __name__ == "__main__":
    main()

