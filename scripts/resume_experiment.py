#!/usr/bin/env python
from __future__ import annotations

import argparse

from evo2_distill.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(run_training(args.config, args.checkpoint))


if __name__ == "__main__":
    main()

