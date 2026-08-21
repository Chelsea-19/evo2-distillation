#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from evo2_distill.utils.io import atomic_copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--drive-results-root", type=Path, required=True)
    args = parser.parse_args()
    destination_root = args.drive_results_root / args.local_run.name
    copied = 0
    for source in args.local_run.rglob("*"):
        if source.is_file():
            atomic_copy(source, destination_root / source.relative_to(args.local_run))
            copied += 1
    print(f"Atomically synced {copied} files to {destination_root}")


if __name__ == "__main__":
    main()

