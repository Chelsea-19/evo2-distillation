#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from evo2_distill.utils.io import atomic_copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=Path("/content/evo2-distillation-data"))
    parser.add_argument("--include-fasta", action="store_true", help="Required only when a valid complete token cache is absent")
    args = parser.parse_args()
    started = time.perf_counter()
    selected = ["frozen_data/targets", "frozen_data/splits", "frozen_data/manifests", "caches/tokens", "models/phase1"]
    if args.include_fasta:
        selected.append("frozen_data/fasta")
    copied = 0
    bytes_copied = 0
    for relative in selected:
        source = args.drive_root / relative
        if not source.exists():
            continue
        for file in source.rglob("*"):
            if file.is_file():
                destination = args.local_root / file.relative_to(args.drive_root)
                if destination.is_file() and destination.stat().st_size == file.stat().st_size:
                    continue
                atomic_copy(file, destination)
                copied += 1
                bytes_copied += file.stat().st_size
    free_gib = shutil.disk_usage(args.local_root).free / 2**30
    print(f"files copied: {copied}")
    print(f"GiB copied: {bytes_copied / 2**30:.3f}")
    print(f"copy time seconds: {time.perf_counter() - started:.1f}")
    print(f"available disk GiB: {free_gib:.3f}")


if __name__ == "__main__":
    main()

