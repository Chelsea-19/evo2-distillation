#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evo2_distill.data.cache import build_token_cache, verify_token_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_token_cache(args.output_dir)
    else:
        result = build_token_cache(
            args.drive_root / "frozen_data/targets/window512_canonical_v1.parquet",
            args.drive_root / "frozen_data/targets/window512_targets_v1.parquet",
            args.drive_root / "frozen_data/manifests/fasta_manifest.csv",
            args.drive_root,
            args.output_dir,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

