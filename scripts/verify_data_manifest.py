#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from evo2_distill.data.safety import verify_test_lock
from evo2_distill.utils.io import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--verify-fasta-hashes", action="store_true")
    args = parser.parse_args()
    root = args.drive_root
    manifest_dir = root / "frozen_data" / "manifests"
    target = root / "frozen_data" / "targets" / "window512_targets_v1.parquet"
    canonical = root / "frozen_data" / "targets" / "window512_canonical_v1.parquet"
    split_path = root / "frozen_data" / "splits" / "dissertation_split_v1.csv"
    fasta_manifest_path = manifest_dir / "fasta_manifest.csv"
    dataset_manifest = json.loads((manifest_dir / "dataset_manifest_v1.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((manifest_dir / "split_manifest_v1.json").read_text(encoding="utf-8"))
    verify_test_lock(manifest_dir / "TEST_SET_LOCKED.txt")

    if pq.ParquetFile(target).metadata.num_rows != 2_347_502:
        raise SystemExit("Frozen target row count is not 2,347,502")
    if pq.ParquetFile(canonical).metadata.num_rows != 2_347_502:
        raise SystemExit("Canonical feature row count is not 2,347,502")
    if sha256_file(target) != dataset_manifest["generated_artifacts"]["target_parquet"]["sha256"]:
        raise SystemExit("Frozen target SHA256 mismatch")

    split = pd.read_csv(split_path)
    counts = split.groupby("split")["assembly_id"].nunique().to_dict()
    if counts != {"development": 207, "test": 44, "validation": 45}:
        raise SystemExit(f"Split counts mismatch: {counts}")
    crossings = int((split.groupby("cluster_id")["split"].nunique() > 1).sum())
    if crossings != 0 or split_manifest["validation"]["cluster_crossings"] != 0:
        raise SystemExit("Mash cluster crossing detected")

    fasta = pd.read_csv(fasta_manifest_path)
    if fasta["assembly_id"].nunique() != 296 or len(fasta) != 296:
        raise SystemExit("FASTA manifest is not exactly one file for each of 296 assemblies")
    for row in fasta.itertuples(index=False):
        path = root / Path(str(row.drive_relative_path))
        if not path.is_file() or path.stat().st_size != int(row.file_size):
            raise SystemExit(f"Missing or wrong-size FASTA: {path}")
        if args.verify_fasta_hashes and sha256_file(path) != row.sha256:
            raise SystemExit(f"FASTA SHA256 mismatch: {path}")
    print(json.dumps({"status": "PASS", "target_rows": 2_347_502, "fasta_assemblies": 296, "split_counts": counts, "cluster_crossings": 0, "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()

