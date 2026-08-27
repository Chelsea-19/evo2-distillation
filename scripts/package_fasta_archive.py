#!/usr/bin/env python
"""Build a deterministic, manifest-verified FASTA transfer archive."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "frozen_data" / "manifests" / "fasta_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 296:
        raise SystemExit(f"Expected 296 FASTA manifest rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.source_root.resolve()
    rows = read_manifest(root)
    verified: list[tuple[str, Path, int]] = []
    total_bytes = 0
    for row in sorted(rows, key=lambda item: item["drive_relative_path"]):
        relative = PurePosixPath(row["drive_relative_path"])
        if relative.parts[:2] != ("frozen_data", "fasta") or ".." in relative.parts:
            raise SystemExit(f"Unsafe FASTA manifest path: {relative}")
        source = root.joinpath(*relative.parts)
        expected_size = int(row["file_size"])
        if not source.is_file() or source.stat().st_size != expected_size:
            raise SystemExit(f"Missing or wrong-size source FASTA: {source}")
        actual_sha256 = sha256_file(source)
        if actual_sha256 != row["sha256"]:
            raise SystemExit(f"Source FASTA SHA256 mismatch: {source}")
        verified.append((relative.as_posix(), source, expected_size))
        total_bytes += expected_size

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative, source, _ in verified:
                    info = archive.gettarinfo(str(source), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
    os.replace(temporary, output)

    archive_sha256 = sha256_file(output)
    metadata = {
        "archive": output.name,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": output.stat().st_size,
        "fasta_files": len(verified),
        "uncompressed_fasta_bytes": total_bytes,
        "paths_root": "frozen_data/fasta",
        "test_accessed": False,
    }
    metadata_path = output.with_name(output.name + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
