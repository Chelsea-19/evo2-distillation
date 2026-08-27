#!/usr/bin/env python
"""Restore missing/wrong-size frozen FASTA files from a verified archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(root: Path) -> dict[str, dict[str, str]]:
    path = root / "frozen_data" / "manifests" / "fasta_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 296:
        raise SystemExit(f"Expected 296 FASTA manifest rows, found {len(rows)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        relative = PurePosixPath(row["drive_relative_path"])
        if relative.parts[:2] != ("frozen_data", "fasta") or ".." in relative.parts:
            raise SystemExit(f"Unsafe FASTA manifest path: {relative}")
        result[relative.as_posix()] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    args = parser.parse_args()

    root = args.drive_root.resolve()
    manifest = read_manifest(root)
    problems: list[str] = []
    for relative, row in manifest.items():
        destination = root.joinpath(*PurePosixPath(relative).parts)
        if not destination.is_file() or destination.stat().st_size != int(row["file_size"]):
            problems.append(relative)

    if not problems:
        print(json.dumps({"status": "PASS", "restored": 0, "fasta_files": 296, "test_accessed": False}))
        return

    archive_path = args.archive.resolve()
    if not archive_path.is_file():
        raise SystemExit(
            f"{len(problems)} FASTA files need repair, but archive is missing: {archive_path}"
        )
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != args.expected_archive_sha256.lower():
        raise SystemExit(
            f"FASTA archive SHA256 mismatch: expected {args.expected_archive_sha256.lower()}, got {archive_sha256}"
        )

    restored = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        unexpected = sorted(set(members) - set(manifest))
        if unexpected:
            raise SystemExit(f"Archive contains unexpected path: {unexpected[0]}")
        for relative in problems:
            row = manifest[relative]
            member = members.get(relative)
            if member is None:
                raise SystemExit(f"Archive is missing required FASTA: {relative}")
            expected_size = int(row["file_size"])
            if member.size != expected_size:
                raise SystemExit(f"Wrong-size FASTA inside archive: {relative}")

            destination = root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            temporary.unlink(missing_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"Cannot read FASTA from archive: {relative}")
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size != expected_size or sha256_file(temporary) != row["sha256"]:
                temporary.unlink(missing_ok=True)
                raise SystemExit(f"Restored FASTA failed manifest verification: {relative}")
            os.replace(temporary, destination)
            restored += 1
            if restored % 25 == 0 or restored == len(problems):
                print(f"Restored {restored}/{len(problems)} FASTA files", flush=True)

    print(json.dumps({"status": "PASS", "restored": restored, "fasta_files": 296, "test_accessed": False}))


if __name__ == "__main__":
    main()
