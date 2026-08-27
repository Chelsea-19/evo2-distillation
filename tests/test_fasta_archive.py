from __future__ import annotations

import csv
import gzip
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_restore_script_rejects_wrong_archive_hash_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    manifest_dir = root / "frozen_data" / "manifests"
    manifest_dir.mkdir(parents=True)
    rows = []
    for index in range(296):
        payload = f">record_{index}\nACGT\n".encode()
        relative = f"frozen_data/fasta/A{index:03d}/A{index:03d}.fna"
        rows.append(
            {
                "assembly_id": f"A{index:03d}",
                "drive_relative_path": relative,
                "file_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    with (manifest_dir / "fasta_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    archive_path = tmp_path / "fasta.tar.gz"
    payload = b">record_0\nACGT\n"
    source = tmp_path / "A000.fna"
    source.write_bytes(payload)
    with gzip.open(archive_path, "wb") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            archive.add(source, arcname="frozen_data/fasta/A000/A000.fna")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "restore_fasta_archive.py"),
            "--drive-root",
            str(root),
            "--archive",
            str(archive_path),
            "--expected-archive-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "archive SHA256 mismatch" in result.stderr
    assert not (root / "frozen_data" / "fasta" / "A000" / "A000.fna").exists()
