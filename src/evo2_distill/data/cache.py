from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evo2_distill.data.fasta import encode_sequence, extract_window, load_fasta
from evo2_distill.utils.io import atomic_json_dump, sha256_file

WINDOW_LENGTH = 512
ENCODING = {"A": 0, "C": 1, "G": 2, "T": 3, "N/other": 4}


def _byte_lookup() -> np.ndarray:
    lookup = np.full(256, 4, dtype=np.uint8)
    for base, token in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3), (b"N", 4)):
        lookup[base[0]] = token
        lookup[base.lower()[0]] = token
    return lookup


def _resolve_fasta_paths(fasta_manifest: pd.DataFrame, drive_root: Path) -> dict[str, Path]:
    required = {"assembly_id", "drive_relative_path", "sha256"}
    missing = required.difference(fasta_manifest.columns)
    if missing:
        raise ValueError(f"FASTA manifest missing columns: {sorted(missing)}")
    mapping: dict[str, Path] = {}
    for row in fasta_manifest.itertuples(index=False):
        path = drive_root / Path(str(row.drive_relative_path))
        if not path.is_file():
            raise FileNotFoundError(path)
        mapping[str(row.assembly_id)] = path
    if len(mapping) != 296:
        raise ValueError(f"Expected 296 assembly FASTAs; found {len(mapping)}")
    return mapping


def build_token_cache(
    canonical_path: str | Path,
    target_path: str | Path,
    fasta_manifest_path: str | Path,
    drive_root: str | Path,
    output_dir: str | Path,
    validate_samples: int = 256,
) -> dict:
    """Build an all-partition cache keyed exactly to frozen canonical row order.

    Cache construction may include TEST sequence, but this function never computes
    predictions or metrics. Downstream Phase 4.5 datasets enforce split gates.
    """
    canonical_path, target_path = Path(canonical_path), Path(target_path)
    fasta_manifest_path, drive_root = Path(fasta_manifest_path), Path(drive_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_columns = [
        "window_id", "assembly_id", "replicon_id", "start", "end", "window_length",
        "gc_content", "k4_rarity", "k6_rarity", "cluster_id", "split",
    ]
    target_columns = [
        "window_id", "raw_ppl", "absolute_residual", "within_genome_absolute_residual_rank"
    ]
    canonical = pd.read_parquet(canonical_path, columns=canonical_columns)
    targets = pd.read_parquet(target_path, columns=target_columns)
    if len(canonical) != 2_347_502 or len(targets) != 2_347_502:
        raise ValueError(f"Frozen row-count mismatch: canonical={len(canonical)}, targets={len(targets)}")
    if not canonical["window_id"].equals(targets["window_id"]):
        raise ValueError("Canonical and target tables are not in identical window_id order")
    if not np.all(canonical["window_length"].to_numpy() == WINDOW_LENGTH):
        raise ValueError("A non-512-bp window was found")

    fasta_manifest = pd.read_csv(fasta_manifest_path)
    fasta_paths = _resolve_fasta_paths(fasta_manifest, drive_root)
    if set(canonical["assembly_id"].unique()) != set(fasta_paths):
        raise ValueError("Canonical assembly IDs do not match the 296-FASTA manifest")

    cache_final = output_dir / "window_tokens.uint8.mmap"
    cache_tmp = output_dir / "window_tokens.uint8.mmap.partial"
    cache_tmp.unlink(missing_ok=True)
    tokens = np.memmap(cache_tmp, mode="w+", dtype=np.uint8, shape=(len(canonical), WINDOW_LENGTH))
    lookup = _byte_lookup()
    offsets = np.arange(WINDOW_LENGTH, dtype=np.int64)

    for assembly_id, assembly_rows in canonical.groupby("assembly_id", sort=True):
        records = load_fasta(fasta_paths[str(assembly_id)])
        for replicon_id, rows in assembly_rows.groupby("replicon_id", sort=False):
            replicon_id = str(replicon_id)
            if replicon_id not in records:
                raise KeyError(f"{assembly_id}/{replicon_id} absent from FASTA")
            encoded_replicon = lookup[np.frombuffer(records[replicon_id].encode("ascii"), dtype=np.uint8)]
            starts = rows["start"].to_numpy(dtype=np.int64) - 1
            ends = rows["end"].to_numpy(dtype=np.int64)
            if np.any(ends - starts != WINDOW_LENGTH) or np.any(starts < 0) or np.any(ends > len(encoded_replicon)):
                raise ValueError(f"Invalid coordinates for {assembly_id}/{replicon_id}")
            positions = rows.index.to_numpy(dtype=np.int64)
            tokens[positions] = encoded_replicon[starts[:, None] + offsets[None, :]]
    tokens.flush()
    del tokens
    os.replace(cache_tmp, cache_final)

    metadata = canonical.copy()
    metadata.insert(0, "cache_index", np.arange(len(metadata), dtype=np.int64))
    for column in target_columns[1:]:
        metadata[column] = targets[column].to_numpy()
    metadata_path = output_dir / "token_cache_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False, compression="zstd")

    sample_positions = np.unique(
        np.linspace(0, len(metadata) - 1, num=min(validate_samples, len(metadata)), dtype=np.int64)
    )
    cache = np.memmap(cache_final, mode="r", dtype=np.uint8, shape=(len(metadata), WINDOW_LENGTH))
    validated = 0
    record_cache: dict[str, dict[str, str]] = {}
    for position in sample_positions:
        row = metadata.iloc[int(position)]
        assembly_id = str(row["assembly_id"])
        if assembly_id not in record_cache:
            record_cache[assembly_id] = load_fasta(fasta_paths[assembly_id])
        sequence = extract_window(
            record_cache[assembly_id][str(row["replicon_id"])], int(row["start"]), int(row["end"])
        )
        np.testing.assert_array_equal(cache[int(position)], encode_sequence(sequence))
        validated += 1
    del cache

    manifest = {
        "format_version": "token_cache_v1",
        "shape": [len(metadata), WINDOW_LENGTH],
        "dtype": "uint8",
        "encoding": ENCODING,
        "window_ordering_definition": "row order of frozen window512_canonical_v1.parquet; cache_index is zero-based row position",
        "canonical_sha256": sha256_file(canonical_path),
        "source_target_sha256": sha256_file(target_path),
        "fasta_manifest_sha256": sha256_file(fasta_manifest_path),
        "cache_sha256": sha256_file(cache_final),
        "metadata_sha256": sha256_file(metadata_path),
        "validated_sample_count": validated,
        "partition_note": "Cache contains sequence tokens only for all frozen rows; Phase 4.5 code gates TEST.",
    }
    atomic_json_dump(manifest, output_dir / "token_cache_manifest.json")
    return manifest


def verify_token_cache(cache_dir: str | Path) -> dict:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "token_cache_manifest.json").read_text(encoding="utf-8"))
    expected_size = int(np.prod(manifest["shape"])) * np.dtype(manifest["dtype"]).itemsize
    cache_path = cache_dir / "window_tokens.uint8.mmap"
    if cache_path.stat().st_size != expected_size:
        raise ValueError("Token-cache size does not match manifest shape/dtype")
    if sha256_file(cache_path) != manifest["cache_sha256"]:
        raise ValueError("Token-cache SHA256 mismatch")
    metadata_path = cache_dir / "token_cache_metadata.parquet"
    if sha256_file(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("Token-cache metadata SHA256 mismatch")
    return manifest

