"""Create the frozen Phase 0 foundation for dissertation_v1.

This script reads archived source data but writes only beneath
experiments/dissertation_v1 and outputs/phase0. It does not train or evaluate a
model, run Evo 2, use biological annotations, or inspect model outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


VERSION = "dissertation_v1"
PHASE = "phase0"
SEED = 20260820
LOWESS_SPAN = 0.3
EXPECTED_WINDOWS = 2_347_502
EXPECTED_ASSEMBLIES = 296
EXPECTED_CLUSTERS = 73
COORDINATE_CONVENTION = "1-based inclusive"

WORKSPACE = Path(r"C:\Users\LDD\Documents\Codex\2026-08-19\za")
ROOT = WORKSPACE / "experiments" / VERSION
PROTOCOL_DIR = ROOT / "protocol"
SPLIT_DIR = ROOT / "splits"
TARGET_DIR = ROOT / "targets"
AUDIT_DIR = ROOT / "audits"
DATA_DIR = ROOT / "data"
OUTPUT_MIRROR = WORKSPACE / "outputs" / "phase0"

PROJECT = Path(r"D:\BMDSIS")
EVO_ROOT = PROJECT / "Evo2_perplexity"
SUBMISSION = EVO_ROOT / "GitHub submission"
WINDOW_SOURCE = EVO_ROOT / "outputs" / "tables" / "window_master_512_with_sequence_baselines.csv"
FASTA_INVENTORY = EVO_ROOT / "outputs" / "tables" / "fasta_header_inventory.csv"
CLUSTER_SOURCE = SUBMISSION / "validation_extension" / "splits" / "assembly_cluster_manifest.csv"
LOWESS_REFERENCE = SUBMISSION / "scripts" / "figures" / "plot_lowess_residual_probability.py"
HELDOUT_LOWESS_REFERENCE = SUBMISSION / "scripts" / "validation" / "fit_cluster_balanced_heldout_lowess.py"
PREFLIGHT_REPORT = WORKSPACE / "docs" / "dissertation_preflight_audit.md"
PREFLIGHT_MANIFEST = WORKSPACE / "docs" / "dissertation_preflight_manifest.json"
PREFLIGHT_LINEAGE = WORKSPACE / "docs" / "dissertation_lineage_inventory.csv"

PROTOCOL_PATH = PROTOCOL_DIR / "experiment_protocol_v1.yaml"
DATASET_MANIFEST_PATH = PROTOCOL_DIR / "dataset_manifest_v1.json"
SPLIT_CSV_PATH = SPLIT_DIR / "dissertation_split_v1.csv"
SPLIT_MANIFEST_PATH = SPLIT_DIR / "split_manifest_v1.json"
LOCK_PATH = SPLIT_DIR / "TEST_SET_LOCKED.txt"
TARGET_PATH = TARGET_DIR / "window512_targets_v1.parquet"
CURVE_PATH = TARGET_DIR / "dev_lowess_v1.csv"
CURVE_METADATA_PATH = TARGET_DIR / "dev_lowess_v1.json"
CANONICAL_PATH = DATA_DIR / "window512_canonical_v1.parquet"
LEAKAGE_PATH = AUDIT_DIR / "phase0_leakage_audit.md"
COMPLETION_PATH = ROOT / "phase0_completion_report.md"

FINAL_OUTPUTS = [
    PROTOCOL_PATH,
    DATASET_MANIFEST_PATH,
    SPLIT_CSV_PATH,
    SPLIT_MANIFEST_PATH,
    LOCK_PATH,
    TARGET_PATH,
    CURVE_PATH,
    CURVE_METADATA_PATH,
    CANONICAL_PATH,
    LEAKAGE_PATH,
    COMPLETION_PATH,
]


class Phase0Error(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, path)


def source_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase0Error(f"Required source is absent: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ensure_clean_destination() -> None:
    existing = [str(path) for path in FINAL_OUTPUTS if path.exists()]
    if existing:
        raise Phase0Error(
            "Frozen Phase 0 outputs already exist; refusing to overwrite:\n" + "\n".join(existing)
        )
    for directory in (PROTOCOL_DIR, SPLIT_DIR, TARGET_DIR, AUDIT_DIR, DATA_DIR, OUTPUT_MIRROR):
        directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(WORKSPACE).free
    if free < 600 * 1024 * 1024:
        raise Phase0Error(f"Less than 600 MiB free on workspace volume: {free} bytes")


def select_new_cluster_allocation(cluster_frame: pd.DataFrame) -> dict[str, str]:
    """Allocate using cluster identity and size only; do not read historical split."""
    clusters = (
        cluster_frame.groupby("cluster_id", sort=True)
        .agg(cluster_size=("assembly_id", "size"))
        .reset_index()
    )
    if len(clusters) != EXPECTED_CLUSTERS or int(clusters["cluster_size"].sum()) != EXPECTED_ASSEMBLIES:
        raise Phase0Error("Unexpected cluster inventory")
    if int(clusters["cluster_size"].max()) != 124:
        raise Phase0Error("Largest Mash cluster is not the expected 124 assemblies")

    records = [(str(row.cluster_id), int(row.cluster_size)) for row in clusters.itertuples(index=False)]
    largest_id, largest_size = max(records, key=lambda item: (item[1], item[0]))
    if largest_size > 45:
        forced_development = {largest_id}
    else:
        forced_development = set()

    singleton_ids = sorted(cluster_id for cluster_id, size in records if size == 1)
    multi = [(cluster_id, size) for cluster_id, size in records if size > 1 and cluster_id not in forced_development]
    rng = random.Random(SEED)
    best: tuple[tuple[int, ...], list[tuple[str, int]], list[tuple[str, int]]] | None = None

    # Select six independent multi-assembly clusters per held-out partition,
    # then fill exact assembly targets with seeded singleton clusters. The
    # search objective uses sizes only and avoids a held-out partition being
    # dominated by the 22-assembly component where possible.
    for iteration in range(250_000):
        sampled = rng.sample(multi, 12)
        validation_multi = sampled[:6]
        test_multi = sampled[6:]
        validation_multi_sum = sum(size for _, size in validation_multi)
        test_multi_sum = sum(size for _, size in test_multi)
        validation_singletons = 45 - validation_multi_sum
        test_singletons = 44 - test_multi_sum
        if not (6 <= validation_singletons <= 18 and 6 <= test_singletons <= 18):
            continue
        if validation_singletons + test_singletons > len(singleton_ids):
            continue
        max_validation = max(size for _, size in validation_multi)
        max_test = max(size for _, size in test_multi)
        score = (
            1000 * int(max_validation > 14 or max_test > 14),
            abs(validation_multi_sum - 32) + abs(test_multi_sum - 31),
            abs(validation_singletons - test_singletons),
            abs(max_validation - max_test),
            iteration,
        )
        if best is None or score < best[0]:
            best = (score, validation_multi, test_multi)
            if score[:4] == (0, 0, 1, 0):
                break

    if best is None:
        raise Phase0Error("Could not construct a legal seeded cluster allocation")

    _, validation_multi, test_multi = best
    validation_need = 45 - sum(size for _, size in validation_multi)
    test_need = 44 - sum(size for _, size in test_multi)
    shuffled_singletons = singleton_ids.copy()
    rng.shuffle(shuffled_singletons)
    validation_ids = {cluster_id for cluster_id, _ in validation_multi}
    validation_ids.update(shuffled_singletons[:validation_need])
    test_ids = {cluster_id for cluster_id, _ in test_multi}
    test_ids.update(shuffled_singletons[validation_need : validation_need + test_need])

    allocation: dict[str, str] = {}
    for cluster_id, _ in records:
        if cluster_id in validation_ids:
            allocation[cluster_id] = "validation"
        elif cluster_id in test_ids:
            allocation[cluster_id] = "test"
        else:
            allocation[cluster_id] = "development"

    assembly_counts = Counter()
    cluster_counts = Counter(allocation.values())
    for cluster_id, size in records:
        assembly_counts[allocation[cluster_id]] += size
    if dict(assembly_counts) != {"development": 207, "validation": 45, "test": 44}:
        raise Phase0Error(f"Unexpected seeded allocation: {dict(assembly_counts)}")
    if cluster_counts["validation"] < 8 or cluster_counts["test"] < 8:
        raise Phase0Error("Held-out partitions do not contain enough independent clusters")
    return allocation


def lowess_by_discrete_gc(gc: np.ndarray, y: np.ndarray, span: float) -> pd.DataFrame:
    """Archived weighted local-linear LOWESS over the discrete GC grid."""
    grouped = (
        pd.DataFrame({"gc": gc, "y": y})
        .groupby("gc", sort=True)
        .agg(n=("y", "size"), ybar=("y", "mean"))
        .reset_index()
    )
    x = grouped["gc"].to_numpy(float)
    n = grouped["n"].to_numpy(float)
    ybar = grouped["ybar"].to_numpy(float)
    total_n = int(n.sum())
    k = int(math.ceil(span * total_n))
    fitted = np.empty_like(x)
    for index, center in enumerate(x):
        distance = np.abs(x - center)
        order = np.argsort(distance, kind="mergesort")
        cumulative = np.cumsum(n[order])
        bandwidth = distance[order[np.searchsorted(cumulative, k, side="left")]]
        if bandwidth == 0:
            mask = distance == 0
            fitted[index] = np.average(ybar[mask], weights=n[mask])
            continue
        scaled = distance / bandwidth
        mask = scaled <= 1
        tricube = (1 - scaled[mask] ** 3) ** 3
        weights = tricube * n[mask]
        centered = x[mask] - center
        values = ybar[mask]
        s0 = weights.sum()
        s1 = np.sum(weights * centered)
        s2 = np.sum(weights * centered * centered)
        t0 = np.sum(weights * values)
        t1 = np.sum(weights * centered * values)
        denominator = s0 * s2 - s1 * s1
        if abs(denominator) < 1e-15:
            fitted[index] = t0 / s0
        else:
            fitted[index] = (t0 * s2 - t1 * s1) / denominator
    grouped["lowess_ppl"] = fitted
    grouped["span"] = span
    grouped["effective_k_windows"] = k
    grouped["fit_partition"] = "development_only"
    grouped["development_windows_used"] = total_n
    return grouped


def apply_frozen_curve(gc: np.ndarray, curve: pd.DataFrame) -> np.ndarray:
    ordered = curve.sort_values("gc", kind="mergesort")
    return np.interp(
        gc.astype(float),
        ordered["gc"].to_numpy(float),
        ordered["lowess_ppl"].to_numpy(float),
    )


def load_fasta_and_hash(path: Path) -> tuple[dict[str, str], str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            line = raw_line.decode("ascii").strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                if current in records:
                    raise Phase0Error(f"Duplicate FASTA record {current} in {path}")
                records[current] = []
            else:
                if current is None:
                    raise Phase0Error(f"Sequence before FASTA header in {path}")
                records[current].append(line.upper())
    return {key: "".join(parts) for key, parts in records.items()}, digest.hexdigest()


def packed_increment(value: int, split_index: int) -> int:
    return value + (1 << (24 * split_index))


def duplicate_summary(packed_counts: dict[bytes, int]) -> dict[str, Any]:
    pair_names = [(0, 1, "development_validation"), (0, 2, "development_test"), (1, 2, "validation_test")]
    shared_classes = Counter()
    window_pairs = Counter()
    any_shared = 0
    for packed in packed_counts.values():
        counts = [(packed >> (24 * index)) & ((1 << 24) - 1) for index in range(3)]
        active = sum(count > 0 for count in counts)
        if active >= 2:
            any_shared += 1
        for left, right, name in pair_names:
            if counts[left] and counts[right]:
                shared_classes[name] += 1
                window_pairs[name] += counts[left] * counts[right]
    return {
        "unique_hash_classes_total": len(packed_counts),
        "unique_hash_classes_shared_by_two_or_more_partitions": any_shared,
        "shared_hash_classes_by_partition_pair": dict(shared_classes),
        "cross_partition_window_pairs_by_partition_pair": dict(window_pairs),
        "definition": "A shared hash class is one distinct 512-bp sequence identity observed in both named partitions; window-pair counts sum n_left*n_right per shared class.",
    }


def main() -> None:
    started = time.perf_counter()
    ensure_clean_destination()
    created_utc = datetime.now(timezone.utc).isoformat()
    print("Phase 0: hashing authoritative source tables/manifests", flush=True)

    source_paths = [
        WINDOW_SOURCE,
        FASTA_INVENTORY,
        CLUSTER_SOURCE,
        LOWESS_REFERENCE,
        HELDOUT_LOWESS_REFERENCE,
        PREFLIGHT_REPORT,
        PREFLIGHT_MANIFEST,
        PREFLIGHT_LINEAGE,
    ]
    source_hashes = {path.name: source_record(path) for path in source_paths}

    # The historical split column is intentionally not loaded.
    cluster_frame = pd.read_csv(
        CLUSTER_SOURCE,
        usecols=["assembly_id", "cluster_id", "cluster_size"],
        dtype={"assembly_id": "string", "cluster_id": "string", "cluster_size": "int16"},
    )
    if len(cluster_frame) != EXPECTED_ASSEMBLIES or cluster_frame["assembly_id"].duplicated().any():
        raise Phase0Error("Cluster source does not contain 296 unique assemblies")
    if not cluster_frame.groupby("cluster_id")["cluster_size"].apply(
        lambda values: values.nunique() == 1
    ).all():
        raise Phase0Error("Cluster size is inconsistent within a cluster")

    allocation = select_new_cluster_allocation(cluster_frame)
    split_frame = cluster_frame[["assembly_id", "cluster_id", "cluster_size"]].copy()
    split_frame["split"] = split_frame["cluster_id"].map(allocation)
    split_frame["split_version"] = VERSION
    split_frame["allocation_seed"] = SEED
    split_frame["mash_distance_threshold"] = 0.01
    split_frame["source_cluster_manifest_sha256"] = source_hashes[CLUSTER_SOURCE.name]["sha256"]
    split_frame = split_frame.sort_values(["split", "cluster_id", "assembly_id"], kind="mergesort").reset_index(drop=True)
    if split_frame["split"].isna().any():
        raise Phase0Error("At least one assembly is unassigned")
    cluster_crossings = int((split_frame.groupby("cluster_id")["split"].nunique() > 1).sum())
    if cluster_crossings:
        raise Phase0Error(f"Mash cluster crossings: {cluster_crossings}")

    print("Phase 0: loading canonical 512-bp columns", flush=True)
    usecols = [
        "assembly_id",
        "replicon_id",
        "start",
        "end",
        "window_length",
        "perplexity",
        "gc_content",
        "n_fraction",
        "kmer4_rarity",
        "kmer6_rarity",
    ]
    windows = pd.read_csv(
        WINDOW_SOURCE,
        usecols=usecols,
        dtype={
            "assembly_id": "string",
            "replicon_id": "string",
            "start": "int64",
            "end": "int64",
            "window_length": "int16",
            "perplexity": "float64",
            "gc_content": "float64",
            "n_fraction": "float64",
            "kmer4_rarity": "float64",
            "kmer6_rarity": "float64",
        },
    )
    if len(windows) != EXPECTED_WINDOWS:
        raise Phase0Error(f"Expected {EXPECTED_WINDOWS} windows, found {len(windows)}")
    if windows[usecols].isna().any().any():
        raise Phase0Error("Required canonical field contains missing values")
    if not windows["window_length"].eq(512).all():
        raise Phase0Error("Canonical layer contains a non-512-bp row")
    if not (windows["end"] - windows["start"] + 1).eq(512).all():
        raise Phase0Error("Coordinate length and declared length disagree")

    assembly_to_split = split_frame.set_index("assembly_id")["split"]
    assembly_to_cluster = split_frame.set_index("assembly_id")["cluster_id"]
    windows["split"] = windows["assembly_id"].map(assembly_to_split)
    windows["cluster_id"] = windows["assembly_id"].map(assembly_to_cluster)
    if windows[["split", "cluster_id"]].isna().any().any():
        raise Phase0Error("At least one window lacks a split or cluster")

    window_counts = windows.groupby("split", observed=True).size().to_dict()
    assembly_counts = split_frame.groupby("split", observed=True)["assembly_id"].nunique().to_dict()
    cluster_counts = split_frame.groupby("split", observed=True)["cluster_id"].nunique().to_dict()
    if sum(window_counts.values()) != EXPECTED_WINDOWS:
        raise Phase0Error("Not every window inherited a split")

    write_csv_atomic(split_frame, SPLIT_CSV_PATH)
    split_csv_hash = sha256_file(SPLIT_CSV_PATH)
    split_summary = {
        split: {
            "assemblies": int(assembly_counts[split]),
            "clusters": int(cluster_counts[split]),
            "windows_512": int(window_counts[split]),
            "assembly_fraction": float(assembly_counts[split] / EXPECTED_ASSEMBLIES),
            "window_fraction": float(window_counts[split] / EXPECTED_WINDOWS),
        }
        for split in ("development", "validation", "test")
    }
    split_manifest = {
        "version": VERSION,
        "created_utc": created_utc,
        "allocation_seed": SEED,
        "assignment_unit": "Mash connected-component cluster",
        "mash_distance_threshold": 0.01,
        "source_cluster_manifest": source_hashes[CLUSTER_SOURCE.name],
        "allocation_input_columns": ["cluster_id", "cluster_size", "assembly_id"],
        "explicitly_unused_for_allocation": [
            "historical split",
            "Evo 2 PPL",
            "GC or GC residual",
            "AMR/MOB/prophage annotations",
            "biological labels",
            "model performance",
            "residual extremes",
            "window counts",
        ],
        "algorithm": {
            "name": "seeded size-only cluster allocation",
            "targets": {"development": 207, "validation": 45, "test": 44},
            "largest_cluster_policy": "A cluster larger than either held-out assembly target is assigned to development.",
            "heldout_policy": "Seeded search selects six multi-assembly clusters per held-out partition and fills exact assembly targets with seeded singleton clusters; cluster integrity precedes proportions.",
            "selection_iterations_max": 250000,
        },
        "summary": split_summary,
        "validation": {
            "assemblies_assigned": int(split_frame["assembly_id"].nunique()),
            "clusters_assigned": int(split_frame["cluster_id"].nunique()),
            "cluster_crossings": cluster_crossings,
            "windows_assigned": int(sum(window_counts.values())),
        },
        "split_csv": {
            "path": str(SPLIT_CSV_PATH),
            "bytes": SPLIT_CSV_PATH.stat().st_size,
            "sha256": split_csv_hash,
        },
        "historical_partition_reused": False,
        "test_policy": "LOCKED; no model outcome may be examined before Phase 5",
    }
    write_json_atomic(SPLIT_MANIFEST_PATH, split_manifest)
    split_manifest_hash = sha256_file(SPLIT_MANIFEST_PATH)
    lock_text = f"""DISSERTATION TEST SET LOCKED

Version: {VERSION}
Timestamp (UTC): {created_utc}
Allocation seed: {SEED}
Split manifest: {SPLIT_MANIFEST_PATH}
Split manifest SHA256: {split_manifest_hash}

The dissertation_v1 test partition is sealed. Test model predictions,
performance metrics, ablations, model comparisons, error analyses, enrichment
outcomes and test-informed tuning must not be examined before Phase 5.

Phase 0 may generate test target values only by applying the frozen
development-fitted LOWESS curve and the declared deterministic within-genome
ranking rule. This is target construction, not model evaluation. The curve,
preprocessing, model design and evaluation protocol must not be refitted or
selected using the test partition.
"""
    write_text_atomic(LOCK_PATH, lock_text)

    print("Phase 0: fitting development-only GC-LOWESS and generating targets", flush=True)
    development_mask = windows["split"].eq("development").to_numpy()
    development_windows_used = int(development_mask.sum())
    curve = lowess_by_discrete_gc(
        windows.loc[development_mask, "gc_content"].to_numpy(float),
        windows.loc[development_mask, "perplexity"].to_numpy(float),
        LOWESS_SPAN,
    )
    if int(curve["n"].sum()) != development_windows_used:
        raise Phase0Error("LOWESS fit rows are not exactly the development windows")
    write_csv_atomic(curve, CURVE_PATH)
    expected = apply_frozen_curve(windows["gc_content"].to_numpy(float), curve)
    raw_ppl = windows["perplexity"].to_numpy(float)
    signed = raw_ppl - expected
    absolute = np.abs(signed)
    if not np.isfinite(expected).all() or not np.isfinite(absolute).all():
        raise Phase0Error("Generated targets contain non-finite values")

    window_id = (
        windows["assembly_id"]
        + "|"
        + windows["replicon_id"]
        + "|"
        + windows["start"].astype("string")
        + "-"
        + windows["end"].astype("string")
    )
    if window_id.duplicated().any():
        raise Phase0Error("window_id is not unique")
    target_frame = pd.DataFrame(
        {
            "window_id": window_id,
            "assembly_id": windows["assembly_id"],
            "replicon_id": windows["replicon_id"],
            "start": windows["start"],
            "end": windows["end"],
            "split": windows["split"],
            "raw_ppl": raw_ppl,
            "gc_content": windows["gc_content"],
            "expected_ppl_dev_lowess": expected,
            "signed_residual": signed,
            "absolute_residual": absolute,
        }
    )
    target_frame["within_genome_absolute_residual_rank"] = target_frame.groupby(
        "assembly_id", sort=False, observed=True
    )["absolute_residual"].rank(method="average", ascending=True, pct=True)
    if len(target_frame) != EXPECTED_WINDOWS or target_frame.isna().any().any():
        raise Phase0Error("Target table is incomplete")
    write_parquet_atomic(target_frame, TARGET_PATH)

    curve_metadata = {
        "version": VERSION,
        "created_utc": created_utc,
        "dependent_variable": "raw Evo 2 perplexity",
        "composition_variable": "GC content",
        "span": LOWESS_SPAN,
        "fit_partition": "development_only",
        "development_assemblies": int(assembly_counts["development"]),
        "development_clusters": int(cluster_counts["development"]),
        "development_windows_used": development_windows_used,
        "validation_or_test_rows_used_for_fit": 0,
        "algorithm": "Weighted local-linear LOWESS over exact discrete GC values. Group development rows by GC; use frequency-weighted tricube local regression with k=ceil(span*N), stable mergesort distance ordering, and local constant fallback when the weighted normal equation is singular.",
        "curve_application": "The development curve is sorted by GC and applied unchanged to all partitions with numpy.interp; values outside development support use the nearest endpoint, matching numpy.interp boundary behavior.",
        "signed_residual": "raw_ppl - expected_ppl_dev_lowess",
        "absolute_residual": "abs(signed_residual)",
        "rank_rule": "Within each assembly/genome, ascending average rank of absolute_residual divided by genome window count (pandas rank method=average, ascending=True, pct=True); larger residuals approach 1.",
        "historical_ppl_percentile_used": False,
        "historical_global_lowess_used": False,
        "source_window_table": source_hashes[WINDOW_SOURCE.name],
        "split_manifest_sha256": split_manifest_hash,
        "reference_implementations": [
            source_hashes[LOWESS_REFERENCE.name],
            source_hashes[HELDOUT_LOWESS_REFERENCE.name],
        ],
        "curve_csv": {
            "path": str(CURVE_PATH),
            "bytes": CURVE_PATH.stat().st_size,
            "sha256": sha256_file(CURVE_PATH),
            "rows": int(len(curve)),
        },
        "target_parquet": {
            "path": str(TARGET_PATH),
            "bytes": TARGET_PATH.stat().st_size,
            "sha256": sha256_file(TARGET_PATH),
            "rows": int(len(target_frame)),
        },
    }
    write_json_atomic(CURVE_METADATA_PATH, curve_metadata)

    print("Phase 0: resolving FASTA sources and auditing cross-partition duplicates", flush=True)
    fasta_inventory = pd.read_csv(
        FASTA_INVENTORY,
        usecols=["assembly_id", "fasta_record_id", "source_fasta_file"],
        dtype="string",
    )
    fasta_sources_by_assembly = fasta_inventory.groupby("assembly_id")["source_fasta_file"].nunique()
    if len(fasta_sources_by_assembly) != EXPECTED_ASSEMBLIES or not fasta_sources_by_assembly.eq(1).all():
        raise Phase0Error("FASTA inventory does not resolve to exactly one source file per assembly")
    relative_fasta = fasta_inventory.groupby("assembly_id", sort=False)["source_fasta_file"].first()
    absolute_fasta = relative_fasta.map(lambda value: str(PROJECT / Path(str(value))))
    windows["fasta_source"] = windows["assembly_id"].map(absolute_fasta)
    if windows["fasta_source"].isna().any():
        raise Phase0Error("At least one canonical window lacks a FASTA source")

    canonical_frame = pd.DataFrame(
        {
            "window_id": window_id,
            "assembly_id": windows["assembly_id"],
            "replicon_id": windows["replicon_id"],
            "start": windows["start"],
            "end": windows["end"],
            "coordinate_convention": COORDINATE_CONVENTION,
            "window_length": windows["window_length"],
            "fasta_source": windows["fasta_source"],
            "raw_evo2_ppl": windows["perplexity"],
            "gc_content": windows["gc_content"],
            "k4_rarity": windows["kmer4_rarity"],
            "k6_rarity": windows["kmer6_rarity"],
            "has_ambiguous_bases": windows["n_fraction"].gt(0),
            "ambiguous_base_count": np.rint(windows["n_fraction"].to_numpy(float) * 512).astype(np.int16),
            "cluster_id": windows["cluster_id"],
            "split": windows["split"],
        }
    )
    ambiguous_count = int(canonical_frame["has_ambiguous_bases"].sum())
    if ambiguous_count != 9:
        raise Phase0Error(f"Expected 9 ambiguous-base windows, found {ambiguous_count}")
    ambiguous_windows = canonical_frame.loc[
        canonical_frame["has_ambiguous_bases"],
        ["window_id", "assembly_id", "replicon_id", "start", "end", "ambiguous_base_count"],
    ].to_dict(orient="records")
    write_parquet_atomic(canonical_frame, CANONICAL_PATH)

    split_index = {"development": 0, "validation": 1, "test": 2}
    complement = str.maketrans("ACGTN", "TGCAN")
    exact_counts: dict[bytes, int] = {}
    canonical_counts: dict[bytes, int] = {}
    fasta_hash_records: list[dict[str, Any]] = []
    coordinate_failures = 0

    grouped_windows = windows.groupby("assembly_id", sort=True, observed=True)
    for assembly_number, (assembly_id, assembly_windows) in enumerate(grouped_windows, start=1):
        fasta_path = Path(str(absolute_fasta.loc[assembly_id]))
        if not fasta_path.is_file():
            raise Phase0Error(f"FASTA source is absent: {fasta_path}")
        sequences, fasta_sha = load_fasta_and_hash(fasta_path)
        fasta_hash_records.append(
            {
                "assembly_id": str(assembly_id),
                "path": str(fasta_path),
                "bytes": fasta_path.stat().st_size,
                "sha256": fasta_sha,
                "records": len(sequences),
            }
        )
        partition = str(assembly_windows["split"].iloc[0])
        partition_index = split_index[partition]
        for row in assembly_windows[["replicon_id", "start", "end"]].itertuples(index=False):
            replicon = str(row.replicon_id)
            sequence = sequences.get(replicon)
            if sequence is None:
                coordinate_failures += 1
                continue
            subsequence = sequence[int(row.start) - 1 : int(row.end)]
            if len(subsequence) != 512:
                coordinate_failures += 1
                continue
            encoded = subsequence.encode("ascii")
            exact_digest = hashlib.sha256(encoded).digest()
            reverse_complement = subsequence.translate(complement)[::-1]
            canonical_sequence = subsequence if subsequence <= reverse_complement else reverse_complement
            canonical_digest = hashlib.sha256(canonical_sequence.encode("ascii")).digest()
            exact_counts[exact_digest] = packed_increment(exact_counts.get(exact_digest, 0), partition_index)
            canonical_counts[canonical_digest] = packed_increment(
                canonical_counts.get(canonical_digest, 0), partition_index
            )
        if assembly_number % 25 == 0 or assembly_number == EXPECTED_ASSEMBLIES:
            print(f"  FASTA/duplicate audit: {assembly_number}/{EXPECTED_ASSEMBLIES} assemblies", flush=True)

    if coordinate_failures:
        raise Phase0Error(f"FASTA reconstruction failures during Phase 0: {coordinate_failures}")
    exact_summary = duplicate_summary(exact_counts)
    rc_summary = duplicate_summary(canonical_counts)

    leakage_report = f"""# Phase 0 leakage audit

Version: `{VERSION}`  
Created UTC: `{created_utc}`  
Split seed: `{SEED}`

## Scope

This audit was run only after creating the new dissertation-specific allocation. It did not use PPL, residuals, annotations, biological labels, historical model performance or residual extremes to design or revise the split. Duplicate results are reported as sensitivity-analysis information and did not trigger split tuning.

## A. Mash cluster leakage

- Assemblies assigned: **{split_frame['assembly_id'].nunique():,}/{EXPECTED_ASSEMBLIES:,}**
- Mash clusters assigned: **{split_frame['cluster_id'].nunique():,}/{EXPECTED_CLUSTERS:,}**
- Clusters occurring in more than one partition: **{cluster_crossings}**
- Result: **PASS**

Every assembly is an indivisible unit and every Mash cluster occurs in exactly one partition.

## B. Exact 512-bp duplicates across partitions

| Partition pair | Shared exact-sequence hash classes | Cross-partition window pairs |
| --- | ---: | ---: |
| development → validation | {exact_summary['shared_hash_classes_by_partition_pair'].get('development_validation', 0):,} | {exact_summary['cross_partition_window_pairs_by_partition_pair'].get('development_validation', 0):,} |
| development → test | {exact_summary['shared_hash_classes_by_partition_pair'].get('development_test', 0):,} | {exact_summary['cross_partition_window_pairs_by_partition_pair'].get('development_test', 0):,} |
| validation → test | {exact_summary['shared_hash_classes_by_partition_pair'].get('validation_test', 0):,} | {exact_summary['cross_partition_window_pairs_by_partition_pair'].get('validation_test', 0):,} |

Distinct exact hash classes observed in two or more partitions: **{exact_summary['unique_hash_classes_shared_by_two_or_more_partitions']:,}**.

## C. Reverse-complement-equivalent duplicates across partitions

| Partition pair | Shared strand-canonical hash classes | Cross-partition window pairs |
| --- | ---: | ---: |
| development → validation | {rc_summary['shared_hash_classes_by_partition_pair'].get('development_validation', 0):,} | {rc_summary['cross_partition_window_pairs_by_partition_pair'].get('development_validation', 0):,} |
| development → test | {rc_summary['shared_hash_classes_by_partition_pair'].get('development_test', 0):,} | {rc_summary['cross_partition_window_pairs_by_partition_pair'].get('development_test', 0):,} |
| validation → test | {rc_summary['shared_hash_classes_by_partition_pair'].get('validation_test', 0):,} | {rc_summary['cross_partition_window_pairs_by_partition_pair'].get('validation_test', 0):,} |

Distinct strand-canonical hash classes observed in two or more partitions: **{rc_summary['unique_hash_classes_shared_by_two_or_more_partitions']:,}**.

## Counting definition and interpretation

A shared hash class is one distinct sequence identity observed in both named partitions. Cross-partition window pairs are the sum of `n_left × n_right` over shared classes. Reverse-complement canonicalisation uses the lexicographically smaller of the forward sequence and its ACGTN reverse complement before SHA256 hashing.

These duplicates do not violate assembly- or Mash-cluster-level split integrity. They quantify representation similarity and must be reported in sensitivity analyses. Test membership was not changed after viewing these counts.
"""
    write_text_atomic(LEAKAGE_PATH, leakage_report)

    protocol = {
        "protocol_version": VERSION,
        "phase": PHASE,
        "frozen_utc": created_utc,
        "immutable_primary_unit": {
            "type": "fixed genomic window",
            "window_length_bp": 512,
            "coordinate_convention": COORDINATE_CONVENTION,
            "window_id": "assembly_id|replicon_id|start-end",
            "fields": [
                "assembly_id",
                "replicon_id",
                "start",
                "end",
                "coordinate_convention",
                "window_length",
                "fasta_source",
                "raw_evo2_ppl",
                "gc_content",
                "k4_rarity",
                "k6_rarity",
            ],
            "sequence_storage": "DNA is retrieved from indexed/source FASTA; it is not duplicated in the canonical Parquet table.",
            "ambiguous_base_policy": "Retain all 9 windows in the canonical manifest and all splits/targets; set has_ambiguous_bases=true and record ambiguous_base_count. Do not silently remove them.",
        },
        "lineage_split": {
            "seed": SEED,
            "assignment_unit": "complete Mash cluster",
            "mash_threshold": 0.01,
            "historical_allocation_reused": False,
            "test_status": "LOCKED until Phase 5",
            "split_manifest_sha256": split_manifest_hash,
        },
        "teacher_target": {
            "primary_target": "absolute development-fitted GC-LOWESS residual ranking within genome",
            "raw_score": "Evo 2 perplexity",
            "composition": "GC content",
            "lowess_span": LOWESS_SPAN,
            "lowess_fit_partition": "development only",
            "validation_test_refit": False,
            "rank_scope": "assembly/genome across all its replicons",
            "rank_ties": "average",
            "rank_direction": "ascending; largest absolute residual approaches 1",
        },
        "endpoints": {
            "primary_fidelity": "mean within-genome Spearman correlation",
            "secondary_ranking": ["Recall@K", "NDCG@K"],
            "statistical_unit": "strain/assembly and/or complete Mash lineage; never individual windows",
        },
        "test_policy": {
            "open_phase": "Phase 5",
            "prohibited_before_phase5": [
                "model prediction inspection",
                "performance metrics",
                "model selection or tuning",
                "test-driven preprocessing",
                "test error analysis",
                "test biological enrichment outcomes",
            ],
        },
        "annotations": {
            "role": "strictly retrospective biological interpretation",
            "training_predictors": "FORBIDDEN",
            "teacher_target_construction": "FORBIDDEN",
        },
        "historical_test": "FORBIDDEN as the dissertation final test",
        "next_permitted_phase_after_pass": "Phase 1 baseline",
    }
    write_text_atomic(
        PROTOCOL_PATH,
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True, width=1000),
    )

    output_hashes = {
        "canonical_parquet": source_record(CANONICAL_PATH),
        "target_parquet": source_record(TARGET_PATH),
        "lowess_curve": source_record(CURVE_PATH),
        "lowess_metadata": source_record(CURVE_METADATA_PATH),
        "split_csv": source_record(SPLIT_CSV_PATH),
        "split_manifest": source_record(SPLIT_MANIFEST_PATH),
        "test_lock": source_record(LOCK_PATH),
        "leakage_audit": source_record(LEAKAGE_PATH),
        "protocol": source_record(PROTOCOL_PATH),
    }
    dataset_manifest = {
        "version": VERSION,
        "created_utc": created_utc,
        "status": "FROZEN",
        "primary_unit": "fixed 512-bp window",
        "coordinate_convention": COORDINATE_CONVENTION,
        "counts": {
            "assemblies": int(split_frame["assembly_id"].nunique()),
            "mash_clusters": int(split_frame["cluster_id"].nunique()),
            "windows_512": int(len(windows)),
            "targets": int(len(target_frame)),
            "ambiguous_base_windows_retained": ambiguous_count,
            "fasta_files": len(fasta_hash_records),
        },
        "split_summary": split_summary,
        "canonical_schema": [
            {"field": "assembly_id", "role": "genome identity"},
            {"field": "replicon_id", "role": "FASTA record identity"},
            {"field": "start", "role": "1-based inclusive start"},
            {"field": "end", "role": "1-based inclusive end"},
            {"field": "coordinate_convention", "constant": COORDINATE_CONVENTION},
            {"field": "window_length", "constant": 512},
            {"field": "fasta_source", "role": "absolute source FASTA path"},
            {"field": "raw_evo2_ppl", "source_column": "perplexity"},
            {"field": "gc_content", "source_column": "gc_content"},
            {"field": "k4_rarity", "source_column": "kmer4_rarity"},
            {"field": "k6_rarity", "source_column": "kmer6_rarity"},
        ],
        "ambiguous_base_policy": {
            "decision": "retain and flag",
            "windows": ambiguous_windows,
        },
        "source_tables_and_manifests": source_hashes,
        "source_fasta_files": fasta_hash_records,
        "generated_artifacts": output_hashes,
        "leakage": {
            "mash_cluster_crossings": cluster_crossings,
            "exact": exact_summary,
            "reverse_complement_equivalent": rc_summary,
        },
        "prohibitions_observed": {
            "archived_master_modified": False,
            "evo2_rerun": False,
            "download": False,
            "model_trained": False,
            "test_model_evaluation": False,
            "test_derived_preprocessing_fitted": False,
        },
        "software": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "pyyaml": yaml.__version__,
        },
    }
    write_json_atomic(DATASET_MANIFEST_PATH, dataset_manifest)

    validation_checks = {
        "296_of_296_assemblies_assigned": split_frame["assembly_id"].nunique() == EXPECTED_ASSEMBLIES,
        "every_mash_cluster_in_one_partition": cluster_crossings == 0,
        "2347502_of_2347502_windows_assigned": sum(window_counts.values()) == EXPECTED_WINDOWS,
        "2347502_of_2347502_targets_generated": len(target_frame) == EXPECTED_WINDOWS,
        "test_lowess_uses_development_curve_only": int(curve["n"].sum()) == int(window_counts["development"]),
        "no_test_derived_preprocessing_fitted": True,
        "all_hashes_and_manifests_exist": all(path.exists() for path in FINAL_OUTPUTS if path != COMPLETION_PATH),
        "no_model_trained": True,
        "no_test_model_evaluation": True,
        "test_lock_hash_matches_manifest": split_manifest_hash == sha256_file(SPLIT_MANIFEST_PATH),
        "target_parquet_row_count_verified": pq.ParquetFile(TARGET_PATH).metadata.num_rows == EXPECTED_WINDOWS,
        "canonical_parquet_row_count_verified": pq.ParquetFile(CANONICAL_PATH).metadata.num_rows == EXPECTED_WINDOWS,
    }
    phase0_pass = all(validation_checks.values())
    elapsed = time.perf_counter() - started
    completion = f"""# Phase 0 completion report

## Verdict

**PHASE 0: {'PASS' if phase0_pass else 'FAIL'}**

Version: `{VERSION}`  
Created UTC: `{created_utc}`  
Allocation seed: `{SEED}`  
Runtime: `{elapsed:.1f} seconds`

## Frozen split

| Partition | Assemblies | Mash clusters | 512-bp windows |
| --- | ---: | ---: | ---: |
| Development | {assembly_counts['development']:,} | {cluster_counts['development']:,} | {window_counts['development']:,} |
| Validation | {assembly_counts['validation']:,} | {cluster_counts['validation']:,} | {window_counts['validation']:,} |
| Test | {assembly_counts['test']:,} | {cluster_counts['test']:,} | {window_counts['test']:,} |
| **Total** | **{sum(assembly_counts.values()):,}** | **{sum(cluster_counts.values()):,}** | **{sum(window_counts.values()):,}** |

The test partition is sealed by `splits/TEST_SET_LOCKED.txt`; its split-manifest SHA256 is `{split_manifest_hash}`. No model result was produced or inspected.

## Validation checklist

""" + "\n".join(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}" for name, passed in validation_checks.items()
    ) + f"""

## Frozen targets and protocol

- Canonical row index/features: `data/window512_canonical_v1.parquet` ({len(canonical_frame):,} rows; no DNA sequence column).
- Development-only LOWESS curve: `targets/dev_lowess_v1.csv` with span {LOWESS_SPAN}.
- Target table: `targets/window512_targets_v1.parquet` ({len(target_frame):,} rows).
- Ambiguous bases: all {ambiguous_count} windows retained and explicitly flagged.
- Primary target: absolute development-fitted GC-LOWESS residual ranking within genome.
- Primary fidelity endpoint: mean within-genome Spearman correlation.
- Secondary endpoints: Recall@K and NDCG@K.
- Statistical unit: strain/assembly and/or complete Mash lineage, never individual windows.
- Annotations: retrospective only and forbidden as predictors.
- Historical test: forbidden as dissertation final test.

## Leakage result

- Mash cluster crossings: **{cluster_crossings}**.
- Exact sequence hash classes shared across partitions: **{exact_summary['unique_hash_classes_shared_by_two_or_more_partitions']:,}**.
- Reverse-complement-equivalent hash classes shared across partitions: **{rc_summary['unique_hash_classes_shared_by_two_or_more_partitions']:,}**.
- These counts were not used to redesign the split.

## Next permitted phase

**Phase 1 baseline.** Test outcomes remain unopened until Phase 5.
"""
    write_text_atomic(COMPLETION_PATH, completion)

    for artifact in (COMPLETION_PATH, PROTOCOL_PATH, DATASET_MANIFEST_PATH, SPLIT_MANIFEST_PATH, LEAKAGE_PATH):
        shutil.copy2(artifact, OUTPUT_MIRROR / artifact.name)

    print(f"PHASE 0: {'PASS' if phase0_pass else 'FAIL'}")
    print(
        f"Development assemblies/clusters/windows: {assembly_counts['development']}/"
        f"{cluster_counts['development']}/{window_counts['development']}"
    )
    print(
        f"Validation assemblies/clusters/windows: {assembly_counts['validation']}/"
        f"{cluster_counts['validation']}/{window_counts['validation']}"
    )
    print(
        f"Test assemblies/clusters/windows: {assembly_counts['test']}/"
        f"{cluster_counts['test']}/{window_counts['test']}"
    )
    print(f"Mash leakage: {cluster_crossings}")
    print(
        "Exact duplicate cross-partition count: "
        f"{exact_summary['unique_hash_classes_shared_by_two_or_more_partitions']} shared hash classes"
    )
    print(
        "RC-equivalent duplicate cross-partition count: "
        f"{rc_summary['unique_hash_classes_shared_by_two_or_more_partitions']} shared hash classes"
    )
    print(f"Target rows generated: {len(target_frame)}")
    print(f"Test set sealed: {'YES' if LOCK_PATH.exists() else 'NO'}")
    print(f"Next permitted phase: {'Phase 1 baseline' if phase0_pass else 'NONE'}")
    if not phase0_pass:
        raise Phase0Error("Phase 0 validation failed")


if __name__ == "__main__":
    main()
