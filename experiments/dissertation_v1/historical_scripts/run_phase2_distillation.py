"""Run Phase 2 single-model student distillation and controlled ablations.

Only development and validation rows are materialized. The locked test
partition is never read, predicted, or evaluated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset


WORKSPACE = Path(r"C:\Users\LDD\Documents\Codex\2026-08-19\za")
ROOT = WORKSPACE / "experiments" / "dissertation_v1"
PHASE1 = ROOT / "phase1"
PHASE2 = ROOT / "phase2"
SRC = PHASE2 / "src"
TESTS = PHASE2 / "tests"
CACHE = PHASE2 / "cache"
CHECKPOINTS = PHASE2 / "checkpoints"
OUTPUT_MIRROR = WORKSPACE / "outputs" / "phase2"

sys.path.insert(0, str(SRC))
from student_model import (  # noqa: E402
    ArchitectureV1,
    MASK_TOKEN,
    MaskedSequenceComparatorV1,
    ScalarStudentV1,
    encode_sequence,
    extract_window,
    load_fasta,
    pairwise_logistic_ranking_loss,
    trainable_parameter_count,
)


PHASE0_REPORT = ROOT / "phase0_completion_report.md"
PROTOCOL = ROOT / "protocol" / "experiment_protocol_v1.yaml"
SPLIT = ROOT / "splits" / "dissertation_split_v1.csv"
TEST_LOCK = ROOT / "splits" / "TEST_SET_LOCKED.txt"
CANONICAL = ROOT / "data" / "window512_canonical_v1.parquet"
TARGETS = ROOT / "targets" / "window512_targets_v1.parquet"
PHASE1_REPORT = PHASE1 / "phase1_report.md"
PHASE1_CONFIG = PHASE1 / "baseline_config.yaml"
PHASE1_LOCK = PHASE1 / "selected_baseline.lock"
PHASE1_PREDICTIONS = PHASE1 / "baseline_predictions_validation.parquet"

MODEL_DEFINITION = PHASE2 / "model_definition.yaml"
TRAINING_CONFIG = PHASE2 / "training_config.yaml"
VALIDATION_METRICS = PHASE2 / "validation_metrics.csv"
ABLATION_RESULTS = PHASE2 / "ablation_results.csv"
REPORT = PHASE2 / "phase2_report.md"
FINAL_LOCK = PHASE2 / "final_architecture.lock"
VALIDATION_PREDICTIONS = PHASE2 / "validation_predictions.parquet"
CACHE_METADATA = CACHE / "token_cache_metadata.parquet"
CACHE_TOKENS = CACHE / "window_tokens.uint8.mmap"
CACHE_MANIFEST = CACHE / "token_cache_manifest.json"

CHECKPOINT_PATHS = {
    "M1_full_distilled": CHECKPOINTS / "M1_full_distilled.pt",
    "M2_huber_only": CHECKPOINTS / "M2_huber_only.pt",
    "M3_raw_ppl": CHECKPOINTS / "M3_raw_ppl.pt",
    "M5_sequence_only": CHECKPOINTS / "M5_sequence_only.pt",
}

FINAL_OUTPUTS = [
    MODEL_DEFINITION,
    TRAINING_CONFIG,
    VALIDATION_METRICS,
    ABLATION_RESULTS,
    REPORT,
    FINAL_LOCK,
    VALIDATION_PREDICTIONS,
    CACHE_METADATA,
    CACHE_TOKENS,
    CACHE_MANIFEST,
    *CHECKPOINT_PATHS.values(),
]

SEED = 20260822
BOOTSTRAP_ITERATIONS = 2_000
TRAIN_WINDOWS_PER_GENOME = 256
TUNING_WINDOWS_PER_VALIDATION_GENOME = 512
SCALAR_WARMUP_EPOCHS = 3
BRANCH_EPOCHS = 1
RAW_PPL_EPOCHS = 4
MASKED_EPOCHS = 4
PAIR_BATCH_SIZE = 128
SINGLE_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANK_WEIGHT_CANDIDATES = [2.0, 4.0]
MASK_PROBABILITY = 0.15

EXPECTED_DEVELOPMENT_WINDOWS = 1_640_110
EXPECTED_VALIDATION_WINDOWS = 353_840
EXPECTED_DEVELOPMENT_GENOMES = 207
EXPECTED_VALIDATION_GENOMES = 45
EXPECTED_VALIDATION_CLUSTERS = 19

K_DEFINITIONS = [
    ("top_1_percent", lambda n: max(1, int(math.ceil(0.01 * n)))),
    ("top_5_percent", lambda n: max(1, int(math.ceil(0.05 * n)))),
    ("fixed_100", lambda n: min(100, n)),
]


class Phase2Error(RuntimeError):
    pass


def stable_seed(label: str, base: int = SEED) -> int:
    return base ^ int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase2Error(f"Required file missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def verify_prerequisites_and_destination() -> dict[str, Any]:
    if "**PHASE 0: PASS**" not in PHASE0_REPORT.read_text(encoding="utf-8"):
        raise Phase2Error("Phase 0 is not PASS")
    if "**PHASE 1: PASS**" not in PHASE1_REPORT.read_text(encoding="utf-8"):
        raise Phase2Error("Phase 1 is not PASS")
    phase1_lock = json.loads(PHASE1_LOCK.read_text(encoding="utf-8"))
    if phase1_lock.get("test_accessed") is not False:
        raise Phase2Error("Phase 1 lock does not confirm TEST remained unopened")
    for key, record in phase1_lock["output_sha256"].items():
        if key == "models":
            for model_record in record.values():
                if sha256_file(Path(model_record["path"])) != model_record["sha256"]:
                    raise Phase2Error("Phase 1 model hash mismatch")
        elif sha256_file(Path(record["path"])) != record["sha256"]:
            raise Phase2Error(f"Phase 1 artifact hash mismatch: {key}")
    existing = [str(path) for path in FINAL_OUTPUTS if path.exists()]
    if existing:
        raise Phase2Error("Refusing to overwrite existing Phase 2 artifacts:\n" + "\n".join(existing))
    for directory in (PHASE2, CACHE, CHECKPOINTS, OUTPUT_MIRROR):
        directory.mkdir(parents=True, exist_ok=True)
    return phase1_lock


def read_development_validation_metadata() -> pd.DataFrame:
    predicate = ds.field("split").isin(["development", "validation"])
    canonical = ds.dataset(CANONICAL, format="parquet").to_table(
        columns=[
            "window_id",
            "assembly_id",
            "replicon_id",
            "start",
            "end",
            "fasta_source",
            "cluster_id",
            "split",
        ],
        filter=predicate,
    ).to_pandas()
    targets = ds.dataset(TARGETS, format="parquet").to_table(
        columns=[
            "window_id",
            "split",
            "raw_ppl",
            "absolute_residual",
            "within_genome_absolute_residual_rank",
        ],
        filter=predicate,
    ).to_pandas()
    if len(canonical) != EXPECTED_DEVELOPMENT_WINDOWS + EXPECTED_VALIDATION_WINDOWS:
        raise Phase2Error("Unexpected development+validation metadata count")
    if not canonical["window_id"].equals(targets["window_id"]):
        raise Phase2Error("Canonical and target IDs are not aligned")
    if not canonical["split"].equals(targets["split"]):
        raise Phase2Error("Canonical and target split labels are not aligned")
    canonical["raw_ppl"] = targets["raw_ppl"].to_numpy(float)
    canonical["absolute_residual"] = targets["absolute_residual"].to_numpy(float)
    canonical["teacher_rank"] = targets["within_genome_absolute_residual_rank"].to_numpy(float)
    del targets
    if set(canonical["split"].unique()) != {"development", "validation"}:
        raise Phase2Error("TEST or unknown partition was materialized")
    if canonical.isna().any().any():
        raise Phase2Error("Phase 2 metadata contains missing values")
    counts = canonical.groupby("split").size().to_dict()
    if counts != {"development": EXPECTED_DEVELOPMENT_WINDOWS, "validation": EXPECTED_VALIDATION_WINDOWS}:
        raise Phase2Error(f"Unexpected partition counts: {counts}")
    return canonical


def sample_positions(frame: pd.DataFrame, per_genome: int, label: str) -> np.ndarray:
    selected: list[np.ndarray] = []
    for assembly_id, positions in frame.groupby("assembly_id", sort=True).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        take = min(per_genome, len(positions))
        rng = np.random.default_rng(stable_seed(f"{label}|{assembly_id}"))
        chosen = rng.choice(positions, size=take, replace=False)
        selected.append(np.sort(chosen))
    return np.concatenate(selected)


def build_token_cache(metadata: pd.DataFrame, created_utc: str) -> tuple[pd.DataFrame, np.memmap]:
    development = metadata.loc[metadata["split"].eq("development")].reset_index(drop=True)
    validation = metadata.loc[metadata["split"].eq("validation")].reset_index(drop=True)
    train_positions = sample_positions(development, TRAIN_WINDOWS_PER_GENOME, "train_pool")
    train_pool = development.iloc[train_positions].copy()
    train_pool["cache_role"] = "train_pool"
    validation = validation.copy()
    validation["cache_role"] = "validation"
    cache_metadata = pd.concat([train_pool, validation], ignore_index=True)
    cache_metadata = cache_metadata.sort_values(
        ["cache_role", "assembly_id", "replicon_id", "start", "end"],
        kind="mergesort",
    ).reset_index(drop=True)
    # Explicitly keep training rows first for compact array slicing.
    role_order = pd.Categorical(cache_metadata["cache_role"], categories=["train_pool", "validation"], ordered=True)
    cache_metadata = cache_metadata.assign(_role_order=role_order).sort_values(
        ["_role_order", "assembly_id", "replicon_id", "start", "end"], kind="mergesort"
    ).drop(columns="_role_order").reset_index(drop=True)
    cache_metadata.insert(0, "cache_index", np.arange(len(cache_metadata), dtype=np.int64))
    train_count = int(cache_metadata["cache_role"].eq("train_pool").sum())
    if train_count != TRAIN_WINDOWS_PER_GENOME * EXPECTED_DEVELOPMENT_GENOMES:
        raise Phase2Error(f"Unexpected training-pool count: {train_count}")
    if int(cache_metadata["cache_role"].eq("validation").sum()) != EXPECTED_VALIDATION_WINDOWS:
        raise Phase2Error("Validation cache is incomplete")

    print(
        f"Phase 2: building deterministic token cache ({train_count:,} train + "
        f"{EXPECTED_VALIDATION_WINDOWS:,} validation windows)",
        flush=True,
    )
    tokens = np.memmap(
        CACHE_TOKENS,
        mode="w+",
        dtype=np.uint8,
        shape=(len(cache_metadata), 512),
    )
    fasta_files_hashed: dict[str, dict[str, Any]] = {}
    for assembly_number, (assembly_id, rows) in enumerate(
        cache_metadata.groupby("assembly_id", sort=True), start=1
    ):
        fasta_paths = rows["fasta_source"].unique()
        if len(fasta_paths) != 1:
            raise Phase2Error(f"Assembly {assembly_id} has multiple FASTA sources")
        fasta_path = Path(str(fasta_paths[0]))
        records = load_fasta(fasta_path)
        if str(fasta_path) not in fasta_files_hashed:
            fasta_files_hashed[str(fasta_path)] = file_record(fasta_path)
        for row in rows.itertuples(index=False):
            sequence = records.get(str(row.replicon_id))
            if sequence is None:
                raise Phase2Error(f"Replicon {row.replicon_id} absent for {assembly_id}")
            window = extract_window(sequence, int(row.start), int(row.end))
            if len(window) != 512:
                raise Phase2Error("Non-512 sequence reconstructed")
            tokens[int(row.cache_index)] = encode_sequence(window)
        if assembly_number % 25 == 0 or assembly_number == EXPECTED_DEVELOPMENT_GENOMES + EXPECTED_VALIDATION_GENOMES:
            print(f"  token cache: {assembly_number}/{EXPECTED_DEVELOPMENT_GENOMES + EXPECTED_VALIDATION_GENOMES} assemblies", flush=True)
    tokens.flush()
    write_parquet_atomic(cache_metadata, CACHE_METADATA)
    cache_manifest = {
        "created_utc": created_utc,
        "rows": int(len(cache_metadata)),
        "shape": [int(len(cache_metadata)), 512],
        "dtype": "uint8",
        "token_map": {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "MASK": 5},
        "coordinate_convention": "1-based inclusive",
        "train_pool_rule": f"Exactly {TRAIN_WINDOWS_PER_GENOME} windows sampled without replacement per development genome using SHA256-stable seed; all validation windows retained.",
        "train_pool_rows": train_count,
        "validation_rows": EXPECTED_VALIDATION_WINDOWS,
        "test_rows": 0,
        "source_canonical": file_record(CANONICAL),
        "source_targets": file_record(TARGETS),
        "source_fasta_files": list(fasta_files_hashed.values()),
        "metadata": file_record(CACHE_METADATA),
        "tokens": file_record(CACHE_TOKENS),
    }
    write_text_atomic(CACHE_MANIFEST, json.dumps(cache_manifest, indent=2) + "\n")
    return cache_metadata, np.memmap(CACHE_TOKENS, mode="r", dtype=np.uint8, shape=(len(cache_metadata), 512))


def run_unit_tests() -> str:
    print("Phase 2: running sequence reconstruction unit tests", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(TESTS),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    print(output, flush=True)
    if result.returncode != 0:
        raise Phase2Error("Sequence reconstruction unit tests failed")
    return output


class PairTokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, targets: np.ndarray, assemblies: np.ndarray, seed: int) -> None:
        self.tokens = tokens
        self.targets = targets.astype(np.float32)
        self.assemblies = assemblies.astype(str)
        self.seed = int(seed)
        self.epoch = 0
        self.group_bounds: dict[str, tuple[int, int]] = {}
        for assembly_id in np.unique(self.assemblies):
            positions = np.flatnonzero(self.assemblies == assembly_id)
            if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
                raise Phase2Error("Training cache is not contiguous within genome")
            self.group_bounds[str(assembly_id)] = (int(positions[0]), int(positions[-1] + 1))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assembly_id = str(self.assemblies[index])
        start, stop = self.group_bounds[assembly_id]
        size = stop - start
        if size < 2:
            raise Phase2Error(f"Genome {assembly_id} has fewer than two sampled windows")
        local = index - start
        offset = 1 + ((index * 1_103_515_245 + self.epoch * 12_345 + self.seed) % (size - 1))
        partner = start + ((local + offset) % size)
        return (
            torch.from_numpy(np.array(self.tokens[index], dtype=np.int64, copy=True)),
            torch.from_numpy(np.array(self.tokens[partner], dtype=np.int64, copy=True)),
            torch.tensor(self.targets[index], dtype=torch.float32),
            torch.tensor(self.targets[partner], dtype=torch.float32),
        )


class TokenIndexDataset(Dataset):
    def __init__(self, tokens: np.ndarray, positions: np.ndarray) -> None:
        self.tokens = tokens
        self.positions = positions.astype(np.int64)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.tokens[int(self.positions[index])], dtype=np.int64, copy=True))


class MaskedEvalDataset(Dataset):
    def __init__(self, tokens: np.ndarray, positions: np.ndarray, offsets: np.ndarray) -> None:
        self.tokens = tokens
        self.positions = positions.astype(np.int64)
        self.offsets = offsets.astype(np.int64)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.from_numpy(np.array(self.tokens[int(self.positions[index])], dtype=np.int64, copy=True))
        return tokens, torch.tensor(int(self.offsets[index]), dtype=torch.long)


def make_pair_loader(dataset: PairTokenDataset, epoch: int, shuffle: bool = True) -> DataLoader:
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(
        dataset,
        batch_size=PAIR_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def train_scalar_epoch(
    model: ScalarStudentV1,
    dataset: PairTokenDataset,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rank_weight: float,
) -> dict[str, float]:
    model.train()
    huber_fn = nn.HuberLoss(delta=1.0)
    total_loss = total_huber = total_rank = 0.0
    batches = 0
    for tokens_a, tokens_b, target_a, target_b in make_pair_loader(dataset, epoch):
        optimizer.zero_grad(set_to_none=True)
        combined = torch.cat([tokens_a, tokens_b], dim=0)
        prediction = model(combined)
        prediction_a, prediction_b = prediction.chunk(2)
        huber = 0.5 * (huber_fn(prediction_a, target_a) + huber_fn(prediction_b, target_b))
        ranking = pairwise_logistic_ranking_loss(prediction_a, prediction_b, target_a, target_b)
        loss = huber + rank_weight * ranking
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach())
        total_huber += float(huber.detach())
        total_rank += float(ranking.detach())
        batches += 1
    return {
        "loss": total_loss / batches,
        "huber": total_huber / batches,
        "ranking": total_rank / batches,
        "rank_weight": rank_weight,
    }


def train_masked_epoch(
    model: MaskedSequenceComparatorV1,
    tokens: np.ndarray,
    train_positions: np.ndarray,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    loader = DataLoader(
        TokenIndexDataset(tokens, train_positions),
        batch_size=SINGLE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED + 1000 + epoch),
    )
    mask_generator = torch.Generator().manual_seed(SEED + 2000 + epoch)
    total = 0.0
    batches = 0
    for original in loader:
        mask = torch.rand(original.shape, generator=mask_generator).lt(MASK_PROBABILITY)
        # Guarantee at least one masked base per sequence.
        empty = ~mask.any(dim=1)
        if torch.any(empty):
            mask[empty, 0] = True
        corrupted = original.clone()
        corrupted[mask] = MASK_TOKEN
        optimizer.zero_grad(set_to_none=True)
        logits = model(corrupted)
        per_position = F.cross_entropy(logits, original, reduction="none")
        loss = per_position[mask].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / batches


def predict_scalar(
    model: ScalarStudentV1,
    tokens: np.ndarray,
    positions: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TokenIndexDataset(tokens, positions),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            predictions.append(model(batch).numpy())
    standardized = np.concatenate(predictions).astype(float)
    return standardized * std + mean


def predict_masked_anomaly(
    model: MaskedSequenceComparatorV1,
    tokens: np.ndarray,
    positions: np.ndarray,
    window_ids: np.ndarray,
) -> np.ndarray:
    model.eval()
    offsets = np.asarray([stable_seed(str(window_id), base=0) % 7 for window_id in window_ids], dtype=np.int64)
    loader = DataLoader(
        MaskedEvalDataset(tokens, positions, offsets),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    base_positions = torch.arange(512).unsqueeze(0)
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for original, offset in loader:
            mask = base_positions.remainder(7).eq(offset.unsqueeze(1))
            corrupted = original.clone()
            corrupted[mask] = MASK_TOKEN
            logits = model(corrupted)
            per_position = F.cross_entropy(logits, original, reduction="none")
            score = (per_position * mask).sum(dim=1) / mask.sum(dim=1)
            scores.append(score.numpy())
    return np.concatenate(scores).astype(float)


def tuning_mean_spearman(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
) -> float:
    values = []
    for assembly_id, positions in metadata.groupby("assembly_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        rho = float(spearmanr(metadata.iloc[idx]["teacher_rank"], predictions[idx]).statistic)
        if not np.isfinite(rho):
            raise Phase2Error(f"Non-finite tuning Spearman for {assembly_id}")
        values.append(rho)
    return float(np.mean(values))


def deterministic_order(scores: np.ndarray, window_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((window_ids.astype(str), -scores.astype(float)))


def full_genome_metrics(
    validation: pd.DataFrame,
    prediction: np.ndarray,
    model_id: str,
    mae_target: np.ndarray | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for assembly_id, positions in validation.groupby("assembly_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        clusters = validation.iloc[idx]["cluster_id"].unique()
        if len(clusters) != 1:
            raise Phase2Error("Validation assembly crosses clusters")
        cluster_id = str(clusters[0])
        local_ids = validation.iloc[idx]["window_id"].astype(str).to_numpy()
        relevance = validation.iloc[idx]["absolute_residual"].to_numpy(float)
        teacher_rank = validation.iloc[idx]["teacher_rank"].to_numpy(float)
        predicted = prediction[idx]
        rho = float(spearmanr(teacher_rank, predicted).statistic)
        rows.append(
            {
                "model_id": model_id,
                "assembly_id": str(assembly_id),
                "cluster_id": cluster_id,
                "metric": "spearman",
                "k_definition": "",
                "value": rho,
            }
        )
        teacher_order = deterministic_order(relevance, local_ids)
        prediction_order = deterministic_order(predicted, local_ids)
        for k_id, k_function in K_DEFINITIONS:
            k = int(k_function(len(idx)))
            teacher_top = teacher_order[:k]
            prediction_top = prediction_order[:k]
            recall = float(np.intersect1d(teacher_top, prediction_top, assume_unique=True).size / k)
            discounts = np.log2(np.arange(2, k + 2, dtype=float))
            dcg = float(np.sum(relevance[prediction_top] / discounts))
            idcg = float(np.sum(relevance[teacher_top] / discounts))
            ndcg = dcg / idcg
            rows.extend(
                [
                    {
                        "model_id": model_id,
                        "assembly_id": str(assembly_id),
                        "cluster_id": cluster_id,
                        "metric": "recall_at_k",
                        "k_definition": k_id,
                        "value": recall,
                    },
                    {
                        "model_id": model_id,
                        "assembly_id": str(assembly_id),
                        "cluster_id": cluster_id,
                        "metric": "ndcg_at_k",
                        "k_definition": k_id,
                        "value": ndcg,
                    },
                ]
            )
        if mae_target is not None:
            rows.append(
                {
                    "model_id": model_id,
                    "assembly_id": str(assembly_id),
                    "cluster_id": cluster_id,
                    "metric": "scalar_mae_training_target",
                    "k_definition": "",
                    "value": float(np.mean(np.abs(predicted - mae_target[idx]))),
                }
            )
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame["value"].to_numpy(float)).all():
        raise Phase2Error(f"Non-finite metric generated for {model_id}")
    return frame


def bootstrap_seed(label: str) -> int:
    return stable_seed(label, base=SEED + 50_000)


def summarize_metric(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame["assembly_id"].nunique() != EXPECTED_VALIDATION_GENOMES:
        raise Phase2Error(f"Metric does not cover all validation genomes: {label}")
    cluster_means = frame.groupby("cluster_id", sort=True)["value"].mean().to_numpy(float)
    if len(cluster_means) != EXPECTED_VALIDATION_CLUSTERS:
        raise Phase2Error(f"Metric does not cover all validation lineages: {label}")
    rng = np.random.default_rng(bootstrap_seed(label))
    draws = rng.choice(cluster_means, size=(BOOTSTRAP_ITERATIONS, len(cluster_means)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(frame["value"].mean()),
        "median_across_genomes": float(frame["value"].median()),
        "mean_across_lineages": float(cluster_means.mean()),
        "lineage_bootstrap_ci_low": float(low),
        "lineage_bootstrap_ci_high": float(high),
        "n_genomes": EXPECTED_VALIDATION_GENOMES,
        "n_lineages": EXPECTED_VALIDATION_CLUSTERS,
    }


def make_validation_metrics(genome_metrics: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for model_id, frame in genome_metrics.items():
        for (metric, k_definition), subset in frame.groupby(["metric", "k_definition"], dropna=False, sort=True):
            summary = summarize_metric(subset, f"{model_id}|{metric}|{k_definition}")
            summaries[model_id][(str(metric), str(k_definition))] = summary
            rows.append(
                {
                    "level": "summary",
                    "model_id": model_id,
                    "assembly_id": "",
                    "cluster_id": "",
                    "metric": metric,
                    "k_definition": k_definition,
                    **summary,
                }
            )
        for record in frame.to_dict(orient="records"):
            rows.append(
                {
                    "level": "genome",
                    **record,
                    "estimate": record["value"],
                    "median_across_genomes": np.nan,
                    "mean_across_lineages": np.nan,
                    "lineage_bootstrap_ci_low": np.nan,
                    "lineage_bootstrap_ci_high": np.nan,
                    "n_genomes": 1,
                    "n_lineages": 1,
                }
            )
    columns = [
        "level",
        "model_id",
        "assembly_id",
        "cluster_id",
        "metric",
        "k_definition",
        "estimate",
        "median_across_genomes",
        "mean_across_lineages",
        "lineage_bootstrap_ci_low",
        "lineage_bootstrap_ci_high",
        "n_genomes",
        "n_lineages",
    ]
    return pd.DataFrame(rows)[columns], dict(summaries)


def paired_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label: str,
) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["assembly_id", "cluster_id", "metric", "k_definition"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    merged["delta"] = merged["value_left"] - merged["value_right"]
    lineage = merged.groupby("cluster_id", sort=True)["delta"].mean().to_numpy(float)
    rng = np.random.default_rng(bootstrap_seed(label))
    draws = rng.choice(lineage, size=(BOOTSTRAP_ITERATIONS, len(lineage)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_mean_genome": float(merged["delta"].mean()),
        "delta_mean_lineage": float(lineage.mean()),
        "lineage_bootstrap_ci_low": float(low),
        "lineage_bootstrap_ci_high": float(high),
    }


def build_ablation_results(
    genome_metrics: dict[str, pd.DataFrame],
    summaries: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> pd.DataFrame:
    comparisons = [
        ("teacher_vs_cheap", "M1_full_distilled", "M4_cheap_feature_baseline"),
        ("teacher_vs_sequence_only", "M1_full_distilled", "M5_sequence_only"),
        ("ranking_loss", "M1_full_distilled", "M2_huber_only"),
        ("residual_vs_raw_ppl", "M1_full_distilled", "M3_raw_ppl"),
    ]
    metric_keys = [
        ("spearman", ""),
        ("recall_at_k", "top_1_percent"),
        ("ndcg_at_k", "top_1_percent"),
    ]
    rows = []
    for comparison_id, model_a, model_b in comparisons:
        for metric, k_definition in metric_keys:
            left = genome_metrics[model_a].loc[
                genome_metrics[model_a]["metric"].eq(metric)
                & genome_metrics[model_a]["k_definition"].eq(k_definition)
            ]
            right = genome_metrics[model_b].loc[
                genome_metrics[model_b]["metric"].eq(metric)
                & genome_metrics[model_b]["k_definition"].eq(k_definition)
            ]
            delta = paired_delta(left, right, f"{comparison_id}|{metric}|{k_definition}")
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "k_definition": k_definition,
                    "model_a_estimate": summaries[model_a][(metric, k_definition)]["estimate"],
                    "model_b_estimate": summaries[model_b][(metric, k_definition)]["estimate"],
                    **delta,
                    "direction_favors_model_a": delta["delta_mean_genome"] > 0,
                }
            )
    return pd.DataFrame(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_id: str,
    architecture: ArchitectureV1,
    history: list[dict[str, Any]],
    target_definition: str,
    target_mean: float | None,
    target_std: float | None,
    rank_weight: float | None,
    created_utc: str,
) -> None:
    payload = {
        "model_id": model_id,
        "created_utc": created_utc,
        "architecture": architecture.__dict__,
        "state_dict": model.state_dict(),
        "parameter_count": trainable_parameter_count(model),
        "training_seed": SEED,
        "target_definition": target_definition,
        "target_mean_development_pool": target_mean,
        "target_std_development_pool": target_std,
        "huber_weight": 1.0 if target_mean is not None else None,
        "ranking_weight": rank_weight,
        "history": history,
        "test_accessed": False,
        "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    started = time.perf_counter()
    phase1_lock = verify_prerequisites_and_destination()
    created_utc = datetime.now(timezone.utc).isoformat()
    set_global_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    print("Phase 2: Phase 0/1 PASS and hashes verified; TEST remains sealed", flush=True)

    metadata = read_development_validation_metadata()
    cache_metadata, token_cache = build_token_cache(metadata, created_utc)
    test_output = run_unit_tests()
    del metadata

    architecture = ArchitectureV1()
    scalar_template = ScalarStudentV1(architecture).to(device)
    masked_template = MaskedSequenceComparatorV1(architecture).to(device)
    scalar_parameter_count = trainable_parameter_count(scalar_template)
    masked_parameter_count = trainable_parameter_count(masked_template)
    if architecture.receptive_field_bp != 511:
        raise Phase2Error("Unexpected receptive field")

    train_mask = cache_metadata["cache_role"].eq("train_pool").to_numpy()
    validation_mask = cache_metadata["cache_role"].eq("validation").to_numpy()
    train_positions = cache_metadata.loc[train_mask, "cache_index"].to_numpy(np.int64)
    validation_positions = cache_metadata.loc[validation_mask, "cache_index"].to_numpy(np.int64)
    train_tokens = np.asarray(token_cache[train_positions], dtype=np.uint8)
    train_metadata = cache_metadata.loc[train_mask].reset_index(drop=True)
    validation_metadata = cache_metadata.loc[validation_mask].reset_index(drop=True)
    if len(validation_metadata) != EXPECTED_VALIDATION_WINDOWS:
        raise Phase2Error("Validation token cache incomplete")

    residual_train = train_metadata["absolute_residual"].to_numpy(float)
    residual_mean = float(residual_train.mean())
    residual_std = float(residual_train.std(ddof=0))
    ppl_train = train_metadata["raw_ppl"].to_numpy(float)
    ppl_mean = float(ppl_train.mean())
    ppl_std = float(ppl_train.std(ddof=0))
    if residual_std <= 0 or ppl_std <= 0:
        raise Phase2Error("Training target has zero variance")
    residual_standardized = (residual_train - residual_mean) / residual_std
    ppl_standardized = (ppl_train - ppl_mean) / ppl_std

    print(
        f"Phase 2: scalar architecture has {scalar_parameter_count:,} trainable parameters, "
        f"{architecture.convolutional_layers} convolutional layers and RF={architecture.receptive_field_bp} bp",
        flush=True,
    )

    # M1/M2 share an identical initialization and three-epoch Huber warm-up.
    set_global_seed(SEED + 1)
    base_model = ScalarStudentV1(architecture).to(device)
    residual_pairs = PairTokenDataset(
        train_tokens,
        residual_standardized,
        train_metadata["assembly_id"].astype(str).to_numpy(),
        SEED + 10,
    )
    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_history = []
    print("Phase 2: training shared residual-Huber warm-up", flush=True)
    for epoch in range(SCALAR_WARMUP_EPOCHS):
        metrics = train_scalar_epoch(base_model, residual_pairs, base_optimizer, epoch, rank_weight=0.0)
        warmup_history.append({"stage": "shared_huber_warmup", "epoch": epoch + 1, **metrics})
        print(f"  warm-up epoch {epoch + 1}/{SCALAR_WARMUP_EPOCHS}: {metrics}", flush=True)
    base_state = copy.deepcopy(base_model.state_dict())
    del base_model, base_optimizer

    # M2 matched branch: one additional Huber-only epoch.
    model_m2 = ScalarStudentV1(architecture).to(device)
    model_m2.load_state_dict(base_state)
    optimizer_m2 = torch.optim.AdamW(model_m2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    m2_history = copy.deepcopy(warmup_history)
    for branch_epoch in range(BRANCH_EPOCHS):
        result = train_scalar_epoch(
            model_m2,
            residual_pairs,
            optimizer_m2,
            SCALAR_WARMUP_EPOCHS + branch_epoch,
            rank_weight=0.0,
        )
        m2_history.append({"stage": "matched_huber_branch", "epoch": SCALAR_WARMUP_EPOCHS + branch_epoch + 1, **result})
        print(f"  M2 matched branch: {result}", flush=True)

    # Predeclared validation tuning subset, used only to choose ranking weight.
    tuning_local_positions = sample_positions(validation_metadata, TUNING_WINDOWS_PER_VALIDATION_GENOME, "rank_weight_tuning")
    tuning_cache_positions = validation_metadata.iloc[tuning_local_positions]["cache_index"].to_numpy(np.int64)
    tuning_metadata = validation_metadata.iloc[tuning_local_positions].reset_index(drop=True)
    m1_candidates: list[dict[str, Any]] = []
    print("Phase 2: tuning ranking-loss weight on the fixed validation subset", flush=True)
    for rank_weight in RANK_WEIGHT_CANDIDATES:
        candidate = ScalarStudentV1(architecture).to(device)
        candidate.load_state_dict(base_state)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        history = copy.deepcopy(warmup_history)
        for branch_epoch in range(BRANCH_EPOCHS):
            result = train_scalar_epoch(
                candidate,
                residual_pairs,
                optimizer,
                SCALAR_WARMUP_EPOCHS + branch_epoch,
                rank_weight=rank_weight,
            )
            history.append({"stage": "ranking_branch", "epoch": SCALAR_WARMUP_EPOCHS + branch_epoch + 1, **result})
        tuning_prediction = predict_scalar(
            candidate,
            token_cache,
            tuning_cache_positions,
            residual_mean,
            residual_std,
        )
        tuning_spearman = tuning_mean_spearman(tuning_metadata, tuning_prediction)
        print(f"  rank_weight={rank_weight:.1f}: tuning mean Spearman={tuning_spearman:.6f}", flush=True)
        m1_candidates.append(
            {
                "rank_weight": rank_weight,
                "tuning_mean_spearman": tuning_spearman,
                "state_dict": copy.deepcopy(candidate.state_dict()),
                "history": history,
            }
        )
        del candidate, optimizer
    selected_m1 = max(m1_candidates, key=lambda item: (item["tuning_mean_spearman"], -item["rank_weight"]))
    selected_rank_weight = float(selected_m1["rank_weight"])
    model_m1 = ScalarStudentV1(architecture).to(device)
    model_m1.load_state_dict(selected_m1["state_dict"])
    m1_history = selected_m1["history"]
    del base_state, m1_candidates

    # M3: same architecture and selected loss weighting, raw PPL supervision.
    print("Phase 2: training M3 raw-PPL supervision", flush=True)
    set_global_seed(SEED + 3)
    model_m3 = ScalarStudentV1(architecture).to(device)
    optimizer_m3 = torch.optim.AdamW(model_m3.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ppl_pairs = PairTokenDataset(
        train_tokens,
        ppl_standardized,
        train_metadata["assembly_id"].astype(str).to_numpy(),
        SEED + 30,
    )
    m3_history = []
    for epoch in range(RAW_PPL_EPOCHS):
        rank_weight = 0.0 if epoch < SCALAR_WARMUP_EPOCHS else selected_rank_weight
        result = train_scalar_epoch(model_m3, ppl_pairs, optimizer_m3, epoch, rank_weight=rank_weight)
        m3_history.append({"stage": "raw_ppl", "epoch": epoch + 1, **result})
        print(f"  M3 epoch {epoch + 1}/{RAW_PPL_EPOCHS}: {result}", flush=True)

    # M5: same encoder, no teacher values, deterministic masked reconstruction.
    print("Phase 2: training M5 sequence-only masked reconstruction", flush=True)
    set_global_seed(SEED + 5)
    model_m5 = MaskedSequenceComparatorV1(architecture).to(device)
    optimizer_m5 = torch.optim.AdamW(model_m5.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    m5_history: list[dict[str, Any]] = []
    local_train_positions = np.arange(len(train_tokens), dtype=np.int64)
    for epoch in range(MASKED_EPOCHS):
        loss = train_masked_epoch(model_m5, train_tokens, local_train_positions, optimizer_m5, epoch)
        m5_history.append({"stage": "masked_reconstruction", "epoch": epoch + 1, "loss": loss})
        print(f"  M5 epoch {epoch + 1}/{MASKED_EPOCHS}: loss={loss:.6f}", flush=True)

    print("Phase 2: full validation inference (M1/M2/M3/M5), TEST excluded", flush=True)
    prediction_m1 = predict_scalar(model_m1, token_cache, validation_positions, residual_mean, residual_std)
    print("  M1 complete", flush=True)
    prediction_m2 = predict_scalar(model_m2, token_cache, validation_positions, residual_mean, residual_std)
    print("  M2 complete", flush=True)
    prediction_m3 = predict_scalar(model_m3, token_cache, validation_positions, ppl_mean, ppl_std)
    print("  M3 complete", flush=True)
    prediction_m5 = predict_masked_anomaly(
        model_m5,
        token_cache,
        validation_positions,
        validation_metadata["window_id"].astype(str).to_numpy(),
    )
    print("  M5 complete", flush=True)

    phase1_predictions = pd.read_parquet(
        PHASE1_PREDICTIONS,
        columns=["window_id", "split", "prediction_baseline_c_hist_gbr"],
    )
    if not phase1_predictions["split"].eq("validation").all():
        raise Phase2Error("Frozen Phase 1 predictions are not validation-only")
    cheap_lookup = phase1_predictions.set_index("window_id")["prediction_baseline_c_hist_gbr"]
    prediction_m4 = cheap_lookup.reindex(validation_metadata["window_id"]).to_numpy(float)
    if not np.isfinite(prediction_m4).all():
        raise Phase2Error("Frozen cheap baseline predictions do not cover validation")
    del phase1_predictions, cheap_lookup

    predictions_by_model = {
        "M1_full_distilled": prediction_m1,
        "M2_huber_only": prediction_m2,
        "M3_raw_ppl": prediction_m3,
        "M4_cheap_feature_baseline": prediction_m4,
        "M5_sequence_only": prediction_m5,
    }
    mae_targets = {
        "M1_full_distilled": validation_metadata["absolute_residual"].to_numpy(float),
        "M2_huber_only": validation_metadata["absolute_residual"].to_numpy(float),
        "M3_raw_ppl": validation_metadata["raw_ppl"].to_numpy(float),
        "M4_cheap_feature_baseline": validation_metadata["absolute_residual"].to_numpy(float),
        "M5_sequence_only": None,
    }
    genome_metrics = {
        model_id: full_genome_metrics(validation_metadata, prediction, model_id, mae_targets[model_id])
        for model_id, prediction in predictions_by_model.items()
    }
    validation_metrics, summaries = make_validation_metrics(genome_metrics)
    ablations = build_ablation_results(genome_metrics, summaries)

    validation_predictions = validation_metadata[
        ["window_id", "assembly_id", "cluster_id", "absolute_residual", "teacher_rank", "raw_ppl"]
    ].copy()
    for model_id, prediction in predictions_by_model.items():
        validation_predictions[f"score_{model_id}"] = prediction
    validation_predictions["split"] = "validation"
    write_parquet_atomic(validation_predictions, VALIDATION_PREDICTIONS)
    write_csv_atomic(validation_metrics, VALIDATION_METRICS)
    write_csv_atomic(ablations, ABLATION_RESULTS)

    save_checkpoint(
        CHECKPOINT_PATHS["M1_full_distilled"], model_m1, "M1_full_distilled", architecture,
        m1_history, "absolute development-fitted GC-LOWESS residual", residual_mean, residual_std,
        selected_rank_weight, created_utc,
    )
    save_checkpoint(
        CHECKPOINT_PATHS["M2_huber_only"], model_m2, "M2_huber_only", architecture,
        m2_history, "absolute development-fitted GC-LOWESS residual", residual_mean, residual_std,
        0.0, created_utc,
    )
    save_checkpoint(
        CHECKPOINT_PATHS["M3_raw_ppl"], model_m3, "M3_raw_ppl", architecture,
        m3_history, "raw Evo 2 perplexity", ppl_mean, ppl_std,
        selected_rank_weight, created_utc,
    )
    save_checkpoint(
        CHECKPOINT_PATHS["M5_sequence_only"], model_m5, "M5_sequence_only", architecture,
        m5_history, "masked nucleotide reconstruction; anomaly=deterministic masked cross-entropy",
        None, None, None, created_utc,
    )

    model_definition = {
        "version": "student_cnn_v1",
        "created_utc": created_utc,
        "proposal_compliant": True,
        "historical_cds_student_reused": False,
        "input": {"length_bp": 512, "tokens": ["A", "C", "G", "T", "N"], "mask_token": "M5 only"},
        "embedding_dim": architecture.embedding_dim,
        "channels": architecture.channels,
        "stem": {"kernel_size": architecture.stem_kernel_size, "stride": architecture.stem_stride, "padding": 3},
        "residual_blocks": len(architecture.dilation_schedule),
        "convolutional_layers": architecture.convolutional_layers,
        "convolutions_per_residual_block": 2,
        "dilation_schedule": list(architecture.dilation_schedule),
        "kernel_size": architecture.kernel_size,
        "receptive_field_bp": architecture.receptive_field_bp,
        "normalisation": f"GroupNorm({architecture.group_norm_groups} groups)",
        "activation": "GELU",
        "dropout": architecture.dropout,
        "pooling": "concatenated global mean and global max pooling",
        "scalar_head": f"Linear({2 * architecture.channels}->{architecture.head_hidden}) + GELU + Dropout + Linear({architecture.head_hidden}->1)",
        "scalar_parameter_count": scalar_parameter_count,
        "sequence_comparator_decoder": "ConvTranspose1d stride 2 + GELU + 1x1 nucleotide logits",
        "sequence_comparator_parameter_count": masked_parameter_count,
        "same_encoder_for_M1_M2_M3_M5": True,
    }
    write_text_atomic(MODEL_DEFINITION, yaml.safe_dump(model_definition, sort_keys=False, width=1000))

    training_config = {
        "version": "phase2_training_v1",
        "created_utc": created_utc,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "seed": SEED,
        "test_accessed": False,
        "partitions_materialized": ["development", "validation"],
        "training_pool": {
            "rule": f"{TRAIN_WINDOWS_PER_GENOME} deterministic windows per development genome",
            "rows": int(len(train_metadata)),
            "genomes": int(train_metadata["assembly_id"].nunique()),
            "clusters": int(train_metadata["cluster_id"].nunique()),
            "limitation": "CPU-bounded Phase 2 architecture/ablation screening; this is not full-development-window training.",
        },
        "validation": {
            "full_rows": int(len(validation_metadata)),
            "genomes": int(validation_metadata["assembly_id"].nunique()),
            "clusters": int(validation_metadata["cluster_id"].nunique()),
            "ranking_weight_tuning_subset": f"{TUNING_WINDOWS_PER_VALIDATION_GENOME} deterministic windows per validation genome",
        },
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": 1.0},
        "scalar_target_standardisation": "development training-pool mean and population standard deviation only",
        "huber": {"delta_standardised_units": 1.0, "weight": 1.0},
        "pairwise_ranking": {
            "loss": "softplus(-sign(y_a-y_b)*(prediction_a-prediction_b))",
            "same_genome_pairs_only": True,
            "candidate_weights": RANK_WEIGHT_CANDIDATES,
            "selected_weight": selected_rank_weight,
            "ranking_emphasis_greater_than_huber": selected_rank_weight > 1.0,
        },
        "epochs": {
            "shared_residual_huber_warmup": SCALAR_WARMUP_EPOCHS,
            "M1_or_M2_matched_branch": BRANCH_EPOCHS,
            "M3_raw_ppl": RAW_PPL_EPOCHS,
            "M5_masked_reconstruction": MASKED_EPOCHS,
        },
        "batch_sizes": {"pair_training": PAIR_BATCH_SIZE, "masked_training": SINGLE_BATCH_SIZE, "validation": EVAL_BATCH_SIZE},
        "M5": {
            "teacher_values_used": False,
            "annotations_used": False,
            "mask_probability_training": MASK_PROBABILITY,
            "validation_anomaly_score": "Mean cross-entropy at positions p mod 7 = SHA256(window_id) mod 7, with those positions masked; one deterministic pass per window.",
        },
        "frozen_K_definitions_from_phase1": phase1_lock["frozen_k_definitions"],
        "primary_selection_criterion": "mean within-genome Spearman on validation",
        "unit_test_output": test_output,
        "software": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "phase0_report": file_record(PHASE0_REPORT),
            "protocol": file_record(PROTOCOL),
            "split": file_record(SPLIT),
            "test_lock": file_record(TEST_LOCK),
            "canonical": file_record(CANONICAL),
            "targets": file_record(TARGETS),
            "phase1_report": file_record(PHASE1_REPORT),
            "phase1_config": file_record(PHASE1_CONFIG),
            "phase1_lock": file_record(PHASE1_LOCK),
            "phase1_predictions": file_record(PHASE1_PREDICTIONS),
            "token_cache_manifest": file_record(CACHE_MANIFEST),
        },
    }
    write_text_atomic(TRAINING_CONFIG, yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True, width=1000))

    def estimate(model_id: str, metric: str, k: str = "") -> float:
        return summaries[model_id][(metric, k)]["estimate"]

    m1_spearman = estimate("M1_full_distilled", "spearman")
    m2_spearman = estimate("M2_huber_only", "spearman")
    m3_spearman = estimate("M3_raw_ppl", "spearman")
    m4_spearman = estimate("M4_cheap_feature_baseline", "spearman")
    m5_spearman = estimate("M5_sequence_only", "spearman")
    rank_delta = m1_spearman - m2_spearman
    residual_raw_delta = m1_spearman - m3_spearman
    teacher_cheap_delta = m1_spearman - m4_spearman
    teacher_sequence_delta = m1_spearman - m5_spearman
    architecture_justified = m1_spearman > m4_spearman and m1_spearman > m5_spearman

    report = f"""# Phase 2 single-model distillation and ablation report

## Verdict

**PHASE 2: PASS**

Created UTC: `{created_utc}`  
Device: **CPU** (no CUDA runtime available)  
Development training pool: **{len(train_metadata):,} windows / {train_metadata['assembly_id'].nunique()} genomes / {train_metadata['cluster_id'].nunique()} lineages**  
Validation: **{len(validation_metadata):,} windows / {EXPECTED_VALIDATION_GENOMES} genomes / {EXPECTED_VALIDATION_CLUSTERS} lineages**  
Locked TEST accessed: **NO**

## Architecture

`student_cnn_v1` is a newly implemented 512-bp DNA encoder, not the historical CDS StudentCNN. It uses an 8-dimensional nucleotide embedding, a kernel-7 stride-2 stem, six dilated residual blocks with dilation schedule `[1, 2, 4, 8, 16, 32]`, 16 channels, GroupNorm, GELU, dropout 0.10, and concatenated global mean/max pooling followed by a scalar MLP. The scalar model has **{scalar_parameter_count:,} trainable parameters**, **{architecture.convolutional_layers} convolutional layers**, and a calculated receptive field of **{architecture.receptive_field_bp} bp**.

The M5 comparator uses the identical encoder and a reconstruction decoder. It was trained only on masked development DNA; no PPL, residual or annotation entered its objective.

## Training design and limitation

Because the available PyTorch runtime is CPU-only on a 4-core laptop, Phase 2 used a frozen, genome-balanced architecture-screening pool of {TRAIN_WINDOWS_PER_GENOME} windows per development genome. All {EXPECTED_VALIDATION_WINDOWS:,} validation windows were evaluated. This limitation is explicit: the results justify controlled Phase 2 model selection under the local compute budget, not a claim that the CNN has exhausted all 1,640,110 development windows.

M1 and M2 share the same three-epoch Huber warm-up and differ only in the matched final branch. M1 ranking weights {RANK_WEIGHT_CANDIDATES} were compared on a deterministic {TUNING_WINDOWS_PER_VALIDATION_GENOME}-window-per-genome validation subset; **{selected_rank_weight:.1f}** was selected. Since Huber weight is 1.0, ranking receives greater emphasis. M3 uses the same backbone and selected ranking weight but raw PPL targets. M4 is loaded unchanged from the frozen Phase 1 result.

## Validation results

| Model | Supervision | Mean within-genome Spearman | Top-1% Recall@K | Top-1% NDCG@K | Diagnostic MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 full distilled | residual Huber + within-genome ranking | {m1_spearman:.6f} | {estimate('M1_full_distilled','recall_at_k','top_1_percent'):.6f} | {estimate('M1_full_distilled','ndcg_at_k','top_1_percent'):.6f} | {estimate('M1_full_distilled','scalar_mae_training_target'):.6f} |
| M2 no ranking | residual Huber only | {m2_spearman:.6f} | {estimate('M2_huber_only','recall_at_k','top_1_percent'):.6f} | {estimate('M2_huber_only','ndcg_at_k','top_1_percent'):.6f} | {estimate('M2_huber_only','scalar_mae_training_target'):.6f} |
| M3 raw teacher | raw-PPL Huber + ranking | {m3_spearman:.6f} | {estimate('M3_raw_ppl','recall_at_k','top_1_percent'):.6f} | {estimate('M3_raw_ppl','ndcg_at_k','top_1_percent'):.6f} | {estimate('M3_raw_ppl','scalar_mae_training_target'):.6f} (PPL units) |
| M4 frozen cheap baseline | Phase 1 GC/k-mer HistGBR | {m4_spearman:.6f} | {estimate('M4_cheap_feature_baseline','recall_at_k','top_1_percent'):.6f} | {estimate('M4_cheap_feature_baseline','ndcg_at_k','top_1_percent'):.6f} | {estimate('M4_cheap_feature_baseline','scalar_mae_training_target'):.6f} |
| M5 sequence-only | masked nucleotide reconstruction | {m5_spearman:.6f} | {estimate('M5_sequence_only','recall_at_k','top_1_percent'):.6f} | {estimate('M5_sequence_only','ndcg_at_k','top_1_percent'):.6f} | n/a |

All ranking metrics were calculated within each validation genome using the Phase 1 frozen top-1%, top-5% and fixed-100 definitions. Windows were not treated as independent inferential replicates; uncertainty summaries use a {BOOTSTRAP_ITERATIONS}-iteration bootstrap over the {EXPECTED_VALIDATION_CLUSTERS} validation lineages.

## Ablation decisions

1. **Teacher distillation versus cheap features:** M1 minus M4 mean Spearman = **{teacher_cheap_delta:+.6f}**. {'M1 exceeds the frozen cheap baseline.' if teacher_cheap_delta > 0 else 'M1 does not exceed the frozen cheap baseline under the Phase 2 CPU-bounded training budget.'}
2. **Teacher supervision versus sequence-only learning:** M1 minus M5 = **{teacher_sequence_delta:+.6f}**. {'Teacher supervision improves ranking.' if teacher_sequence_delta > 0 else 'Teacher supervision does not improve ranking in this run.'}
3. **Pairwise ranking loss:** M1 minus M2 = **{rank_delta:+.6f}**. {'The selected ranking loss improves fidelity.' if rank_delta > 0 else 'The ranking-loss branch does not improve mean Spearman over matched Huber-only training.'}
4. **Residual versus raw-PPL supervision:** M1 minus M3 = **{residual_raw_delta:+.6f}**. {'Residual supervision improves the proposal-aligned ranking.' if residual_raw_delta > 0 else 'Residual supervision does not outperform raw-PPL supervision in this run.'}
5. **CNN justification:** **{'YES' if architecture_justified else 'LIMITED'}**. {'The compact sequence model beats both cheap and sequence-only comparators on the primary validation endpoint.' if architecture_justified else 'The architecture is reproducible and proposal-compliant, but the present screening budget does not establish superiority over both comparators.'}

The exact paired genome/lineage deltas and confidence intervals are in `ablation_results.csv`; biological claims are outside this phase.

## Phase 3 freeze

The architecture is frozen as `student_cnn_v1`: 16 channels, dilation `[1,2,4,8,16,32]`, GroupNorm, dropout 0.10, mean+max pooling and the recorded scalar head. The Phase 3 primary objective is M1 with Huber weight 1.0 and ranking weight **{selected_rank_weight:.1f}**. Checkpoint hashes, target standardisation, training budget, seeds and K definitions are sealed in `final_architecture.lock`.

## Boundary

Phase 2 stops here. No test performance was accessed. Phase 3 must use the frozen architecture/objective or create an explicitly versioned deviation without consulting TEST.
"""
    write_text_atomic(REPORT, report)

    output_records = {
        "model_definition": file_record(MODEL_DEFINITION),
        "training_config": file_record(TRAINING_CONFIG),
        "validation_metrics": file_record(VALIDATION_METRICS),
        "ablation_results": file_record(ABLATION_RESULTS),
        "validation_predictions": file_record(VALIDATION_PREDICTIONS),
        "report": file_record(REPORT),
        "checkpoints": {key: file_record(path) for key, path in CHECKPOINT_PATHS.items()},
        "token_cache_manifest": file_record(CACHE_MANIFEST),
    }
    lock_payload = {
        "lock": "DISSERTATION PHASE 2 ARCHITECTURE FROZEN",
        "created_utc": created_utc,
        "architecture_version": "student_cnn_v1",
        "architecture": model_definition,
        "phase3_primary_model": "M1_full_distilled",
        "phase3_objective": {"Huber_weight": 1.0, "within_genome_pairwise_ranking_weight": selected_rank_weight},
        "training_seed": SEED,
        "frozen_k_definitions": phase1_lock["frozen_k_definitions"],
        "validation_mean_spearman": m1_spearman,
        "cheap_baseline_spearman": m4_spearman,
        "sequence_only_spearman": m5_spearman,
        "ranking_loss_delta": rank_delta,
        "residual_vs_raw_delta": residual_raw_delta,
        "architecture_justified_against_both_comparators": architecture_justified,
        "outputs": output_records,
        "test_accessed": False,
        "statement": "Architecture, primary objective and Phase 1 K definitions are frozen for Phase 3. TEST remains sealed.",
    }
    write_text_atomic(FINAL_LOCK, json.dumps(lock_payload, indent=2) + "\n")

    pass_checks = {
        "phase0_pass": True,
        "phase1_pass": True,
        "unit_tests_pass": "OK" in test_output,
        "train_genomes": train_metadata["assembly_id"].nunique() == EXPECTED_DEVELOPMENT_GENOMES,
        "validation_rows": len(validation_predictions) == EXPECTED_VALIDATION_WINDOWS,
        "validation_genomes": validation_predictions["assembly_id"].nunique() == EXPECTED_VALIDATION_GENOMES,
        "validation_lineages": validation_predictions["cluster_id"].nunique() == EXPECTED_VALIDATION_CLUSTERS,
        "metrics_finite": np.isfinite(validation_metrics["estimate"].to_numpy(float)).all(),
        "ranking_weight_greater_than_huber": selected_rank_weight > 1.0,
        "same_genome_pair_sampler": True,
        "architecture_lock_exists": FINAL_LOCK.is_file(),
        "test_not_accessed": True,
    }
    if not all(pass_checks.values()):
        raise Phase2Error(f"Phase 2 validation failed: {pass_checks}")

    for artifact in (MODEL_DEFINITION, TRAINING_CONFIG, VALIDATION_METRICS, ABLATION_RESULTS, REPORT, FINAL_LOCK):
        shutil.copy2(artifact, OUTPUT_MIRROR / artifact.name)

    elapsed = time.perf_counter() - started
    print(f"Phase 2 runtime seconds: {elapsed:.1f}")
    print("PHASE 2: PASS")
    print(f"Best distilled validation Spearman: {max(m1_spearman, m2_spearman):.6f}")
    print(f"Cheap baseline Spearman: {m4_spearman:.6f}")
    print(f"Non-distilled comparator Spearman: {m5_spearman:.6f}")
    print(f"Ranking-loss delta: {rank_delta:+.6f}")
    print(f"Residual-vs-raw delta: {residual_raw_delta:+.6f}")
    print("Architecture frozen: YES")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
