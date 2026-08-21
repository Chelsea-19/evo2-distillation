"""Train and freeze the Phase 3 five-member deep ensemble.

Only the frozen Phase 2 development token pool and the full validation cache are
used. Locked TEST rows are never materialized, predicted, or evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import torch
import yaml
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader

WORKSPACE = Path(r"C:\Users\LDD\Documents\Codex\2026-08-19\za")
ROOT = WORKSPACE / "experiments" / "dissertation_v1"
PHASE2 = ROOT / "phase2"
PHASE3 = ROOT / "phase3"
CHECKPOINTS = PHASE3 / "checkpoints"
OUTPUT_MIRROR = WORKSPACE / "outputs" / "phase3"
RUN_SPEC = PHASE3 / "phase3_run_spec.json"

ENSEMBLE_CONFIG = PHASE3 / "ensemble_config.yaml"
MEMBER_METRICS = PHASE3 / "member_metrics.csv"
VALIDATION_PREDICTIONS = PHASE3 / "validation_ensemble_predictions.parquet"
DIAGNOSTICS = PHASE3 / "ensemble_diagnostics.md"
REPORT = PHASE3 / "phase3_report.md"
ENSEMBLE_LOCK = PHASE3 / "ensemble.lock"

PROTOCOL = ROOT / "protocol" / "experiment_protocol_v1.yaml"
PHASE2_LOCK = PHASE2 / "final_architecture.lock"
PHASE2_CONFIG = PHASE2 / "training_config.yaml"
PHASE2_MODEL = PHASE2 / "model_definition.yaml"
PHASE2_M1 = PHASE2 / "checkpoints" / "M1_full_distilled.pt"
CACHE_METADATA = PHASE2 / "cache" / "token_cache_metadata.parquet"
CACHE_TOKENS = PHASE2 / "cache" / "window_tokens.uint8.mmap"
CACHE_MANIFEST = PHASE2 / "cache" / "token_cache_manifest.json"

sys.path.insert(0, str(ROOT / "scripts"))
import run_phase2_distillation as p2  # noqa: E402

SEEDS = [11, 23, 37, 53, 71]
EPOCHS = 4
HUBER_WARMUP_EPOCHS = 3
RANK_WEIGHT = 4.0
HUBER_WEIGHT = 1.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
TUNING_WINDOWS_PER_GENOME = 512
BOOTSTRAP_ITERATIONS = 2_000
EXPECTED_TRAIN_ROWS = 52_992
EXPECTED_TRAIN_GENOMES = 207
EXPECTED_VALIDATION_ROWS = 353_840
EXPECTED_VALIDATION_GENOMES = 45
EXPECTED_VALIDATION_LINEAGES = 19

FINAL_OUTPUTS = [
    ENSEMBLE_CONFIG,
    MEMBER_METRICS,
    VALIDATION_PREDICTIONS,
    DIAGNOSTICS,
    REPORT,
    ENSEMBLE_LOCK,
]


class Phase3Error(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase3Error(f"Required file missing: {path}")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def iter_output_records(outputs: dict[str, Any]):
    for key, record in outputs.items():
        if key == "checkpoints":
            yield from record.values()
        else:
            yield record


def verify_phase2_and_prepare() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if any(path.exists() for path in FINAL_OUTPUTS):
        existing = [str(path) for path in FINAL_OUTPUTS if path.exists()]
        raise Phase3Error("Refusing to overwrite frozen/final Phase 3 outputs:\n" + "\n".join(existing))
    lock = json.loads(PHASE2_LOCK.read_text(encoding="utf-8"))
    if lock.get("test_accessed") is not False:
        raise Phase3Error("Phase 2 lock does not confirm TEST remained sealed")
    for record in iter_output_records(lock["outputs"]):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise Phase3Error(f"Phase 2 locked artifact mismatch: {path}")
    config = yaml.safe_load(PHASE2_CONFIG.read_text(encoding="utf-8"))
    model_definition = yaml.safe_load(PHASE2_MODEL.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol_text = PROTOCOL.read_text(encoding="utf-8").lower()
    if "ensemble_seed" in protocol_text or "ensemble seeds" in protocol_text:
        raise Phase3Error("Protocol appears to define ensemble seeds; explicit parser update required")
    expected = {
        "channels": 16,
        "dilation_schedule": [1, 2, 4, 8, 16, 32],
        "convolutional_layers": 13,
        "receptive_field_bp": 511,
        "scalar_parameter_count": 11_873,
    }
    for key, value in expected.items():
        if model_definition.get(key) != value:
            raise Phase3Error(f"Frozen architecture mismatch for {key}")
    if config["pairwise_ranking"]["selected_weight"] != RANK_WEIGHT:
        raise Phase3Error("Frozen ranking weight mismatch")
    if config["epochs"]["shared_residual_huber_warmup"] != HUBER_WARMUP_EPOCHS:
        raise Phase3Error("Frozen warm-up schedule mismatch")
    if config["optimizer"]["learning_rate"] != LEARNING_RATE or config["optimizer"]["weight_decay"] != WEIGHT_DECAY:
        raise Phase3Error("Frozen optimizer mismatch")
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    if cache_manifest.get("test_rows") != 0:
        raise Phase3Error("Phase 2 cache manifest contains TEST rows")
    for directory in (PHASE3, CHECKPOINTS, OUTPUT_MIRROR):
        directory.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "version": "phase3_run_spec_v1",
        "seeds": SEEDS,
        "seed_source": "Task-prespecified example; protocol contains no ensemble-specific seeds",
        "architecture_version": "student_cnn_v1",
        "phase2_lock_sha256": sha256_file(PHASE2_LOCK),
        "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
        "epochs": EPOCHS,
        "warmup_epochs": HUBER_WARMUP_EPOCHS,
        "huber_weight": HUBER_WEIGHT,
        "ranking_weight_final_epoch": RANK_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "test_accessed": False,
    }
    if RUN_SPEC.exists():
        existing = json.loads(RUN_SPEC.read_text(encoding="utf-8"))
        if existing != run_spec:
            raise Phase3Error("Resume run specification differs from the frozen Phase 3 specification")
    else:
        write_text_atomic(RUN_SPEC, json.dumps(run_spec, indent=2) + "\n")
    return lock, config, protocol


def member_checkpoint_path(seed: int, epoch: int) -> Path:
    return CHECKPOINTS / f"member_seed{seed}_epoch{epoch:02d}.pt"


def save_member_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch_completed: int,
    history: list[dict[str, Any]],
    cumulative_runtime_seconds: float,
    target_mean: float,
    target_std: float,
) -> None:
    payload = {
        "checkpoint_kind": "phase3_resume_safe_member",
        "architecture_version": "student_cnn_v1",
        "phase2_lock_sha256": sha256_file(PHASE2_LOCK),
        "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
        "seed": seed,
        "epoch_completed": epoch_completed,
        "total_epochs": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "cumulative_runtime_seconds": cumulative_runtime_seconds,
        "target_mean": target_mean,
        "target_std": target_std,
        "hyperparameters": {
            "huber_weight": HUBER_WEIGHT,
            "ranking_weight_final_epoch": RANK_WEIGHT,
            "warmup_epochs": HUBER_WARMUP_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
        },
        "test_accessed": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_latest_checkpoint(seed: int) -> tuple[dict[str, Any] | None, Path | None]:
    paths = sorted(CHECKPOINTS.glob(f"member_seed{seed}_epoch*.pt"))
    if not paths:
        return None, None
    latest = paths[-1]
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("phase2_lock_sha256") != sha256_file(PHASE2_LOCK):
        raise Phase3Error(f"Resume checkpoint metadata mismatch: {latest}")
    if payload.get("cache_manifest_sha256") != sha256_file(CACHE_MANIFEST) or payload.get("test_accessed") is not False:
        raise Phase3Error(f"Resume checkpoint data boundary mismatch: {latest}")
    if payload.get("epoch_completed") != int(latest.stem.rsplit("epoch", 1)[1]):
        raise Phase3Error(f"Resume checkpoint filename/epoch mismatch: {latest}")
    return payload, latest


def train_epoch(
    model: p2.ScalarStudentV1,
    dataset: p2.PairTokenDataset,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch_index: int,
    rank_weight: float,
) -> dict[str, float]:
    model.train()
    dataset.set_epoch(epoch_index)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed * 100_000 + epoch_index),
    )
    huber_fn = nn.HuberLoss(delta=1.0)
    totals = {"loss": 0.0, "huber": 0.0, "ranking": 0.0}
    batches = 0
    for tokens_a, tokens_b, target_a, target_b in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.cat([tokens_a, tokens_b], dim=0))
        prediction_a, prediction_b = prediction.chunk(2)
        huber = 0.5 * (huber_fn(prediction_a, target_a) + huber_fn(prediction_b, target_b))
        ranking = p2.pairwise_logistic_ranking_loss(prediction_a, prediction_b, target_a, target_b)
        loss = HUBER_WEIGHT * huber + rank_weight * ranking
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["huber"] += float(huber.detach())
        totals["ranking"] += float(ranking.detach())
        batches += 1
    return {key: value / batches for key, value in totals.items()}


def predict(
    model: p2.ScalarStudentV1,
    tokens: np.ndarray,
    positions: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        p2.TokenIndexDataset(tokens, positions),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            outputs.append(model(batch).numpy())
    return np.concatenate(outputs).astype(float) * target_std + target_mean


def mean_within_genome_spearman(metadata: pd.DataFrame, scores: np.ndarray) -> float:
    values = []
    for _, positions in metadata.groupby("assembly_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        rho = float(spearmanr(metadata.iloc[idx]["teacher_rank"], scores[idx]).statistic)
        if not np.isfinite(rho):
            raise Phase3Error("Non-finite within-genome Spearman")
        values.append(rho)
    return float(np.mean(values))


def summarize_full_metrics(model_id: str, metadata: pd.DataFrame, scores: np.ndarray) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    frame = p2.full_genome_metrics(
        metadata,
        scores,
        model_id,
        metadata["absolute_residual"].to_numpy(float),
    )
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for (metric, k_definition), subset in frame.groupby(["metric", "k_definition"], dropna=False, sort=True):
        summaries[(str(metric), str(k_definition))] = p2.summarize_metric(
            subset,
            f"phase3|{model_id}|{metric}|{k_definition}",
        )
    return frame, summaries


def empty_metric_row() -> dict[str, Any]:
    return {
        "record_type": "",
        "entity_id": "",
        "seed": np.nan,
        "epoch": np.nan,
        "validation_scope": "",
        "metric": "",
        "k_definition": "",
        "estimate": np.nan,
        "median_across_genomes": np.nan,
        "mean_across_lineages": np.nan,
        "lineage_bootstrap_ci_low": np.nan,
        "lineage_bootstrap_ci_high": np.nan,
        "n_genomes": np.nan,
        "n_lineages": np.nan,
        "training_loss": np.nan,
        "huber_loss": np.nan,
        "ranking_loss": np.nan,
        "rank_weight": np.nan,
        "epoch_runtime_seconds": np.nan,
        "cumulative_training_runtime_seconds": np.nan,
        "checkpoint_sha256": "",
        "status": "",
        "notes": "",
    }


def main() -> None:
    started = time.perf_counter()
    phase2_lock, phase2_config, protocol = verify_phase2_and_prepare()
    created_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    print("Phase 3: Phase 2 lock and hashes verified; TEST remains sealed", flush=True)
    print(f"Phase 3: frozen seeds {SEEDS}", flush=True)

    cache_metadata = pd.read_parquet(CACHE_METADATA)
    if set(cache_metadata["split"].unique()) != {"development", "validation"}:
        raise Phase3Error("Cache contains an unexpected partition")
    train_mask = cache_metadata["cache_role"].eq("train_pool").to_numpy()
    validation_mask = cache_metadata["cache_role"].eq("validation").to_numpy()
    train_metadata = cache_metadata.loc[train_mask].reset_index(drop=True)
    validation_metadata = cache_metadata.loc[validation_mask].reset_index(drop=True)
    if len(train_metadata) != EXPECTED_TRAIN_ROWS or train_metadata["assembly_id"].nunique() != EXPECTED_TRAIN_GENOMES:
        raise Phase3Error("Frozen training pool does not match Phase 2")
    if len(validation_metadata) != EXPECTED_VALIDATION_ROWS:
        raise Phase3Error("Validation cache is incomplete")
    token_cache = np.memmap(CACHE_TOKENS, mode="r", dtype=np.uint8, shape=(len(cache_metadata), 512))
    train_positions = train_metadata["cache_index"].to_numpy(np.int64)
    validation_positions = validation_metadata["cache_index"].to_numpy(np.int64)
    train_tokens = np.asarray(token_cache[train_positions], dtype=np.uint8)

    phase2_m1 = torch.load(PHASE2_M1, map_location="cpu", weights_only=False)
    target_mean = float(phase2_m1["target_mean_development_pool"])
    target_std = float(phase2_m1["target_std_development_pool"])
    residual = train_metadata["absolute_residual"].to_numpy(float)
    standardized_target = (residual - target_mean) / target_std
    if not np.allclose(target_mean, residual.mean()) or not np.allclose(target_std, residual.std(ddof=0)):
        raise Phase3Error("Frozen target standardization differs from Phase 2")

    tuning_local = p2.sample_positions(validation_metadata, TUNING_WINDOWS_PER_GENOME, "rank_weight_tuning")
    tuning_metadata = validation_metadata.iloc[tuning_local].reset_index(drop=True)
    tuning_positions = tuning_metadata["cache_index"].to_numpy(np.int64)
    architecture = p2.ArchitectureV1()
    member_predictions: dict[int, np.ndarray] = {}
    member_histories: dict[int, list[dict[str, Any]]] = {}
    member_checkpoints: dict[int, Path] = {}
    member_inference_seconds: dict[int, float] = {}

    for member_index, seed in enumerate(SEEDS, start=1):
        print(f"Phase 3: member {member_index}/5 seed={seed}", flush=True)
        set_seed(seed)
        model = p2.ScalarStudentV1(architecture)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        dataset = p2.PairTokenDataset(
            train_tokens,
            standardized_target,
            train_metadata["assembly_id"].astype(str).to_numpy(),
            seed,
        )
        resume, resume_path = load_latest_checkpoint(seed)
        history: list[dict[str, Any]] = []
        cumulative_runtime = 0.0
        start_epoch = 0
        if resume is not None:
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            history = list(resume["history"])
            cumulative_runtime = float(resume["cumulative_runtime_seconds"])
            start_epoch = int(resume["epoch_completed"])
            print(f"  resumed from {resume_path.name} (epoch {start_epoch}/{EPOCHS})", flush=True)
        for epoch_index in range(start_epoch, EPOCHS):
            epoch_started = time.perf_counter()
            rank_weight = 0.0 if epoch_index < HUBER_WARMUP_EPOCHS else RANK_WEIGHT
            losses = train_epoch(model, dataset, optimizer, seed, epoch_index, rank_weight)
            tuning_scores = predict(model, token_cache, tuning_positions, target_mean, target_std)
            tuning_spearman = mean_within_genome_spearman(tuning_metadata, tuning_scores)
            epoch_runtime = time.perf_counter() - epoch_started
            cumulative_runtime += epoch_runtime
            history.append(
                {
                    "epoch": epoch_index + 1,
                    "training_loss": losses["loss"],
                    "huber_loss": losses["huber"],
                    "ranking_loss": losses["ranking"],
                    "rank_weight": rank_weight,
                    "validation_subset_mean_within_genome_spearman": tuning_spearman,
                    "epoch_runtime_seconds": epoch_runtime,
                    "cumulative_training_runtime_seconds": cumulative_runtime,
                }
            )
            checkpoint = member_checkpoint_path(seed, epoch_index + 1)
            save_member_checkpoint(
                checkpoint,
                model,
                optimizer,
                seed,
                epoch_index + 1,
                history,
                cumulative_runtime,
                target_mean,
                target_std,
            )
            print(
                f"  epoch {epoch_index + 1}/{EPOCHS}: loss={losses['loss']:.6f}, "
                f"tuning Spearman={tuning_spearman:.6f}, runtime={epoch_runtime:.1f}s",
                flush=True,
            )
        final_checkpoint = member_checkpoint_path(seed, EPOCHS)
        if not final_checkpoint.is_file():
            raise Phase3Error(f"Final checkpoint missing for seed {seed}")
        inference_started = time.perf_counter()
        scores = predict(model, token_cache, validation_positions, target_mean, target_std)
        inference_seconds = time.perf_counter() - inference_started
        if len(scores) != EXPECTED_VALIDATION_ROWS or not np.isfinite(scores).all() or np.std(scores) == 0:
            raise Phase3Error(f"Invalid validation predictions for seed {seed}")
        member_predictions[seed] = scores
        member_histories[seed] = history
        member_checkpoints[seed] = final_checkpoint
        member_inference_seconds[seed] = inference_seconds
        print(f"  full validation inference complete in {inference_seconds:.1f}s", flush=True)

    prediction_matrix = np.column_stack([member_predictions[seed] for seed in SEEDS])
    ensemble_mean = prediction_matrix.mean(axis=1)
    ensemble_variance = prediction_matrix.var(axis=1, ddof=1)
    if not np.isfinite(ensemble_mean).all() or not np.isfinite(ensemble_variance).all() or np.any(ensemble_variance < 0):
        raise Phase3Error("Invalid ensemble mean or variance")

    genome_metrics: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for seed in SEEDS:
        entity = f"member_seed_{seed}"
        genome_metrics[entity], summaries[entity] = summarize_full_metrics(entity, validation_metadata, member_predictions[seed])
    genome_metrics["ensemble_mean"], summaries["ensemble_mean"] = summarize_full_metrics(
        "ensemble_mean", validation_metadata, ensemble_mean
    )

    metric_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        checkpoint_hash = sha256_file(member_checkpoints[seed])
        for curve in member_histories[seed]:
            row = empty_metric_row()
            row.update(
                {
                    "record_type": "learning_curve",
                    "entity_id": f"member_seed_{seed}",
                    "seed": seed,
                    "epoch": curve["epoch"],
                    "validation_scope": f"fixed_{TUNING_WINDOWS_PER_GENOME}_windows_per_validation_genome",
                    "metric": "mean_within_genome_spearman",
                    "estimate": curve["validation_subset_mean_within_genome_spearman"],
                    "training_loss": curve["training_loss"],
                    "huber_loss": curve["huber_loss"],
                    "ranking_loss": curve["ranking_loss"],
                    "rank_weight": curve["rank_weight"],
                    "epoch_runtime_seconds": curve["epoch_runtime_seconds"],
                    "cumulative_training_runtime_seconds": curve["cumulative_training_runtime_seconds"],
                    "checkpoint_sha256": checkpoint_hash if curve["epoch"] == EPOCHS else "see_epoch_checkpoint",
                    "status": "valid",
                    "notes": "CPU runtime; peak GPU memory not available",
                }
            )
            metric_rows.append(row)

    for entity, entity_summaries in summaries.items():
        seed_value = int(entity.rsplit("_", 1)[1]) if entity.startswith("member_seed_") else np.nan
        checkpoint_hash = sha256_file(member_checkpoints[int(seed_value)]) if np.isfinite(seed_value) else ""
        for (metric, k_definition), summary in entity_summaries.items():
            row = empty_metric_row()
            row.update(
                {
                    "record_type": "final_validation_metric",
                    "entity_id": entity,
                    "seed": seed_value,
                    "epoch": EPOCHS if np.isfinite(seed_value) else np.nan,
                    "validation_scope": "full_validation",
                    "metric": metric,
                    "k_definition": k_definition,
                    "estimate": summary["estimate"],
                    "median_across_genomes": summary["median_across_genomes"],
                    "mean_across_lineages": summary["mean_across_lineages"],
                    "lineage_bootstrap_ci_low": summary["lineage_bootstrap_ci_low"],
                    "lineage_bootstrap_ci_high": summary["lineage_bootstrap_ci_high"],
                    "n_genomes": summary["n_genomes"],
                    "n_lineages": summary["n_lineages"],
                    "cumulative_training_runtime_seconds": member_histories[int(seed_value)][-1]["cumulative_training_runtime_seconds"] if np.isfinite(seed_value) else np.nan,
                    "checkpoint_sha256": checkpoint_hash,
                    "status": "valid",
                    "notes": f"full inference seconds={member_inference_seconds[int(seed_value)]:.3f}; peak GPU memory n/a" if np.isfinite(seed_value) else "mean of five frozen member ranking scores",
                }
            )
            metric_rows.append(row)

    global_pearson_values = []
    within_genome_pair_values = []
    pairwise_rows = []
    for left_index, left_seed in enumerate(SEEDS):
        for right_seed in SEEDS[left_index + 1 :]:
            left = member_predictions[left_seed]
            right = member_predictions[right_seed]
            global_pearson = float(np.corrcoef(left, right)[0, 1])
            within_values = []
            for _, positions in validation_metadata.groupby("assembly_id", sort=True).indices.items():
                idx = np.asarray(positions, dtype=np.int64)
                within_values.append(float(spearmanr(left[idx], right[idx]).statistic))
            within_mean = float(np.mean(within_values))
            global_pearson_values.append(global_pearson)
            within_genome_pair_values.append(within_mean)
            pairwise_rows.append((left_seed, right_seed, global_pearson, within_mean))
            for metric_name, value in [
                ("pairwise_global_pearson", global_pearson),
                ("pairwise_mean_within_genome_spearman", within_mean),
            ]:
                row = empty_metric_row()
                row.update(
                    {
                        "record_type": "ensemble_diagnostic",
                        "entity_id": f"seed_{left_seed}_vs_seed_{right_seed}",
                        "validation_scope": "full_validation",
                        "metric": metric_name,
                        "estimate": value,
                        "status": "valid",
                    }
                )
                metric_rows.append(row)

    quantiles = [("min", 0.0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q90", 0.9), ("q95", 0.95), ("q99", 0.99), ("max", 1.0)]
    variance_values = {label: float(np.quantile(ensemble_variance, q)) for label, q in quantiles}
    variance_values["mean"] = float(ensemble_variance.mean())
    for label, value in variance_values.items():
        row = empty_metric_row()
        row.update(
            {
                "record_type": "ensemble_diagnostic",
                "entity_id": "ensemble_mean",
                "validation_scope": "full_validation",
                "metric": "prediction_variance_distribution",
                "k_definition": label,
                "estimate": value,
                "status": "valid",
                "notes": "sample variance (ddof=1) across five scalar ranking scores",
            }
        )
        metric_rows.append(row)

    member_spearman = {seed: summaries[f"member_seed_{seed}"][("spearman", "")]["estimate"] for seed in SEEDS}
    ensemble_spearman = summaries["ensemble_mean"][("spearman", "")]["estimate"]
    best_seed = max(member_spearman, key=member_spearman.get)
    best_single = member_spearman[best_seed]
    mean_single = float(np.mean(list(member_spearman.values())))
    checkpoint_hashes = {seed: sha256_file(member_checkpoints[seed]) for seed in SEEDS}
    unique_checkpoints = len(set(checkpoint_hashes.values())) == len(SEEDS)
    distinct_predictions = all(
        not np.array_equal(member_predictions[left], member_predictions[right])
        for i, left in enumerate(SEEDS)
        for right in SEEDS[i + 1 :]
    )
    pathological = {
        seed: not (
            np.isfinite(member_predictions[seed]).all()
            and np.std(member_predictions[seed]) > 0
            and all(np.isfinite(curve["training_loss"]) for curve in member_histories[seed])
            and member_checkpoints[seed].is_file()
        )
        for seed in SEEDS
    }
    valid_members = sum(not value for value in pathological.values())
    if valid_members != 5:
        raise Phase3Error(f"One or more prespecified members failed validity checks: {pathological}")

    predictions = validation_metadata[
        ["window_id", "assembly_id", "cluster_id", "absolute_residual", "teacher_rank"]
    ].copy()
    predictions["split"] = "validation"
    for seed in SEEDS:
        predictions[f"ranking_score_seed_{seed}"] = member_predictions[seed]
    predictions["ensemble_mean_ranking_score"] = ensemble_mean
    predictions["ensemble_prediction_variance"] = ensemble_variance
    write_parquet_atomic(predictions, VALIDATION_PREDICTIONS)

    columns = list(empty_metric_row().keys())
    member_metrics = pd.DataFrame(metric_rows)[columns]
    if not np.isfinite(member_metrics.loc[member_metrics["estimate"].notna(), "estimate"].to_numpy(float)).all():
        raise Phase3Error("Non-finite Phase 3 metric")
    write_csv_atomic(member_metrics, MEMBER_METRICS)

    ensemble_config = {
        "version": "phase3_ensemble_v1",
        "created_utc": created_utc,
        "seeds": SEEDS,
        "seed_source": "Task-provided example reused because experiment_protocol_v1.yaml defines no ensemble seeds",
        "member_count": 5,
        "test_accessed": False,
        "partitions_materialized": ["development", "validation"],
        "architecture": {
            "version": phase2_lock["architecture_version"],
            "definition_sha256": sha256_file(PHASE2_MODEL),
            "altered_after_phase2": False,
            "parameter_count": phase2_lock["architecture"]["scalar_parameter_count"],
        },
        "data": {
            "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
            "development_rows": len(train_metadata),
            "development_genomes": train_metadata["assembly_id"].nunique(),
            "validation_rows": len(validation_metadata),
            "validation_genomes": validation_metadata["assembly_id"].nunique(),
            "validation_lineages": validation_metadata["cluster_id"].nunique(),
            "same_rows_for_all_members": True,
        },
        "target": {
            "definition": "absolute development-fitted GC-LOWESS residual",
            "development_pool_mean": target_mean,
            "development_pool_population_std": target_std,
            "same_for_all_members": True,
        },
        "training": {
            "epochs": EPOCHS,
            "schedule": "epochs 1-3 Huber only; epoch 4 Huber + same-genome pairwise ranking",
            "huber_weight": HUBER_WEIGHT,
            "huber_delta_standardized_units": 1.0,
            "pairwise_ranking_weight_final_epoch": RANK_WEIGHT,
            "same_genome_pairs_only": True,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP,
            "batch_size": BATCH_SIZE,
            "only_seed_dependent_elements": ["parameter initialization", "same-genome pair sampling", "minibatch ordering"],
        },
        "resume_safety": {
            "checkpoint_each_epoch": True,
            "atomic_checkpoint_write": True,
            "metadata_verified_before_resume": True,
        },
        "uncertainty": {
            "primary_score": "sample variance (ddof=1) across five member scalar ranking scores",
            "calibration": "none",
        },
        "hardware": {
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "gpu_available": False,
            "peak_gpu_memory_mb": None,
        },
        "frozen_k_definitions": phase2_lock["frozen_k_definitions"],
        "software": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "pyarrow": str(pa.__version__),
            "scipy": str(scipy.__version__),
        },
        "inputs": {
            "protocol": file_record(PROTOCOL),
            "phase2_lock": file_record(PHASE2_LOCK),
            "phase2_training_config": file_record(PHASE2_CONFIG),
            "phase2_model_definition": file_record(PHASE2_MODEL),
            "phase2_primary_checkpoint": file_record(PHASE2_M1),
            "cache_manifest": file_record(CACHE_MANIFEST),
        },
        "members": {
            f"seed_{seed}": {
                "checkpoint": file_record(member_checkpoints[seed]),
                "training_runtime_seconds": member_histories[seed][-1]["cumulative_training_runtime_seconds"],
                "validation_inference_seconds": member_inference_seconds[seed],
                "validation_mean_spearman": member_spearman[seed],
                "status": "valid",
            }
            for seed in SEEDS
        },
    }
    write_text_atomic(ENSEMBLE_CONFIG, yaml.safe_dump(ensemble_config, sort_keys=False, allow_unicode=True, width=1000))

    pair_table = "\n".join(
        f"| {left} | {right} | {pearson:.6f} | {within:.6f} |"
        for left, right, pearson, within in pairwise_rows
    )
    member_table_rows = []
    report_table_rows = []
    for seed in SEEDS:
        entity = f"member_seed_{seed}"
        top1_recall = summaries[entity][("recall_at_k", "top_1_percent")]["estimate"]
        top1_ndcg = summaries[entity][("ndcg_at_k", "top_1_percent")]["estimate"]
        runtime_seconds = member_histories[seed][-1]["cumulative_training_runtime_seconds"]
        member_table_rows.append(
            f"| {seed} | {checkpoint_hashes[seed][:12]}... | {member_spearman[seed]:.6f} | "
            f"{top1_recall:.6f} | {top1_ndcg:.6f} | {runtime_seconds:.1f} |"
        )
        report_table_rows.append(
            f"| Seed {seed} | {member_spearman[seed]:.6f} | {top1_recall:.6f} | {top1_ndcg:.6f} |"
        )
    member_table = "\n".join(member_table_rows)
    report_member_table = "\n".join(report_table_rows)
    diagnostics = f"""# Phase 3 ensemble diagnostics

## Member validity

- Five valid members: **{valid_members}/5**
- Seeds: `{SEEDS}`
- Unique final checkpoint hashes: **{'YES' if unique_checkpoints else 'NO'}**
- Byte-identical prediction vectors: **{'NO' if distinct_predictions else 'YES'}**
- Pathological numerical/training failures: **{sum(pathological.values())}**
- GPU memory: **not available (CPU runtime)**

No seed was discarded or selected based on its validation performance.

## Final validation metrics

| Seed | Checkpoint SHA-256 prefix | Mean within-genome Spearman | Top-1% Recall | Top-1% NDCG | Training runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
{member_table}
| Ensemble | mean of all five | {ensemble_spearman:.6f} | {summaries['ensemble_mean'][("recall_at_k", "top_1_percent")]['estimate']:.6f} | {summaries['ensemble_mean'][("ndcg_at_k", "top_1_percent")]['estimate']:.6f} | n/a |

Best single member is seed **{best_seed}** at **{best_single:.6f}**. Ensemble minus best single = **{ensemble_spearman - best_single:+.6f}**; ensemble minus mean single = **{ensemble_spearman - mean_single:+.6f}**. The ensemble is retained exactly as prespecified regardless of whether this delta is positive.

## Pairwise prediction correlations

| Seed A | Seed B | Global Pearson | Mean within-genome Spearman |
| ---: | ---: | ---: | ---: |
{pair_table}

Global Pearson range: **{min(global_pearson_values):.6f}–{max(global_pearson_values):.6f}**. Mean within-genome member-pair Spearman range: **{min(within_genome_pair_values):.6f}–{max(within_genome_pair_values):.6f}**. Members are correlated but not identical.

## Prediction variance

Primary uncertainty is the sample variance across the five scalar ranking predictions. Distribution across {len(ensemble_variance):,} validation windows:

| Statistic | Variance |
| --- | ---: |
| Minimum | {variance_values['min']:.10f} |
| 25th percentile | {variance_values['q25']:.10f} |
| Median | {variance_values['median']:.10f} |
| 75th percentile | {variance_values['q75']:.10f} |
| 90th percentile | {variance_values['q90']:.10f} |
| 95th percentile | {variance_values['q95']:.10f} |
| 99th percentile | {variance_values['q99']:.10f} |
| Maximum | {variance_values['max']:.10f} |
| Mean | {variance_values['mean']:.10f} |

All variance values are finite and non-negative. No test-dependent calibration was performed.
"""
    write_text_atomic(DIAGNOSTICS, diagnostics)

    report = f"""# Phase 3 frozen five-member deep ensemble report

## Verdict

**PHASE 3: PASS**

Created UTC: `{created_utc}`  
Five valid members: **{valid_members}/5**  
Frozen seeds: **{SEEDS}**  
Development training pool: **{len(train_metadata):,} windows / {EXPECTED_TRAIN_GENOMES} genomes**  
Full validation: **{len(validation_metadata):,} windows / {EXPECTED_VALIDATION_GENOMES} genomes / {EXPECTED_VALIDATION_LINEAGES} lineages**  
Locked TEST accessed: **NO**

## Frozen design

All members use the unchanged Phase 2 `student_cnn_v1` architecture (11,873 parameters), the same deterministic development windows, residual targets, target standardisation, optimizer, batch size, four-epoch schedule, Huber loss and final-epoch within-genome ranking weight 4.0. Only initialization, same-genome pair sampling and minibatch order vary with seed. Epoch checkpoints are atomic and resume-safe.

The protocol contains no ensemble-specific seeds, so the five seeds explicitly proposed in the Phase 3 task—11, 23, 37, 53 and 71—were frozen before training. No member was discarded because of its performance.

## Validation result

| Entity | Mean within-genome Spearman | Top-1% Recall | Top-1% NDCG |
| --- | ---: | ---: | ---: |
{report_member_table}
| Ensemble mean | **{ensemble_spearman:.6f}** | **{summaries['ensemble_mean'][("recall_at_k", "top_1_percent")]['estimate']:.6f}** | **{summaries['ensemble_mean'][("ndcg_at_k", "top_1_percent")]['estimate']:.6f}** |

Best single-member Spearman is **{best_single:.6f}** (seed {best_seed}). The ensemble changes Spearman by **{ensemble_spearman - best_single:+.6f}** versus the best member and **{ensemble_spearman - mean_single:+.6f}** versus the mean member. This observation did not alter architecture, seeds, membership or hyperparameters.

## Sanity checks and uncertainty

All five checkpoint hashes and prediction vectors differ. Pairwise global prediction correlations span {min(global_pearson_values):.6f}–{max(global_pearson_values):.6f}; no numerical or zero-variance seed failure occurred. The primary uncertainty score is the sample variance across the five ranking scores. Its median is {variance_values['median']:.10f}, 95th percentile is {variance_values['q95']:.10f}, and maximum is {variance_values['max']:.10f}. It is available for every validation window.

The local runtime is CPU-only, so peak GPU memory is not applicable. Phase 3 retains Phase 2's explicit limitation: this is a genome-balanced 52,992-window development training pool, not training on all 1,640,110 development windows.

## Freeze

The five seeds, member checkpoint hashes, ensemble mean rule, sample-variance uncertainty definition, Phase 2 architecture and frozen K definitions are sealed in `ensemble.lock`. No calibration was chosen from TEST, and no TEST outcome was accessed.
"""
    write_text_atomic(REPORT, report)

    output_records = {
        "ensemble_config": file_record(ENSEMBLE_CONFIG),
        "member_metrics": file_record(MEMBER_METRICS),
        "validation_ensemble_predictions": file_record(VALIDATION_PREDICTIONS),
        "ensemble_diagnostics": file_record(DIAGNOSTICS),
        "phase3_report": file_record(REPORT),
    }
    ensemble_lock = {
        "lock": "DISSERTATION PHASE 3 ENSEMBLE FROZEN",
        "created_utc": created_utc,
        "seeds": SEEDS,
        "five_valid_members": valid_members,
        "architecture_version": phase2_lock["architecture_version"],
        "architecture_altered": False,
        "member_checkpoints": {f"seed_{seed}": file_record(member_checkpoints[seed]) for seed in SEEDS},
        "ensemble_prediction": "arithmetic mean of five scalar ranking scores",
        "primary_uncertainty": "sample variance (ddof=1) across five scalar ranking scores",
        "frozen_k_definitions": phase2_lock["frozen_k_definitions"],
        "best_single_member_seed": best_seed,
        "best_single_member_validation_spearman": best_single,
        "ensemble_validation_spearman": ensemble_spearman,
        "ensemble_minus_best_single": ensemble_spearman - best_single,
        "prediction_variance_available": True,
        "outputs": output_records,
        "phase2_lock": file_record(PHASE2_LOCK),
        "test_accessed": False,
        "statement": "Five-member ensemble frozen after validation. Architecture and membership were not altered using ensemble outcomes. TEST remains sealed.",
    }
    write_text_atomic(ENSEMBLE_LOCK, json.dumps(ensemble_lock, indent=2) + "\n")

    final_checks = {
        "valid_members": valid_members == 5,
        "unique_checkpoint_hashes": unique_checkpoints,
        "distinct_predictions": distinct_predictions,
        "variance_available": len(ensemble_variance) == EXPECTED_VALIDATION_ROWS,
        "variance_finite_nonnegative": np.isfinite(ensemble_variance).all() and np.all(ensemble_variance >= 0),
        "validation_only": predictions["split"].eq("validation").all(),
        "test_not_accessed": True,
        "lock_exists": ENSEMBLE_LOCK.is_file(),
    }
    if not all(final_checks.values()):
        raise Phase3Error(f"Phase 3 final checks failed: {final_checks}")

    for artifact in FINAL_OUTPUTS:
        shutil.copy2(artifact, OUTPUT_MIRROR / artifact.name)

    elapsed = time.perf_counter() - started
    print(f"Phase 3 total runtime seconds: {elapsed:.1f}")
    print("PHASE 3: PASS")
    print(f"Five valid members: {valid_members}/5")
    print(f"Best single-member Spearman: {best_single:.6f}")
    print(f"Ensemble Spearman: {ensemble_spearman:.6f}")
    print("Prediction variance available: YES")
    print("Ensemble frozen: YES")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
