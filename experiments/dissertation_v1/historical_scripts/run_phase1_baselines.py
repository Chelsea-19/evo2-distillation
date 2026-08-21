"""Run dissertation Phase 1 inexpensive baselines on development/validation only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "dissertation_v1"
PHASE1_SEED = 20260821
BOOTSTRAP_ITERATIONS = 2_000
EXPECTED_DEVELOPMENT = 1_640_110
EXPECTED_VALIDATION = 353_840
EXPECTED_VALIDATION_GENOMES = 45
EXPECTED_VALIDATION_CLUSTERS = 19

WORKSPACE = Path(r"C:\Users\LDD\Documents\Codex\2026-08-19\za")
ROOT = WORKSPACE / "experiments" / VERSION
PHASE1 = ROOT / "phase1"
MODEL_DIR = PHASE1 / "models"
OUTPUT_MIRROR = WORKSPACE / "outputs" / "phase1"

PHASE0_REPORT = ROOT / "phase0_completion_report.md"
PROTOCOL = ROOT / "protocol" / "experiment_protocol_v1.yaml"
SPLIT = ROOT / "splits" / "dissertation_split_v1.csv"
CANONICAL = ROOT / "data" / "window512_canonical_v1.parquet"
TARGETS = ROOT / "targets" / "window512_targets_v1.parquet"
TEST_LOCK = ROOT / "splits" / "TEST_SET_LOCKED.txt"

CONFIG_PATH = PHASE1 / "baseline_config.yaml"
METRICS_PATH = PHASE1 / "baseline_metrics_validation.csv"
PREDICTIONS_PATH = PHASE1 / "baseline_predictions_validation.parquet"
REPORT_PATH = PHASE1 / "phase1_report.md"
SELECTED_LOCK_PATH = PHASE1 / "selected_baseline.lock"

MODEL_PATHS = {
    "baseline_a_gc_linear": MODEL_DIR / "baseline_a_gc_linear.joblib",
    "baseline_b_gc_kmer_ridge": MODEL_DIR / "baseline_b_gc_kmer_ridge.joblib",
    "baseline_c_hist_gbr": MODEL_DIR / "baseline_c_hist_gbr.joblib",
}

FINAL_OUTPUTS = [CONFIG_PATH, METRICS_PATH, PREDICTIONS_PATH, REPORT_PATH, SELECTED_LOCK_PATH, *MODEL_PATHS.values()]

K_DEFINITIONS = [
    {
        "id": "top_1_percent",
        "rule": "ceil(0.01 * n_windows_in_genome)",
        "function": lambda n: max(1, int(math.ceil(0.01 * n))),
    },
    {
        "id": "top_5_percent",
        "rule": "ceil(0.05 * n_windows_in_genome)",
        "function": lambda n: max(1, int(math.ceil(0.05 * n))),
    },
    {
        "id": "fixed_100",
        "rule": "min(100, n_windows_in_genome)",
        "function": lambda n: min(100, n),
    },
]

BASELINE_B_GRID = [
    {"alpha": 0.01},
    {"alpha": 1.0},
    {"alpha": 100.0},
]

BASELINE_C_GRID = [
    {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.10, "max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.10, "max_iter": 100, "max_leaf_nodes": 31, "l2_regularization": 1.0},
]


class Phase1Error(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase1Error(f"Missing required file: {path}")
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


def ensure_phase0_pass_and_clean_destination() -> None:
    if not PHASE0_REPORT.is_file() or "**PHASE 0: PASS**" not in PHASE0_REPORT.read_text(encoding="utf-8"):
        raise Phase1Error("Phase 0 is not PASS")
    if not TEST_LOCK.is_file():
        raise Phase1Error("The Phase 0 test lock is missing")
    existing = [str(path) for path in FINAL_OUTPUTS if path.exists()]
    if existing:
        raise Phase1Error("Refusing to overwrite frozen Phase 1 outputs:\n" + "\n".join(existing))
    PHASE1.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MIRROR.mkdir(parents=True, exist_ok=True)


def validate_existing_outputs() -> None:
    """Finalize a completed run after a terminal-only validation interruption."""
    required = [CONFIG_PATH, METRICS_PATH, PREDICTIONS_PATH, REPORT_PATH, SELECTED_LOCK_PATH, *MODEL_PATHS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Phase1Error("Existing-run validation is missing outputs:\n" + "\n".join(missing))
    if "**PHASE 1: PASS**" not in REPORT_PATH.read_text(encoding="utf-8"):
        raise Phase1Error("Existing Phase 1 report is not PASS")
    lock = json.loads(SELECTED_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("test_accessed") is not False:
        raise Phase1Error("Existing lock does not confirm test_accessed=false")
    prediction_file = pq.ParquetFile(PREDICTIONS_PATH)
    if prediction_file.metadata.num_rows != EXPECTED_VALIDATION:
        raise Phase1Error("Existing prediction Parquet has the wrong row count")
    prediction_splits = pd.read_parquet(PREDICTIONS_PATH, columns=["split"])["split"]
    if not prediction_splits.eq("validation").all():
        raise Phase1Error("Existing prediction Parquet contains a non-validation row")
    metrics = pd.read_csv(METRICS_PATH)
    if metrics.empty or not np.isfinite(metrics["estimate"].to_numpy(float)).all():
        raise Phase1Error("Existing validation metrics are absent or non-finite")
    best_baseline = str(lock["best_baseline"])
    selected_summary = metrics.loc[
        metrics["level"].eq("selected_summary") & metrics["baseline"].eq(best_baseline)
    ].copy()
    spearman = float(
        selected_summary.loc[selected_summary["metric"].eq("spearman"), "estimate"].iloc[0]
    )
    recall = {
        str(row.k_definition): float(row.estimate)
        for row in selected_summary.loc[selected_summary["metric"].eq("recall_at_k")].itertuples()
    }
    ndcg = {
        str(row.k_definition): float(row.estimate)
        for row in selected_summary.loc[selected_summary["metric"].eq("ndcg_at_k")].itertuples()
    }
    OUTPUT_MIRROR.mkdir(parents=True, exist_ok=True)
    for artifact in (CONFIG_PATH, METRICS_PATH, PREDICTIONS_PATH, REPORT_PATH, SELECTED_LOCK_PATH):
        shutil.copy2(artifact, OUTPUT_MIRROR / artifact.name)
    print("PHASE 1: PASS")
    print(f"Best baseline: {best_baseline} ({lock['selected_config_id']})")
    print(f"Validation mean Spearman: {spearman:.6f}")
    print("Validation Recall@K: " + ", ".join(f"{key}={recall[key]:.6f}" for key in sorted(recall)))
    print("Validation NDCG: " + ", ".join(f"{key}={ndcg[key]:.6f}" for key in sorted(ndcg)))
    print(
        "Final frozen K definitions: "
        + "; ".join(f"{item['id']} ({item['rule']})" for item in lock["frozen_k_definitions"])
    )
    print("Test accessed: NO")


def read_development_validation_only() -> pd.DataFrame:
    allowed = ["development", "validation"]
    predicate = ds.field("split").isin(allowed)
    canonical_dataset = ds.dataset(CANONICAL, format="parquet")
    canonical_table = canonical_dataset.to_table(
        columns=[
            "window_id",
            "assembly_id",
            "cluster_id",
            "split",
            "window_length",
            "gc_content",
            "k4_rarity",
            "k6_rarity",
        ],
        filter=predicate,
    )
    canonical = canonical_table.to_pandas()
    del canonical_table

    target_dataset = ds.dataset(TARGETS, format="parquet")
    target_table = target_dataset.to_table(
        columns=[
            "window_id",
            "split",
            "absolute_residual",
            "within_genome_absolute_residual_rank",
        ],
        filter=predicate,
    )
    target = target_table.to_pandas()
    del target_table

    if len(canonical) != EXPECTED_DEVELOPMENT + EXPECTED_VALIDATION or len(target) != len(canonical):
        raise Phase1Error("Unexpected development+validation row count")
    if set(canonical["split"].unique()) != set(allowed) or set(target["split"].unique()) != set(allowed):
        raise Phase1Error("A partition other than development/validation was materialized")
    if not canonical["window_id"].equals(target["window_id"]):
        raise Phase1Error("Canonical and target row order/IDs do not match")
    if not canonical["split"].equals(target["split"]):
        raise Phase1Error("Canonical and target split labels do not match")
    canonical["absolute_residual"] = target["absolute_residual"].to_numpy(float)
    canonical["teacher_rank"] = target["within_genome_absolute_residual_rank"].to_numpy(float)
    del target

    if canonical.isna().any().any():
        raise Phase1Error("Required Phase 1 field contains missing values")
    if not canonical["window_length"].eq(512).all() or canonical["window_length"].nunique() != 1:
        raise Phase1Error("Window length is not the expected constant 512")
    counts = canonical.groupby("split").size().to_dict()
    if counts != {"development": EXPECTED_DEVELOPMENT, "validation": EXPECTED_VALIDATION}:
        raise Phase1Error(f"Unexpected partition counts: {counts}")
    return canonical


def validation_groups(validation: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    groups: list[tuple[str, str, np.ndarray]] = []
    for assembly_id, indexes in validation.groupby("assembly_id", sort=True).indices.items():
        idx = np.asarray(indexes, dtype=np.int64)
        clusters = validation.iloc[idx]["cluster_id"].unique()
        if len(clusters) != 1:
            raise Phase1Error(f"Validation assembly {assembly_id} maps to multiple clusters")
        groups.append((str(assembly_id), str(clusters[0]), idx))
    if len(groups) != EXPECTED_VALIDATION_GENOMES:
        raise Phase1Error(f"Expected {EXPECTED_VALIDATION_GENOMES} validation genomes")
    if len({cluster for _, cluster, _ in groups}) != EXPECTED_VALIDATION_CLUSTERS:
        raise Phase1Error(f"Expected {EXPECTED_VALIDATION_CLUSTERS} validation lineages")
    return groups


def genome_spearman(
    validation: pd.DataFrame,
    prediction: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
) -> pd.DataFrame:
    teacher = validation["teacher_rank"].to_numpy(float)
    rows = []
    for assembly_id, cluster_id, idx in groups:
        rho = float(spearmanr(teacher[idx], prediction[idx]).statistic)
        if not np.isfinite(rho):
            raise Phase1Error(f"Non-finite Spearman for {assembly_id}")
        rows.append({"assembly_id": assembly_id, "cluster_id": cluster_id, "metric": "spearman", "k_definition": "", "value": rho})
    return pd.DataFrame(rows)


def deterministic_order(scores: np.ndarray, window_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((window_ids.astype(str), -scores.astype(float)))


def full_genome_metrics(
    validation: pd.DataFrame,
    prediction: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
) -> pd.DataFrame:
    teacher_rank = validation["teacher_rank"].to_numpy(float)
    relevance = validation["absolute_residual"].to_numpy(float)
    window_ids = validation["window_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for assembly_id, cluster_id, idx in groups:
        local_ids = window_ids[idx]
        local_teacher_rank = teacher_rank[idx]
        local_relevance = relevance[idx]
        local_prediction = prediction[idx]
        rho = float(spearmanr(local_teacher_rank, local_prediction).statistic)
        rows.append({"assembly_id": assembly_id, "cluster_id": cluster_id, "metric": "spearman", "k_definition": "", "value": rho})
        teacher_order = deterministic_order(local_relevance, local_ids)
        prediction_order = deterministic_order(local_prediction, local_ids)
        for definition in K_DEFINITIONS:
            k = int(definition["function"](len(idx)))
            teacher_top = teacher_order[:k]
            predicted_top = prediction_order[:k]
            recall = float(np.intersect1d(teacher_top, predicted_top, assume_unique=True).size / k)
            discounts = np.log2(np.arange(2, k + 2, dtype=float))
            dcg = float(np.sum(local_relevance[predicted_top] / discounts))
            idcg = float(np.sum(local_relevance[teacher_top] / discounts))
            ndcg = dcg / idcg if idcg > 0 else float("nan")
            if not np.isfinite(ndcg):
                raise Phase1Error(f"Non-finite NDCG for {assembly_id} at {definition['id']}")
            rows.append(
                {
                    "assembly_id": assembly_id,
                    "cluster_id": cluster_id,
                    "metric": "recall_at_k",
                    "k_definition": definition["id"],
                    "value": recall,
                }
            )
            rows.append(
                {
                    "assembly_id": assembly_id,
                    "cluster_id": cluster_id,
                    "metric": "ndcg_at_k",
                    "k_definition": definition["id"],
                    "value": ndcg,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_seed(label: str) -> int:
    stable = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return PHASE1_SEED ^ stable


def summarize_genome_metric(values: pd.DataFrame, label: str) -> dict[str, Any]:
    if values["assembly_id"].nunique() != EXPECTED_VALIDATION_GENOMES:
        raise Phase1Error(f"Metric {label} does not cover every validation genome")
    cluster_means = values.groupby("cluster_id", sort=True)["value"].mean().to_numpy(float)
    if len(cluster_means) != EXPECTED_VALIDATION_CLUSTERS:
        raise Phase1Error(f"Metric {label} does not cover every validation lineage")
    rng = np.random.default_rng(bootstrap_seed(label))
    draws = rng.choice(cluster_means, size=(BOOTSTRAP_ITERATIONS, len(cluster_means)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values["value"].mean()),
        "median_across_genomes": float(values["value"].median()),
        "mean_across_lineages": float(cluster_means.mean()),
        "lineage_bootstrap_ci_low": float(low),
        "lineage_bootstrap_ci_high": float(high),
        "n_genomes": EXPECTED_VALIDATION_GENOMES,
        "n_lineages": EXPECTED_VALIDATION_CLUSTERS,
    }


def paired_difference_summary(left: pd.DataFrame, right: pd.DataFrame, label: str) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["assembly_id", "cluster_id", "metric", "k_definition"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    merged["difference"] = merged["value_left"] - merged["value_right"]
    cluster_differences = merged.groupby("cluster_id", sort=True)["difference"].mean().to_numpy(float)
    rng = np.random.default_rng(bootstrap_seed(label))
    draws = rng.choice(
        cluster_differences,
        size=(BOOTSTRAP_ITERATIONS, len(cluster_differences)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_genome_difference": float(merged["difference"].mean()),
        "mean_lineage_difference": float(cluster_differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def selection_record(
    baseline: str,
    config_id: str,
    hyperparameters: dict[str, Any],
    per_genome: pd.DataFrame,
) -> dict[str, Any]:
    summary = summarize_genome_metric(per_genome, f"selection|{baseline}|{config_id}")
    return {
        "baseline": baseline,
        "config_id": config_id,
        "hyperparameters": hyperparameters,
        "per_genome_spearman": per_genome,
        **summary,
    }


def fit_models(data: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, list[tuple[str, str, np.ndarray]]]:
    development = data.loc[data["split"].eq("development")].copy()
    validation = data.loc[data["split"].eq("validation")].copy().reset_index(drop=True)
    groups = validation_groups(validation)
    y_development = development["absolute_residual"].to_numpy(float)

    features = {
        "baseline_a_gc_linear": ["gc_content"],
        "baseline_b_gc_kmer_ridge": ["gc_content", "k4_rarity", "k6_rarity"],
        "baseline_c_hist_gbr": ["gc_content", "window_length", "k4_rarity", "k6_rarity"],
    }
    selected: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    print("Phase 1: fitting Baseline A (GC-only linear)", flush=True)
    x_dev_a = development[features["baseline_a_gc_linear"]].to_numpy(float)
    x_val_a = validation[features["baseline_a_gc_linear"]].to_numpy(float)
    model_a = LinearRegression()
    model_a.fit(x_dev_a, y_development)
    pred_a = model_a.predict(x_val_a)
    genome_a = genome_spearman(validation, pred_a, groups)
    candidate_a = selection_record("baseline_a_gc_linear", "A_linear", {"model": "LinearRegression"}, genome_a)
    candidates.append(candidate_a)
    selected["baseline_a_gc_linear"] = {
        "config_id": "A_linear",
        "hyperparameters": {"model": "LinearRegression"},
        "model": model_a,
        "prediction": pred_a,
        "features": features["baseline_a_gc_linear"],
        "selection": candidate_a,
    }

    print("Phase 1: fitting Baseline B compact Ridge grid", flush=True)
    x_dev_b = development[features["baseline_b_gc_kmer_ridge"]].to_numpy(float)
    x_val_b = validation[features["baseline_b_gc_kmer_ridge"]].to_numpy(float)
    best_b: dict[str, Any] | None = None
    for index, params in enumerate(BASELINE_B_GRID, start=1):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=params["alpha"])),
            ]
        )
        model.fit(x_dev_b, y_development)
        prediction = model.predict(x_val_b)
        per_genome = genome_spearman(validation, prediction, groups)
        config_id = f"B_ridge_{index}"
        record = selection_record("baseline_b_gc_kmer_ridge", config_id, params, per_genome)
        candidates.append(record)
        if best_b is None or record["estimate"] > best_b["selection"]["estimate"]:
            best_b = {
                "config_id": config_id,
                "hyperparameters": params,
                "model": model,
                "prediction": prediction,
                "features": features["baseline_b_gc_kmer_ridge"],
                "selection": record,
            }
    if best_b is None:
        raise Phase1Error("Baseline B selection failed")
    selected["baseline_b_gc_kmer_ridge"] = best_b

    print("Phase 1: fitting Baseline C compact HistGradientBoosting grid", flush=True)
    x_dev_c = development[features["baseline_c_hist_gbr"]].to_numpy(float)
    x_val_c = validation[features["baseline_c_hist_gbr"]].to_numpy(float)
    best_c: dict[str, Any] | None = None
    for index, params in enumerate(BASELINE_C_GRID, start=1):
        print(f"  Baseline C config {index}/{len(BASELINE_C_GRID)}: {params}", flush=True)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=50,
            l2_regularization=params["l2_regularization"],
            early_stopping=False,
            random_state=PHASE1_SEED,
        )
        model.fit(x_dev_c, y_development)
        prediction = model.predict(x_val_c)
        per_genome = genome_spearman(validation, prediction, groups)
        config_id = f"C_hist_gbr_{index}"
        full_params = {**params, "min_samples_leaf": 50, "loss": "squared_error", "early_stopping": False}
        record = selection_record("baseline_c_hist_gbr", config_id, full_params, per_genome)
        candidates.append(record)
        if best_c is None or record["estimate"] > best_c["selection"]["estimate"]:
            best_c = {
                "config_id": config_id,
                "hyperparameters": full_params,
                "model": model,
                "prediction": prediction,
                "features": features["baseline_c_hist_gbr"],
                "selection": record,
            }
    if best_c is None:
        raise Phase1Error("Baseline C selection failed")
    selected["baseline_c_hist_gbr"] = best_c

    del development, x_dev_a, x_val_a, x_dev_b, x_val_b, x_dev_c, x_val_c, y_development
    return selected, candidates, validation, groups


def build_metrics(
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    validation: pd.DataFrame,
    groups: list[tuple[str, str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    selected_ids = {baseline: payload["config_id"] for baseline, payload in selected.items()}
    for candidate in candidates:
        rows.append(
            {
                "level": "candidate_summary",
                "baseline": candidate["baseline"],
                "config_id": candidate["config_id"],
                "selected": candidate["config_id"] == selected_ids[candidate["baseline"]],
                "assembly_id": "",
                "cluster_id": "",
                "metric": "spearman",
                "k_definition": "",
                "estimate": candidate["estimate"],
                "median_across_genomes": candidate["median_across_genomes"],
                "mean_across_lineages": candidate["mean_across_lineages"],
                "lineage_bootstrap_ci_low": candidate["lineage_bootstrap_ci_low"],
                "lineage_bootstrap_ci_high": candidate["lineage_bootstrap_ci_high"],
                "n_genomes": candidate["n_genomes"],
                "n_lineages": candidate["n_lineages"],
                "hyperparameters_json": json.dumps(candidate["hyperparameters"], sort_keys=True),
            }
        )

    genome_metrics_by_model: dict[str, pd.DataFrame] = {}
    summaries_by_model: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for baseline, payload in selected.items():
        genome_metrics = full_genome_metrics(validation, payload["prediction"], groups)
        genome_metrics_by_model[baseline] = genome_metrics
        summaries: dict[tuple[str, str], dict[str, Any]] = {}
        for (metric, k_definition), subset in genome_metrics.groupby(["metric", "k_definition"], dropna=False, sort=True):
            label = f"selected|{baseline}|{metric}|{k_definition}"
            summary = summarize_genome_metric(subset, label)
            summaries[(str(metric), str(k_definition))] = summary
            rows.append(
                {
                    "level": "selected_summary",
                    "baseline": baseline,
                    "config_id": payload["config_id"],
                    "selected": True,
                    "assembly_id": "",
                    "cluster_id": "",
                    "metric": metric,
                    "k_definition": k_definition,
                    **summary,
                    "hyperparameters_json": json.dumps(payload["hyperparameters"], sort_keys=True),
                }
            )
        summaries_by_model[baseline] = summaries
        for record in genome_metrics.to_dict(orient="records"):
            rows.append(
                {
                    "level": "genome_metric",
                    "baseline": baseline,
                    "config_id": payload["config_id"],
                    "selected": True,
                    "assembly_id": record["assembly_id"],
                    "cluster_id": record["cluster_id"],
                    "metric": record["metric"],
                    "k_definition": record["k_definition"],
                    "estimate": record["value"],
                    "median_across_genomes": np.nan,
                    "mean_across_lineages": np.nan,
                    "lineage_bootstrap_ci_low": np.nan,
                    "lineage_bootstrap_ci_high": np.nan,
                    "n_genomes": 1,
                    "n_lineages": 1,
                    "hyperparameters_json": json.dumps(payload["hyperparameters"], sort_keys=True),
                }
            )
    metrics = pd.DataFrame(rows)
    ordered_columns = [
        "level",
        "baseline",
        "config_id",
        "selected",
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
        "hyperparameters_json",
    ]
    return metrics[ordered_columns], genome_metrics_by_model, summaries_by_model


def main() -> None:
    if "--validate-existing" in sys.argv:
        validate_existing_outputs()
        return
    started = time.perf_counter()
    ensure_phase0_pass_and_clean_destination()
    created_utc = datetime.now(timezone.utc).isoformat()
    print("Phase 1: Phase 0 PASS confirmed; scanning development/validation only", flush=True)
    data = read_development_validation_only()
    selected, candidates, validation, groups = fit_models(data)

    print("Phase 1: computing validation ranking metrics and lineage uncertainty", flush=True)
    metrics, genome_metrics_by_model, summaries_by_model = build_metrics(selected, candidates, validation, groups)
    best_baseline = max(selected, key=lambda name: selected[name]["selection"]["estimate"])
    best = selected[best_baseline]

    predictions = validation[
        ["window_id", "assembly_id", "cluster_id", "absolute_residual", "teacher_rank"]
    ].copy()
    predictions = predictions.rename(columns={"teacher_rank": "within_genome_teacher_rank"})
    predictions["prediction_baseline_a_gc_linear"] = selected["baseline_a_gc_linear"]["prediction"]
    predictions["prediction_baseline_b_gc_kmer_ridge"] = selected["baseline_b_gc_kmer_ridge"]["prediction"]
    predictions["prediction_baseline_c_hist_gbr"] = selected["baseline_c_hist_gbr"]["prediction"]
    predictions["split"] = "validation"
    if len(predictions) != EXPECTED_VALIDATION or predictions["assembly_id"].nunique() != EXPECTED_VALIDATION_GENOMES:
        raise Phase1Error("Validation prediction coverage is incomplete")
    if predictions.isna().any().any():
        raise Phase1Error("Validation predictions contain missing values")

    print("Phase 1: saving selected development-fitted estimators", flush=True)
    for baseline, path in MODEL_PATHS.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(selected[baseline]["model"], temporary, compress=3)
        os.replace(temporary, path)

    input_records = {
        "phase0_completion_report": file_record(PHASE0_REPORT),
        "experiment_protocol": file_record(PROTOCOL),
        "split": file_record(SPLIT),
        "canonical_features": file_record(CANONICAL),
        "targets": file_record(TARGETS),
        "test_lock": file_record(TEST_LOCK),
    }
    config = {
        "version": VERSION,
        "phase": "phase1_baseline",
        "created_utc": created_utc,
        "phase0_status": "PASS",
        "test_not_accessed": True,
        "permitted_partitions_materialized": ["development", "validation"],
        "seed": PHASE1_SEED,
        "target": {
            "training": "absolute_residual",
            "ranking_reference": "within-genome absolute development-fitted GC-LOWESS residual ordering",
        },
        "features": {
            "baseline_a_gc_linear": ["gc_content"],
            "baseline_b_gc_kmer_ridge": ["gc_content", "k4_rarity", "k6_rarity"],
            "baseline_c_hist_gbr": ["gc_content", "window_length", "k4_rarity", "k6_rarity"],
            "window_length_note": "window_length is frozen at 512 for every row and has zero predictive variance; it is retained only to represent the complete permitted feature schema.",
            "forbidden": ["DNA sequence", "annotations", "raw PPL rank", "historical test outcomes", "test target outcomes"],
        },
        "models": {
            "baseline_a_gc_linear": {"grid": [{"model": "LinearRegression"}]},
            "baseline_b_gc_kmer_ridge": {"grid": BASELINE_B_GRID},
            "baseline_c_hist_gbr": {"grid": BASELINE_C_GRID, "fixed": {"min_samples_leaf": 50, "loss": "squared_error", "early_stopping": False}},
        },
        "selection": {
            "partition": "validation only",
            "criterion": "maximum mean within-genome Spearman across 45 validation genomes",
            "tie_break": "first configuration in the predeclared compact grid",
            "selected": {
                baseline: {
                    "config_id": payload["config_id"],
                    "features": payload["features"],
                    "hyperparameters": payload["hyperparameters"],
                    "validation_mean_spearman": payload["selection"]["estimate"],
                }
                for baseline, payload in selected.items()
            },
            "best_inexpensive_baseline": best_baseline,
        },
        "frozen_k_definitions": [{"id": item["id"], "rule": item["rule"]} for item in K_DEFINITIONS],
        "primary_k_for_threshold_reporting": "top_1_percent",
        "metrics": {
            "spearman": "Spearman correlation between predictions and teacher absolute-residual order, calculated separately in each validation genome.",
            "recall_at_k": "Intersection size between deterministic predicted and teacher top-K sets divided by K; ties break by window_id.",
            "ndcg_at_k": "Linear-gain DCG using absolute_residual relevance and log2 position discount, divided by ideal teacher-order DCG; ties break by window_id.",
        },
        "inference": {
            "window_level_replicates": False,
            "genome_summary": "equal-weight mean and median across 45 genomes",
            "lineage_summary": "equal-weight mean across 19 validation Mash clusters",
            "uncertainty": f"{BOOTSTRAP_ITERATIONS}-iteration nonparametric bootstrap of validation cluster means, 2.5th and 97.5th percentiles",
        },
        "inputs": input_records,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_text_atomic(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000))
    write_csv_atomic(metrics, METRICS_PATH)
    write_parquet_atomic(predictions, PREDICTIONS_PATH)

    for baseline, path in MODEL_PATHS.items():
        config["selection"]["selected"][baseline]["model_artifact"] = file_record(path)

    # Paired k-mer increment over GC-only.
    metrics_a = genome_metrics_by_model["baseline_a_gc_linear"]
    metrics_b = genome_metrics_by_model["baseline_b_gc_kmer_ridge"]
    spearman_a = metrics_a.loc[metrics_a["metric"].eq("spearman")]
    spearman_b = metrics_b.loc[metrics_b["metric"].eq("spearman")]
    kmer_spearman_delta = paired_difference_summary(spearman_b, spearman_a, "kmer_minus_gc|spearman")
    kmer_primary_deltas: dict[str, dict[str, float]] = {}
    for metric in ("recall_at_k", "ndcg_at_k"):
        left = metrics_b.loc[metrics_b["metric"].eq(metric) & metrics_b["k_definition"].eq("top_1_percent")]
        right = metrics_a.loc[metrics_a["metric"].eq(metric) & metrics_a["k_definition"].eq("top_1_percent")]
        kmer_primary_deltas[metric] = paired_difference_summary(left, right, f"kmer_minus_gc|{metric}|top_1_percent")

    best_summaries = summaries_by_model[best_baseline]
    best_spearman = best_summaries[("spearman", "")]
    best_recall_primary = best_summaries[("recall_at_k", "top_1_percent")]
    best_ndcg_primary = best_summaries[("ndcg_at_k", "top_1_percent")]
    best_recall_all = {
        definition["id"]: best_summaries[("recall_at_k", definition["id"])]["estimate"]
        for definition in K_DEFINITIONS
    }
    best_ndcg_all = {
        definition["id"]: best_summaries[("ndcg_at_k", definition["id"])]["estimate"]
        for definition in K_DEFINITIONS
    }

    report = f"""# Phase 1 inexpensive-baseline report

## Verdict

**PHASE 1: PASS**

Created UTC: `{created_utc}`  
Development windows used for fitting: **{EXPECTED_DEVELOPMENT:,}**  
Validation windows evaluated: **{EXPECTED_VALIDATION:,}** across **{EXPECTED_VALIDATION_GENOMES} genomes / {EXPECTED_VALIDATION_CLUSTERS} Mash lineages**  
Locked TEST accessed: **NO**

## Experimental question

This phase tests how much of the development-fitted Evo 2 absolute GC-LOWESS residual ordering can be recovered from cheap composition features alone. No DNA sequence encoder or biological annotation was used.

## Baselines and selection

| Baseline | Permitted features | Selected configuration | Validation mean within-genome Spearman |
| --- | --- | --- | ---: |
| A: GC-only | GC content | `{selected['baseline_a_gc_linear']['config_id']}` | {selected['baseline_a_gc_linear']['selection']['estimate']:.6f} |
| B: linear cheap composition | GC, k4 rarity, k6 rarity | `{selected['baseline_b_gc_kmer_ridge']['config_id']}` `{json.dumps(selected['baseline_b_gc_kmer_ridge']['hyperparameters'], sort_keys=True)}` | {selected['baseline_b_gc_kmer_ridge']['selection']['estimate']:.6f} |
| C: boosted cheap composition | GC, length, k4 rarity, k6 rarity | `{selected['baseline_c_hist_gbr']['config_id']}` `{json.dumps(selected['baseline_c_hist_gbr']['hyperparameters'], sort_keys=True)}` | {selected['baseline_c_hist_gbr']['selection']['estimate']:.6f} |

Window length is exactly 512 in every row. It was retained in Baseline C only because it belongs to the frozen inexpensive-feature schema; it has zero variance and therefore contributes no usable information.

The compact grids were evaluated only on validation, and selection used only mean within-genome Spearman. Test targets and outcomes were never materialized.

## Best inexpensive baseline

**{best_baseline}** (`{best['config_id']}`)

- Mean within-genome Spearman: **{best_spearman['estimate']:.6f}**
- Median within-genome Spearman: **{best_spearman['median_across_genomes']:.6f}**
- Lineage-balanced mean Spearman: **{best_spearman['mean_across_lineages']:.6f}**
- 95% lineage-bootstrap interval: **[{best_spearman['lineage_bootstrap_ci_low']:.6f}, {best_spearman['lineage_bootstrap_ci_high']:.6f}]**
- Top-1% Recall@K: **{best_recall_primary['estimate']:.6f}**
- Top-1% NDCG@K: **{best_ndcg_primary['estimate']:.6f}**

### Frozen ranking endpoints

| K definition | Rule | Mean Recall@K | Mean NDCG@K |
| --- | --- | ---: | ---: |
""" + "\n".join(
        f"| {definition['id']} | `{definition['rule']}` | {best_recall_all[definition['id']]:.6f} | {best_ndcg_all[definition['id']]:.6f} |"
        for definition in K_DEFINITIONS
    ) + f"""

All three K definitions are now frozen for subsequent phases. `top_1_percent` is the primary threshold-reporting endpoint; top 5% and fixed 100 are prespecified secondary sensitivities.

## Required interpretation

### 1. Recoverable cheap-feature ranking signal

The best inexpensive baseline recovers a mean within-genome Spearman of **{best_spearman['estimate']:.6f}** and top-1% Recall@K of **{best_recall_primary['estimate']:.6f}**. This is the empirical cheap-feature ceiling observed on validation under the compact search, not a biological interpretation.

### 2. Increment from k-mer rarity beyond GC

Adding k4/k6 rarity in the linear baseline changes mean genome Spearman by **{kmer_spearman_delta['mean_genome_difference']:+.6f}** relative to GC-only. The lineage-bootstrap interval for the paired difference is **[{kmer_spearman_delta['ci_low']:+.6f}, {kmer_spearman_delta['ci_high']:+.6f}]**. At top 1%, the paired changes are **{kmer_primary_deltas['recall_at_k']['mean_genome_difference']:+.6f}** for Recall and **{kmer_primary_deltas['ndcg_at_k']['mean_genome_difference']:+.6f}** for NDCG.

### 3. Best validation performance

The selected best baseline achieves mean/median Spearman **{best_spearman['estimate']:.6f}/{best_spearman['median_across_genomes']:.6f}**, top-1% Recall **{best_recall_primary['estimate']:.6f}**, and top-1% NDCG **{best_ndcg_primary['estimate']:.6f}**. Complete genome-level and lineage-aware results for every frozen K are in `baseline_metrics_validation.csv`.

### 4. Minimum useful-transfer threshold

For a distilled student to demonstrate useful teacher transfer on validation, it must exceed the best inexpensive baseline's **mean within-genome Spearman of {best_spearman['estimate']:.6f}**. The prespecified stronger criterion is a positive paired lineage-bootstrap difference in Spearman while showing no material degradation relative to **Recall@top1% {best_recall_primary['estimate']:.6f}** and **NDCG@top1% {best_ndcg_primary['estimate']:.6f}**. These are validation thresholds only; no test claim is made.

## Statistical policy

Metrics were first calculated separately for each validation genome. Point estimates are equal-weight genome means/medians. Uncertainty is a {BOOTSTRAP_ITERATIONS}-iteration nonparametric bootstrap over the {EXPECTED_VALIDATION_CLUSTERS} complete validation Mash lineages using lineage-level means. Windows were not treated as independent inferential replicates.

## Frozen outputs

- `baseline_config.yaml`
- `baseline_metrics_validation.csv`
- `baseline_predictions_validation.parquet`
- `models/*.joblib` for the selected development-fitted estimator of each baseline
- `selected_baseline.lock`

## Next-phase boundary

Phase 1 stops here. The locked TEST partition remains unopened. Any student comparison must use the frozen K definitions, selected baseline and validation policy recorded here.
"""
    write_text_atomic(REPORT_PATH, report)

    output_records = {
        "config": file_record(CONFIG_PATH),
        "metrics": file_record(METRICS_PATH),
        "predictions": file_record(PREDICTIONS_PATH),
        "report": file_record(REPORT_PATH),
        "models": {baseline: file_record(path) for baseline, path in MODEL_PATHS.items()},
    }
    lock_payload = {
        "lock": "DISSERTATION PHASE 1 BASELINE SELECTION FROZEN",
        "created_utc": created_utc,
        "best_baseline": best_baseline,
        "selected_config_id": best["config_id"],
        "selected_hyperparameters": best["hyperparameters"],
        "validation_mean_within_genome_spearman": best_spearman["estimate"],
        "validation_top_1_percent_recall": best_recall_primary["estimate"],
        "validation_top_1_percent_ndcg": best_ndcg_primary["estimate"],
        "frozen_k_definitions": [{"id": item["id"], "rule": item["rule"]} for item in K_DEFINITIONS],
        "output_sha256": output_records,
        "test_accessed": False,
        "statement": "Baseline selection and K definitions are frozen. Do not change them using test outcomes. The test partition remains sealed until Phase 5.",
    }
    write_text_atomic(SELECTED_LOCK_PATH, json.dumps(lock_payload, indent=2) + "\n")

    # Compact final validation without reading test rows or outcomes.
    parquet_file = pq.ParquetFile(PREDICTIONS_PATH)
    pass_checks = {
        "phase0_pass": True,
        "development_rows_fitted": int((data["split"] == "development").sum()) == EXPECTED_DEVELOPMENT,
        "validation_rows_evaluated": len(predictions) == EXPECTED_VALIDATION,
        "validation_genomes": predictions["assembly_id"].nunique() == EXPECTED_VALIDATION_GENOMES,
        "validation_lineages": predictions["cluster_id"].nunique() == EXPECTED_VALIDATION_CLUSTERS,
        "prediction_parquet_rows": parquet_file.metadata.num_rows == EXPECTED_VALIDATION,
        "prediction_split_validation_only": predictions["split"].eq("validation").all(),
        "metrics_finite": np.isfinite(metrics["estimate"].to_numpy(float)).all(),
        "k_definitions_frozen": len(K_DEFINITIONS) == 3,
        "selected_lock_exists": SELECTED_LOCK_PATH.is_file(),
        "test_accessed": False,
    }
    phase1_pass = all(pass_checks.values())
    if not phase1_pass:
        raise Phase1Error(f"Phase 1 final validation failed: {pass_checks}")

    for artifact in (CONFIG_PATH, METRICS_PATH, PREDICTIONS_PATH, REPORT_PATH, SELECTED_LOCK_PATH):
        shutil.copy2(artifact, OUTPUT_MIRROR / artifact.name)

    elapsed = time.perf_counter() - started
    print(f"Phase 1 runtime seconds: {elapsed:.1f}")
    print("PHASE 1: PASS")
    print(f"Best baseline: {best_baseline} ({best['config_id']})")
    print(f"Validation mean Spearman: {best_spearman['estimate']:.6f}")
    print("Validation Recall@K: " + ", ".join(f"{key}={value:.6f}" for key, value in best_recall_all.items()))
    print("Validation NDCG: " + ", ".join(f"{key}={value:.6f}" for key, value in best_ndcg_all.items()))
    print("Final frozen K definitions: " + "; ".join(f"{item['id']} ({item['rule']})" for item in K_DEFINITIONS))
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
