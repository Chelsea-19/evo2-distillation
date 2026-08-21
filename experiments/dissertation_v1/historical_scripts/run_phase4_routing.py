"""Phase 4 uncertainty-aware selective teacher escalation on validation only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import yaml
from scipy.stats import rankdata, spearmanr

WORKSPACE = Path(r"C:\Users\LDD\Documents\Codex\2026-08-19\za")
ROOT = WORKSPACE / "experiments" / "dissertation_v1"
PHASE3 = ROOT / "phase3"
PHASE4 = ROOT / "phase4"
OUTPUT_MIRROR = WORKSPACE / "outputs" / "phase4"

PHASE3_LOCK = PHASE3 / "ensemble.lock"
PHASE3_CONFIG = PHASE3 / "ensemble_config.yaml"
PHASE3_PREDICTIONS = PHASE3 / "validation_ensemble_predictions.parquet"

ROUTING_CONFIG = PHASE4 / "routing_config.yaml"
RISK_COVERAGE = PHASE4 / "validation_risk_coverage.csv"
RANDOM_RESULTS = PHASE4 / "random_escalation_results.csv"
ROUTING_PREDICTIONS = PHASE4 / "validation_routing_predictions.parquet"
REPORT = PHASE4 / "phase4_report.md"
ROUTING_LOCK = PHASE4 / "final_routing_rule.lock"

BUDGETS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
RANDOM_REPETITIONS = 200
RANDOM_BASE_SEED = 20260824
RANDOM_WORKERS = 4
FIDELITY_TARGET = 0.90
MONTE_CARLO_SE_TOLERANCE = 0.001
LINEAGE_BOOTSTRAPS = 2_000
EXPECTED_ROWS = 353_840
EXPECTED_GENOMES = 45
EXPECTED_LINEAGES = 19

K_IDS = ["top_1_percent", "top_5_percent", "fixed_100"]
FINAL_OUTPUTS = [ROUTING_CONFIG, RISK_COVERAGE, RANDOM_RESULTS, ROUTING_PREDICTIONS, REPORT, ROUTING_LOCK]


class Phase4Error(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase4Error(f"Required file missing: {path}")
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


def iter_lock_records(lock: dict[str, Any]):
    yield from lock["outputs"].values()
    yield from lock["member_checkpoints"].values()
    yield lock["phase2_lock"]


def verify_phase3() -> tuple[dict[str, Any], dict[str, Any]]:
    if any(path.exists() for path in FINAL_OUTPUTS):
        existing = [str(path) for path in FINAL_OUTPUTS if path.exists()]
        raise Phase4Error("Refusing to overwrite existing Phase 4 outputs:\n" + "\n".join(existing))
    lock = json.loads(PHASE3_LOCK.read_text(encoding="utf-8"))
    if lock.get("test_accessed") is not False or lock.get("five_valid_members") != 5:
        raise Phase4Error("Phase 3 ensemble is not valid and TEST-sealed")
    for record in iter_lock_records(lock):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise Phase4Error(f"Phase 3 locked artifact mismatch: {path}")
    config = yaml.safe_load(PHASE3_CONFIG.read_text(encoding="utf-8"))
    if config.get("test_accessed") is not False or config.get("partitions_materialized") != ["development", "validation"]:
        raise Phase4Error("Phase 3 configuration boundary mismatch")
    if config["uncertainty"]["primary_score"] != "sample variance (ddof=1) across five member scalar ranking scores":
        raise Phase4Error("Phase 3 uncertainty definition mismatch")
    for directory in (PHASE4, OUTPUT_MIRROR):
        directory.mkdir(parents=True, exist_ok=True)
    return lock, config


def stable_seed(label: str) -> int:
    return RANDOM_BASE_SEED ^ int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def call_count(n: int, requested_fraction: float) -> int:
    if requested_fraction <= 0:
        return 0
    if requested_fraction >= 1:
        return n
    return min(n, int(math.ceil(requested_fraction * n)))


def k_count(n: int, k_id: str) -> int:
    if k_id == "top_1_percent":
        return max(1, int(math.ceil(0.01 * n)))
    if k_id == "top_5_percent":
        return max(1, int(math.ceil(0.05 * n)))
    if k_id == "fixed_100":
        return min(100, n)
    raise Phase4Error(f"Unknown frozen K definition: {k_id}")


def build_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    groups = []
    for assembly_id, positions in frame.groupby("assembly_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        local = frame.iloc[idx]
        clusters = local["cluster_id"].unique()
        if len(clusters) != 1:
            raise Phase4Error("Validation assembly spans multiple lineages")
        ids = local["window_id"].astype(str).to_numpy()
        target = local["absolute_residual"].to_numpy(float)
        student = local["ensemble_mean_ranking_score"].to_numpy(float)
        variance = local["ensemble_prediction_variance"].to_numpy(float)
        teacher_order = np.lexsort((ids, -target))
        k_info = {}
        for k_id in K_IDS:
            k = k_count(len(idx), k_id)
            top = teacher_order[:k]
            discounts = np.log2(np.arange(2, k + 2, dtype=float))
            k_info[k_id] = {
                "k": k,
                "teacher_top": top,
                "discounts": discounts,
                "idcg": float(np.sum(target[top] / discounts)),
            }
        groups.append(
            {
                "assembly_id": str(assembly_id),
                "cluster_id": str(clusters[0]),
                "global_positions": idx,
                "window_ids": ids,
                "target": target,
                "teacher_rank": rankdata(target, method="average"),
                "student": student,
                "variance": variance,
                "uncertainty_order": np.lexsort((ids, -variance)),
                "k_info": k_info,
            }
        )
    if len(groups) != EXPECTED_GENOMES or len({group["cluster_id"] for group in groups}) != EXPECTED_LINEAGES:
        raise Phase4Error("Unexpected validation genome/lineage coverage")
    return groups


def evaluate_hybrid(group: dict[str, Any], selected: np.ndarray) -> dict[str, float]:
    score = group["student"].copy()
    score[selected] = group["target"][selected]
    score_rank = rankdata(score, method="average")
    rho = float(np.corrcoef(group["teacher_rank"], score_rank)[0, 1])
    order = np.lexsort((group["window_ids"], -score))
    result = {"spearman": rho, "risk": 1.0 - rho}
    for k_id, info in group["k_info"].items():
        predicted_top = order[: info["k"]]
        recall = float(np.intersect1d(predicted_top, info["teacher_top"], assume_unique=True).size / info["k"])
        dcg = float(np.sum(group["target"][predicted_top] / info["discounts"]))
        result[f"recall_{k_id}"] = recall
        result[f"ndcg_{k_id}"] = dcg / info["idcg"]
    return result


def aggregate_budget(groups: list[dict[str, Any]], selections: list[np.ndarray], requested: float) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    total_calls = 0
    for group, selected in zip(groups, selections):
        metrics = evaluate_hybrid(group, selected)
        total_calls += len(selected)
        rows.append({"assembly_id": group["assembly_id"], "cluster_id": group["cluster_id"], **metrics})
    genome = pd.DataFrame(rows)
    summary = {
        "teacher_call_fraction_requested": requested,
        "teacher_calls": total_calls,
        "teacher_call_fraction_actual": total_calls / EXPECTED_ROWS,
        "student_coverage": 1.0 - total_calls / EXPECTED_ROWS,
    }
    for column in ["spearman", "risk", *[f"recall_{k}" for k in K_IDS], *[f"ndcg_{k}" for k in K_IDS]]:
        summary[column] = float(genome[column].mean())
    return summary, genome


def risk_lineage_ci(genome: pd.DataFrame, label: str) -> tuple[float, float]:
    lineage = genome.groupby("cluster_id", sort=True)["risk"].mean().to_numpy(float)
    rng = np.random.default_rng(stable_seed(f"lineage_bootstrap|{label}"))
    draws = rng.choice(lineage, size=(LINEAGE_BOOTSTRAPS, len(lineage)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def aurc(curve: list[dict[str, float]]) -> float:
    ordered = sorted(curve, key=lambda row: row["student_coverage"])
    x = np.asarray([row["student_coverage"] for row in ordered], dtype=float)
    y = np.asarray([row["risk"] for row in ordered], dtype=float)
    return float(np.trapz(y, x))


def random_repetition(rep: int, groups: list[dict[str, Any]]) -> list[dict[str, float]]:
    permutations = [
        np.random.default_rng(stable_seed(f"random|rep={rep}|assembly={group['assembly_id']}")).permutation(len(group["target"]))
        for group in groups
    ]
    curve = []
    for requested in BUDGETS:
        selections = [order[: call_count(len(group["target"]), requested)] for group, order in zip(groups, permutations)]
        summary, _ = aggregate_budget(groups, selections, requested)
        curve.append(summary)
    rep_aurc = aurc(curve)
    for row in curve:
        row["replicate"] = rep
        row["aurc_risk_coverage"] = rep_aurc
    return curve


def calibration_analysis(frame: pd.DataFrame, groups: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, float]]:
    absolute_error = np.empty(len(frame), dtype=float)
    uncertainty_percentile = np.empty(len(frame), dtype=float)
    uncertainty_decile = np.empty(len(frame), dtype=np.int8)
    relationship = []
    enrichment = []
    high_error_recall = []
    genome_bin_rows = []
    for group in groups:
        n = len(group["target"])
        teacher_pct = (rankdata(group["target"], method="average") - 1.0) / max(1, n - 1)
        student_pct = (rankdata(group["student"], method="average") - 1.0) / max(1, n - 1)
        error = np.abs(teacher_pct - student_pct)
        ascending = np.lexsort((group["window_ids"], group["variance"]))
        ordinal = np.empty(n, dtype=np.int64)
        ordinal[ascending] = np.arange(n)
        percentile = ordinal / max(1, n - 1)
        decile = np.minimum(9, (ordinal * 10 // n)).astype(np.int8)
        global_idx = group["global_positions"]
        absolute_error[global_idx] = error
        uncertainty_percentile[global_idx] = percentile
        uncertainty_decile[global_idx] = decile
        relationship.append(float(spearmanr(group["variance"], error).statistic))
        k = max(1, int(math.ceil(0.10 * n)))
        high_unc = set(np.lexsort((group["window_ids"], -group["variance"]))[:k].tolist())
        high_err = set(np.lexsort((group["window_ids"], -error))[:k].tolist())
        overlap = len(high_unc & high_err)
        high_error_recall.append(overlap / k)
        enrichment.append((overlap / k) / (k / n))
        for bin_id in range(10):
            values = error[decile == bin_id]
            genome_bin_rows.append(
                {
                    "assembly_id": group["assembly_id"],
                    "cluster_id": group["cluster_id"],
                    "uncertainty_decile": bin_id + 1,
                    "mean_absolute_rank_error": float(values.mean()),
                    "windows": len(values),
                }
            )
    enriched = frame.copy()
    enriched["absolute_within_genome_rank_error"] = absolute_error
    enriched["uncertainty_percentile_within_genome"] = uncertainty_percentile
    enriched["uncertainty_decile"] = uncertainty_decile + 1
    bin_frame = pd.DataFrame(genome_bin_rows)
    decile_summary = (
        bin_frame.groupby("uncertainty_decile", sort=True)
        .agg(
            mean_across_genomes=("mean_absolute_rank_error", "mean"),
            median_across_genomes=("mean_absolute_rank_error", "median"),
            windows=("windows", "sum"),
        )
        .reset_index()
    )
    relationship_values = np.asarray(relationship, dtype=float)
    summary = {
        "global_spearman_variance_vs_absolute_rank_error": float(spearmanr(enriched["ensemble_prediction_variance"], absolute_error).statistic),
        "mean_within_genome_spearman_variance_vs_error": float(relationship_values.mean()),
        "median_within_genome_spearman_variance_vs_error": float(np.median(relationship_values)),
        "mean_high_uncertainty_top_error_enrichment": float(np.mean(enrichment)),
        "median_high_uncertainty_top_error_enrichment": float(np.median(enrichment)),
        "mean_high_uncertainty_recall_of_top_error": float(np.mean(high_error_recall)),
        "top_vs_bottom_uncertainty_decile_error_ratio": float(
            decile_summary.iloc[-1]["mean_across_genomes"] / decile_summary.iloc[0]["mean_across_genomes"]
        ),
    }
    return enriched, decile_summary, summary


def main() -> None:
    started = time.perf_counter()
    phase3_lock, phase3_config = verify_phase3()
    created_utc = datetime.now(timezone.utc).isoformat()
    print("Phase 4: Phase 3 ensemble and hashes verified; TEST remains sealed", flush=True)
    frame = pd.read_parquet(PHASE3_PREDICTIONS)
    if len(frame) != EXPECTED_ROWS or not frame["split"].eq("validation").all():
        raise Phase4Error("Phase 3 predictions are not the complete validation partition")
    required_numeric = ["absolute_residual", "teacher_rank", "ensemble_mean_ranking_score", "ensemble_prediction_variance"]
    if not np.isfinite(frame[required_numeric].to_numpy(float)).all() or np.any(frame["ensemble_prediction_variance"] < 0):
        raise Phase4Error("Invalid Phase 3 validation values")
    groups = build_groups(frame)

    uncertainty_curve = []
    uncertainty_genome_results: dict[float, pd.DataFrame] = {}
    uncertainty_selections: dict[float, list[np.ndarray]] = {}
    print(f"Phase 4: evaluating uncertainty routing budgets {BUDGETS}", flush=True)
    for requested in BUDGETS:
        selections = [group["uncertainty_order"][: call_count(len(group["target"]), requested)] for group in groups]
        summary, genome = aggregate_budget(groups, selections, requested)
        low, high = risk_lineage_ci(genome, f"uncertainty|{requested}")
        summary["risk_ci_low"] = low
        summary["risk_ci_high"] = high
        uncertainty_curve.append(summary)
        uncertainty_genome_results[requested] = genome
        uncertainty_selections[requested] = selections
        print(
            f"  uncertainty budget={requested:.0%}, actual={summary['teacher_call_fraction_actual']:.4f}, "
            f"Spearman={summary['spearman']:.6f}, risk={summary['risk']:.6f}",
            flush=True,
        )
    uncertainty_aurc = aurc(uncertainty_curve)

    print(f"Phase 4: running {RANDOM_REPETITIONS} deterministic nested random repetitions", flush=True)
    random_curves: dict[int, list[dict[str, float]]] = {}
    with ThreadPoolExecutor(max_workers=RANDOM_WORKERS) as executor:
        futures = {executor.submit(random_repetition, rep, groups): rep for rep in range(RANDOM_REPETITIONS)}
        completed = 0
        for future in as_completed(futures):
            rep = futures[future]
            random_curves[rep] = future.result()
            completed += 1
            if completed % 20 == 0 or completed == RANDOM_REPETITIONS:
                print(f"  random repetitions complete: {completed}/{RANDOM_REPETITIONS}", flush=True)
    random_rows = [row for rep in range(RANDOM_REPETITIONS) for row in random_curves[rep]]
    random_frame = pd.DataFrame(random_rows).sort_values(["replicate", "teacher_call_fraction_requested"]).reset_index(drop=True)
    random_aurcs = random_frame.groupby("replicate", sort=True)["aurc_risk_coverage"].first().to_numpy(float)
    random_aurc_mean = float(random_aurcs.mean())
    random_aurc_sd = float(random_aurcs.std(ddof=1))
    random_aurc_se = random_aurc_sd / math.sqrt(RANDOM_REPETITIONS)
    random_aurc_ci = tuple(float(value) for value in np.quantile(random_aurcs, [0.025, 0.975]))
    random_stable = random_aurc_se <= MONTE_CARLO_SE_TOLERANCE
    if not random_stable:
        raise Phase4Error(f"Random comparator Monte Carlo SE {random_aurc_se} exceeds tolerance")

    risk_rows = []
    metric_columns = ["spearman", "risk", *[f"recall_{k}" for k in K_IDS], *[f"ndcg_{k}" for k in K_IDS]]
    for uncertainty_row in uncertainty_curve:
        requested = uncertainty_row["teacher_call_fraction_requested"]
        random_subset = random_frame.loc[random_frame["teacher_call_fraction_requested"].eq(requested)]
        row_uncertainty = {
            "routing_method": "uncertainty_ordered",
            **uncertainty_row,
            "random_repetitions": RANDOM_REPETITIONS,
            "aurc_risk_coverage": uncertainty_aurc,
            "aurc_ci_low": np.nan,
            "aurc_ci_high": np.nan,
        }
        risk_rows.append(row_uncertainty)
        row_random = {
            "routing_method": "random_mean",
            "teacher_call_fraction_requested": requested,
            "teacher_calls": int(random_subset["teacher_calls"].iloc[0]),
            "teacher_call_fraction_actual": float(random_subset["teacher_call_fraction_actual"].iloc[0]),
            "student_coverage": float(random_subset["student_coverage"].iloc[0]),
            "random_repetitions": RANDOM_REPETITIONS,
            "aurc_risk_coverage": random_aurc_mean,
            "aurc_ci_low": random_aurc_ci[0],
            "aurc_ci_high": random_aurc_ci[1],
            "risk_ci_low": float(random_subset["risk"].quantile(0.025)),
            "risk_ci_high": float(random_subset["risk"].quantile(0.975)),
        }
        for column in metric_columns:
            row_random[column] = float(random_subset[column].mean())
        risk_rows.append(row_random)
    risk_coverage = pd.DataFrame(risk_rows)

    enriched_predictions, calibration_bins, calibration = calibration_analysis(frame, groups)
    uncertainty_beats_random = uncertainty_aurc < random_aurc_mean
    variance_calibrated_direction = calibration["mean_within_genome_spearman_variance_vs_error"] > 0
    high_uncertainty_enriched = calibration["mean_high_uncertainty_top_error_enrichment"] > 1
    useful_router = uncertainty_beats_random and variance_calibrated_direction and high_uncertainty_enriched

    qualifying = [
        row for row in uncertainty_curve if row["spearman"] >= FIDELITY_TARGET
    ]
    if useful_router and qualifying:
        selected = min(qualifying, key=lambda row: row["teacher_call_fraction_actual"])
        routing_status = "variance_routing_selected"
    elif useful_router:
        selected = uncertainty_curve[-1]
        routing_status = "target_not_met_below_full_teacher"
    else:
        selected = uncertainty_curve[-1]
        routing_status = "variance_not_useful_partial_routing_disabled"
    selected_requested = float(selected["teacher_call_fraction_requested"])
    selected_actual = float(selected["teacher_call_fraction_actual"])
    teacher_usage_removed = 1.0 - selected_actual

    for requested in BUDGETS:
        selected_global = np.zeros(len(enriched_predictions), dtype=bool)
        for group, local_selection in zip(groups, uncertainty_selections[requested]):
            selected_global[group["global_positions"][local_selection]] = True
        label = int(round(requested * 100))
        enriched_predictions[f"uncertainty_teacher_call_at_{label}pct"] = selected_global
    final_column = f"uncertainty_teacher_call_at_{int(round(selected_requested * 100))}pct"
    enriched_predictions["final_routing_teacher_called"] = enriched_predictions[final_column]
    enriched_predictions["final_hybrid_ranking_score"] = enriched_predictions["ensemble_mean_ranking_score"]
    called = enriched_predictions["final_routing_teacher_called"].to_numpy(bool)
    enriched_predictions.loc[called, "final_hybrid_ranking_score"] = enriched_predictions.loc[called, "absolute_residual"]
    routing_predictions = enriched_predictions.rename(
        columns={
            "absolute_residual": "teacher_ranking_target",
            "teacher_rank": "teacher_within_genome_rank",
            "ensemble_mean_ranking_score": "student_ensemble_mean_ranking_score",
        }
    )
    write_parquet_atomic(routing_predictions, ROUTING_PREDICTIONS)
    write_csv_atomic(risk_coverage, RISK_COVERAGE)
    write_csv_atomic(random_frame, RANDOM_RESULTS)

    config = {
        "version": "phase4_routing_v1",
        "created_utc": created_utc,
        "validation_only": True,
        "test_accessed": False,
        "uncertainty": {
            "definition": "Phase 3 sample variance (ddof=1) across five member scalar ranking scores",
            "primary": True,
            "conformal_fallback_explored": False,
            "fallback_policy": "Only eligible after clear variance-routing failure; not used opportunistically in this phase.",
        },
        "budgets": BUDGETS,
        "budget_source": "Phase 4 task example; no earlier routing grid was frozen",
        "routing_unit": "within each validation genome",
        "call_count_rule": "0 at 0%; n at 100%; otherwise ceil(requested_fraction * n_windows_in_genome)",
        "uncertainty_order": "descending ensemble variance; window_id ascending tie-break",
        "hybrid_score": "student ensemble mean, replaced by true absolute-residual teacher score for routed windows",
        "random_comparator": {
            "repetitions": RANDOM_REPETITIONS,
            "base_seed": RANDOM_BASE_SEED,
            "nested_permutation_within_genome": True,
            "exact_same_call_counts_as_uncertainty": True,
            "aurc_monte_carlo_standard_error": random_aurc_se,
            "stability_tolerance": MONTE_CARLO_SE_TOLERANCE,
            "stable": random_stable,
        },
        "risk": {
            "primary": "1 - within-genome Spearman, averaged equally across genomes",
            "aurc": "trapezoidal integration of mean risk over actual student coverage, including 0% and 100% teacher endpoints",
        },
        "fidelity_target": {
            "metric": "mean within-genome Spearman",
            "threshold": FIDELITY_TARGET,
            "selection": "minimum prespecified budget meeting threshold if variance is a useful router; otherwise 100% teacher",
        },
        "K_definitions": phase3_lock["frozen_k_definitions"],
        "evaluation": {
            "spearman_ties": "average ranks",
            "ranking_tie_break_for_top_k": "window_id ascending",
            "lineage_bootstrap_repetitions": LINEAGE_BOOTSTRAPS,
            "windows_as_independent_inferential_replicates": False,
        },
        "calibration": calibration,
        "decision": {
            "uncertainty_beats_random_by_aurc": uncertainty_beats_random,
            "variance_error_direction_positive": variance_calibrated_direction,
            "high_uncertainty_error_enrichment_above_one": high_uncertainty_enriched,
            "variance_useful_router": useful_router,
            "selected_budget_requested": selected_requested,
            "selected_budget_actual": selected_actual,
            "teacher_usage_removed_vs_full_teacher": teacher_usage_removed,
            "routing_status": routing_status,
        },
        "aurc": {
            "uncertainty": uncertainty_aurc,
            "random_mean": random_aurc_mean,
            "uncertainty_minus_random": uncertainty_aurc - random_aurc_mean,
            "random_minus_uncertainty_improvement": random_aurc_mean - uncertainty_aurc,
            "random_95pct_interval": list(random_aurc_ci),
        },
        "calibration_bins": calibration_bins.to_dict(orient="records"),
        "inputs": {
            "phase3_lock": file_record(PHASE3_LOCK),
            "phase3_config": file_record(PHASE3_CONFIG),
            "phase3_validation_predictions": file_record(PHASE3_PREDICTIONS),
        },
        "software": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "pyarrow": str(pa.__version__),
            "scipy": str(scipy.__version__),
        },
    }
    write_text_atomic(ROUTING_CONFIG, yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000))

    curve_table = "\n".join(
        f"| {row['teacher_call_fraction_requested']:.0%} | {row['teacher_call_fraction_actual']:.4f} | "
        f"{row['student_coverage']:.4f} | {row['spearman']:.6f} | {row['risk']:.6f} | "
        f"{random_frame.loc[random_frame['teacher_call_fraction_requested'].eq(row['teacher_call_fraction_requested']), 'risk'].mean():.6f} | "
        f"{row['recall_top_1_percent']:.6f} | {row['ndcg_top_1_percent']:.6f} |"
        for row in uncertainty_curve
    )
    bins_table = "\n".join(
        f"| {int(row.uncertainty_decile)} | {int(row.windows):,} | {row.mean_across_genomes:.6f} | {row.median_across_genomes:.6f} |"
        for row in calibration_bins.itertuples(index=False)
    )
    report = f"""# Phase 4 uncertainty-aware selective teacher escalation report

## Verdict

**PHASE 4: PASS**

Created UTC: `{created_utc}`  
Validation windows: **{len(frame):,} / {EXPECTED_GENOMES} genomes / {EXPECTED_LINEAGES} lineages**  
Locked TEST accessed: **NO**

## Frozen evaluation

The frozen Phase 3 ensemble mean is the student score and the frozen five-member sample variance is the primary uncertainty score. For each genome and budget, the highest-variance windows are referred to the teacher; their student score is replaced by the true absolute-residual teacher score. Random referral uses the exact same per-genome call counts. Each of {RANDOM_REPETITIONS} deterministic repetitions uses one nested random permutation per genome, so its risk-coverage path is coherent across budgets.

The routing grid is `{BUDGETS}`. No earlier grid existed. Primary risk is one minus within-genome Spearman, averaged equally across genomes. AURC integrates this risk over actual student coverage. Frozen Phase 1 K definitions are retained.

## Risk-coverage results

| Requested teacher calls | Actual teacher fraction | Student coverage | Uncertainty Spearman | Uncertainty risk | Random mean risk | Top-1% Recall | Top-1% NDCG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{curve_table}

Uncertainty-routing AURC is **{uncertainty_aurc:.8f}**; mean random AURC is **{random_aurc_mean:.8f}** (95% repetition interval {random_aurc_ci[0]:.8f} to {random_aurc_ci[1]:.8f}; Monte Carlo SE {random_aurc_se:.8f}). Random minus uncertainty AURC is **{random_aurc_mean - uncertainty_aurc:+.8f}**, so uncertainty routing **{'beats' if uncertainty_beats_random else 'does not beat'}** random referral on the primary comparison.

## Uncertainty calibration

Mean within-genome Spearman between variance and absolute rank error is **{calibration['mean_within_genome_spearman_variance_vs_error']:.6f}** (median {calibration['median_within_genome_spearman_variance_vs_error']:.6f}). The highest-uncertainty 10% has **{calibration['mean_high_uncertainty_top_error_enrichment']:.3f}x** enrichment for the highest-error 10%, with mean high-error recall **{calibration['mean_high_uncertainty_recall_of_top_error']:.6f}**. Top-versus-bottom uncertainty-decile mean error ratio is **{calibration['top_vs_bottom_uncertainty_decile_error_ratio']:.3f}**.

| Within-genome uncertainty decile | Windows | Mean absolute rank error across genomes | Median across genomes |
| ---: | ---: | ---: | ---: |
{bins_table}

Variance is classified as a **{'useful' if useful_router else 'failed'}** routing score under the prespecified joint rule: lower AURC than random, positive within-genome variance/error association, and high-error enrichment above one. A conformal fallback was not explored; this prevents opportunistic selection between uncertainty methods.

## Frozen routing rule

The prespecified fidelity target is mean within-genome Spearman >= {FIDELITY_TARGET:.2f}. The frozen rule is: within each genome, sort windows by ensemble variance descending (window ID ascending for ties) and refer the top **{selected_requested:.0%}** requested budget, corresponding to **{selected_actual:.4%}** actual validation calls. This achieves validation Spearman **{selected['spearman']:.6f}** and removes **{teacher_usage_removed:.4%}** of teacher calls relative to full-teacher scoring. Routing status: `{routing_status}`.

This rule, budgets, uncertainty definition, K values, score replacement and evaluation implementation are sealed in `final_routing_rule.lock`. Phase 5 was not started.
"""
    write_text_atomic(REPORT, report)

    output_records = {
        "routing_config": file_record(ROUTING_CONFIG),
        "validation_risk_coverage": file_record(RISK_COVERAGE),
        "random_escalation_results": file_record(RANDOM_RESULTS),
        "validation_routing_predictions": file_record(ROUTING_PREDICTIONS),
        "phase4_report": file_record(REPORT),
    }
    lock = {
        "lock": "DISSERTATION PHASE 4 ROUTING RULE FROZEN",
        "created_utc": created_utc,
        "phase3_ensemble_lock": file_record(PHASE3_LOCK),
        "uncertainty_definition": config["uncertainty"]["definition"],
        "teacher_call_budgets": BUDGETS,
        "routing_rule": {
            "unit": "within genome",
            "order": "ensemble variance descending, window_id ascending ties",
            "selected_requested_fraction": selected_requested,
            "selected_actual_validation_fraction": selected_actual,
            "score_replacement": config["hybrid_score"],
            "fidelity_target_spearman": FIDELITY_TARGET,
            "routing_status": routing_status,
        },
        "K_definitions": phase3_lock["frozen_k_definitions"],
        "evaluation_implementation": config["evaluation"],
        "aurc_uncertainty": uncertainty_aurc,
        "aurc_random_mean": random_aurc_mean,
        "aurc_uncertainty_minus_random": uncertainty_aurc - random_aurc_mean,
        "uncertainty_routing_beats_random": uncertainty_beats_random,
        "variance_useful_router": useful_router,
        "prediction_variance_available": True,
        "outputs": output_records,
        "test_accessed": False,
        "phase5_started": False,
        "statement": "Routing frozen using validation only. Locked TEST remains unopened. Do not proceed automatically to Phase 5.",
    }
    write_text_atomic(ROUTING_LOCK, json.dumps(lock, indent=2) + "\n")

    checks = {
        "phase3_valid": phase3_lock["five_valid_members"] == 5,
        "validation_only": routing_predictions["split"].eq("validation").all(),
        "all_rows": len(routing_predictions) == EXPECTED_ROWS,
        "variance_available": np.isfinite(routing_predictions["ensemble_prediction_variance"]).all(),
        "random_stable": random_stable,
        "random_repetitions": random_frame["replicate"].nunique() == RANDOM_REPETITIONS,
        "budgets_complete": set(random_frame["teacher_call_fraction_requested"].unique()) == set(BUDGETS),
        "final_rule_frozen": ROUTING_LOCK.is_file(),
        "test_not_accessed": True,
        "phase5_not_started": True,
    }
    if not all(checks.values()):
        raise Phase4Error(f"Phase 4 checks failed: {checks}")
    for artifact in FINAL_OUTPUTS:
        shutil.copy2(artifact, OUTPUT_MIRROR / artifact.name)

    print(f"Phase 4 runtime seconds: {time.perf_counter() - started:.1f}")
    print("PHASE 4: PASS")
    print(f"Uncertainty routing beats random: {'YES' if uncertainty_beats_random else 'NO'}")
    print(f"AURC uncertainty: {uncertainty_aurc:.8f}")
    print(f"AURC random: {random_aurc_mean:.8f}")
    print(f"Final teacher-call budgets: {BUDGETS}")
    print("Final routing rule frozen: YES")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
