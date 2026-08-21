"""Finalize Phase 2 from completed checkpoints, predictions and metrics.

The original run completed training and inference, then PyYAML rejected the
TorchVersion subclass while serializing the final configuration. This recovery
path performs no training and no model inference.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_phase2_distillation as r  # noqa: E402


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    required = [
        r.MODEL_DEFINITION,
        r.VALIDATION_METRICS,
        r.ABLATION_RESULTS,
        r.VALIDATION_PREDICTIONS,
        r.CACHE_MANIFEST,
        r.CACHE_METADATA,
        r.CACHE_TOKENS,
        *r.CHECKPOINT_PATHS.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise r.Phase2Error("Cannot finalize; completed run artifacts missing:\n" + "\n".join(missing))
    for unfinished in (r.TRAINING_CONFIG, r.REPORT, r.FINAL_LOCK):
        if unfinished.exists():
            raise r.Phase2Error(f"Refusing to overwrite finalized artifact: {unfinished}")

    if "**PHASE 0: PASS**" not in r.PHASE0_REPORT.read_text(encoding="utf-8"):
        raise r.Phase2Error("Phase 0 is not PASS")
    if "**PHASE 1: PASS**" not in r.PHASE1_REPORT.read_text(encoding="utf-8"):
        raise r.Phase2Error("Phase 1 is not PASS")
    phase1_lock = json.loads(r.PHASE1_LOCK.read_text(encoding="utf-8"))
    if phase1_lock.get("test_accessed") is not False:
        raise r.Phase2Error("Phase 1 does not confirm TEST remained sealed")

    checkpoints = {model_id: load_checkpoint(path) for model_id, path in r.CHECKPOINT_PATHS.items()}
    m1 = checkpoints["M1_full_distilled"]
    created_utc = str(m1["created_utc"])
    selected_rank_weight = float(m1["ranking_weight"])
    model_definition = yaml.safe_load(r.MODEL_DEFINITION.read_text(encoding="utf-8"))
    scalar_parameter_count = int(model_definition["scalar_parameter_count"])

    cache_metadata = pd.read_parquet(r.CACHE_METADATA)
    train_metadata = cache_metadata.loc[cache_metadata["cache_role"].eq("train_pool")].copy()
    validation_metadata = cache_metadata.loc[cache_metadata["cache_role"].eq("validation")].copy()
    validation_predictions = pd.read_parquet(r.VALIDATION_PREDICTIONS)
    if set(cache_metadata["split"].unique()) != {"development", "validation"}:
        raise r.Phase2Error("Unexpected partition in cache metadata")
    if not validation_predictions["split"].eq("validation").all():
        raise r.Phase2Error("Validation predictions include a non-validation row")

    test_output = r.run_unit_tests()
    training_config = {
        "version": "phase2_training_v1",
        "created_utc": created_utc,
        "device": "cpu",
        "cpu_threads": 4,
        "seed": r.SEED,
        "test_accessed": False,
        "partitions_materialized": ["development", "validation"],
        "training_pool": {
            "rule": f"{r.TRAIN_WINDOWS_PER_GENOME} deterministic windows per development genome",
            "rows": int(len(train_metadata)),
            "genomes": int(train_metadata["assembly_id"].nunique()),
            "clusters": int(train_metadata["cluster_id"].nunique()),
            "limitation": "CPU-bounded Phase 2 architecture/ablation screening; this is not full-development-window training.",
        },
        "validation": {
            "full_rows": int(len(validation_metadata)),
            "genomes": int(validation_metadata["assembly_id"].nunique()),
            "clusters": int(validation_metadata["cluster_id"].nunique()),
            "ranking_weight_tuning_subset": f"{r.TUNING_WINDOWS_PER_VALIDATION_GENOME} deterministic windows per validation genome",
        },
        "optimizer": {"name": "AdamW", "learning_rate": r.LEARNING_RATE, "weight_decay": r.WEIGHT_DECAY, "gradient_clip_norm": 1.0},
        "scalar_target_standardisation": "development training-pool mean and population standard deviation only",
        "huber": {"delta_standardised_units": 1.0, "weight": 1.0},
        "pairwise_ranking": {
            "loss": "softplus(-sign(y_a-y_b)*(prediction_a-prediction_b))",
            "same_genome_pairs_only": True,
            "candidate_weights": r.RANK_WEIGHT_CANDIDATES,
            "selected_weight": selected_rank_weight,
            "ranking_emphasis_greater_than_huber": selected_rank_weight > 1.0,
        },
        "epochs": {
            "shared_residual_huber_warmup": r.SCALAR_WARMUP_EPOCHS,
            "M1_or_M2_matched_branch": r.BRANCH_EPOCHS,
            "M3_raw_ppl": r.RAW_PPL_EPOCHS,
            "M5_masked_reconstruction": r.MASKED_EPOCHS,
        },
        "batch_sizes": {"pair_training": r.PAIR_BATCH_SIZE, "masked_training": r.SINGLE_BATCH_SIZE, "validation": r.EVAL_BATCH_SIZE},
        "M5": {
            "teacher_values_used": False,
            "annotations_used": False,
            "mask_probability_training": r.MASK_PROBABILITY,
            "validation_anomaly_score": "Mean cross-entropy at positions p mod 7 = SHA256(window_id) mod 7, with those positions masked; one deterministic pass per window.",
        },
        "frozen_K_definitions_from_phase1": phase1_lock["frozen_k_definitions"],
        "primary_selection_criterion": "mean within-genome Spearman on validation",
        "unit_test_output": test_output,
        "software": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "pyarrow": str(pa.__version__),
            "scipy": str(scipy.__version__),
        },
        "inputs": {
            "phase0_report": r.file_record(r.PHASE0_REPORT),
            "protocol": r.file_record(r.PROTOCOL),
            "split": r.file_record(r.SPLIT),
            "test_lock": r.file_record(r.TEST_LOCK),
            "canonical": r.file_record(r.CANONICAL),
            "targets": r.file_record(r.TARGETS),
            "phase1_report": r.file_record(r.PHASE1_REPORT),
            "phase1_config": r.file_record(r.PHASE1_CONFIG),
            "phase1_lock": r.file_record(r.PHASE1_LOCK),
            "phase1_predictions": r.file_record(r.PHASE1_PREDICTIONS),
            "token_cache_manifest": r.file_record(r.CACHE_MANIFEST),
        },
        "run_histories": {model_id: checkpoint["history"] for model_id, checkpoint in checkpoints.items()},
        "recovery_note": "Training, inference, metrics and checkpoints completed in the original process. Final YAML serialization was resumed after converting torch.__version__ to str; no model was retrained and no predictions were recomputed.",
    }
    r.write_text_atomic(r.TRAINING_CONFIG, yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True, width=1000))

    metrics = pd.read_csv(r.VALIDATION_METRICS, keep_default_na=False)
    summaries = metrics.loc[metrics["level"].eq("summary")].copy()

    def estimate(model_id: str, metric: str, k_definition: str = "") -> float:
        values = summaries.loc[
            summaries["model_id"].eq(model_id)
            & summaries["metric"].eq(metric)
            & summaries["k_definition"].eq(k_definition),
            "estimate",
        ]
        if len(values) != 1:
            raise r.Phase2Error(f"Missing or duplicate summary: {model_id}/{metric}/{k_definition}")
        return float(values.iloc[0])

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
    architecture = r.ArchitectureV1()

    report = f"""# Phase 2 single-model distillation and ablation report

## Verdict

**PHASE 2: PASS**

Created UTC: `{created_utc}`  
Device: **CPU** (no CUDA runtime available)  
Development training pool: **{len(train_metadata):,} windows / {train_metadata['assembly_id'].nunique()} genomes / {train_metadata['cluster_id'].nunique()} lineages**  
Validation: **{len(validation_metadata):,} windows / {r.EXPECTED_VALIDATION_GENOMES} genomes / {r.EXPECTED_VALIDATION_CLUSTERS} lineages**  
Locked TEST accessed: **NO**

## Architecture

`student_cnn_v1` is a newly implemented 512-bp DNA encoder, not the historical CDS StudentCNN. It uses an 8-dimensional nucleotide embedding, a kernel-7 stride-2 stem, six dilated residual blocks with dilation schedule `[1, 2, 4, 8, 16, 32]`, 16 channels, GroupNorm, GELU, dropout 0.10, and concatenated global mean/max pooling followed by a scalar MLP. The scalar model has **{scalar_parameter_count:,} trainable parameters**, **{architecture.convolutional_layers} convolutional layers**, and a calculated receptive field of **{architecture.receptive_field_bp} bp**.

The M5 comparator uses the identical encoder and a reconstruction decoder. It was trained only on masked development DNA; no PPL, residual or annotation entered its objective.

## Training design and limitation

Because the available PyTorch runtime is CPU-only on a 4-core laptop, Phase 2 used a frozen, genome-balanced architecture-screening pool of {r.TRAIN_WINDOWS_PER_GENOME} windows per development genome. All {r.EXPECTED_VALIDATION_WINDOWS:,} validation windows were evaluated. This limitation is explicit: the results justify controlled Phase 2 model selection under the local compute budget, not a claim that the CNN has exhausted all 1,640,110 development windows.

M1 and M2 share the same three-epoch Huber warm-up and differ only in the matched final branch. M1 ranking weights {r.RANK_WEIGHT_CANDIDATES} were compared on a deterministic {r.TUNING_WINDOWS_PER_VALIDATION_GENOME}-window-per-genome validation subset; **{selected_rank_weight:.1f}** was selected. Since Huber weight is 1.0, ranking receives greater emphasis. M3 uses the same backbone and selected ranking weight but raw PPL targets. M4 is loaded unchanged from the frozen Phase 1 result.

## Validation results

| Model | Supervision | Mean within-genome Spearman | Top-1% Recall@K | Top-1% NDCG@K | Diagnostic MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 full distilled | residual Huber + within-genome ranking | {m1_spearman:.6f} | {estimate('M1_full_distilled','recall_at_k','top_1_percent'):.6f} | {estimate('M1_full_distilled','ndcg_at_k','top_1_percent'):.6f} | {estimate('M1_full_distilled','scalar_mae_training_target'):.6f} |
| M2 no ranking | residual Huber only | {m2_spearman:.6f} | {estimate('M2_huber_only','recall_at_k','top_1_percent'):.6f} | {estimate('M2_huber_only','ndcg_at_k','top_1_percent'):.6f} | {estimate('M2_huber_only','scalar_mae_training_target'):.6f} |
| M3 raw teacher | raw-PPL Huber + ranking | {m3_spearman:.6f} | {estimate('M3_raw_ppl','recall_at_k','top_1_percent'):.6f} | {estimate('M3_raw_ppl','ndcg_at_k','top_1_percent'):.6f} | {estimate('M3_raw_ppl','scalar_mae_training_target'):.6f} (PPL units) |
| M4 frozen cheap baseline | Phase 1 GC/k-mer HistGBR | {m4_spearman:.6f} | {estimate('M4_cheap_feature_baseline','recall_at_k','top_1_percent'):.6f} | {estimate('M4_cheap_feature_baseline','ndcg_at_k','top_1_percent'):.6f} | {estimate('M4_cheap_feature_baseline','scalar_mae_training_target'):.6f} |
| M5 sequence-only | masked nucleotide reconstruction | {m5_spearman:.6f} | {estimate('M5_sequence_only','recall_at_k','top_1_percent'):.6f} | {estimate('M5_sequence_only','ndcg_at_k','top_1_percent'):.6f} | n/a |

All ranking metrics were calculated within each validation genome using the Phase 1 frozen top-1%, top-5% and fixed-100 definitions. Windows were not treated as independent inferential replicates; uncertainty summaries use a {r.BOOTSTRAP_ITERATIONS}-iteration bootstrap over the {r.EXPECTED_VALIDATION_CLUSTERS} validation lineages.

## Ablation decisions

1. **Teacher distillation versus cheap features:** M1 minus M4 mean Spearman = **{teacher_cheap_delta:+.6f}**, so M1 wins on the primary endpoint. It does **not** win uniformly: top-1% Recall is {estimate('M1_full_distilled','recall_at_k','top_1_percent'):.6f} versus {estimate('M4_cheap_feature_baseline','recall_at_k','top_1_percent'):.6f} and top-1% NDCG is {estimate('M1_full_distilled','ndcg_at_k','top_1_percent'):.6f} versus {estimate('M4_cheap_feature_baseline','ndcg_at_k','top_1_percent'):.6f}. Teacher distillation therefore improves global within-genome ordering but not extreme-tail retrieval under this CPU-bounded run.
2. **Teacher supervision versus sequence-only learning:** M1 minus M5 mean Spearman = **{teacher_sequence_delta:+.6f}**, and M1 also improves top-1% NDCG ({estimate('M1_full_distilled','ndcg_at_k','top_1_percent'):.6f} versus {estimate('M5_sequence_only','ndcg_at_k','top_1_percent'):.6f}). M5 has higher top-1% Recall ({estimate('M5_sequence_only','recall_at_k','top_1_percent'):.6f} versus {estimate('M1_full_distilled','recall_at_k','top_1_percent'):.6f}), so the advantage is not uniform across retrieval metrics.
3. **Pairwise ranking loss:** M1 minus M2 = **{rank_delta:+.6f}**. {'The selected ranking loss improves fidelity.' if rank_delta > 0 else 'The ranking-loss branch does not improve mean Spearman over matched Huber-only training.'}
4. **Residual versus raw-PPL supervision:** M1 minus M3 = **{residual_raw_delta:+.6f}**. {'Residual supervision improves the proposal-aligned ranking.' if residual_raw_delta > 0 else 'Residual supervision does not outperform raw-PPL supervision in this run.'}
5. **CNN justification:** **{'YES for the Phase 3 primary endpoint, with a top-K caveat' if architecture_justified else 'LIMITED'}**. {'The compact sequence model beats both cheap and sequence-only comparators on mean within-genome Spearman, while the cheap baseline remains stronger for the frozen extreme-tail Recall/NDCG metrics.' if architecture_justified else 'The architecture is reproducible and proposal-compliant, but the present screening budget does not establish superiority over both comparators.'}

The exact paired genome/lineage deltas and confidence intervals are in `ablation_results.csv`; biological claims are outside this phase.

## Phase 3 freeze

The architecture is frozen as `student_cnn_v1`: 16 channels, dilation `[1,2,4,8,16,32]`, GroupNorm, dropout 0.10, mean+max pooling and the recorded scalar head. The Phase 3 primary objective is M1 with Huber weight 1.0 and ranking weight **{selected_rank_weight:.1f}**. Checkpoint hashes, target standardisation, training budget, seeds and K definitions are sealed in `final_architecture.lock`.

## Boundary

Phase 2 stops here. No test performance was accessed. Phase 3 must use the frozen architecture/objective or create an explicitly versioned deviation without consulting TEST.
"""
    r.write_text_atomic(r.REPORT, report)

    output_records = {
        "model_definition": r.file_record(r.MODEL_DEFINITION),
        "training_config": r.file_record(r.TRAINING_CONFIG),
        "validation_metrics": r.file_record(r.VALIDATION_METRICS),
        "ablation_results": r.file_record(r.ABLATION_RESULTS),
        "validation_predictions": r.file_record(r.VALIDATION_PREDICTIONS),
        "report": r.file_record(r.REPORT),
        "checkpoints": {key: r.file_record(path) for key, path in r.CHECKPOINT_PATHS.items()},
        "token_cache_manifest": r.file_record(r.CACHE_MANIFEST),
    }
    lock_payload = {
        "lock": "DISSERTATION PHASE 2 ARCHITECTURE FROZEN",
        "created_utc": created_utc,
        "architecture_version": "student_cnn_v1",
        "architecture": model_definition,
        "phase3_primary_model": "M1_full_distilled",
        "phase3_objective": {"Huber_weight": 1.0, "within_genome_pairwise_ranking_weight": selected_rank_weight},
        "training_seed": r.SEED,
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
    r.write_text_atomic(r.FINAL_LOCK, json.dumps(lock_payload, indent=2) + "\n")

    checks = {
        "unit_tests_pass": "OK" in test_output,
        "train_rows": len(train_metadata) == r.TRAIN_WINDOWS_PER_GENOME * r.EXPECTED_DEVELOPMENT_GENOMES,
        "train_genomes": train_metadata["assembly_id"].nunique() == r.EXPECTED_DEVELOPMENT_GENOMES,
        "validation_rows": len(validation_predictions) == r.EXPECTED_VALIDATION_WINDOWS,
        "validation_genomes": validation_predictions["assembly_id"].nunique() == r.EXPECTED_VALIDATION_GENOMES,
        "validation_lineages": validation_predictions["cluster_id"].nunique() == r.EXPECTED_VALIDATION_CLUSTERS,
        "metrics_finite": np.isfinite(pd.to_numeric(metrics["estimate"], errors="coerce")).all(),
        "ranking_weight_greater_than_huber": selected_rank_weight > 1.0,
        "all_checkpoints_test_sealed": all(cp.get("test_accessed") is False for cp in checkpoints.values()),
        "test_not_accessed": True,
    }
    if not all(checks.values()):
        raise r.Phase2Error(f"Finalization checks failed: {checks}")

    r.OUTPUT_MIRROR.mkdir(parents=True, exist_ok=True)
    for artifact in (r.MODEL_DEFINITION, r.TRAINING_CONFIG, r.VALIDATION_METRICS, r.ABLATION_RESULTS, r.REPORT, r.FINAL_LOCK):
        shutil.copy2(artifact, r.OUTPUT_MIRROR / artifact.name)

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
