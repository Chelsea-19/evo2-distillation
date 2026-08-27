#!/usr/bin/env python
"""Generate the validation-only Phase 4.6 rank/tail Colab notebook."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "notebooks" / "00_run_all_phase4_5.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def base_cells() -> list[dict]:
    document = json.loads(MASTER.read_text(encoding="utf-8"))
    cells = copy.deepcopy(document["cells"][3:17])
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return cells


def notebook_text() -> str:
    cells = [
        markdown(
            """# Evo 2 dissertation — Phase 4.6 rank/tail Colab runner

This notebook runs a compact, pre-specified follow-up designed to improve both
within-genome ranking and extreme-tail retrieval. It reuses the verified Drive
data workflow from `00_run_all_phase4_5.ipynb`, then trains two validation-only
rank/tail candidates with a larger compact CNN, development-only tail sampling,
gap-weighted pairwise loss, an auxiliary tail head, and a frozen cheap-baseline
rank fusion.

**Safety boundary:** development is used for fitting and validation for model
selection. TEST remains locked and this notebook contains no Phase 5 unlock.
Results are not guaranteed to exceed the existing capacity model; failure is
retained and reported rather than hidden.
"""
        ),
        code(
            """# ===== USER CONTROLS =====
REPO_URL = "https://github.com/Chelsea-19/evo2-distillation.git"
REPO_BRANCH = "main"
DRIVE_ROOT = "/content/drive/MyDrive/evo2-distillation"
REPO_ROOT = "/content/evo2-distillation"
LOCAL_DATA = "/content/evo2-distillation-data"
LOCAL_RUNS = "/content/evo2-distillation-runs-phase4-6"
FASTA_ARCHIVE_NAME = "frozen_fasta_v1.tar.gz"
FASTA_ARCHIVE_SHA256 = "56a95502c4478a9564d94c7c35c93ba4548e52e094a56039a5b3ec607bbfbb67"

REQUIRE_GPU = True
AUTO_DISCOVER_DRIVE_ROOT = True
AUTO_RESTORE_FASTA_ARCHIVE = True
VERIFY_ALL_FASTA_HASHES = True
RUN_REPOSITORY_TESTS = True
RUN_CANDIDATES = True
CANDIDATE_VARIANTS = ["rank_tail_balanced", "rank_tail_focus"]
SKIP_COMPLETED_CONFIGS = True
SYNC_INTERVAL_SECONDS = 300

# Keep these False on the first pass. Review the candidate table before freezing.
APPROVE_PHASE4_6_SELECTION = False
RUN_FIVE_MEMBER_ENSEMBLE = False

assert REPO_URL == "https://github.com/Chelsea-19/evo2-distillation.git"
assert REPO_BRANCH and "test" not in REPO_BRANCH.lower()
"""
        ),
        markdown(
            """## Usage

1. Select an A100 GPU when available and run all cells with both final flags `False`.
2. The notebook resumes matching incomplete Drive runs and skips matching completed runs.
3. Review `results/phase4_6/validation_candidate_summary.csv` against the frozen references.
4. Only if the result is acceptable, set both final flags `True` and rerun from the freeze cell.
5. Do not rename a split or open TEST; the repository rejects it.
"""
        ),
    ]
    cells.extend(base_cells())
    cells.extend(
        [
            markdown("## Resume-safe Phase 4.6 candidate execution\n"),
            code(
                """import yaml

os.environ["EVO2_DATA_ROOT"] = LOCAL_DATA
os.environ["EVO2_LOCAL_RUN_ROOT"] = LOCAL_RUNS
pathlib.Path(LOCAL_RUNS).mkdir(parents=True, exist_ok=True)
drive_results = pathlib.Path(DRIVE_ROOT, "results/phase4_6")
drive_results.mkdir(parents=True, exist_ok=True)
pathlib.Path(DRIVE_ROOT, "models/phase4_6").mkdir(parents=True, exist_ok=True)


def read_yaml(path):
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def matching_drive_runs(config):
    matches = []
    for run_dir in sorted(drive_results.glob("*")):
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue
        try:
            if read_yaml(config_path) == config:
                matches.append(run_dir)
        except Exception:
            continue
    return matches


def sync_local_runs(quiet=False):
    for run_dir in pathlib.Path(LOCAL_RUNS).glob("*"):
        if not run_dir.is_dir():
            continue
        command = ["python", f"{REPO_ROOT}/scripts/sync_results_to_drive.py", "--local-run", str(run_dir), "--drive-results-root", str(drive_results)]
        if quiet:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            run_cmd(command)


class PeriodicDriveSync:
    def __init__(self, seconds):
        self.seconds = max(60, int(seconds))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.wait(self.seconds):
            sync_local_runs(quiet=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=10)
        sync_local_runs(quiet=False)


def run_config(config_path):
    config_path = pathlib.Path(config_path)
    config = read_yaml(config_path)
    assert config.get("validation_only") is True
    assert config.get("test_accessed") is False
    assert config.get("experiment_type") == "rank_tail_v2"
    matches = matching_drive_runs(config)
    complete = [p for p in matches if (p / "metrics.json").is_file() and (p / "run_manifest.json").is_file()]
    if complete and SKIP_COMPLETED_CONFIGS:
        print("SKIP completed config:", config["variant_name"], complete[-1])
        return complete[-1]
    command = ["python", f"{REPO_ROOT}/scripts/run_phase4_5.py", "--config", str(config_path)]
    incomplete = [p for p in matches if not (p / "metrics.json").is_file()]
    if incomplete:
        source_run = incomplete[-1]
        local_run = pathlib.Path(LOCAL_RUNS, source_run.name)
        shutil.copytree(source_run, local_run, dirs_exist_ok=True)
        checkpoints = sorted((local_run / "checkpoint").glob("epoch_*.pt"))
        if checkpoints:
            command = ["python", f"{REPO_ROOT}/scripts/resume_experiment.py", "--config", str(config_path), "--checkpoint", str(checkpoints[-1])]
            print("RESUME:", checkpoints[-1])
    with PeriodicDriveSync(SYNC_INTERVAL_SECONDS):
        run_cmd(command)
    complete = [p for p in matching_drive_runs(config) if (p / "metrics.json").is_file()]
    if not complete:
        raise RuntimeError(f"No completed Drive result found for {config_path}")
    return complete[-1]


if RUN_CANDIDATES:
    for variant in CANDIDATE_VARIANTS:
        if variant not in {"rank_tail_balanced", "rank_tail_focus"}:
            raise ValueError(f"Unsupported pre-specified variant: {variant}")
        run_config(pathlib.Path(REPO_ROOT, "configs/phase4_6", f"{variant}.yaml"))
else:
    print("Candidate execution disabled; existing Drive results will be analyzed.")
"""
            ),
            markdown("## Validation-only comparison and selection preview\n"),
            code(
                """import pandas as pd

rows = []
for run_dir in drive_results.glob("*"):
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"
    if not (config_path.is_file() and metrics_path.is_file() and manifest_path.is_file()):
        continue
    config = read_yaml(config_path)
    if config.get("variant_name") not in CANDIDATE_VARIANTS:
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metrics.get("test_accessed") is False and manifest.get("test_accessed") is False
    rows.append({
        "run_id": run_dir.name,
        "variant": config["variant_name"],
        "parameters": manifest["parameter_count"],
        "mean_spearman": metrics["mean_within_genome_spearman"],
        "raw_student_spearman": metrics.get("raw_student_mean_within_genome_spearman"),
        "recall_top_1_percent": metrics["mean_recall_top_1_percent"],
        "ndcg_top_1_percent": metrics["mean_ndcg_top_1_percent"],
        "run_dir": str(run_dir),
    })

summary = pd.DataFrame(rows)
if summary.empty:
    raise RuntimeError("No completed Phase 4.6 candidate results found on Drive")
summary = summary.sort_values(
    ["mean_spearman", "recall_top_1_percent", "ndcg_top_1_percent", "run_id"],
    ascending=[False, False, False, True],
).drop_duplicates("variant", keep="first").reset_index(drop=True)
summary["delta_vs_phase4_5_capacity_spearman"] = summary["mean_spearman"] - 0.504923
summary["delta_vs_phase1_cheap_recall_1pct"] = summary["recall_top_1_percent"] - 0.057567
display(summary.drop(columns=["run_dir"]))
summary.to_csv(drive_results / "validation_candidate_summary.csv", index=False)

selected = summary.iloc[0]
print("VALIDATION SELECTION PREVIEW:", selected["variant"], selected["run_id"])
print("Frozen references: capacity Spearman=0.504923; cheap-baseline Recall@1%=0.057567")
print("Primary criterion: maximum mean within-genome Spearman")
print("Tie-breakers: Recall@1%, NDCG@1%, run_id")
print("TEST ACCESSED: NO")
"""
            ),
            markdown("## Freeze the winner and optionally train five seeds (explicit gate)\n"),
            code(
                """selected_config = None
if APPROVE_PHASE4_6_SELECTION:
    selected_run = pathlib.Path(selected["run_dir"])
    selected_config = read_yaml(selected_run / "config.yaml")
    selected_config["seeds"] = [11, 23, 37, 53, 71]
    config_text = yaml.safe_dump(selected_config, sort_keys=False)
    lock = {
        "lock_version": "phase4_6_rank_tail_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_partition": "validation",
        "selected_variant": selected["variant"],
        "selected_run_id": selected["run_id"],
        "mean_spearman": float(selected["mean_spearman"]),
        "recall_top_1_percent": float(selected["recall_top_1_percent"]),
        "ndcg_top_1_percent": float(selected["ndcg_top_1_percent"]),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "git_commit": GIT_COMMIT,
        "test_accessed": False,
    }
    model_dir = pathlib.Path(DRIVE_ROOT, "models/phase4_6")
    (model_dir / "selected_rank_tail_v1.yaml").write_text(config_text, encoding="utf-8")
    atomic_json(model_dir / "selected_rank_tail_v1.lock", lock)
    print("PHASE 4.6 WINNER FROZEN:", lock)
else:
    print("GATE CLOSED. Review validation_candidate_summary.csv first.")

if RUN_FIVE_MEMBER_ENSEMBLE:
    if not APPROVE_PHASE4_6_SELECTION:
        raise RuntimeError("RUN_FIVE_MEMBER_ENSEMBLE requires APPROVE_PHASE4_6_SELECTION=True")
    member_dir = pathlib.Path(LOCAL_DATA, "models/phase4_6/member_configs")
    member_dir.mkdir(parents=True, exist_ok=True)
    for seed in [11, 23, 37, 53, 71]:
        member = dict(selected_config)
        member.pop("seeds", None)
        member["seed"] = seed
        member["variant_name"] = f"{selected_config['variant_name']}_ensemble_seed{seed}"
        member_path = member_dir / f"member_seed{seed}.yaml"
        member_path.write_text(yaml.safe_dump(member, sort_keys=False), encoding="utf-8")
        run_config(member_path)
    sync_local_runs(quiet=False)
    print("FIVE-MEMBER TRAINING COMPLETE OR RESUMED; TEST ACCESSED: NO")
else:
    print("Five-member ensemble not requested in this pass.")
"""
            ),
            markdown("## Five-member validation ensemble analysis\n"),
            code(
                """if RUN_FIVE_MEMBER_ENSEMBLE:
    import numpy as np
    from evo2_distill.evaluation.ranking import evaluate_validation_predictions

    expected_seeds = [11, 23, 37, 53, 71]
    member_runs = {}
    for run_dir in sorted(drive_results.glob("*")):
        config_path = run_dir / "config.yaml"
        prediction_path = run_dir / "predictions/validation_predictions.parquet"
        manifest_path = run_dir / "run_manifest.json"
        if not (config_path.is_file() and prediction_path.is_file() and manifest_path.is_file()):
            continue
        config = read_yaml(config_path)
        seed = config.get("seed")
        expected_variant = f"{selected_config['variant_name']}_ensemble_seed{seed}"
        if seed in expected_seeds and config.get("variant_name") == expected_variant:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest.get("test_accessed") is False
            member_runs[seed] = run_dir

    if len(member_runs) != 5:
        raise RuntimeError(f"Expected five matching members, found {len(member_runs)}/5")
    frames = [pd.read_parquet(member_runs[seed] / "predictions/validation_predictions.parquet").sort_values("window_id").reset_index(drop=True) for seed in expected_seeds]
    ensemble = frames[0][["window_id", "assembly_id", "cluster_id", "teacher_target"]].copy()
    matrix = []
    for seed, frame in zip(expected_seeds, frames):
        assert frame["window_id"].equals(ensemble["window_id"])
        assert set(frame["split"].unique()) == {"validation"}
        ensemble[f"member_seed_{seed}"] = frame["prediction"].to_numpy(float)
        matrix.append(frame["prediction"].to_numpy(float))
    matrix = np.vstack(matrix)
    ensemble["prediction"] = matrix.mean(axis=0)
    ensemble["prediction_variance"] = matrix.var(axis=0, ddof=1)
    ensemble["split"] = "validation"
    per_genome, metrics = evaluate_validation_predictions(ensemble)
    metrics.update({"members": 5, "seeds": expected_seeds, "test_accessed": False})
    output = drive_results / "final_ensemble_summary"
    output.mkdir(parents=True, exist_ok=True)
    ensemble.to_parquet(output / "validation_ensemble_predictions.parquet", index=False)
    per_genome.to_csv(output / "ensemble_metrics_by_genome.csv", index=False)
    atomic_json(output / "ensemble_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    print("Prediction variance available: YES")
    print("TEST ACCESSED: NO")
else:
    print("Ensemble analysis waits until the explicit ensemble gate is opened.")
"""
            ),
            markdown("## Final status\n"),
            code(
                """print("Git commit:", GIT_COMMIT)
print("Drive root:", DRIVE_ROOT)
print("Phase 4.6 summary:", drive_results / "validation_candidate_summary.csv")
print("TEST ACCESSED: NO")
print("TEST REMAINS LOCKED: YES")
print("PHASE 5 STARTED: NO")
if not APPROVE_PHASE4_6_SELECTION:
    print("NEXT: review the candidate table before opening the freeze gate.")
elif not RUN_FIVE_MEMBER_ENSEMBLE:
    print("NEXT: optionally set RUN_FIVE_MEMBER_ENSEMBLE=True and rerun from the freeze cell.")
else:
    print("PHASE 4.6 VALIDATION WORKFLOW COMPLETE")
"""
            ),
        ]
    )
    document = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "05_phase4_6_rank_tail.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks" / "05_phase4_6_rank_tail.ipynb")
    args = parser.parse_args()
    expected = notebook_text()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"Generated notebook is stale: {args.output}")
        print(f"Notebook generation check PASS: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
