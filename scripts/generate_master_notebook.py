#!/usr/bin/env python
"""Generate the single, validation-only Colab orchestration notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# Evo 2 dissertation — unified Colab runner

This is the single entry point for the post-migration workflow. It verifies the
frozen Phase 0–4 records, prepares data, builds/verifies the token cache, runs
Phase 4.5 candidates, summarizes validation metrics, and optionally freezes and
trains the final five-member validation ensemble.

**Safety boundary:** development is used for fitting, validation for selection,
and TEST remains locked. This notebook contains no Phase 5 unlock.
"""
    ),
    code(
        """# ===== USER CONTROLS =====
REPO_URL = "https://github.com/Chelsea-19/evo2-distillation.git"
REPO_BRANCH = "main"
DRIVE_ROOT = "/content/drive/MyDrive/evo2-distillation"
REPO_ROOT = "/content/evo2-distillation"
LOCAL_DATA = "/content/evo2-distillation-data"
LOCAL_RUNS = "/content/evo2-distillation-runs"
FASTA_ARCHIVE_NAME = "frozen_fasta_v1.tar.gz"
FASTA_ARCHIVE_SHA256 = "56a95502c4478a9564d94c7c35c93ba4548e52e094a56039a5b3ec607bbfbb67"

REQUIRE_GPU = True
AUTO_DISCOVER_DRIVE_ROOT = True
AUTO_RESTORE_FASTA_ARCHIVE = True
VERIFY_ALL_FASTA_HASHES = True
RUN_REPOSITORY_TESTS = True
RUN_CANDIDATES = True
CANDIDATE_EXPERIMENTS = [
    "data_scaling",
    "capacity_scaling",
    "tail_aware",
    "baseline_correction",
]
SKIP_COMPLETED_CONFIGS = True
SYNC_INTERVAL_SECONDS = 300

# Leave both False for the first pass. Review the validation table first.
APPROVE_FINAL_SELECTION = False
RUN_FINAL_ENSEMBLE = False
RUN_ENSEMBLE_ANALYSIS = True

assert REPO_URL == "https://github.com/Chelsea-19/evo2-distillation.git"
assert REPO_BRANCH and "test" not in REPO_BRANCH.lower()
"""
    ),
    markdown(
        """## How to use this notebook

1. First pass: keep the two final-selection flags `False` and run all cells.
2. If the Drive FASTA folder is incomplete, upload `frozen_fasta_v1.tar.gz`
   into `DRIVE_ROOT`; the notebook restores only missing/wrong-size files.
3. Review the validation-only candidate table produced below.
4. To accept the deterministic winner, set both flags to `True`, rerun the
   controls cell, then rerun from **Freeze the validation-selected student**.
5. Never change a split to TEST. The code rejects TEST even if requested.
"""
    ),
    code(
        """# Mount Drive and inspect the runtime.
from google.colab import drive
drive.mount("/content/drive")

import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else 0)
print("/content free GiB:", round(shutil.disk_usage("/content").free / 2**30, 2))
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU runtime: Runtime > Change runtime type > GPU")
"""
    ),
    code(
        """# Command helper: streams the actual error instead of hiding it in CalledProcessError.
def run_cmd(command, *, cwd=None, env=None):
    command = [str(item) for item in command]
    print("\\n$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return_code = process.wait()
    if return_code:
        tail = "".join(lines[-30:])
        raise RuntimeError(f"Command failed with exit code {return_code}. Last output:\\n{tail}")
    return "".join(lines)


def atomic_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
"""
    ),
    markdown("## Acquire the exact GitHub repository and install dependencies\n"),
    code(
        """repo = pathlib.Path(REPO_ROOT)
if (repo / ".git").is_dir():
    remote_url = run_cmd(["git", "-C", REPO_ROOT, "remote", "get-url", "origin"]).strip()
    if remote_url.rstrip("/").removesuffix(".git") != REPO_URL.rstrip("/").removesuffix(".git"):
        raise RuntimeError(f"Unexpected repository remote: {remote_url}")
    run_cmd(["git", "-C", REPO_ROOT, "checkout", REPO_BRANCH])
    run_cmd(["git", "-C", REPO_ROOT, "pull", "--ff-only", "origin", REPO_BRANCH])
elif repo.exists():
    raise RuntimeError(f"{REPO_ROOT} exists but is not a Git repository. Restart the runtime or choose an empty REPO_ROOT.")
else:
    run_cmd(["git", "clone", "--branch", REPO_BRANCH, "--single-branch", REPO_URL, REPO_ROOT])

run_cmd(["python", "-m", "pip", "install", "-r", f"{REPO_ROOT}/requirements/requirements-colab.txt"])
run_cmd(["python", "-m", "pip", "install", "-e", REPO_ROOT])
run_cmd(["python", f"{REPO_ROOT}/scripts/verify_environment.py", "--require-gpu"])
GIT_COMMIT = run_cmd(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"]).strip()
print("Git commit:", GIT_COMMIT)
"""
    ),
    markdown("## Locate and validate the Google Drive payload\n"),
    code(
        """required_markers = [
    "frozen_data/manifests/dataset_manifest_v1.json",
    "frozen_data/manifests/split_manifest_v1.json",
    "frozen_data/manifests/fasta_manifest.csv",
    "frozen_data/manifests/TEST_SET_LOCKED.txt",
    "frozen_data/splits/dissertation_split_v1.csv",
    "frozen_data/targets/window512_targets_v1.parquet",
    "frozen_data/targets/window512_canonical_v1.parquet",
]


def valid_drive_root(candidate):
    candidate = pathlib.Path(candidate)
    return candidate.is_dir() and all((candidate / marker).is_file() for marker in required_markers)


requested_root = pathlib.Path(DRIVE_ROOT)
if not valid_drive_root(requested_root) and AUTO_DISCOVER_DRIVE_ROOT:
    search_root = pathlib.Path("/content/drive/MyDrive")
    matches = []
    for manifest in search_root.glob("*/frozen_data/manifests/dataset_manifest_v1.json"):
        candidate = manifest.parents[2]
        if valid_drive_root(candidate):
            matches.append(candidate)
    legacy = pathlib.Path("/content/drive/MyDrive/evo2-distillation")
    if valid_drive_root(legacy):
        matches.append(legacy)
    matches = sorted(set(matches))
    if len(matches) == 1:
        requested_root = matches[0]
        print("Auto-discovered Drive root:", requested_root)
    elif not matches:
        raise RuntimeError(
            "Drive payload not found. Upload the CONTENTS of the prepared folder to "
            "/content/drive/MyDrive/evo2-distillation so that frozen_data/ is directly inside it."
        )
    else:
        raise RuntimeError(f"Multiple valid Drive payloads found; set DRIVE_ROOT explicitly: {matches}")

DRIVE_ROOT = str(requested_root)
for relative in [
    "caches/tokens",
    "models/phase4_5",
    "checkpoints/phase4_5",
    "predictions/phase4_5",
    "results/phase4_5",
    "logs/phase4_5",
    "environments",
]:
    pathlib.Path(DRIVE_ROOT, relative).mkdir(parents=True, exist_ok=True)

print("Drive root:", DRIVE_ROOT)
for marker in required_markers:
    print("PASS", marker)

if AUTO_RESTORE_FASTA_ARCHIVE:
    run_cmd([
        "python",
        f"{REPO_ROOT}/scripts/restore_fasta_archive.py",
        "--drive-root",
        DRIVE_ROOT,
        "--archive",
        str(pathlib.Path(DRIVE_ROOT, FASTA_ARCHIVE_NAME)),
        "--expected-archive-sha256",
        FASTA_ARCHIVE_SHA256,
    ])
"""
    ),
    markdown("## Verify frozen Phase 0–4 gates and TEST lock\n"),
    code(
        """phase_reports = {
    0: "experiments/dissertation_v1/phase0/phase0_completion_report.md",
    1: "experiments/dissertation_v1/phase1/phase1_report.md",
    2: "experiments/dissertation_v1/phase2/phase2_report.md",
    3: "experiments/dissertation_v1/phase3/phase3_report.md",
    4: "experiments/dissertation_v1/phase4/phase4_report.md",
}
required_locks = [
    "experiments/dissertation_v1/phase1/selected_baseline.lock",
    "experiments/dissertation_v1/phase2/final_architecture.lock",
    "experiments/dissertation_v1/phase3/ensemble.lock",
    "experiments/dissertation_v1/phase4/final_routing_rule.lock",
]
for phase, relative in phase_reports.items():
    text = pathlib.Path(REPO_ROOT, relative).read_text(encoding="utf-8")
    assert f"PHASE {phase}: PASS" in text, f"Phase {phase} is not PASS"
    assert "TEST accessed: **YES**" not in text and "Locked TEST accessed: **YES**" not in text
    print(f"PHASE {phase}: PASS")
for relative in required_locks:
    assert pathlib.Path(REPO_ROOT, relative).is_file(), f"Missing frozen lock: {relative}"
test_lock = pathlib.Path(DRIVE_ROOT, "frozen_data/manifests/TEST_SET_LOCKED.txt")
assert test_lock.is_file() and "LOCK" in test_lock.read_text(encoding="utf-8").upper()
print("TEST REMAINS LOCKED: YES")
"""
    ),
    markdown("## Verify all frozen data and FASTA hashes\n"),
    code(
        """verify_command = [
    "python",
    f"{REPO_ROOT}/scripts/verify_data_manifest.py",
    "--drive-root",
    DRIVE_ROOT,
]
if VERIFY_ALL_FASTA_HASHES:
    verify_command.append("--verify-fasta-hashes")
run_cmd(verify_command)
"""
    ),
    markdown("## Stage data to Colab local SSD and prepare the complete token cache\n"),
    code(
        """local_data = pathlib.Path(LOCAL_DATA)
local_cache = local_data / "caches/tokens"
drive_cache = pathlib.Path(DRIVE_ROOT, "caches/tokens")
drive_cache_manifest = drive_cache / "token_cache_manifest.json"

stage_command = [
    "python",
    f"{REPO_ROOT}/scripts/stage_drive_data.py",
    "--drive-root",
    DRIVE_ROOT,
    "--local-root",
    LOCAL_DATA,
]
if not drive_cache_manifest.is_file():
    stage_command.append("--include-fasta")
run_cmd(stage_command)

local_cache_manifest = local_cache / "token_cache_manifest.json"
cache_command = [
    "python",
    f"{REPO_ROOT}/scripts/build_token_cache.py",
    "--drive-root",
    LOCAL_DATA,
    "--output-dir",
    str(local_cache),
]
if local_cache_manifest.is_file():
    cache_command.append("--verify-only")
run_cmd(cache_command)

if not drive_cache_manifest.is_file():
    from evo2_distill.utils.io import atomic_copy
    drive_cache.mkdir(parents=True, exist_ok=True)
    for source in local_cache.glob("*"):
        if source.is_file():
            print("Persisting cache artifact:", source.name)
            atomic_copy(source, drive_cache / source.name)

run_cmd([
    "python",
    f"{REPO_ROOT}/scripts/build_token_cache.py",
    "--drive-root",
    LOCAL_DATA,
    "--output-dir",
    str(local_cache),
    "--verify-only",
])
print("CACHE READY; TEST ACCESSED: NO")
"""
    ),
    markdown("## Repository safety tests and GPU benchmark\n"),
    code(
        """if RUN_REPOSITORY_TESTS:
    run_cmd(["python", "-m", "pytest", "-q", f"{REPO_ROOT}/tests"])

benchmark_path = "/content/evo2-distillation-gpu-benchmark.json"
run_cmd([
    "python",
    f"{REPO_ROOT}/scripts/benchmark_gpu.py",
    "--output",
    benchmark_path,
    "--device",
    "cuda",
])
from evo2_distill.utils.io import atomic_copy
atomic_copy(benchmark_path, pathlib.Path(DRIVE_ROOT, "results/phase4_5/gpu_benchmark.json"))
"""
    ),
    markdown("## Resume-safe Phase 4.5 candidate execution\n"),
    code(
        """import yaml

os.environ["EVO2_DATA_ROOT"] = LOCAL_DATA
os.environ["EVO2_LOCAL_RUN_ROOT"] = LOCAL_RUNS
pathlib.Path(LOCAL_RUNS).mkdir(parents=True, exist_ok=True)
drive_results = pathlib.Path(DRIVE_ROOT, "results/phase4_5")


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
        command = [
            "python",
            f"{REPO_ROOT}/scripts/sync_results_to_drive.py",
            "--local-run",
            str(run_dir),
            "--drive-results-root",
            str(drive_results),
        ]
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
    assert config.get("experiment_type") != "test"

    matches = matching_drive_runs(config)
    complete = [path for path in matches if (path / "metrics.json").is_file() and (path / "run_manifest.json").is_file()]
    if complete and SKIP_COMPLETED_CONFIGS:
        print("SKIP completed config:", config["experiment_type"], complete[-1])
        return complete[-1]

    incomplete = [path for path in matches if not (path / "metrics.json").is_file()]
    if config.get("experiment_type") == "final_ensemble" and "seeds" not in config:
        command = ["python", f"{REPO_ROOT}/scripts/run_experiment.py", "--config", str(config_path)]
    else:
        command = ["python", f"{REPO_ROOT}/scripts/run_phase4_5.py", "--config", str(config_path)]
    if incomplete:
        source_run = incomplete[-1]
        local_run = pathlib.Path(LOCAL_RUNS, source_run.name)
        shutil.copytree(source_run, local_run, dirs_exist_ok=True)
        checkpoints = sorted((local_run / "checkpoint").glob("epoch_*.pt"))
        if checkpoints:
            command = [
                "python",
                f"{REPO_ROOT}/scripts/resume_experiment.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoints[-1]),
            ]
            print("RESUME:", checkpoints[-1])

    with PeriodicDriveSync(SYNC_INTERVAL_SECONDS):
        run_cmd(command)
    refreshed = matching_drive_runs(config)
    complete = [path for path in refreshed if (path / "metrics.json").is_file()]
    if not complete:
        raise RuntimeError(f"No completed Drive result found for {config_path}")
    return complete[-1]


candidate_outputs = []
if RUN_CANDIDATES:
    for experiment in CANDIDATE_EXPERIMENTS:
        if experiment not in {"data_scaling", "capacity_scaling", "tail_aware", "baseline_correction"}:
            raise ValueError(f"Unsupported candidate: {experiment}")
        candidate_outputs.append(run_config(pathlib.Path(REPO_ROOT, "configs/phase4_5", f"{experiment}.yaml")))
else:
    print("Candidate execution disabled; existing Drive results will be analyzed.")
"""
    ),
    markdown("## Validation-only candidate summary and deterministic selection preview\n"),
    code(
        """import pandas as pd

candidate_rows = []
candidate_types = set(CANDIDATE_EXPERIMENTS)
for run_dir in drive_results.glob("*"):
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"
    if not (config_path.is_file() and metrics_path.is_file() and manifest_path.is_file()):
        continue
    config = read_yaml(config_path)
    if config.get("experiment_type") not in candidate_types:
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert manifest.get("test_accessed") is False and metrics.get("test_accessed") is False
    candidate_rows.append({
        "run_id": run_dir.name,
        "experiment_type": config["experiment_type"],
        "seed": config["seed"],
        "mean_within_genome_spearman": metrics["mean_within_genome_spearman"],
        "median_within_genome_spearman": metrics["median_within_genome_spearman"],
        "mean_recall_top_1_percent": metrics["mean_recall_top_1_percent"],
        "mean_ndcg_top_1_percent": metrics["mean_ndcg_top_1_percent"],
        "run_dir": str(run_dir),
    })

candidate_summary = pd.DataFrame(candidate_rows)
if candidate_summary.empty:
    raise RuntimeError("No completed Phase 4.5 candidate results were found on Drive.")
candidate_summary = candidate_summary.sort_values(
    [
        "mean_within_genome_spearman",
        "mean_recall_top_1_percent",
        "mean_ndcg_top_1_percent",
        "run_id",
    ],
    ascending=[False, False, False, True],
).drop_duplicates("experiment_type", keep="first").reset_index(drop=True)
display(candidate_summary.drop(columns=["run_dir"]))
candidate_summary.to_csv(drive_results / "validation_candidate_summary.csv", index=False)

selected_row = candidate_summary.iloc[0]
print("VALIDATION SELECTION PREVIEW:", selected_row["experiment_type"], selected_row["run_id"])
print("Primary criterion: maximum mean within-genome Spearman")
print("Tie-breakers: top-1% Recall, top-1% NDCG, run_id")
print("TEST ACCESSED: NO")
"""
    ),
    markdown("## Freeze the validation-selected student (explicit gate)\n"),
    code(
        """selected_final_config = None
if APPROVE_FINAL_SELECTION:
    selected_run = pathlib.Path(selected_row["run_dir"])
    selected_candidate_config = read_yaml(selected_run / "config.yaml")
    selected_variant = selected_candidate_config["experiment_type"]
    selected_final_config = dict(selected_candidate_config)
    selected_final_config["experiment_type"] = "final_ensemble"
    selected_final_config["student_variant"] = selected_variant
    selected_final_config["seed"] = 11
    selected_final_config["seeds"] = [11, 23, 37, 53, 71]
    selected_final_config.setdefault("paths", {})["final_student_lock"] = "${EVO2_DATA_ROOT}/models/phase4_5/final_student_v2.lock"

    config_text = yaml.safe_dump(selected_final_config, sort_keys=False)
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    lock_payload = {
        "lock_version": "final_student_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_partition": "validation",
        "primary_criterion": "maximum mean within-genome Spearman",
        "tie_breakers": ["top-1% Recall", "top-1% NDCG", "run_id"],
        "selected_run_id": selected_row["run_id"],
        "selected_variant": selected_variant,
        "selected_mean_within_genome_spearman": float(selected_row["mean_within_genome_spearman"]),
        "final_config_sha256": config_sha256,
        "git_commit": GIT_COMMIT,
        "test_accessed": False,
    }

    drive_model_dir = pathlib.Path(DRIVE_ROOT, "models/phase4_5")
    drive_model_dir.mkdir(parents=True, exist_ok=True)
    drive_config = drive_model_dir / "final_student_v2.yaml"
    drive_config.write_text(config_text, encoding="utf-8")
    atomic_json(drive_model_dir / "final_student_v2.lock", lock_payload)

    local_model_dir = pathlib.Path(LOCAL_DATA, "models/phase4_5")
    local_model_dir.mkdir(parents=True, exist_ok=True)
    local_config = local_model_dir / "final_student_v2.yaml"
    local_config.write_text(config_text, encoding="utf-8")
    atomic_json(local_model_dir / "final_student_v2.lock", lock_payload)
    print("FINAL STUDENT FROZEN:", lock_payload)
else:
    print("GATE CLOSED: review validation_candidate_summary.csv before freezing the final student.")
    print("To continue, set APPROVE_FINAL_SELECTION=True and RUN_FINAL_ENSEMBLE=True, then rerun from this cell.")
"""
    ),
    markdown("## Train/resume the selected five-member ensemble\n"),
    code(
        """if RUN_FINAL_ENSEMBLE:
    if not APPROVE_FINAL_SELECTION:
        raise RuntimeError("RUN_FINAL_ENSEMBLE requires APPROVE_FINAL_SELECTION=True")
    local_config = pathlib.Path(LOCAL_DATA, "models/phase4_5/final_student_v2.yaml")
    local_lock = pathlib.Path(LOCAL_DATA, "models/phase4_5/final_student_v2.lock")
    assert local_config.is_file() and local_lock.is_file()
    final_template = read_yaml(local_config)
    assert final_template.get("seeds") == [11, 23, 37, 53, 71]
    member_config_dir = pathlib.Path(LOCAL_DATA, "models/phase4_5/member_configs")
    member_config_dir.mkdir(parents=True, exist_ok=True)
    for seed in final_template["seeds"]:
        member_config = dict(final_template)
        member_config.pop("seeds", None)
        member_config["seed"] = seed
        member_config_path = member_config_dir / f"final_student_v2_seed{seed}.yaml"
        member_config_path.write_text(yaml.safe_dump(member_config, sort_keys=False), encoding="utf-8")
        run_config(member_config_path)
    sync_local_runs(quiet=False)
    print("Five-member validation ensemble complete or resumed; TEST ACCESSED: NO")
else:
    print("Final ensemble not requested in this pass.")
"""
    ),
    markdown("## Final ensemble validation analysis\n"),
    code(
        """if RUN_ENSEMBLE_ANALYSIS and pathlib.Path(DRIVE_ROOT, "models/phase4_5/final_student_v2.lock").is_file():
    lock_payload = json.loads(pathlib.Path(DRIVE_ROOT, "models/phase4_5/final_student_v2.lock").read_text(encoding="utf-8"))
    final_template = read_yaml(pathlib.Path(DRIVE_ROOT, "models/phase4_5/final_student_v2.yaml"))
    expected_seeds = [11, 23, 37, 53, 71]
    members = {}
    for run_dir in sorted(drive_results.glob("*")):
        config_path = run_dir / "config.yaml"
        prediction_path = run_dir / "predictions/validation_predictions.parquet"
        manifest_path = run_dir / "run_manifest.json"
        if not (config_path.is_file() and prediction_path.is_file() and manifest_path.is_file()):
            continue
        config = read_yaml(config_path)
        if config.get("experiment_type") != "final_ensemble":
            continue
        seed = config.get("seed")
        comparable = dict(final_template)
        comparable.pop("seeds", None)
        comparable["seed"] = seed
        actual = dict(config)
        actual.pop("seeds", None)
        if seed in expected_seeds and actual == comparable:
            members[seed] = run_dir

    if len(members) == 5:
        member_frames = []
        for seed in expected_seeds:
            frame = pd.read_parquet(members[seed] / "predictions/validation_predictions.parquet")
            assert set(frame["split"].unique()) == {"validation"}
            member_frames.append(frame.sort_values("window_id").reset_index(drop=True))
        reference = member_frames[0][["window_id", "assembly_id", "cluster_id", "teacher_target"]].copy()
        prediction_matrix = []
        for seed, frame in zip(expected_seeds, member_frames):
            assert frame["window_id"].equals(reference["window_id"])
            prediction_matrix.append(frame["prediction"].to_numpy())
            reference[f"member_seed_{seed}"] = frame["prediction"].to_numpy()
        import numpy as np
        prediction_matrix = np.vstack(prediction_matrix)
        reference["prediction"] = prediction_matrix.mean(axis=0)
        reference["prediction_variance"] = prediction_matrix.var(axis=0, ddof=1)
        reference["split"] = "validation"

        from evo2_distill.evaluation.ranking import evaluate_validation_predictions
        per_genome, ensemble_metrics = evaluate_validation_predictions(reference)
        ensemble_metrics.update({
            "members": 5,
            "seeds": expected_seeds,
            "uncertainty": "sample variance across member predictions",
            "selected_variant": lock_payload["selected_variant"],
            "test_accessed": False,
        })
        output_dir = drive_results / "final_ensemble_summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        reference.to_parquet(output_dir / "validation_ensemble_predictions.parquet", index=False)
        per_genome.to_csv(output_dir / "ensemble_metrics_by_genome.csv", index=False)
        atomic_json(output_dir / "ensemble_metrics.json", ensemble_metrics)
        print(json.dumps(ensemble_metrics, indent=2))
        print("Prediction variance available: YES")
    else:
        print(f"Ensemble analysis waiting for five complete matching members: {len(members)}/5")
else:
    print("Ensemble analysis skipped until a final student lock exists.")
"""
    ),
    markdown("## Final status\n"),
    code(
        """print("Git commit:", GIT_COMMIT)
print("Drive root:", DRIVE_ROOT)
print("Candidate summary:", drive_results / "validation_candidate_summary.csv")
print("TEST ACCESSED: NO")
print("TEST REMAINS LOCKED: YES")
print("PHASE 5 STARTED: NO")
if not APPROVE_FINAL_SELECTION:
    print("NEXT: review the candidate table, then explicitly open the final-selection gate.")
elif not RUN_FINAL_ENSEMBLE:
    print("NEXT: set RUN_FINAL_ENSEMBLE=True and rerun from the final-selection section.")
else:
    print("VALIDATION-ONLY PHASE 4.5 WORKFLOW COMPLETE")
"""
    ),
]


def notebook_text() -> str:
    document = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "00_run_all_phase4_5.ipynb", "provenance": []},
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "00_run_all_phase4_5.ipynb",
    )
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
