#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path

import yaml

from evo2_distill.training.trainer import run_training

ALLOWED_TYPES = {"data_scaling", "capacity_scaling", "tail_aware", "baseline_correction", "final_ensemble"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["development", "validation", "test"])
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("REFUSED: Phase 4.5 cannot access --split test; no Phase 5 unlock is implemented.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_type") not in ALLOWED_TYPES:
        raise SystemExit(f"Unsupported Phase 4.5 experiment_type: {config.get('experiment_type')}")
    if config.get("experiment_type") == "final_ensemble":
        lock = Path(os.path.expandvars(config["paths"]["final_student_lock"]))
        if not lock.is_file():
            raise SystemExit("REFUSED: final_student_v2.lock is required before final ensemble training.")
    if args.dry_run:
        print(f"DRY RUN PASS: {args.config}; TEST ACCESSED: NO")
        return
    if config["experiment_type"] == "final_ensemble":
        seeds = config.get("seeds")
        if seeds != [11, 23, 37, 53, 71]:
            raise SystemExit("Final ensemble seeds must be exactly [11, 23, 37, 53, 71]")
        if args.resume:
            raise SystemExit("Resume one ensemble member with scripts/resume_experiment.py and its seeded run config")
        outputs = []
        temporary_root = Path(os.environ.get("EVO2_LOCAL_RUN_ROOT", "/content/evo2-distillation-runs"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            member = copy.deepcopy(config)
            member["seed"] = seed
            member.pop("seeds", None)
            with tempfile.NamedTemporaryFile("w", suffix=f"_seed{seed}.yaml", dir=temporary_root, delete=False, encoding="utf-8") as handle:
                yaml.safe_dump(member, handle, sort_keys=False)
                member_config = Path(handle.name)
            try:
                outputs.append(str(run_training(member_config)))
            finally:
                member_config.unlink(missing_ok=True)
        print("\n".join(outputs))
    else:
        print(run_training(args.config, args.resume))


if __name__ == "__main__":
    main()
