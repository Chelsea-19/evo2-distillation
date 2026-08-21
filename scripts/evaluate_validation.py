#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evo2_distill.data.safety import require_phase4_5_split
from evo2_distill.evaluation.ranking import evaluate_validation_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = require_phase4_5_split(args.split)
    if split != "validation":
        raise SystemExit("This command reports model metrics on validation only")
    frame = pd.read_parquet(args.predictions)
    if "split" in frame and not frame["split"].eq("validation").all():
        raise SystemExit("Prediction file contains non-validation rows")
    per_genome, summary = evaluate_validation_predictions(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    per_genome.to_csv(args.output / "validation_metrics_by_genome.csv", index=False)
    (args.output / "validation_metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

