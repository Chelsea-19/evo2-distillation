from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FROZEN_K = {
    "top_1_percent": lambda n: int(math.ceil(0.01 * n)),
    "top_5_percent": lambda n: int(math.ceil(0.05 * n)),
    "fixed_100": lambda n: min(100, n),
}


def _top_indices(values: np.ndarray, window_ids: np.ndarray, k: int) -> np.ndarray:
    return np.lexsort((window_ids.astype(str), -values))[:k]


def _ndcg(prediction: np.ndarray, relevance: np.ndarray, window_ids: np.ndarray, k: int) -> float:
    predicted = _top_indices(prediction, window_ids, k)
    ideal = _top_indices(relevance, window_ids, k)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[predicted] * discount))
    idcg = float(np.sum(relevance[ideal] * discount))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_validation_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {"window_id", "assembly_id", "cluster_id", "teacher_target", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame missing columns: {sorted(missing)}")
    rows: list[dict[str, float | str | int]] = []
    for assembly_id, group in frame.groupby("assembly_id", sort=True):
        teacher = group["teacher_target"].to_numpy(float)
        prediction = group["prediction"].to_numpy(float)
        window_ids = group["window_id"].astype(str).to_numpy()
        correlation = float(spearmanr(prediction, teacher).statistic)
        row: dict[str, float | str | int] = {
            "assembly_id": str(assembly_id),
            "cluster_id": str(group["cluster_id"].iloc[0]),
            "windows": len(group),
            "spearman": correlation,
            "mae": float(np.mean(np.abs(prediction - teacher))),
        }
        for name, rule in FROZEN_K.items():
            k = rule(len(group))
            teacher_top = set(_top_indices(teacher, window_ids, k).tolist())
            prediction_top = set(_top_indices(prediction, window_ids, k).tolist())
            row[f"recall_{name}"] = len(teacher_top.intersection(prediction_top)) / k
            row[f"ndcg_{name}"] = _ndcg(prediction, teacher, window_ids, k)
        rows.append(row)
    per_genome = pd.DataFrame(rows)
    summary = {
        "mean_within_genome_spearman": float(per_genome["spearman"].mean()),
        "median_within_genome_spearman": float(per_genome["spearman"].median()),
        "mean_mae_diagnostic": float(per_genome["mae"].mean()),
        "mean_recall_top_1_percent": float(per_genome["recall_top_1_percent"].mean()),
        "mean_ndcg_top_1_percent": float(per_genome["ndcg_top_1_percent"].mean()),
        "validation_genomes": int(len(per_genome)),
        "test_accessed": False,
    }
    return per_genome, summary

