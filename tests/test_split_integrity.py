from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_split_counts_and_cluster_integrity() -> None:
    split = pd.read_csv(ROOT / "manifests" / "dissertation_split_v1.csv")
    counts = split.groupby("split")["assembly_id"].nunique().to_dict()
    assert counts == {"development": 207, "test": 44, "validation": 45}
    assert split["assembly_id"].nunique() == 296
    assert int((split.groupby("cluster_id")["split"].nunique() > 1).sum()) == 0

