from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from evo2_distill.data.safety import require_phase4_5_split


class TokenWindowDataset(Dataset):
    """Memory-mapped 512-bp tokens with development/validation gating."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        target_column: str = "absolute_residual",
        additional_columns: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = require_phase4_5_split(split)
        manifest = json.loads((self.cache_dir / "token_cache_manifest.json").read_text(encoding="utf-8"))
        self.shape = tuple(int(v) for v in manifest["shape"])
        self.tokens = np.memmap(
            self.cache_dir / "window_tokens.uint8.mmap",
            mode="r",
            dtype=np.uint8,
            shape=self.shape,
        )
        columns = ["cache_index", "window_id", "assembly_id", "cluster_id", "split", target_column]
        columns.extend(additional_columns or [])
        columns = list(dict.fromkeys(columns))
        frame = pd.read_parquet(self.cache_dir / "token_cache_metadata.parquet", columns=columns)
        self.frame = frame.loc[frame["split"].eq(self.split)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows for permitted split {self.split}")
        self.target_column = target_column
        self.assembly_codes, self.assembly_ids = pd.factorize(self.frame["assembly_id"], sort=True)

    def refresh_assembly_codes(self) -> None:
        self.assembly_codes, self.assembly_ids = pd.factorize(self.frame["assembly_id"], sort=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        cache_index = int(row["cache_index"])
        return {
            "tokens": torch.from_numpy(np.array(self.tokens[cache_index], dtype=np.int64, copy=True)),
            "target": torch.tensor(float(row[self.target_column]), dtype=torch.float32),
            "assembly_code": torch.tensor(int(self.assembly_codes[index]), dtype=torch.int64),
            "window_id": str(row["window_id"]),
            "assembly_id": str(row["assembly_id"]),
            "cluster_id": str(row["cluster_id"]),
            "cache_index": cache_index,
        }


class GenomePairBatchSampler(Sampler[list[int]]):
    """Yield adjacent pairs from the same genome for pairwise ranking loss."""

    def __init__(self, assembly_codes: np.ndarray, batch_size: int, seed: int, drop_last: bool = True) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Pairwise batches require an even batch_size >= 2")
        self.batch_size = batch_size
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self.groups = [np.flatnonzero(assembly_codes == code) for code in np.unique(assembly_codes)]
        if any(len(group) < 2 for group in self.groups):
            raise ValueError("Every genome must contain at least two windows")
        self.total = len(assembly_codes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.total // self.batch_size if self.drop_last else math.ceil(self.total / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(len(self)):
            batch: list[int] = []
            chosen_groups = rng.integers(0, len(self.groups), size=self.batch_size // 2)
            for group_index in chosen_groups:
                left, right = rng.choice(self.groups[int(group_index)], size=2, replace=False)
                batch.extend((int(left), int(right)))
            yield batch
