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
        include_metadata: bool = True,
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
        self.include_metadata = include_metadata
        self.refresh_assembly_codes()

    def refresh_assembly_codes(self) -> None:
        self.assembly_codes, self.assembly_ids = pd.factorize(self.frame["assembly_id"], sort=True)
        self.cache_indices = self.frame["cache_index"].to_numpy(dtype=np.int64, copy=True)
        self.targets = self.frame[self.target_column].to_numpy(dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        cache_index = int(self.cache_indices[index])
        item: dict[str, object] = {
            # Keep the host-side representation compact and cast once on GPU.
            "tokens": torch.from_numpy(np.array(self.tokens[cache_index], dtype=np.uint8, copy=True)),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }
        if self.include_metadata:
            row = self.frame.iloc[index]
            item.update(
                {
                    "assembly_code": torch.tensor(int(self.assembly_codes[index]), dtype=torch.int64),
                    "window_id": str(row["window_id"]),
                    "assembly_id": str(row["assembly_id"]),
                    "cluster_id": str(row["cluster_id"]),
                    "cache_index": cache_index,
                }
            )
        return item


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


class TailAwareGenomePairBatchSampler(GenomePairBatchSampler):
    """Mix ordinary within-genome pairs with pre-specified high-vs-low pairs.

    The tail sets are computed from DEVELOPMENT targets only.  Validation labels
    are never used by this sampler.
    """

    def __init__(
        self,
        assembly_codes: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        seed: int,
        tail_quantile: float = 0.90,
        tail_pair_fraction: float = 0.50,
        drop_last: bool = True,
    ) -> None:
        super().__init__(assembly_codes, batch_size, seed, drop_last)
        if not 0.5 < tail_quantile < 1.0:
            raise ValueError("tail_quantile must be in (0.5, 1)")
        if not 0.0 <= tail_pair_fraction <= 1.0:
            raise ValueError("tail_pair_fraction must be in [0, 1]")
        self.tail_pair_fraction = float(tail_pair_fraction)
        target_values = np.asarray(targets, dtype=np.float64)
        self.tail_groups: list[tuple[np.ndarray, np.ndarray]] = []
        for group in self.groups:
            threshold = float(np.quantile(target_values[group], tail_quantile))
            high = group[target_values[group] >= threshold]
            low = group[target_values[group] < threshold]
            if len(high) and len(low):
                self.tail_groups.append((high, low))
        if not self.tail_groups:
            raise ValueError("No genome has both tail and non-tail development windows")

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(len(self)):
            batch: list[int] = []
            for _ in range(self.batch_size // 2):
                if rng.random() < self.tail_pair_fraction:
                    high, low = self.tail_groups[int(rng.integers(0, len(self.tail_groups)))]
                    pair = (int(rng.choice(high)), int(rng.choice(low)))
                    if rng.random() < 0.5:
                        pair = pair[::-1]
                else:
                    group = self.groups[int(rng.integers(0, len(self.groups)))]
                    left, right = rng.choice(group, size=2, replace=False)
                    pair = (int(left), int(right))
                batch.extend(pair)
            yield batch
