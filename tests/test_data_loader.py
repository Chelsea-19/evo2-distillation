import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evo2_distill.data.dataset import GenomePairBatchSampler, TailAwareGenomePairBatchSampler, TokenWindowDataset
from evo2_distill.utils.io import sha256_file


def _small_cache(root: Path) -> None:
    shape = (8, 512)
    tokens = np.memmap(root / "window_tokens.uint8.mmap", mode="w+", dtype=np.uint8, shape=shape)
    for index in range(shape[0]):
        tokens[index] = index % 5
    tokens.flush(); del tokens
    frame = pd.DataFrame({
        "cache_index": range(8),
        "window_id": [f"w{i}" for i in range(8)],
        "assembly_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "cluster_id": ["x", "x", "y", "y", "z", "z", "q", "q"],
        "split": ["development"] * 4 + ["validation"] * 2 + ["test"] * 2,
        "absolute_residual": np.linspace(0, 1, 8),
    })
    frame.to_parquet(root / "token_cache_metadata.parquet", index=False)
    manifest = {"shape": list(shape), "dtype": "uint8", "cache_sha256": sha256_file(root / "window_tokens.uint8.mmap"), "metadata_sha256": sha256_file(root / "token_cache_metadata.parquet")}
    (root / "token_cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_development_and_validation_tokens(tmp_path: Path) -> None:
    _small_cache(tmp_path)
    development = TokenWindowDataset(tmp_path, "development")
    validation = TokenWindowDataset(tmp_path, "validation")
    assert len(development) == 4 and len(validation) == 2
    assert tuple(development[0]["tokens"].shape) == (512,)
    assert development[0]["tokens"].dtype == torch.uint8
    sampler = GenomePairBatchSampler(development.assembly_codes, batch_size=4, seed=11)
    batch = next(iter(sampler))
    assert development.assembly_codes[batch[0]] == development.assembly_codes[batch[1]]
    assert development.assembly_codes[batch[2]] == development.assembly_codes[batch[3]]


def test_training_dataset_omits_unused_metadata(tmp_path: Path) -> None:
    _small_cache(tmp_path)
    development = TokenWindowDataset(tmp_path, "development", include_metadata=False)
    assert set(development[0]) == {"tokens", "target"}


def test_tail_aware_sampler_keeps_pairs_within_genome(tmp_path: Path) -> None:
    _small_cache(tmp_path)
    development = TokenWindowDataset(tmp_path, "development", include_metadata=False)
    sampler = TailAwareGenomePairBatchSampler(
        development.assembly_codes,
        development.targets,
        batch_size=4,
        seed=11,
        tail_quantile=0.75,
        tail_pair_fraction=1.0,
    )
    batch = next(iter(sampler))
    for index in range(0, len(batch), 2):
        left, right = batch[index:index + 2]
        assert development.assembly_codes[left] == development.assembly_codes[right]
        assert development.targets[left] != development.targets[right]
