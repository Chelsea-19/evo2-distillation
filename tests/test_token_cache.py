from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from evo2_distill.data.cache import verify_token_cache
from evo2_distill.utils.io import sha256_file


def test_cache_checksum_verification(tmp_path: Path) -> None:
    cache = np.memmap(tmp_path / "window_tokens.uint8.mmap", mode="w+", dtype=np.uint8, shape=(2, 512))
    cache[:] = 0; cache.flush(); del cache
    pd.DataFrame({"cache_index": [0, 1]}).to_parquet(tmp_path / "token_cache_metadata.parquet", index=False)
    manifest = {
        "shape": [2, 512], "dtype": "uint8",
        "cache_sha256": sha256_file(tmp_path / "window_tokens.uint8.mmap"),
        "metadata_sha256": sha256_file(tmp_path / "token_cache_metadata.parquet"),
    }
    (tmp_path / "token_cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_token_cache(tmp_path)["shape"] == [2, 512]
    with (tmp_path / "window_tokens.uint8.mmap").open("r+b") as handle:
        handle.write(b"X")
    with pytest.raises(ValueError):
        verify_token_cache(tmp_path)

