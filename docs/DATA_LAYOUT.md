# Data layout

GitHub contains code, configs, small manifests, frozen reports, tests, and
notebooks. Google Drive contains the 296 FASTAs, frozen target and canonical
feature Parquets, split/runtime manifests, frozen baseline artifact, selected
legacy checkpoints, token-cache destination, and persistent run outputs.
The optional `frozen_fasta_v1.tar.gz` transfer archive at the Drive root is a
reliable replacement for browser folder uploads; it contains the same 296
manifest-pinned FASTAs and is used only to repair missing/wrong-size files.

Colab stages required files from Drive into `/content/evo2-distillation-data/`.
The complete cache consists of:

```text
caches/tokens/window_tokens.uint8.mmap
caches/tokens/token_cache_metadata.parquet
caches/tokens/token_cache_manifest.json
```

Rows are keyed to the exact row order of `window512_canonical_v1.parquet`.
Coordinates remain 1-based inclusive and every interval is exactly 512 bp.
Encoding is A=0, C=1, G=2, T=3, N/other=4 using `uint8`.

The existing Phase 2 workstation cache is not transferred because it contains
only the CPU-bounded development subset plus validation, not all 2,347,502
frozen windows.

