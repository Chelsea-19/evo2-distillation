# Reproducibility

Every run records its config, config SHA256, Git commit, dataset/split/target/cache
manifest hashes, seed, device, GPU/VRAM, PyTorch/CUDA, precision, batch size,
worker count, row counts, and `test_accessed=false`.

Scientific parameters are read from versioned YAML. Hardware adaptation is
limited to batch size, workers, prefetch, and AMP mode. The final ensemble uses
exactly seeds 11, 23, 37, 53, and 71 after `final_student_v2.lock` exists.

The frozen Phase 0–4 artifacts under `experiments/dissertation_v1/` are copied
unchanged and labelled historical read-only records. Large data/checkpoints are
identified by SHA256 in Drive manifests instead of being committed to GitHub.

