# Phase 4.5 execution controls

Phase 4.5 is not executed by package preparation. Notebook 02 exposes independent
booleans for data scaling, capacity scaling, tail-aware training, baseline
correction, final ensemble, and uncertainty analysis. Final ensemble and UQ are
off by default.

For Colab, `notebooks/00_run_all_phase4_5.ipynb` is the recommended single entry
point. It performs the environment/data/cache gates, runs or resumes candidates,
syncs checkpoints periodically, and writes a validation-only selection preview.
Its final-student and ensemble switches are off by default and require explicit
approval after the candidate table is reviewed.

The command-line interface is:

```bash
python scripts/run_phase4_5.py --config configs/phase4_5/data_scaling.yaml
```

Each run writes a unique directory containing the exact config, run manifest,
metrics, training log, checkpoints, and validation predictions. Run the GPU
benchmark first. Candidate selection must be documented on validation before
creating `final_student_v2.lock`; notebook 03 refuses to proceed without it.
The unified notebook records the selected run, metric, variant, Git commit and
final configuration hash in that lock. A selected tail-aware or baseline-
correction variant is preserved in final ensemble members through the explicit
`student_variant` field.

Notebook 04 performs validation-only aggregation, duplicate-free sensitivity,
lineage bootstrap, AURC bootstrap, uncertainty diagnostics, and a conformal
fallback only if the protocol decision permits it. It must never load TEST
predictions.
