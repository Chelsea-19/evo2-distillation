# Evo 2 dissertation — Colab Phase 4.5 package

This repository contains code, frozen small configurations, manifests, tests,
and Google Colab notebooks for validation-only Phase 4.5 development of
“Uncertainty-Aware Distillation of Evo 2 for Genomic Region Prioritisation in
*Acinetobacter baumannii*”. Large immutable data live in Google Drive.

## Safety boundary

- Train on `development` only.
- Select and evaluate candidates on `validation` only.
- The dissertation `test` partition is locked until a future, explicit Phase 5.
- Phase 4.5 commands reject `--split test`; no unlock is implemented here.
- This package does not run Evo 2 and does not modify archived source data.

## Quick start

1. Upload the prepared Drive payload to `MyDrive/evo2-distillation/`. For a
   reliable FASTA transfer, upload `frozen_fasta_v1.tar.gz` at that root; the
   unified notebook restores only missing/wrong-size FASTAs from it.
2. Open [`notebooks/00_run_all_phase4_5.ipynb`](https://colab.research.google.com/github/Chelsea-19/evo2-distillation/blob/main/notebooks/00_run_all_phase4_5.ipynb) in Colab Pro.
3. Enable a GPU runtime and run the notebook from the top.
4. Review its validation-only candidate table before opening the explicit final
   student/ensemble gate.

The numbered environment, data, runner, ensemble, and analysis notebooks remain
available as modular recovery/debugging entry points. The unified runner is the
recommended path and already contains the public repository URL.

See `docs/COLAB_SETUP.md` and `docs/TEST_SET_POLICY.md`.
