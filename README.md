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

1. Upload the prepared Drive payload to `MyDrive/evo2_dissertation/`.
2. Push this directory to a GitHub repository.
3. Open `notebooks/00_colab_environment_check.ipynb` in Colab Pro.
4. Continue with notebooks 01 and 02 only after their verification gates pass.
5. Run notebook 03 only after `final_student_v2.lock` has been created by an
   explicit validation-only candidate-selection decision.

See `docs/COLAB_SETUP.md` and `docs/TEST_SET_POLICY.md`.

