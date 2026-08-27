# Colab Pro setup

## Required persistent layout

Upload the prepared Drive payload so that the following exists:

```text
/content/drive/MyDrive/evo2-distillation/
  frozen_fasta_v1.tar.gz
  frozen_data/
  caches/tokens/
  models/
  checkpoints/phase4_5/
  predictions/phase4_5/
  results/phase4_5/
  logs/phase4_5/
```

The GitHub repository is cloned from
`https://github.com/Chelsea-19/evo2-distillation.git`; no credentials are stored
in the notebook.

## Runtime sequence

1. In Colab choose **Runtime → Change runtime type → GPU**.
2. Open and run `notebooks/00_run_all_phase4_5.ipynb`.
3. Mount Drive at `/content/drive`.
4. Clone the repository to `/content/evo2-distillation`.
5. Install `requirements/requirements-colab.txt` and the repository editable.
6. If any FASTA is missing or has the wrong size, verify the transfer archive
   SHA256 and restore only the affected files. Then run the complete manifest
   and FASTA hash verification.
7. Stage data once to `/content/evo2-distillation-data`; training must not use
   mounted Drive as its high-frequency filesystem.
8. Load or build the complete token cache and verify its hashes.
9. Run the non-scientific GPU benchmark and copy its JSON to Drive.
10. Run selected Phase 4.5 configurations. The unified notebook syncs run
    directories periodically and after each completed or failed command.
11. Review validation metrics before explicitly approving the final student.
12. Train the five-member ensemble only after the validation-selection lock is
    written. Phase 5 is never started automatically.

## Checkpoint and resume

Each epoch checkpoint stores model, optimizer, scheduler, epoch, global step,
seed, config hash, best validation metric, and random states. Resume with:

```bash
python scripts/resume_experiment.py \
  --config configs/phase4_5/data_scaling.yaml \
  --checkpoint /content/evo2-distillation-runs/<run_id>/checkpoint/epoch_008.pt
```

After disconnect, restage data/cache, copy the desired checkpoint from Drive to
`/content`, and resume with the identical config. A config-hash mismatch must be
treated as a different run.

## Common recovery

- CUDA unavailable: enable a GPU runtime and restart notebook 00.
- CUDA OOM: rerun the benchmark and reduce only batch size/prefetch/workers or
  precision; do not change scientific hyperparameters.
- Missing/wrong-size FASTAs: upload `frozen_fasta_v1.tar.gz` to the Drive root
  and rerun notebook 00. The archive and every restored file are SHA256 checked;
  already-correct FASTAs are not rewritten.
- Other corrupt/incomplete Drive file: remove only the `.partial` copy, then
  restage; final files are created by atomic rename.
- Invalid cache: rebuild from the frozen canonical table plus the 296 verified
  FASTAs; do not use the historical partial CPU cache.
- Colab disk pressure: retain the token cache and Parquets; omit staged FASTA
  after a verified cache exists.

TEST remains locked throughout Phase 4.5.
