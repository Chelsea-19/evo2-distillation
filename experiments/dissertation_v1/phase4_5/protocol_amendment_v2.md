# Phase 4.5 protocol amendment v2

Status: **SKELETON — NOT YET EXECUTED**  
Scope: **development training and validation-only selection**

Phase 4.5 is a validation-only development phase. The locked dissertation TEST
partition remains unopened, and no Phase 4.5 command may produce TEST
predictions, metrics, calibration, thresholds, or model-selection evidence.

The amendment is motivated by four limitations observed in immutable Phase 0–4
records:

- training-data underutilisation in the CPU-bounded Phase 2/3 screens;
- poor Top-K recovery despite modest Spearman fidelity;
- seed instability across the frozen Phase 3 ensemble;
- failed raw ensemble-variance error calibration in Phase 4.

Candidate work may examine, on development/validation only, data scaling,
capacity scaling, tail-aware objectives, and frozen cheap-baseline correction.
Exact candidate grids and the final student decision must be recorded before the
final ensemble is trained. Phase 0–4 files are historical, immutable records and
must not be edited or overwritten.

No Phase 5 execution is permitted automatically. This package intentionally
contains no TEST unlock mechanism.

