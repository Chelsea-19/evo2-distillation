# TEST-set policy

The dissertation TEST partition contains 44 assemblies and 353,552 frozen
windows. It remains locked until a future explicit Phase 5.

Phase 4.5 may use development rows for fitting and validation rows for selection
and analysis. It must not load TEST targets into evaluation, report TEST metrics,
use TEST for early stopping or thresholds, or use TEST for uncertainty/conformal
calibration. A frozen split manifest may be read only to verify integrity.

Sequence tokens for all partitions may coexist in the immutable cache, but the
dataset API rejects `test`. `scripts/run_phase4_5.py --split test` always exits.
This repository implements no unlock path.

