# Phase 0 completion report

## Verdict

**PHASE 0: PASS**

Version: `dissertation_v1`  
Created UTC: `2026-08-20T01:21:00.303565+00:00`  
Allocation seed: `20260820`  
Runtime: `134.1 seconds`

## Frozen split

| Partition | Assemblies | Mash clusters | 512-bp windows |
| --- | ---: | ---: | ---: |
| Development | 207 | 35 | 1,640,110 |
| Validation | 45 | 19 | 353,840 |
| Test | 44 | 19 | 353,552 |
| **Total** | **296** | **73** | **2,347,502** |

The test partition is sealed by `splits/TEST_SET_LOCKED.txt`; its split-manifest SHA256 is `5658f1d4c74f44d700791a800601f6270b847f3de7a9970f4c64db856f73b2d4`. No model result was produced or inspected.

## Validation checklist

- [x] 296 of 296 assemblies assigned
- [x] every mash cluster in one partition
- [x] 2347502 of 2347502 windows assigned
- [x] 2347502 of 2347502 targets generated
- [x] test lowess uses development curve only
- [x] no test derived preprocessing fitted
- [x] all hashes and manifests exist
- [x] no model trained
- [x] no test model evaluation
- [x] test lock hash matches manifest
- [x] target parquet row count verified
- [x] canonical parquet row count verified

## Frozen targets and protocol

- Canonical row index/features: `data/window512_canonical_v1.parquet` (2,347,502 rows; no DNA sequence column).
- Development-only LOWESS curve: `targets/dev_lowess_v1.csv` with span 0.3.
- Target table: `targets/window512_targets_v1.parquet` (2,347,502 rows).
- Ambiguous bases: all 9 windows retained and explicitly flagged.
- Primary target: absolute development-fitted GC-LOWESS residual ranking within genome.
- Primary fidelity endpoint: mean within-genome Spearman correlation.
- Secondary endpoints: Recall@K and NDCG@K.
- Statistical unit: strain/assembly and/or complete Mash lineage, never individual windows.
- Annotations: retrospective only and forbidden as predictors.
- Historical test: forbidden as dissertation final test.

## Leakage result

- Mash cluster crossings: **0**.
- Exact sequence hash classes shared across partitions: **10,153**.
- Reverse-complement-equivalent hash classes shared across partitions: **15,553**.
- These counts were not used to redesign the split.

## Next permitted phase

**Phase 1 baseline.** Test outcomes remain unopened until Phase 5.
