# Phase 3 ensemble diagnostics

## Member validity

- Five valid members: **5/5**
- Seeds: `[11, 23, 37, 53, 71]`
- Unique final checkpoint hashes: **YES**
- Byte-identical prediction vectors: **NO**
- Pathological numerical/training failures: **0**
- GPU memory: **not available (CPU runtime)**

No seed was discarded or selected based on its validation performance.

## Final validation metrics

| Seed | Checkpoint SHA-256 prefix | Mean within-genome Spearman | Top-1% Recall | Top-1% NDCG | Training runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 11 | 7323980992f3... | 0.153028 | 0.008163 | 0.284482 | 945.5 |
| 23 | 28e2c66012af... | 0.220839 | 0.009783 | 0.301576 | 957.1 |
| 37 | 6e84577aff38... | 0.249072 | 0.009278 | 0.334902 | 803.8 |
| 53 | d6a772b890e0... | 0.253864 | 0.007245 | 0.338340 | 782.4 |
| 71 | 12c043a9d648... | 0.226376 | 0.007871 | 0.348776 | 827.5 |
| Ensemble | mean of all five | 0.261910 | 0.010876 | 0.346832 | n/a |

Best single member is seed **53** at **0.253864**. Ensemble minus best single = **+0.008046**; ensemble minus mean single = **+0.041275**. The ensemble is retained exactly as prespecified regardless of whether this delta is positive.

## Pairwise prediction correlations

| Seed A | Seed B | Global Pearson | Mean within-genome Spearman |
| ---: | ---: | ---: | ---: |
| 11 | 23 | 0.505749 | 0.531658 |
| 11 | 37 | 0.469771 | 0.529557 |
| 11 | 53 | 0.491367 | 0.559207 |
| 11 | 71 | 0.299983 | 0.466393 |
| 23 | 37 | 0.742632 | 0.796027 |
| 23 | 53 | 0.857469 | 0.859770 |
| 23 | 71 | 0.664540 | 0.745831 |
| 37 | 53 | 0.822658 | 0.872461 |
| 37 | 71 | 0.840674 | 0.879339 |
| 53 | 71 | 0.729716 | 0.785925 |

Global Pearson range: **0.299983–0.857469**. Mean within-genome member-pair Spearman range: **0.466393–0.879339**. Members are correlated but not identical.

## Prediction variance

Primary uncertainty is the sample variance across the five scalar ranking predictions. Distribution across 353,840 validation windows:

| Statistic | Variance |
| --- | ---: |
| Minimum | 0.0000061334 |
| 25th percentile | 0.0018434620 |
| Median | 0.0028340920 |
| 75th percentile | 0.0038117842 |
| 90th percentile | 0.0046648912 |
| 95th percentile | 0.0051651196 |
| 99th percentile | 0.0061689499 |
| Maximum | 0.0190152805 |
| Mean | 0.0028639070 |

All variance values are finite and non-negative. No test-dependent calibration was performed.
