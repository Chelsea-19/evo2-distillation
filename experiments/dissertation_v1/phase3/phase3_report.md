# Phase 3 frozen five-member deep ensemble report

## Verdict

**PHASE 3: PASS**

Created UTC: `2026-08-20T05:44:27.426316+00:00`  
Five valid members: **5/5**  
Frozen seeds: **[11, 23, 37, 53, 71]**  
Development training pool: **52,992 windows / 207 genomes**  
Full validation: **353,840 windows / 45 genomes / 19 lineages**  
Locked TEST accessed: **NO**

## Frozen design

All members use the unchanged Phase 2 `student_cnn_v1` architecture (11,873 parameters), the same deterministic development windows, residual targets, target standardisation, optimizer, batch size, four-epoch schedule, Huber loss and final-epoch within-genome ranking weight 4.0. Only initialization, same-genome pair sampling and minibatch order vary with seed. Epoch checkpoints are atomic and resume-safe.

The protocol contains no ensemble-specific seeds, so the five seeds explicitly proposed in the Phase 3 task—11, 23, 37, 53 and 71—were frozen before training. No member was discarded because of its performance.

## Validation result

| Entity | Mean within-genome Spearman | Top-1% Recall | Top-1% NDCG |
| --- | ---: | ---: | ---: |
| Seed 11 | 0.153028 | 0.008163 | 0.284482 |
| Seed 23 | 0.220839 | 0.009783 | 0.301576 |
| Seed 37 | 0.249072 | 0.009278 | 0.334902 |
| Seed 53 | 0.253864 | 0.007245 | 0.338340 |
| Seed 71 | 0.226376 | 0.007871 | 0.348776 |
| Ensemble mean | **0.261910** | **0.010876** | **0.346832** |

Best single-member Spearman is **0.253864** (seed 53). The ensemble changes Spearman by **+0.008046** versus the best member and **+0.041275** versus the mean member. This observation did not alter architecture, seeds, membership or hyperparameters.

## Sanity checks and uncertainty

All five checkpoint hashes and prediction vectors differ. Pairwise global prediction correlations span 0.299983–0.857469; no numerical or zero-variance seed failure occurred. The primary uncertainty score is the sample variance across the five ranking scores. Its median is 0.0028340920, 95th percentile is 0.0051651196, and maximum is 0.0190152805. It is available for every validation window.

The local runtime is CPU-only, so peak GPU memory is not applicable. Phase 3 retains Phase 2's explicit limitation: this is a genome-balanced 52,992-window development training pool, not training on all 1,640,110 development windows.

## Freeze

The five seeds, member checkpoint hashes, ensemble mean rule, sample-variance uncertainty definition, Phase 2 architecture and frozen K definitions are sealed in `ensemble.lock`. No calibration was chosen from TEST, and no TEST outcome was accessed.
