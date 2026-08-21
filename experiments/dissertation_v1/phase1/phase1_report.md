# Phase 1 inexpensive-baseline report

## Verdict

**PHASE 1: PASS**

Created UTC: `2026-08-20T01:47:00.836581+00:00`  
Development windows used for fitting: **1,640,110**  
Validation windows evaluated: **353,840** across **45 genomes / 19 Mash lineages**  
Locked TEST accessed: **NO**

## Experimental question

This phase tests how much of the development-fitted Evo 2 absolute GC-LOWESS residual ordering can be recovered from cheap composition features alone. No DNA sequence encoder or biological annotation was used.

## Baselines and selection

| Baseline | Permitted features | Selected configuration | Validation mean within-genome Spearman |
| --- | --- | --- | ---: |
| A: GC-only | GC content | `A_linear` | 0.136527 |
| B: linear cheap composition | GC, k4 rarity, k6 rarity | `B_ridge_3` `{"alpha": 100.0}` | 0.127939 |
| C: boosted cheap composition | GC, length, k4 rarity, k6 rarity | `C_hist_gbr_4` `{"early_stopping": false, "l2_regularization": 1.0, "learning_rate": 0.1, "loss": "squared_error", "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 50}` | 0.207867 |

Window length is exactly 512 in every row. It was retained in Baseline C only because it belongs to the frozen inexpensive-feature schema; it has zero variance and therefore contributes no usable information.

The compact grids were evaluated only on validation, and selection used only mean within-genome Spearman. Test targets and outcomes were never materialized.

## Best inexpensive baseline

**baseline_c_hist_gbr** (`C_hist_gbr_4`)

- Mean within-genome Spearman: **0.207867**
- Median within-genome Spearman: **0.205822**
- Lineage-balanced mean Spearman: **0.206462**
- 95% lineage-bootstrap interval: **[0.201295, 0.211565]**
- Top-1% Recall@K: **0.057567**
- Top-1% NDCG@K: **0.383897**

### Frozen ranking endpoints

| K definition | Rule | Mean Recall@K | Mean NDCG@K |
| --- | --- | ---: | ---: |
| top_1_percent | `ceil(0.01 * n_windows_in_genome)` | 0.057567 | 0.383897 |
| top_5_percent | `ceil(0.05 * n_windows_in_genome)` | 0.164390 | 0.420993 |
| fixed_100 | `min(100, n_windows_in_genome)` | 0.069333 | 0.385532 |

All three K definitions are now frozen for subsequent phases. `top_1_percent` is the primary threshold-reporting endpoint; top 5% and fixed 100 are prespecified secondary sensitivities.

## Required interpretation

### 1. Recoverable cheap-feature ranking signal

The best inexpensive baseline recovers a mean within-genome Spearman of **0.207867** and top-1% Recall@K of **0.057567**. This is the empirical cheap-feature ceiling observed on validation under the compact search, not a biological interpretation.

### 2. Increment from k-mer rarity beyond GC

Adding k4/k6 rarity in the linear baseline changes mean genome Spearman by **-0.008588** relative to GC-only. The lineage-bootstrap interval for the paired difference is **[-0.013591, +0.000680]**. At top 1%, the paired changes are **+0.009264** for Recall and **+0.110589** for NDCG.

### 3. Best validation performance

The selected best baseline achieves mean/median Spearman **0.207867/0.205822**, top-1% Recall **0.057567**, and top-1% NDCG **0.383897**. Complete genome-level and lineage-aware results for every frozen K are in `baseline_metrics_validation.csv`.

### 4. Minimum useful-transfer threshold

For a distilled student to demonstrate useful teacher transfer on validation, it must exceed the best inexpensive baseline's **mean within-genome Spearman of 0.207867**. The prespecified stronger criterion is a positive paired lineage-bootstrap difference in Spearman while showing no material degradation relative to **Recall@top1% 0.057567** and **NDCG@top1% 0.383897**. These are validation thresholds only; no test claim is made.

## Statistical policy

Metrics were first calculated separately for each validation genome. Point estimates are equal-weight genome means/medians. Uncertainty is a 2000-iteration nonparametric bootstrap over the 19 complete validation Mash lineages using lineage-level means. Windows were not treated as independent inferential replicates.

## Frozen outputs

- `baseline_config.yaml`
- `baseline_metrics_validation.csv`
- `baseline_predictions_validation.parquet`
- `models/*.joblib` for the selected development-fitted estimator of each baseline
- `selected_baseline.lock`

## Next-phase boundary

Phase 1 stops here. The locked TEST partition remains unopened. Any student comparison must use the frozen K definitions, selected baseline and validation policy recorded here.
