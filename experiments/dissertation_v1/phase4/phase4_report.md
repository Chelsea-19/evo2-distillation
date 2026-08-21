# Phase 4 uncertainty-aware selective teacher escalation report

## Verdict

**PHASE 4: PASS**

Created UTC: `2026-08-20T07:43:52.141169+00:00`  
Validation windows: **353,840 / 45 genomes / 19 lineages**  
Locked TEST accessed: **NO**

## Frozen evaluation

The frozen Phase 3 ensemble mean is the student score and the frozen five-member sample variance is the primary uncertainty score. For each genome and budget, the highest-variance windows are referred to the teacher; their student score is replaced by the true absolute-residual teacher score. Random referral uses the exact same per-genome call counts. Each of 200 deterministic repetitions uses one nested random permutation per genome, so its risk-coverage path is coherent across budgets.

The routing grid is `[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`. No earlier grid existed. Primary risk is one minus within-genome Spearman, averaged equally across genomes. AURC integrates this risk over actual student coverage. Frozen Phase 1 K definitions are retained.

## Risk-coverage results

| Requested teacher calls | Actual teacher fraction | Student coverage | Uncertainty Spearman | Uncertainty risk | Random mean risk | Top-1% Recall | Top-1% NDCG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.0000 | 1.0000 | 0.261910 | 0.738090 | 0.738090 | 0.010876 | 0.346832 |
| 1% | 0.0101 | 0.9899 | 0.271540 | 0.728460 | 0.728100 | 0.014589 | 0.413360 |
| 5% | 0.0500 | 0.9500 | 0.312304 | 0.687696 | 0.688874 | 0.043899 | 0.560637 |
| 10% | 0.1000 | 0.9000 | 0.361992 | 0.638008 | 0.640940 | 0.085261 | 0.664717 |
| 20% | 0.2000 | 0.8000 | 0.455625 | 0.544375 | 0.549023 | 0.165135 | 0.768291 |
| 30% | 0.3001 | 0.6999 | 0.543757 | 0.456243 | 0.462218 | 0.249996 | 0.825193 |
| 50% | 0.5000 | 0.5000 | 0.701849 | 0.298151 | 0.304297 | 0.434498 | 0.891165 |
| 75% | 0.7500 | 0.2500 | 0.866255 | 0.133745 | 0.135941 | 0.682563 | 0.945006 |
| 100% | 1.0000 | 0.0000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |

Uncertainty-routing AURC is **0.32412211**; mean random AURC is **0.32767879** (95% repetition interval 0.32695319 to 0.32843466; Monte Carlo SE 0.00002769). Random minus uncertainty AURC is **+0.00355668**, so uncertainty routing **beats** random referral on the primary comparison.

## Uncertainty calibration

Mean within-genome Spearman between variance and absolute rank error is **-0.052783** (median -0.053452). The highest-uncertainty 10% has **0.532x** enrichment for the highest-error 10%, with mean high-error recall **0.053229**. Top-versus-bottom uncertainty-decile mean error ratio is **0.760**.

| Within-genome uncertainty decile | Windows | Mean absolute rank error across genomes | Median across genomes |
| ---: | ---: | ---: | ---: |
| 1 | 35,401 | 0.333571 | 0.331349 |
| 2 | 35,382 | 0.300701 | 0.300661 |
| 3 | 35,390 | 0.287006 | 0.287632 |
| 4 | 35,379 | 0.275939 | 0.275761 |
| 5 | 35,376 | 0.270356 | 0.269526 |
| 6 | 35,395 | 0.267104 | 0.266958 |
| 7 | 35,382 | 0.261972 | 0.262574 |
| 8 | 35,387 | 0.258208 | 0.259315 |
| 9 | 35,385 | 0.254853 | 0.253263 |
| 10 | 35,363 | 0.253390 | 0.253382 |

Variance is classified as a **failed** routing score under the prespecified joint rule: lower AURC than random, positive within-genome variance/error association, and high-error enrichment above one. A conformal fallback was not explored; this prevents opportunistic selection between uncertainty methods.

## Frozen routing rule

The prespecified fidelity target is mean within-genome Spearman >= 0.90. The frozen rule is: within each genome, sort windows by ensemble variance descending (window ID ascending for ties) and refer the top **100%** requested budget, corresponding to **100.0000%** actual validation calls. This achieves validation Spearman **1.000000** and removes **0.0000%** of teacher calls relative to full-teacher scoring. Routing status: `variance_not_useful_partial_routing_disabled`.

This rule, budgets, uncertainty definition, K values, score replacement and evaluation implementation are sealed in `final_routing_rule.lock`. Phase 5 was not started.
