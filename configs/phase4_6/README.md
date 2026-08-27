# Phase 4.6 validation-only rank/tail amendment

This compact follow-up is pre-specified before running its candidates. It does
not modify the locked Phase 0–4 artifacts and cannot access TEST.

Both candidates use development-only supervision and the same 512-bp token
cache. They add same-genome tail-aware pair sampling, target-gap-weighted
pairwise ranking, a shared auxiliary top-decile head, global development-fitted
tail weighting, and a fixed 0.10 within-genome percentile-rank fusion with the
frozen Phase 1 HistGBR baseline. The candidates differ only in compact capacity
and pre-declared loss emphasis.

Selection uses validation mean within-genome Spearman, followed by Recall@1%,
NDCG@1%, and run ID. The existing Phase 4.5 capacity result (Spearman 0.504923)
and Phase 1 Recall@1% (0.057567) are displayed as frozen references. No result
is promised to exceed either reference, and unsuccessful runs remain reported.
