# Phase 2 single-model distillation and ablation report

## Verdict

**PHASE 2: PASS**

Created UTC: `2026-08-20T02:12:09.445495+00:00`  
Device: **CPU** (no CUDA runtime available)  
Development training pool: **52,992 windows / 207 genomes / 35 lineages**  
Validation: **353,840 windows / 45 genomes / 19 lineages**  
Locked TEST accessed: **NO**

## Architecture

`student_cnn_v1` is a newly implemented 512-bp DNA encoder, not the historical CDS StudentCNN. It uses an 8-dimensional nucleotide embedding, a kernel-7 stride-2 stem, six dilated residual blocks with dilation schedule `[1, 2, 4, 8, 16, 32]`, 16 channels, GroupNorm, GELU, dropout 0.10, and concatenated global mean/max pooling followed by a scalar MLP. The scalar model has **11,873 trainable parameters**, **13 convolutional layers**, and a calculated receptive field of **511 bp**.

The M5 comparator uses the identical encoder and a reconstruction decoder. It was trained only on masked development DNA; no PPL, residual or annotation entered its objective.

## Training design and limitation

Because the available PyTorch runtime is CPU-only on a 4-core laptop, Phase 2 used a frozen, genome-balanced architecture-screening pool of 256 windows per development genome. All 353,840 validation windows were evaluated. This limitation is explicit: the results justify controlled Phase 2 model selection under the local compute budget, not a claim that the CNN has exhausted all 1,640,110 development windows.

M1 and M2 share the same three-epoch Huber warm-up and differ only in the matched final branch. M1 ranking weights [2.0, 4.0] were compared on a deterministic 512-window-per-genome validation subset; **4.0** was selected. Since Huber weight is 1.0, ranking receives greater emphasis. M3 uses the same backbone and selected ranking weight but raw PPL targets. M4 is loaded unchanged from the frozen Phase 1 result.

## Validation results

| Model | Supervision | Mean within-genome Spearman | Top-1% Recall@K | Top-1% NDCG@K | Diagnostic MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 full distilled | residual Huber + within-genome ranking | 0.242513 | 0.012095 | 0.355867 | 0.213801 |
| M2 no ranking | residual Huber only | 0.222658 | 0.008448 | 0.352850 | 0.210186 |
| M3 raw teacher | raw-PPL Huber + ranking | 0.075116 | 0.009541 | 0.154949 | 0.383575 (PPL units) |
| M4 frozen cheap baseline | Phase 1 GC/k-mer HistGBR | 0.207867 | 0.057567 | 0.383897 | 0.216025 |
| M5 sequence-only | masked nucleotide reconstruction | -0.012169 | 0.034018 | 0.250353 | n/a |

All ranking metrics were calculated within each validation genome using the Phase 1 frozen top-1%, top-5% and fixed-100 definitions. Windows were not treated as independent inferential replicates; uncertainty summaries use a 2000-iteration bootstrap over the 19 validation lineages.

## Ablation decisions

1. **Teacher distillation versus cheap features:** M1 minus M4 mean Spearman = **+0.034647**, so M1 wins on the primary endpoint. It does **not** win uniformly: top-1% Recall is 0.012095 versus 0.057567 and top-1% NDCG is 0.355867 versus 0.383897. Teacher distillation therefore improves global within-genome ordering but not extreme-tail retrieval under this CPU-bounded run.
2. **Teacher supervision versus sequence-only learning:** M1 minus M5 mean Spearman = **+0.254683**, and M1 also improves top-1% NDCG (0.355867 versus 0.250353). M5 has higher top-1% Recall (0.034018 versus 0.012095), so the advantage is not uniform across retrieval metrics.
3. **Pairwise ranking loss:** M1 minus M2 = **+0.019856**. The selected ranking loss improves fidelity.
4. **Residual versus raw-PPL supervision:** M1 minus M3 = **+0.167398**. Residual supervision improves the proposal-aligned ranking.
5. **CNN justification:** **YES for the Phase 3 primary endpoint, with a top-K caveat**. The compact sequence model beats both cheap and sequence-only comparators on mean within-genome Spearman, while the cheap baseline remains stronger for the frozen extreme-tail Recall/NDCG metrics.

The exact paired genome/lineage deltas and confidence intervals are in `ablation_results.csv`; biological claims are outside this phase.

## Phase 3 freeze

The architecture is frozen as `student_cnn_v1`: 16 channels, dilation `[1,2,4,8,16,32]`, GroupNorm, dropout 0.10, mean+max pooling and the recorded scalar head. The Phase 3 primary objective is M1 with Huber weight 1.0 and ranking weight **4.0**. Checkpoint hashes, target standardisation, training budget, seeds and K definitions are sealed in `final_architecture.lock`.

## Boundary

Phase 2 stops here. No test performance was accessed. Phase 3 must use the frozen architecture/objective or create an explicitly versioned deviation without consulting TEST.
