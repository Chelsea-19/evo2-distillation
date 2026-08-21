# Phase 0 leakage audit

Version: `dissertation_v1`  
Created UTC: `2026-08-20T01:21:00.303565+00:00`  
Split seed: `20260820`

## Scope

This audit was run only after creating the new dissertation-specific allocation. It did not use PPL, residuals, annotations, biological labels, historical model performance or residual extremes to design or revise the split. Duplicate results are reported as sensitivity-analysis information and did not trigger split tuning.

## A. Mash cluster leakage

- Assemblies assigned: **296/296**
- Mash clusters assigned: **73/73**
- Clusters occurring in more than one partition: **0**
- Result: **PASS**

Every assembly is an indivisible unit and every Mash cluster occurs in exactly one partition.

## B. Exact 512-bp duplicates across partitions

| Partition pair | Shared exact-sequence hash classes | Cross-partition window pairs |
| --- | ---: | ---: |
| development → validation | 4,659 | 10,371 |
| development → test | 5,157 | 10,651 |
| validation → test | 1,125 | 1,649 |

Distinct exact hash classes observed in two or more partitions: **10,153**.

## C. Reverse-complement-equivalent duplicates across partitions

| Partition pair | Shared strand-canonical hash classes | Cross-partition window pairs |
| --- | ---: | ---: |
| development → validation | 7,540 | 15,740 |
| development → test | 7,632 | 16,224 |
| validation → test | 1,907 | 2,693 |

Distinct strand-canonical hash classes observed in two or more partitions: **15,553**.

## Counting definition and interpretation

A shared hash class is one distinct sequence identity observed in both named partitions. Cross-partition window pairs are the sum of `n_left × n_right` over shared classes. Reverse-complement canonicalisation uses the lexicographically smaller of the forward sequence and its ACGTN reverse complement before SHA256 hashing.

These duplicates do not violate assembly- or Mash-cluster-level split integrity. They quantify representation similarity and must be reported in sensitivity analyses. Test membership was not changed after viewing these counts.
