# Results index

One row per experiment. The point of this table is to stop scores from different
protocols being compared as if they were the same measurement. **Read the
grouping policy, task, OOV handling and training budget before comparing any two
numbers.**

Interpretation status values:

- **supported** — replicated across the tested partitions and conventions.
- **provisional** — point estimate is informative but the calibrated inference
  that would justify the claim is not in place yet.
- **preliminary** — known evaluation problems; do not cite.
- **negative** — the tested model did not beat its baseline.
- **exploratory** — descriptive only, no claim.

Pre-correction copies of every result below are preserved under
`outputs/archive_pre_correction_4f81402/`; the paths in the table are the
**corrected** locations.

## Core evaluation experiments

| # | Experiment | Dataset / subset | Grouping policy | Task | Primary metric | OOV handling | Training budget | Tuning procedure | Result location | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Group audit | 4,624 known-direction gap-split spans | artifact + inscription + exact-sequence connected components | Component-size audit | Largest component token share | n/a | n/a | none | `outputs/group_audit/` | supported |
| 2 | Bigram–trigram inference | 4,624 spans, 16,469 tokens | artifact-grouped (duplicates merged) | Held-out token log loss, bigram vs trigram | Macro group-mean paired diff (bits/token) | OOV handled by MKN; no zero-probability tokens | Full outer folds | none (fixed MKN); 5 predefined fold seeds as sensitivity | `outputs/ngram_inference/` | **provisional** |
| 3 | Restoration robustness | Same spans, 4 protocols (readable / complete / unique / complete-unique) + leave-one-site-out | artifact-grouped (duplicates merged) | Masked single-sign restoration top-1 | Context top-1 vs frequency and position | **OOV targets count as failures** | Training partition only | none | `outputs/robustness/` | **supported** |
| 4 | Sign embeddings | Same spans | artifact-grouped (duplicates merged) | Masked restoration top-1 vs bigram | Top-1 / MRR | OOV failures reported | Nested inner selection | Nested grouped inner fit/validation | `outputs/embeddings/` | **negative** |
| 5 | Masked-sign transformer | Same spans | artifact-grouped (duplicates merged) | Masked restoration top-1 vs bigram | Top-1 | `<UNK>` context tokens; OOV targets are failures; `<PAD>/<MASK>/<UNK>` never candidates | Nested inner selection, refit on outer train | Nested grouped inner selection of config **and** stopping epoch | `outputs/transformer/` | **negative** |
| 6 | Preprocessing sensitivity (2×2×2) | Same spans | artifact-grouped | Masked restoration | Context − frequency and context − position (pp) | OOV failures | Per-cell training partition | none | `outputs/sensitivity/` | **supported** |
| 7 | Motif / stratified evaluation | Metadata sidecar subset | artifact-grouped | Masked restoration + direction cross-check | Top-1 by motif | OOV failures | Per-stratum training | none | `outputs/stratified/` | **exploratory** |
| 8 | Direction diagnostics | Same spans, orientation variants | **union of duplicate relations across orientation variants**, shared group keys | Cross-direction transfer + three boundary models + position classes | Top-1 by orientation and position class | **OOV-separated (overall vs shared-vocabulary targets)** | Controlled training sizes | none | `outputs/direction_diagnostics/` | **exploratory** |
| 9 | Same-family transcription sensitivity | 1,841 matched inscriptions (369 held out/seed) | **two protocols: artifact-only, and artifact + union of duplicates from either transcription** | Masked restoration under two ICIT transcriptions | Context top-1 | OOV ~2.8%, reported per subgroup | 80/20 split on the shared pair table, 10 seeds | none | `outputs/transcription_sensitivity/` | **exploratory** |
| 10 | Compact structural models | 4,624 spans | artifact-grouped, **full records throughout** | Held-out bits/token vs bigram | Bits/token, perplexity | **training-only vocab; `<UNK>` always present; per-token scoring in every model**; OOV reported per model | **matched cap and full outer train, both reported** | HMM state count + init and position alpha on inner **grouped** splits | `outputs/compact_models/` | **negative** |
| 11 | Synthetic calibration | Synthetic corpora sampled from the empirical profile | artifact-grouped (each synthetic inscription is its own component) | Three separate decision rules (difference / improvement / structural) | Rejection rates with Wilson intervals | n/a | Full synthetic corpora | Threshold from a calibration null set; FP rate from a **disjoint** evaluation set | `outputs/power_analysis/` | **run (6 reps/cell, 1× only)** |
| 12 | Cross-corpus transcription audit | Primary vs external ICIT | n/a | Relation classification per CISI | Relation counts | n/a | n/a | none | `outputs/audit/` | **exploratory** |

## Headline numbers and what they mean

| Quantity | Value | Protocol | Comparable to |
| --- | --- | --- | --- |
| Frequency restoration top-1 | 10.82% ± 1.09 | Robustness, gap-split, artifact-grouped | rows 3–5 |
| Position restoration top-1 | 12.84% ± 1.32 | Robustness, gap-split, artifact-grouped | rows 3–5 |
| Bigram (context) restoration top-1 | 29.21% ± 1.87 | Robustness, gap-split, artifact-grouped | rows 3–5 |
| Transformer restoration top-1 (nested) | 23.65% ± 2.87 | Transformer, fully nested | bigram 29.21% only |
| Skip-gram restoration top-1 | ~9.05% | Embeddings, nested | bigram 29.21% only |
| PPMI/SVD restoration top-1 | ~8.30% | Embeddings, nested | bigram 29.21% only |
| Macro group trigram advantage | +0.02702 bits/token, 95% CI [+0.01048, +0.04395] | n-gram inference, primary fold seed 0 | **not** the restoration accuracies |
| Token-weighted trigram advantage | +0.03623 bits/token, 95% CI [−0.01173, +0.09211] — **includes zero** | n-gram inference, primary fold seed 0 | **not** the macro estimand |
| Macro effect across 5 fold seeds | mean +0.02867, range [+0.02592, +0.03147] | n-gram inference sensitivity | each seed separately, never pooled |
| Largest component share | 22.1% of scored tokens (INDUS-0038, 3,645 tokens, 1,575 spans) | Group audit / n-gram inference | — |
| Largest component own effect | **−0.09386** bits/token | n-gram inference | — |
| Macro effect excluding largest component | +0.02707 (vs +0.02702 including) | Labelled sensitivity analysis | not the primary result |
| Token-weighted excluding largest component | +0.07321 (vs +0.03623 including) | Labelled sensitivity analysis | not the primary result |
| Transcription-subset context top-1 (strict union grouping) | 30.16% (primary) / 30.18% (external) | Transcription sensitivity, 1,841 matched inscriptions | **NOT** comparable to 29.21% |
| Transcription-subset context top-1 (artifact-only grouping) | 39.59% (primary) / 38.34% (external) | Same data, looser grouping | **leaks 56–74 sequences per seed** — do not quote |
| Compact models vs bigram (matched budget) | bigram 7.06 vs 8.16–8.90 bits/token | Compact runner, capped subset | **negative result** |
| Compact models vs bigram (full budget) | bigram 5.97 vs 7.29–8.09 bits/token | Compact runner, full outer train | **negative result** |
| Structural test of the trigram gain | observed +0.0241; 0/40 null draws reached it; **p = 0.0244** | Calibration, fitted first-order null | conditional on the fitted null |
| Improvement-rule false-positive rate | **0.000** across 5 valid null cells (0/30 replicates) | Calibration, 1× size | the rule's own null |
| Structural-rule false-positive rate | 0.10 (2/20) on an independent evaluation half | Calibration | imprecise at n = 20 |

### Why the transcription number changed from ~40% to ~30%

The earlier ~39.6% came from a grouping policy that permits exact-sequence duplicates to
cross the shared fold boundary: **56–74 duplicate sequences per seed** appear in both
training and test. Under `artifact_plus_union_duplicates`, which groups the union of
duplicate relations found in **either** transcription, that crossing is **0** and the
figure falls to 30.16%/30.18%. Both transcriptions agree to within 0.02 pp, and context
still beats frequency (+19.6 pp) and position (+15.4 pp).

### Why ~30% ≠ 29.2%

Even the corrected transcription figure is a **different measurement**, not a better model:

- different subset (1,841 matched inscriptions, 369 held out per seed, after excluding
  534 ambiguous-primary and 1,668 unmatched-primary CISI);
- different grouping policy and a different split fraction (80/20, ten seeds) from the
  headline grouped evaluation;
- its exclusions select for records that have a clean one-to-one external match, which is
  not a random subset of the corpus.

The two numbers must not be quoted side by side as a comparison.

## OOV conventions across experiments

| Experiment | Convention |
| --- | --- |
| Robustness, embeddings, transformer, transcription | Unseen target signs are **failures**; `<UNK>` is never a correct restoration |
| N-gram inference | MKN smoothing; a zero-probability token raises rather than silently scoring |
| Compact models | Training-only vocabulary, `<UNK>` always present with a pseudo-count, likelihood evaluated over mapped `<UNK>` outcomes in **every** model; one unseen sign never discards a sequence's known-token contributions. Restoration is scored separately, so an unseen original sign can never be credited |
| Calibration | Not applicable (synthetic corpora are generated from the fitted profile) |

## Run scale actually achieved

| Experiment | Brief asks for | Actually run | Consequence |
| --- | --- | --- | --- |
| Calibration grid | ≥100 replicates × 0.5×/1×/2× × multiple strengths | **112 replicates (6/cell) at 1× only** | structural test rests on 20+20 null draws; enough for the qualitative reading, not for a precise λ ≈ 0.43 estimate |
| N-gram inference | several predefined fold seeds | **5 fold seeds, full corpus** | meets the requirement |
| Transcription sensitivity | several seeds, both protocols | **10 seeds, both protocols** | meets the requirement |
| Direction diagnostics | several seeds | **10 seeds, three boundary models** | meets the requirement |
| Compact models | matched and full budgets | **5 repeats × 5 folds × both budgets** | meets the requirement |

## Provenance

Every JSON summary records: corpus SHA-256, relevant source-file SHA-256, Python
and dependency versions, random seeds, fold identities or fold seeds, the grouping
policy, training sizes, and run status. The pre-correction baseline is fixed in
`docs/BASELINE.md` and hashed in
`outputs/archive_pre_correction_4f81402/BASELINE_MANIFEST.json`.
