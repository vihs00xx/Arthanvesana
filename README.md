# Arthānveṣaṇa

**AI-assisted structural analysis of the Sindhu-Sarasvatī script using statistical baselines, sign embeddings, and self-supervised learning.**


Inspired by the Sanskrit idea of searching and inquiry, **Arthānveṣaṇa** ("search for meaning") reflects this project's purpose: to investigate ancient signs through evidence, uncover structural patterns, and generate testable hypotheses.

## Status

Corpus ingestion and validation, statistical baselines with shuffled controls, grouped evaluation, PPMI/SVD and skip-gram sign embeddings, clustering with permutation-null checks, PCA/UMAP visualization, and a masked-sign transformer are implemented. A metadata sidecar, motif-stratified evaluation, and a cross-corpus transcription audit are also included.

**Claims status.** The primary supported finding is that immediate context beats frequency and position. The higher-order (trigram) claim is **provisional** because the current calibration is pilot-scale. Transformer and embedding results are a **negative** result under fully nested evaluation, and compact models are a **negative** result under both matched and full training budgets. See `docs/METHODS.md` and the consolidated `docs/PROJECT_REPORT.md`.

## Source corpus

- Upstream: [ShaktiOSindia/indus-sign-regimes-deposit](https://github.com/ShaktiOSindia/indus-sign-regimes-deposit)
- File: `sanitized_corpus.json` (5,704 catalogued inscriptions; ~18k sign tokens; 713 distinct signs; `000` marks an illegible sign, not a sign)
- Upstream commit: `e48b3ec1e90f368079f5126790613cded6f6f56c` (2026-08-12)
- SHA-256: `345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

Run scripts from the repository root, e.g. `python scripts/build_corpus.py`.

## Testing

Install development tools with `python -m pip install -r requirements-dev.txt` (or `python -m pip install -e ".[dev]"`).

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

Type checking currently covers five statistical core modules under `strict` (`src/arthanvesana/stats/entropy.py`, `src/arthanvesana/replicate/llr.py`, `src/arthanvesana/replicate/entropy2.py`, `src/arthanvesana/replicate/metrics.py`, `src/arthanvesana/replicate/segment.py`). The newer `simulate` and `compact` modules import numpy and are only partially annotated; they are not yet under `strict`, and numpy's bundled stubs require Python ≥3.12 to parse while the checked configuration targets 3.11. Full annotation of those modules is outstanding work. Run `python scripts/run_stats.py`, `python scripts/run_replication.py`, and `python scripts/run_upgrade.py` to regenerate local reports. Defaults use known-direction, gap-split spans and artifact/duplicate-grouped holdouts; these results are not directly comparable to the earlier record-level splits.

## Restoration robustness

Run `python scripts/run_robustness.py --repeats 10` to compare the context model with training-only frequency and observed-position baselines. The same artifact/duplicate-grouped partitions are evaluated as readable spans, complete inscriptions, unique sequences, and complete unique sequences. Each model predicts the same masked positions; unseen target signs count as failures.

The runner also holds out each site with at least 100 eligible inscriptions, removing linked artifacts and duplicate groups from training. Local `outputs/robustness/` files contain a text report and JSON with all split identities, paired differences, overlap checks, source/data hashes, and dependency versions. Reported standard deviations describe split variability, not confidence intervals over independent experiments. Use `--help` for input, output, seed, split-fraction, and site-threshold options.

## Sign embeddings

Run `python scripts/run_embeddings.py --repeats 10` to train PPMI/SVD and skip-gram sign vectors and compare embedding-based restoration against the bigram on identical grouped splits and masked positions. The runner also clusters both vector sets (k-means sweep plus hierarchical), tests cluster alignment with positional roles against a permutation null, and writes PCA/UMAP figures colored by role. Local `outputs/embeddings/` files contain a text report and JSON with the config sweeps, per-run metrics, paired differences, stability checks, cluster assignments, and full-data neighborhoods. Reported standard deviations describe split variability, not confidence intervals.

## Masked-sign transformer

Run `python scripts/run_transformer.py --repeats 10` to train a tiny transformer encoder (1–2 layers, dim 32–64) with a mask-one-sign objective and compare it against the bigram on identical grouped splits and masked positions. Training uses early stopping on a grouped validation split; parameter counts and epochs are reported alongside accuracy. Requires torch (CPU wheel, see `requirements.txt`). Local `outputs/transformer/` files contain a text report and JSON with the config selection, per-run metrics, paired differences, and training diagnostics. Reported standard deviations describe split variability, not confidence intervals.

**Current fully nested result: the transformer loses to the bigram.** Restoration top-1 is 23.65% ± 2.87 for the transformer versus 29.21% ± 1.87 for the bigram, losing on 10 of 10 outer splits (transformer minus bigram = −5.56 pp). Earlier non-nested runs reported 19.4%; that figure is superseded by the nested result. No outer-test record enters configuration selection or early stopping.

## Data enrichment and audit

Run `python scripts/build_metadata.py` to join external motif and direction fields (HuggingFace `joyboseroy/indus_decipher`, gitignored under `data/external/`) into `data/processed/inscription_metadata.csv`, keyed by CISI number. Run `python scripts/run_stratified.py` for motif-stratified restoration and a direction cross-check, and `python scripts/run_audit.py` for the cross-corpus transcription audit (same-family agreement plus an independent length-level check against the mayig CISI transcription). Local `outputs/stratified/` and `outputs/audit/` hold the reports. See `data/PROVENANCE.md` for sources and limitations.

## Group audit, sequence-order sensitivity, and grouped n-gram inference

Run `python scripts/run_group_audit.py` to audit the connected artifact/inscription/duplicate components used by every grouped split. It reports component-size percentiles, the largest components and which identity relation created them, and warns when any component exceeds one fold's target size. Exact-sequence connectivity merges many records into large components, which is why grouped test sets vary in size; grouping is deliberately not weakened to balance folds.

Run `python scripts/run_sensitivity.py --repeats 10` for the 2×2×2 preprocessing matrix covering **sequence-order processing** (`reading_order_normalized` vs `physical_as_stored`), **direction inclusion**, and **completeness filtering**. This is not a comparison of independent transcription traditions; both sequence-order levels order the *same* transcription.

Run `python scripts/run_ngram_inference.py` for grouped cross-fitted bigram-vs-trigram log-loss inference: connected components are indivisible groups assigned to five folds largest-component-first with seeded tie-breaking, every record gets exactly one out-of-fold prediction, and token differences are aggregated within each group before inference. The paired difference is `d = log2 P_trigram - log2 P_bigram` bits/token.

Two estimands are reported and they answer different questions: the **macro (unweighted) group mean** ("does the trigram help a typical connected group?", primary) and the **token-weighted mean** ("does it help a typical token?", secondary). They are never averaged together. The complete cross-fitting procedure is repeated under five predefined fold seeds; each seed is an independent run and repeated predictions are **never** pooled. The largest connected component is reported on its own, in the full estimate, and in an explicitly labelled sensitivity analysis excluding it — the full-data estimate stays primary. Per-token scores with inscription and group identifiers are saved for audit, and per-token frequency bands use each fold's training data only.

The group-level sign-flip p-value is retained as a **descriptive** statistic and is **not cited as evidence**: its null assumes group differences are symmetric about zero, which a first-order generator does not guarantee. Whether any rejection rule is calibrated is a separate whole-pipeline simulation question.

**Corrected results** (4,624 spans, 16,469 tokens, 2,262 components, zero leakage):

| Quantity | Value |
| --- | --- |
| Macro group effect (primary, fold seed 0) | **+0.02702** bits/token, 95% CI [+0.01048, +0.04395] |
| Across 5 fold seeds | mean +0.02867, range [+0.02592, +0.03147] |
| Token-weighted (secondary) | +0.03623 bits/token, 95% CI [−0.01173, +0.09211] — **includes zero** |
| Largest component (INDUS-0038, 22.1% of tokens) | own effect **−0.09386** |
| Excluding it (labelled sensitivity analysis) | macro +0.02707 (unchanged), token-weighted **+0.07321** |

The largest connected group materially affects the **token-weighted** conclusion (it nearly doubles when the group is excluded) but barely moves the macro estimand. The full-data estimate remains primary; the group is not dropped to improve significance.

## Direction diagnostics

Run `python scripts/run_direction_diagnostics.py --repeats 10` to investigate the as-stored vs reading-order-normalized restoration gap. On **identical, union-grouped partitions** the as-stored advantage is about **1.2 pp** (0.3193 vs 0.3076 top-1), smaller than the 2.7–3.1 pp previously quoted from non-aligned splits.

All orientation variants are split with the **same union-derived group keys**, so neither ordering can leak an equivalent sequence across the shared split. The runner reports:

- **Three boundary models** under reversal: `asymmetric` (the default, mean delta −0.0134), `none` (+0.0094) and `symmetric` (−0.0046). **No model is exactly reversal-invariant**, including `symmetric`: the context model's left term is `P(w | prev)` and its right term is `P(next | w)`, which are transpose-related and coincide only under detailed balance. The edge-symmetric models do show a smaller delta than the default.
- **Cross-direction transfer with OOV separated.** Comparing conditions at *matched* OOV (both 0.1815) shows 0.0697 vs 0.2808 on shared-vocabulary targets, so the ordering effect is real rather than a vocabulary-coverage artefact.
- **Position classes** reported separately: singleton 0.0536, first 0.1604, interior 0.3476, last 0.3485.

Diagnostic only: no pipeline default changes.

## Same-family transcription sensitivity

Run `python scripts/run_transcription_sensitivity.py --repeats 10` to test whether context beats frequency and position under *both* same-family ICIT transcriptions. Pairs are formed in an **explicit artifact/inscription-level pair table before gap splitting**; only CISI with exactly one unambiguous record on each side enter the comparison. Multiple primary records are never merged merely because their sequences agree, and one external inscription is never copied onto several primary spans. The external side keeps its own (absent) completeness information rather than inheriting the primary's boundary flags. Agreement between these ICIT-derived sources is **not** independent inter-annotator agreement.

**The corrected result is ~30%, not ~40%.** Two grouping protocols are reported:

| Protocol | Sequences crossing folds per seed | Primary context top-1 | External context top-1 |
| --- | --- | --- | --- |
| `artifact_only` | **56–74** | 0.3959 | 0.3834 |
| `artifact_plus_union_duplicates` | **0** | **0.3016** | **0.3018** |

The earlier ~39.6% figure came from artifact-only grouping, which leaks 56–74 duplicate sequences across the shared split and inflates the score. Under strict union grouping both transcriptions agree almost exactly (0.3016 vs 0.3018) and context still beats frequency (+19.6 pp) and position (+15.4 pp) — so the main result **does survive both transcription versions under aligned strict grouping**.

**Do not compare this subset's accuracy with the ~29% headline restoration result.** They are different measurements: this subset is 1,841 matched inscriptions (369 held out per seed) after excluding 534 ambiguous-primary and 1,668 unmatched-primary CISI, with a different grouping policy and split fraction. A higher number on a smaller, differently-selected subset is not evidence of a stronger model.

## Power calibration and compact models

Run `python scripts/run_power_analysis.py` for synthetic calibration. The runner answers three **separate** questions and never treats them as interchangeable:

1. **Predictive difference** — do the fitted bigram and trigram differ in held-out performance, in either direction? (two-sided sign-flip)
2. **Predictive improvement** — does the trigram improve held-out performance? (one-sided positive sign-flip)
3. **Structural departure** — is the observed gain unusually large relative to a fitted first-order generative null passed through the complete estimation pipeline? (whole-pipeline simulation)

A rate is labelled a **false-positive rate only where the simulated scenario satisfies that rule's null**. A first-order generator can produce a genuine predictive *disadvantage* for a more complex fitted model, so a two-sided rejection under such a generator is **not** a false discovery of higher-order structure. Position-conditioned, slot-generator, and shuffled-real corpora are named controls, not pure first-order nulls. Runs are resumable, with one JSON file per replicate under `replicates/`; pass `--replicates 100` for the full grid, and `--null-replicates N` to enable the structural test.

**Corrected calibration results** (112 replicates at 1× size):

- The **difference** rule rejects in 100% of cells, including the zero-effect ones. It is therefore not a discovery criterion — exactly as predicted, since the over-parameterized fitted trigram genuinely loses under a first-order generator.
- The **improvement** rule has a **0.000 false-positive rate across all five valid null cells** (unigram, markov, position_only, shuffled_real, trigram λ=0; 0/30 replicates). Power rises 0.000 → 0.333 → 1.000 between λ=0.35 and λ=0.50.
- The deterministic **slot generator is a named alternative, not a null**: its positive-rejection rate is 1.000 with a *positive* mean effect, so that column is power, not a false-positive rate.
- **Structural test**: observed real-corpus effect +0.0241 against a fitted first-order null (mean −0.0914, threshold −0.0862); 0 of 40 null draws reached it → **p = 0.0244**, structural departure supported at α = 0.05. The rule's false-positive rate on the independent evaluation half was 0.10 (2/20), imprecise at that sample size. The p-value is conditional on the fitted null and is **not** proof of a linguistic mechanism.

Run `python scripts/run_compact_models.py` to compare compact structural models against the bigram under **two comparable budgets**: all models trained on the same capped subset (whole components, realized sizes reported), and all models trained on the full outer-training partition (except the HMM, which is capped for tractability and says so). Evaluation takes **full records**, so artifact identity and grouping survive the whole runner, and HMM state count plus position smoothing strength are selected on an inner **grouped** split of each outer training partition.

**Result: the bigram beats every compact model under both budgets.**

| Budget | bigram | position exact | position relative | exact+complete | HMM |
| --- | --- | --- | --- | --- | --- |
| matched (cap 600 records) | **7.06** | 8.49 | 8.90 | 8.42 | 8.16 |
| full | **5.97** | 7.51 | 7.29 | 7.56 | 8.09 |

All models share one training-only vocabulary policy: `<UNK>` is always present with a pseudo-count, likelihood is evaluated over mapped `<UNK>` outcomes, and one unseen sign never discards the known-token contributions of its sequence. OOV rates are reported per model (0.130 matched, 0.020 full). Restoration scoring is separate — an unseen original sign can never count as a correct restoration.

## Reproducibility and result provenance

- `docs/PROJECT_REPORT.md` records the pre-correction baseline, current results, limitations, and next steps in one consolidated report.
- `outputs/archive_pre_correction_4f81402/` holds the pre-correction results verbatim, with every file hashed in `BASELINE_MANIFEST.json`.
- Each experiment's dataset, grouping policy, metric, OOV handling, budget, output location, and interpretation status are summarized in `docs/PROJECT_REPORT.md`.
- `docs/METHODS.md` is a **retrospective** methods and claims specification, written after the initial analyses, not a preregistration.


## License

Apache 2.0
