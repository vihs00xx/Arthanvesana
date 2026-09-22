# Arthānveṣaṇa project report

This is the single consolidated project report. It combines the former baseline,
handoff, and results-index documents into one readable record of provenance,
methods, results, limitations, and next actions. `docs/METHODS.md` remains the
detailed technical specification; this file is the project-level summary.

## 1. Project status

The repository is synchronized with GitHub on `main`. The corpus and processed
data are pinned and documented in `data/PROVENANCE.md`. Generated experiment
outputs live under `outputs/<experiment>/`; they are reproducible artifacts and
are intentionally not committed to Git.

The pre-correction state is preserved at
`outputs/archive_pre_correction_4f81402/`, with hashes in its
`BASELINE_MANIFEST.json`. It must not be mixed with the corrected results.

## 2. What the project asks

The central question is whether local sign context contains predictive structure
beyond sign frequency and position. The project tests this with held-out
restoration and log-loss experiments, while checking leakage, transcription
choices, orientation, model class, and synthetic calibration.

This work does **not** decode the Indus script or claim a sign-to-language,
sign-to-morpheme, or sign-to-meaning mapping.

## 3. Current evidence in plain language

### Supported primary finding

Immediate neighboring context predicts a masked sign substantially better than
frequency or position baselines. The main corrected restoration result is about
29.2% top-1 for the context model, versus about 10.8% for frequency and 12.8%
for position. The result survives the documented preprocessing and transcription
sensitivity checks.

### Provisional higher-order finding

A trigram can improve held-out log loss over a bigram under grouped cross-fitting.
The primary macro effect is approximately +0.027 bits/token, with a descriptive
cluster-bootstrap interval of about [+0.010, +0.044]. The token-weighted interval
includes zero, and the result is therefore provisional rather than a definitive
claim about universal higher-order structure. The structural calibration is also
pilot-scale (six replicates per cell at the 1× size in the current run).

### Negative model-comparison findings

The fully nested transformer loses to the bigram (about 23.65% versus 29.21%,
losing all ten outer splits). PPMI/SVD and skip-gram embeddings also lose. The
compact relative-position, complete-context, and HMM models are worse in both
matched-budget and full-budget comparisons. HMM convergence is incomplete in
part of the search, so this is a model-specific negative result, not a claim
that all structural models are impossible.

### Sensitivity and diagnostics

- The largest connected component contains about 22.1% of analyzed tokens. It
  is reported in the full estimate and in a labelled leave-one-component-out
  sensitivity analysis.
- The strict matched-transcription comparison remains positive at roughly 30%
  context accuracy under both transcriptions. A looser artifact-only protocol
  has cross-fold sequence overlap and is not used for claims.
- Direction transfer and boundary results are exploratory. Normalized versus
  as-stored orientation remains a real unresolved source of variation.

## 4. Implementation and analysis system

The project is implemented as a small, reproducible Python package under
`src/arthanvesana/`, with command-line runners under `scripts/` and regression
tests under `tests/`. The implementation has five layers:

1. **Data and provenance** — parsing, gap-aware preprocessing, metadata joins,
   corpus validation, source hashes, and a documented processed corpus.
2. **Shared evaluation infrastructure** — artifact/inscription/exact-sequence
   connected components, indivisible group assignment, seeded largest-first
   fold balancing, leakage diagnostics, out-of-fold predictions, and audit
   tables with record and group identifiers.
3. **Statistical baselines** — frequency, position, smoothed bigram and
   trigram models, held-out restoration, log loss, MRR, OOV accounting, and
   group-level inference.
4. **Model and robustness runners** — transformer, PPMI/SVD, skip-gram,
   compact position/complete-context/HMM models, direction diagnostics,
   transcription matching, motif stratification, and leave-one-site-out
   robustness.
5. **Simulation and quality control** — resumable synthetic calibration,
   disjoint calibration/evaluation sets, explicit decision rules, unit tests,
   Ruff linting, and mypy checks.

The central anti-leakage design is that related records are connected before
fold assignment. Every member of a connected component stays in one fold;
duplicate sequences therefore cannot appear in both training and test data.
Configuration selection and early stopping use inner grouped partitions, never
outer test records. The n-gram analysis reports macro group means as its primary
estimand and token-weighted means separately, rather than silently pooling them.

## 5. Quantitative results

### 5.1 Main restoration result

On the corrected grouped evaluation, immediate context reaches approximately
29.21% top-1 restoration accuracy, compared with 10.82% for frequency and
12.84% for position. This is an absolute gain of roughly 18.4 percentage
points over frequency and 16.4 points over position. These are predictive
associations in the corpus, not evidence that the signs have been decoded.

### 5.2 Bigram versus trigram log loss

The analysis contains 4,624 spans, 16,469 tokens, and 2,262 connected groups,
with zero detected leakage. The primary macro effect (trigram improvement in
bits/token) is +0.02702 with a descriptive cluster-bootstrap 95% interval of
[+0.01048, +0.04395]. Across five pre-specified fold seeds, effects range from
 +0.02592 to +0.03147 (mean +0.02867). The secondary token-weighted estimate
is +0.03623 with interval [−0.01173, +0.09211], which includes zero.

The largest component (`INDUS-0038`) contains 22.1% of tokens and has an own
effect of −0.09386. Removing it changes the macro estimate to +0.02707 but the
token-weighted estimate to +0.07321. This is why the full-data macro estimate
remains primary and the leave-one-component result is labelled sensitivity.

### 5.3 Learned representations and transformer

The fully nested transformer obtains 23.65% ± 2.87 top-1 versus 29.21% ± 1.87
for the bigram and loses on all 10 outer splits (−5.56 percentage points).
PPMI/SVD obtains about 8.30% and skip-gram about 9.05%. These are negative
results under the tested architectures and budgets; they do not establish that
no neural or distributional representation could ever help.

### 5.4 Transcription and direction sensitivity

Under strict matched transcription grouping, the primary and external
transcriptions obtain 30.16% and 30.18% context accuracy, respectively, with no
sequence crossing folds. The older 39–40% result came from an artifact-only
protocol with 56–74 cross-fold duplicate sequences and is not valid evidence.

For orientation diagnostics, the as-stored versus normalized context results
are approximately 31.93% and 30.76% on identical union-grouped partitions. The
approximately 1.2-point gap is exploratory; boundary-model and OOV-separated
analyses show that ordering and vocabulary coverage both matter.

### 5.5 Compact structural models

Held-out bits/token (lower is better) under matched and full budgets are:

| Budget | Bigram | Position exact | Position relative | Complete-context | HMM |
|---|---:|---:|---:|---:|---:|
| Matched cap | **7.0609** | 8.4865 | 8.8997 | 8.4246 | 8.1635 |
| Full outer training | **5.9663** | 7.5094 | 7.2874 | 7.5562 | 8.0941 |

The bigram wins every comparison. HMM fits are not all converged (100 of 200
inner trials converged), so the HMM conclusion should remain bounded to the
implemented search and reported convergence quality.

### 5.6 Synthetic calibration and statistical significance

The calibration runner separates three questions: any predictive difference,
positive improvement, and structural departure from a fitted first-order null.
The current pilot contains six replicates per cell at the 1× size for the
resumable grid, plus disjoint structural calibration/evaluation draws.

The improvement rule had 0/30 false positives across the five valid null cells;
power increased from 0.000 at λ=0.35 to 0.333 at λ=0.50 and 1.000 at stronger
effects. The observed structural effect was +0.0241 against a fitted-null mean
of −0.0914, giving a conditional structural p≈0.0244. However, the independent
evaluation false-positive estimate was 2/20 = 0.10, so this p-value is not yet
precisely calibrated. It is evidence of departure under the chosen fitted null,
not proof of a linguistic mechanism or a universal significance claim.

## 6. Scientific significance

The strongest defensible conclusion is that sign sequences in this corpus are
not well described by independent sign frequency or position alone: neighboring
context carries reproducible predictive information. The effect survives strict
grouped evaluation and transcription sensitivity, making it a meaningful
empirical property of the dataset.

The project does **not** show that the Indus script has been deciphered. It does
not identify a language, grammar, morphemes, meanings, or a unique generative
mechanism. The trigram result suggests possible higher-order structure but is
limited by token-weighted uncertainty and incomplete calibration. The negative
transformer, embedding, and compact-model results are useful controls: the
signal is not automatically recovered by every more flexible model, and the
simple bigram is a strong baseline.

## 7. Limitations affecting interpretation

- The connected-group structure is highly imbalanced; one component contains
  about one-fifth of all tokens.
- The full calibration grid (100 replicates × three sizes × multiple strengths)
  has not been completed.
- HMM optimization is partly non-converged.
- Direction normalization is not settled and remains exploratory.
- The corpus is a finite, curated epigraphic dataset; results may not
  generalize to other corpora or future readings.
- Confidence intervals quantify the stated resampling procedure; they are not
  automatically population-level intervals over all possible inscriptions.

## 8. Experiment inventory

| Experiment | Main output | Interpretation |
|---|---|---|
| Group audit | `outputs/group_audit/` | leakage and component-size audit |
| Bigram/trigram inference | `outputs/ngram_inference/` | provisional higher-order result |
| Restoration robustness | `outputs/robustness/` | primary supported result |
| Embeddings | `outputs/embeddings/` | negative under nested evaluation |
| Transformer | `outputs/transformer/` | negative under nested evaluation |
| Preprocessing sensitivity | `outputs/sensitivity/` | primary result survives matrix |
| Motif/stratified analysis | `outputs/stratified/` | exploratory |
| Direction diagnostics | `outputs/direction_diagnostics/` | exploratory |
| Matched transcription | `outputs/transcription_sensitivity/` | exploratory sensitivity |
| Compact models | `outputs/compact_models/` | negative, with HMM caveat |
| Synthetic calibration | `outputs/power_analysis/` | pilot calibration; not complete |
| Cross-corpus audit | `outputs/audit/` | exploratory transcription audit |

Each runner writes a text report and machine-readable JSON in its output
directory. The scripts and tests are the authoritative reproducibility layer;
the generated reports are not source code and are ignored by Git.

## 9. Provenance and pre-correction baseline

The archived baseline was recorded at commit `4f81402` under Python 3.13.3.
It included 265 passing tests plus 19 sandbox-only `tmp_path` errors; the
corrected repository now passes the full suite in the workspace environment.
The archive preserves the earlier approximately 10.8% frequency, 12.8% position,
29.2% bigram, 23.65% nested-transformer, 9.05% skip-gram, and 8.30% PPMI/SVD
headline values, along with the known pre-correction calibration and compact
model limitations.

## 10. What is complete and what is not

Complete: grouped folds and leakage diagnostics; corrected n-gram inference;
matched transcription protocols; direction diagnostics; compact-model repair;
nested transformer and embedding evaluations; provenance and tests.

Not complete: the full 100-replicate, three-size, multi-strength calibration
grid. Until that grid is run, calibration-based claims must remain explicitly
pilot/provisional. HMM convergence and orientation normalization also deserve
follow-up before stronger model-family claims are made.

## 11. Recommended next sequence

1. Run the resumable calibration grid and record its exact command, environment,
   and completed-cell counts.
2. Resolve the orientation convention and rerun direction diagnostics under the
   pre-registered choice.
3. Investigate HMM non-convergence or narrow the claim to converged fits.
4. Regenerate all outputs from a clean environment and update this report's
   headline numbers only from those regenerated artifacts.
5. Treat the context result as the central result, the trigram result as
   provisional, and all learned/compact model comparisons as bounded negative
   evidence.

## 12. Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe scripts/run_group_audit.py
.\.venv\Scripts\python.exe scripts/run_ngram_inference.py
.\.venv\Scripts\python.exe scripts/run_power_analysis.py
```

For the complete experiment order and model definitions, read
`docs/METHODS.md` and the `--help` output of each runner.
