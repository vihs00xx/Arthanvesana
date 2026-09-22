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

## 4. Experiment inventory

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

## 5. Provenance and pre-correction baseline

The archived baseline was recorded at commit `4f81402` under Python 3.13.3.
It included 265 passing tests plus 19 sandbox-only `tmp_path` errors; the
corrected repository now passes the full suite in the workspace environment.
The archive preserves the earlier approximately 10.8% frequency, 12.8% position,
29.2% bigram, 23.65% nested-transformer, 9.05% skip-gram, and 8.30% PPMI/SVD
headline values, along with the known pre-correction calibration and compact
model limitations.

## 6. What is complete and what is not

Complete: grouped folds and leakage diagnostics; corrected n-gram inference;
matched transcription protocols; direction diagnostics; compact-model repair;
nested transformer and embedding evaluations; provenance and tests.

Not complete: the full 100-replicate, three-size, multi-strength calibration
grid. Until that grid is run, calibration-based claims must remain explicitly
pilot/provisional. HMM convergence and orientation normalization also deserve
follow-up before stronger model-family claims are made.

## 7. Recommended next sequence

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

## 8. Reproduction commands

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
