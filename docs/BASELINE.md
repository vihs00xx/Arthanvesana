# Baseline record — pre-correction state

Recorded before any correction in this brief was applied. Its purpose is to let a
reader distinguish **pre-correction** results from **corrected** results, and to
pin the environment in which the pre-correction numbers were produced.

## Git state

| Field | Value |
| --- | --- |
| Commit | `4f81402d4efd01f5ac43ce340fae9338fa4c611f` |
| HEAD subject | Record false-positive calibration: randomization p-value is anti-conservative, use null band |
| Working tree | clean (`git status --porcelain` empty) |

Recent history at baseline:

```
4f81402 Record false-positive calibration: randomization p-value is anti-conservative, use null band
e26184f Cap HMM training sequences and surface the limit in the report
1e0cb76 Document direction diagnostics, transcription sensitivity, power calibration, and compact models
5f2d4c2 Add compact structural models (relative-position, HMM) under grouped nested evaluation
f0df077 Tighten boundary-probe assertions in direction diagnostics tests
```

## Environment

| Field | Value |
| --- | --- |
| Python | 3.13.3 (CPython) |
| Interpreter | `.venv/Scripts/python.exe` |
| Platform | Windows (see `BASELINE_MANIFEST.json` for the full string) |
| Dependency versions | recorded per-package in `BASELINE_MANIFEST.json` |

## Checks run at baseline

| Check | Command | Result |
| --- | --- | --- |
| Tests | `python -m pytest -q` | **265 passed, 19 errors** |
| Lint | `python -m ruff check .` | All checks passed |
| Types | `python -m mypy` | Success: no issues found in 5 source files |

**The 19 test errors are environmental, not code failures.** Every one is
`PermissionError: [WinError 5] Access is denied` raised by pytest's `tmp_path`
fixture, which defaults to a directory outside the session workspace that the
DSH file sandbox refuses to enumerate. The tests themselves are not failing:
with an in-workspace `--basetemp` the same tests execute and the progress
output contains no failure markers, though pytest's own teardown cleanup still
trips the sandbox. This limitation is unchanged by the corrections below and
should be reported as such, not silently counted as a regression.

Reproduce the environmental diagnosis:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=no -p no:cacheprovider 2>$null |
  Select-String -Pattern 'passed|failed|error'
```

## Pre-correction results (preserved)

The full pre-correction results tree was copied verbatim to
`outputs/archive_pre_correction_4f81402/`, and every file in it is hashed in
`outputs/archive_pre_correction_4f81402/BASELINE_MANIFEST.json`. Corrected runs
write to the ordinary `outputs/<experiment>/` paths, so the archive remains the
only record of the earlier state.

Headline numbers carried into this brief, with their archived source:

| Result | Value | Archived source |
| --- | --- | --- |
| Frequency restoration top-1 | ~10.8% | `robustness/`, `transcription_sensitivity/` |
| Position restoration top-1 | ~12.8% | `robustness/` |
| Bigram restoration top-1 | ~29.2% | `robustness/`, `embeddings/`, `transformer/` |
| Fully nested transformer top-1 | ~23.65% | `transformer/` |
| Skip-gram restoration top-1 | ~9.05% | `embeddings/` |
| PPMI/SVD restoration top-1 | ~8.30% | `embeddings/` |
| Group-average trigram advantage | ~0.0279 bits/token | `ngram_inference/` |
| Token-weighted trigram interval | includes zero | `ngram_inference/` |
| Largest connected component | 22.1% of analyzed tokens | `group_audit/` |
| Same-family transcription comparison | ~39.6% vs ~39.7% context accuracy | `transcription_sensitivity/` |
| Saved synthetic calibration | four repetitions per condition, one corpus size | `power_analysis/` |
| Compact-model results | preliminary, known evaluation problems | `compact_models/` |

### Pre-correction power calibration (verbatim)

```
Replicates per cell: 4; sizes: [1.0]; permutations: 1000.
cell | n_runs | mean effect (bits/token) | MCSE | detection rate
markov|lam=0.0|size=1.0          | 4 | -0.08597 | 0.00261 | 1.000
position_only|lam=0.0|size=1.0   | 4 | -0.06906 | 0.00404 | 1.000
shuffled_real|lam=0.0|size=1.0   | 4 | -0.03229 | 0.00382 | 1.000
trigram_mixture|lam=0.0|size=1.0 | 4 | -0.09811 | 0.00476 | 1.000
trigram_mixture|lam=0.25|size=1.0| 4 | -0.04136 | 0.00441 | 1.000
trigram_mixture|lam=0.5|size=1.0 | 4 | +0.06738 | 0.00674 | 1.000
trigram_mixture|lam=1.0|size=1.0 | 4 | +0.52496 | 0.00727 | 1.000
unigram|lam=0.0|size=1.0         | 4 | -0.06318 | 0.00163 | 1.000
```

This is the artefact the brief identifies as mislabelled: a two-sided
`p < 0.05` rate of 1.000 is reported as a "detection rate" and hence as a
false-positive rate at `lambda=0`, even though the underlying signed effects at
`lambda=0` are **negative**. A fitted higher-order model losing to a fitted
first-order model is not a false discovery of higher-order structure.

### Pre-correction compact-model comparison (verbatim)

```
HMM training is capped at 600 sequences (the bigram and relative-position
models use the full outer training partition).

Model | bits/token mean +/- SD | perplexity | parameters
bigram             | 5.8839 +/- 0.0156 |  59.06 | see MKN bigram
relative_position  | 7.3749 +/- 0.0107 | 165.99 | 66490
hmm                | 7.2726 +/- 0.0406 | 154.67 | 1324.08
```

The HMM budget (600 sequences) and the baselines' budget (full outer train) are
not comparable, so this table cannot support a model-family conclusion.

## Implementation checklist

Maps each requirement of the brief to its code, tests, and outputs. Status
values: **done**, **partial**, **pending**.

| # | Requirement | Code | Tests | Outputs | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Reproducible baseline | `scripts/build_baseline_manifest.py`, this file | — | `outputs/archive_pre_correction_4f81402/BASELINE_MANIFEST.json` | done |
| 2 | Three separate statistical questions | `scripts/run_power_analysis.py`, `src/arthanvesana/simulate/pipeline.py` | `tests/test_power_analysis.py` | `outputs/power_analysis/` | done |
| 2 | No negative effect counted as a discovery | `run_power_analysis.summarize` | `test_negative_effect_never_counts_as_discovery` | `power_analysis/` | done |
| 2 | Plus-one Monte Carlo p, never zero | `pipeline._signflip_p` | `test_crossfit_is_group_level_and_p_never_zero` | `power_analysis/` | done |
| 3 | Shared grouping / folds / OOF scoring | `src/arthanvesana/stats/grouping.py` | `tests/test_grouping.py` | `outputs/group_audit/` | done |
| 3 | Largest-component-first seeded balancing | `grouping.assign_folds` | `test_large_components_stay_intact` | `group_audit/`, `ngram_inference/` | done |
| 4 | Bigram–trigram experiment strengthening | `scripts/run_ngram_inference.py` | `tests/test_ngram_inference.py` | `outputs/ngram_inference/` | partial |
| 5 | Compact-model comparison repair | `src/arthanvesana/replicate/compact.py` | `tests/test_compact_models.py` | `outputs/compact_models/` | pending |
| 6 | Matched-transcription comparison | `scripts/run_transcription_sensitivity.py` | `tests/test_transcription_sensitivity.py` | `outputs/transcription_sensitivity/` | pending |
| 7 | Direction diagnostics refinement | `scripts/run_direction_diagnostics.py` | `tests/test_direction_diagnostics.py` | `outputs/direction_diagnostics/` | pending |
| 8 | Synthetic calibration rebuild | `src/arthanvesana/simulate/generators.py`, `run_power_analysis.py` | `tests/test_power_analysis.py` | `outputs/power_analysis/` | partial |
| 9 | Scientific regression tests | `tests/` | — | — | partial |
| 10 | Documentation reconciliation | `README.md`, `docs/METHODS.md`, `docs/RESULTS_INDEX.md` | — | — | pending |
| 11 | Regenerate results in dependency order | `scripts/*` | — | `outputs/*` | pending |
| 12 | Verification and handoff | — | — | `docs/HANDOFF.md` | pending |

## Constraints observed

Per the brief: no new corpus was downloaded, nothing was published, no commit
was pushed, and no unrelated user change was modified. The corpus is pinned to
the hash recorded in `README.md` and `docs/METHODS.md` and is re-verified on
every rebuild.