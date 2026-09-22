# Handoff report

Plain-language report on the correction pass, for a reader who wants to know what the
evidence now supports. Written after Steps 1–11 of the brief. Numbers here are the
**corrected** ones; the pre-correction state is preserved in `docs/BASELINE.md` and
`outputs/archive_pre_correction_4f81402/`.

---

## 1. Does neighboring-sign context still beat frequency and position?

**Yes — this is the strongest and least changed result.** The context bigram reaches
**29.21% ± 1.87** restoration top-1 against **10.82% ± 1.09** for frequency and
**12.84% ± 1.32** for position: **+18.4 pp** and **+16.4 pp**, winning on 10 of 10
grouped splits in every one of four data protocols (readable spans, complete
inscriptions, unique sequences, complete unique sequences) and in the leave-one-site-out
evaluation. The 2×2×2 preprocessing matrix holds across all 8 cells (+17.3 to +22.2 pp
over frequency, +15.0 to +19.3 pp over position).

This result was **not** affected by the corrections; it is carried forward unchanged.

## 2. Does the transformer still lose to the bigram?

**Yes.** Under fully nested selection the transformer reaches **23.65% ± 2.87** against
the bigram's **29.21% ± 1.87**, losing on **10 of 10** outer splits (−5.56 pp). The
earlier non-nested figure of 19.4% is superseded. Embeddings are worse still (~9%).
At ~18k tokens over ~700 signs this is a corpus-size statement, not a claim about model
classes in general.

## 3. Is the trigram improvement supported after the corrected calibration?

**Yes, provisionally — and the evidence is now a calibrated test rather than a p-value.**

| Evidence | Value |
| --- | --- |
| Macro group effect (primary fold seed 0) | **+0.02702** bits/token, 95% CI [+0.01048, +0.04395] |
| Across 5 fold seeds | mean +0.02867, range [+0.02592, +0.03147] |
| Token-weighted (secondary) | +0.03623, 95% CI [−0.01173, +0.09211] — **includes zero** |
| Improvement-rule false-positive rate | **0.000** (0/30) across five valid null cells |
| Structural test | observed +0.0241; **0 of 40** null draws reached it; **p = 0.0244** |
| Structural-rule false-positive rate | 0.10 (2/20) on a disjoint evaluation half |

The old two-sided "difference" test rejects in **100% of cells, including every
zero-effect cell**, so it is worthless as a discovery criterion — exactly as the brief
predicted, because an over-parameterized fitted trigram genuinely *loses* under a
first-order generator. The corrected one-sided improvement rule is well behaved, and the
structural test now supports a departure from the fitted first-order null.

**Why still provisional:** the structural test rests on only 20 calibration and 20
evaluation null draws, the p-value is conditional on a null fitted to finite data that
reproduces neither artifact nor duplicate structure, and the observed effect sits near
the edge of the powered range (the λ = 0.40 cell detects 33% of the time). It is evidence
that the effect is unusual under *that* null — not proof of a linguistic mechanism.

## 4. How much does the largest connected group affect conclusions?

Substantially — but only for one of the two estimands. Component `INDUS-0038` holds
**3,645 tokens = 22.1%** of the corpus across 1,575 spans, created by exact-sequence
duplicate chaining. **Its own effect is negative: −0.09386 bits/token.**

| | Macro group effect | Token-weighted effect |
| --- | --- | --- |
| Including it (primary) | +0.02702 | +0.03623 |
| Excluding it (labelled sensitivity) | +0.02707 (**unchanged**) | +0.07321 (**doubles**) |

Because the macro estimand weights every component once, one huge group barely moves it.
Because the token-weighted estimand weights by tokens, that group dominates it — and it
pulls the estimate *down*. This is precisely why the two estimands are reported
separately and never averaged. The full-data estimate remains primary; the group is not
dropped to improve significance.

## 5. Does the main result survive both transcription versions under aligned strict grouping?

**Yes.** With the corrected explicit pair table and the strict protocol:

| Protocol | Sequences crossing folds/seed | Primary context top-1 | External context top-1 |
| --- | --- | --- | --- |
| `artifact_only` | **56–74** | 0.3959 | 0.3834 |
| `artifact_plus_union_duplicates` | **0** | **0.3016** | **0.3018** |

Both transcriptions agree to within **0.02 pp**, and context still beats frequency
(+19.6 pp) and position (+15.4 pp) on both. Record accounting is explicit: 1,841 matched
inscriptions, 369 held out per seed, 17,342 prediction events, after excluding 534
ambiguous-primary and 1,668 unmatched-primary CISI.

**The old ~40% figure was inflated by leakage.** Artifact-only grouping let 56–74
duplicate sequences sit in both training and test. The honest number is **~30%**.

## 6. What explains — and what remains unexplained about — the orientation difference?

**Explained:** the as-stored advantage on identical aligned partitions is small, about
**1.2 pp** (0.3193 vs 0.3076), not the 2.7–3.1 pp quoted from non-aligned splits. The
earlier claim that **vocabulary coverage** explains cross-direction transfer is
**refuted**: at *matched* OOV (both 0.1815), normalized-direction training scores 0.0697
on shared-vocabulary targets while as-stored scores 0.2808. A 21 pp gap at equal OOV is
an ordering effect, not a coverage effect.

The orientation behaviour is also partly a property of the model, not the data. Under
reversal, **no boundary model is exactly invariant** — including `symmetric`. Mean
deltas: `asymmetric` (default) −0.0134, `none` +0.0094, `symmetric` −0.0046. Equalizing
the edge terms shrinks the delta but cannot zero it, because the context model's left
term is `P(w | prev)` and its right term is `P(next | w)`; those are transpose-related
and coincide only under detailed balance.

**Unexplained:** the gap concentrates in short spans and complete records, and several
sites reverse sign (Kalibangan, Dholavira, Lothal, Chanhu-daro go the other way). First
positions are much harder than interior or last positions (0.1604 vs 0.3476 vs 0.3485).
Direction conventions were verified against `data/PROVENANCE.md`. **The pipeline default
is unchanged** — another ordering predicting better is not evidence the default is wrong,
and this remains an open archaeological question, not a software one.

## 7. Do compact models remain worse under fair training and scoring conditions?

**Yes — this is now a clean negative result rather than an artefact.** Previously the HMM
was capped at 600 sequences while the baselines used the full partition, and the API took
plain sequences, discarding artifact identity. Both defects are fixed, and two comparable
budgets are reported:

| Budget | bigram | position exact | position relative | exact+complete | HMM |
| --- | --- | --- | --- | --- | --- |
| matched (same capped subset) | **7.06** | 8.49 | 8.90 | 8.42 | 8.16 |
| full (outer train, HMM capped) | **5.97** | 7.51 | 7.29 | 7.56 | 8.09 |

The bigram wins under **both**. All models now share one training-only vocabulary policy
(`<UNK>` always present, per-token scoring, one unseen sign never discards a sequence's
known tokens), with OOV rates reported per model (0.130 matched, 0.020 full).

## 8. Which results changed after corrections?

| Quantity | Before | After | Why |
| --- | --- | --- | --- |
| Transcription context top-1 | ~39.6% | **30.2%** | duplicate leakage across folds removed |
| Orientation advantage | 2.7–3.1 pp | **~1.2 pp** | identical aligned partitions |
| Cross-direction explanation | "vocabulary coverage" | **ordering effect** | OOV-matched comparison |
| Last-position accuracy | `None` | **0.3485** | broken position filter fixed |
| Trigram claim basis | withdrawn null band (4 sims) | **calibrated structural test, p = 0.0244** | three questions separated |
| "Detection rate" 1.000 | called false-positive rate | **not an FP rate; difference rule rejects everywhere** | negative effects were counted as discoveries |
| Compact models | "preliminary", unfair budgets | **negative under two fair budgets** | real identities + matched budgets |
| Artifact types | all "unknown" | **resolved** (SEAL:S, TAB:C, …) | `artefact_type` never carried into analysis records |
| Largest component | not isolated | **own effect −0.0939** | newly reported |
| Trigram macro effect | +0.0279 | +0.02702 | larger-component-first fold balancing |

## 9. Which claims are supported, provisional, or unsupported?

**Supported**
- Immediate context beats frequency and position, across all protocols and partitions.
- The transformer and embeddings lose to the bigram under nested evaluation.
- Compact models lose to the bigram under matched and full budgets.
- The largest component materially affects the token-weighted estimand.

**Provisional**
- A trigram carries a small amount of predictive information beyond the bigram: calibrated
  test rejects at p = 0.0244, improvement rule has 0.000 FP rate, but the null is fitted
  and the sample is small.

**Unsupported / withdrawn**
- That a 100% two-sided rejection rate proves the test anti-conservative — **withdrawn**;
  it was a category error (see §3).
- That the observed effect sat above a "calibrated null band" — **withdrawn**; that band
  was the min/max of four simulations.
- That vocabulary coverage explains cross-direction transfer — **refuted** at matched OOV.
- That compact models are competitive — **not supported**.
- That exact reversal invariance is achievable with this context model — **not supported**;
  the left/right terms are transpose-related.

**Explicit non-claims (unchanged):** no decipherment, no semantic readings, no language
identification, no demonstrated words, morphemes, affixes or phrases.

## 10. What should be done next?

1. **Complete the calibration grid.** Currently 112 replicates (6/cell) at 1× only; the
   brief asks ≥100/cell at 0.5×/1×/2×. Resumable — existing replicates are reused:
   ```powershell
   .\.venv\Scripts\python.exe scripts\run_power_analysis.py `
     --sizes 0.5 1.0 2.0 --replicates 100 `
     --scenarios unigram position_only markov hmm_slots shuffled_real trigram_mixture `
     --lambdas 0.0 0.25 0.30 0.35 0.40 0.5 1.0 `
     --null-replicates 200 --permutations 2000 --bootstrap 2000
   ```
   ~4,000 runs at ~100 s each ≈ **110 h serial**, ~18 h across 6 parallel shards.
2. **Tighten the structural test** once (1) lands: 200 null draws per half instead of 20,
   so the false-positive rate is estimated to better than ±0.03.
3. **Investigate the position asymmetry** — first positions (0.1604) are far worse than
   interior (0.3476) and last (0.3485). Is that the `<S>` start distribution, or genuine
   sequence-initial uncertainty?
4. **Convert the position-model/HMM negative result into a paper-grade statement** by
   adding a parameter-matched control, so "bigram wins" is not confounded with model size.
5. **Do not rerun** the transformer or embeddings: their evaluation path is unaffected by
   these corrections and their manifests still match the code.

---

## Requirement checklist

Status key: **code** = implemented and tested; **run** = executed on the real corpus;
**partial** = code complete but the run is smaller than the brief specifies.

| Brief item | Code | Tests | Run | Status |
| --- | --- | --- | --- | --- |
| §1 reproducible baseline + archived outputs | ✅ | — | ✅ | complete |
| §2 three statistical questions; FP labelling; no negative-as-positive | ✅ | ✅ | ✅ | complete |
| §2 plus-one MC p, never zero | ✅ | ✅ | ✅ | complete |
| §3 shared grouping / folds / OOF scoring / leakage / group aggregation | ✅ | ✅ | ✅ | complete |
| §3 largest-component-first seeded balancing + load/imbalance reporting | ✅ | ✅ | ✅ | complete |
| §3 explicit grouping policies; not interchangeable | ✅ | ✅ | ✅ | complete |
| §4 two estimands + explanation | ✅ | ✅ | ✅ | complete |
| §4 auditable per-token scores with identifiers | ✅ | — | ✅ | complete |
| §4 several predefined fold seeds, never pooled | ✅ | ✅ | ✅ | complete |
| §4 largest-component own / included / excluded sensitivity | ✅ | — | ✅ | complete |
| §4 interval limits explained | ✅ | — | ✅ | complete |
| §4 artifact-type metadata retained | ✅ | — | ✅ | complete |
| §4 training-only frequency bands; length labels; sample sizes | ✅ | — | ✅ | complete |
| §5 real identities, shared grouping/folds, grouped inner selection | ✅ | ✅ | ✅ | complete |
| §5 matched and full budget experiments | ✅ | ✅ | ✅ | complete |
| §5 shared `<UNK>` policy; per-token scoring; OOV reported | ✅ | ✅ | ✅ | complete |
| §5 position models named separately; hierarchical smoothing; alpha tuned | ✅ | ✅ | ✅ | complete |
| §5 HMM multiple deterministic inits; convergence; parameter counts | ✅ | ✅ | ✅ | complete |
| §6 explicit artifact-level pair table before gap splitting | ✅ | ✅ | ✅ | complete |
| §6 one unambiguous record per side; no span fan-out; no flag transfer | ✅ | ✅ | ✅ | complete |
| §6 shared partitions; two protocols; union of duplicate relations | ✅ | ✅ | ✅ | complete |
| §6 real subgroup metrics; sample sizes; small strata marked descriptive | ✅ | ✅ | ✅ | complete |
| §7 shared aligned groups; union duplicate controls | ✅ | ✅ | ✅ | complete |
| §7 OOV-matched transfer; shared-vocabulary accuracy | ✅ | ✅ | ✅ | complete |
| §7 three boundary experiments with stated invariants | ✅ | ✅ | ✅ | complete |
| §7 first / interior / last / singleton reporting; last-position fixed | ✅ | ✅ | ✅ | complete |
| §7 direction conventions verified against source docs | ✅ | — | ✅ | complete |
| §8 generator descriptions corrected; state-update bug fixed | ✅ | ✅ | ✅ | complete |
| §8 zero-strength mixture verified first-order | ✅ | ✅ | ✅ | complete |
| §8 scenarios incl. slot generator and shuffled controls | ✅ | ✅ | ✅ | complete |
| §8 structural test with disjoint calibration/evaluation | ✅ | ✅ | ✅ | complete |
| §8 ≥100 reps per cell × 0.5×/1×/2× | ✅ | — | ⚠️ **6/cell, 1× only** | **partial** |
| §8 benchmark + resumable per-replicate output | ✅ | — | ✅ | complete |
| §8 positive/negative rates, FP rate, power, CIs, failures | ✅ | ✅ | ✅ | complete |
| §9 scientific regression tests | ✅ | ✅ | — | complete |
| §10 README / METHODS / results index reconciled | ✅ | — | — | complete |
| §11 regenerate results in dependency order | ✅ | — | ✅ 6 experiments | complete |
| §12 pytest / ruff / mypy | — | ✅ 327 collected; 141 verified in change set | ✅ | complete |
| §12 handoff report | — | — | — | this document |

### What is *not* complete

- **The calibration grid is partial**: 6 replicates per cell at 1× size, not ≥100 across
  0.5×/1×/2×. The structural test therefore uses 20 + 20 null draws. This is reported as
  provisional everywhere and the resume command is in §10.
- **`docs/RESULTS_INDEX.md`** records the achieved run scale per experiment, so no reader
  mistakes a partial run for a complete one.

### Honest note on the environment

The DSH file sandbox denies pytest's `tmp_path` fixture, so ~20 tests cannot execute under
it; they pass when the suite is run outside the sandbox. `docs/BASELINE.md` records this
as an environmental limitation, not a code failure.