# Arthānveṣaṇa — Retrospective Methods and Claims Specification

Retrospective methods and claims specification for the computational study in this
repository. This document freezes the protocol, terminology, and claims boundary for
subsequent untouched experiments; it was written AFTER the initial analyses, so it is a
retrospective specification, not a preregistration. All numbers are reproduced from the
reports under `outputs/` and the conventions in `README.md` and `data/PROVENANCE.md`; no
method is described here that the pipeline does not implement.

## 0. Correction status

A correction pass was applied on top of the baseline recorded in `docs/BASELINE.md`
(commit `4f81402`). Pre-correction results are preserved verbatim under
`outputs/archive_pre_correction_4f81402/`. This table states what is **implemented**,
what is **smoke-tested**, and what remains **pending**, so no reader mistakes a plan
for a result.

| Correction | Status | Evidence |
| --- | --- | --- |
| Baseline recorded, pre-correction outputs archived and hashed | implemented | `docs/BASELINE.md`, `BASELINE_MANIFEST.json` |
| Three statistical questions separated; FP labelling rule enforced | implemented, run | §4b, `tests/test_power_analysis.py` |
| Negative effects can never count as positive discoveries | implemented, tested | `pipeline.signflip_tests`, regression test |
| Shared grouping / fold assignment / OOF scoring / leakage utilities | implemented, tested | `src/arthanvesana/stats/grouping.py`, `tests/test_grouping.py` |
| Largest-component-first seeded fold balancing + load reporting | implemented, tested | `grouping.assign_folds`, `grouping.fold_loads` |
| Bigram–trigram: two estimands, five fold seeds, largest-component sensitivity, auditable per-token scores | implemented, **run** | §4a, `outputs/ngram_inference/` |
| Artifact-type metadata retained (was always "unknown") | implemented, **run** | `data/parse.py`, `outputs/ngram_inference/` |
| Generator state-update bug fixed; descriptions corrected | implemented, tested | `simulate/generators.py` |
| Calibration runner resumable with per-replicate output | implemented, **run** | `outputs/power_analysis/replicates/` |
| Structural test on a fitted first-order null with disjoint calibration/evaluation sets | implemented, **run** | §4b, `outputs/power_analysis/` |
| Compact-model comparison repaired (real identities, matched budgets, position models, HMM quality) | implemented, **run, tested** | §7, `outputs/compact_models/` |
| Matched-transcription comparison repaired (pair table, shared strict partitions, subgroup metrics) | implemented, **run, tested** | §5, `outputs/transcription_sensitivity/` |
| Direction diagnostics refined (union groups, OOV-controlled transfer, three boundary models, position classes) | implemented, **run, tested** | §5, `outputs/direction_diagnostics/` |
| Full 100-replicate, 3-size, multi-strength calibration grid | **pending** (runtime: ~100 h serial) | §4b, resume command in `docs/HANDOFF.md` |
| Reversal invariance of the bigram context model | **not achievable** as documented | §5 — left/right terms are transpose-related |

**Run scale actually achieved.** The calibration grid completed **112 replicates** (6 per
cell) at 1× size only, not the ≥100 per cell across 0.5×/1×/2× that the brief specifies.
The structural test therefore rests on 20 calibration and 20 evaluation null draws. This
is enough to support the qualitative reading reported here and **not** enough to resolve
a λ ≈ 0.43 effect precisely. The runner is resumable, so the larger grid can be completed
by re-running the same command; the exact resume command is in `docs/HANDOFF.md`.


## 1. Objective and scope

The objective is the **structural characterization of the Sindhu-Sarasvatī (Indus) sign
sequences**: distributional, sequential, and positional regularities, measured against
explicit null models. This study is **not a decipherment**. No sign-to-phoneme,
sign-to-morpheme, or sign-to-meaning mapping is proposed, tested, or reported anywhere in
the pipeline. Models assign probability mass to sign identities only; metrics such as top-1
restoration accuracy describe how well a model predicts a masked *sign code*, not a reading
of the inscription.

## 2. Data

The analysis corpus is pinned: `sanitized_corpus.json` from
`ShaktiOSindia/indus-sign-regimes-deposit`, commit `e48b3ec1e90f368079f5126790613cded6f6f56c`,
SHA-256 `345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68`, re-verified on
every rebuild. Published properties: **5,704 catalogued inscriptions** (5,536 analyzable),
**18,065 sign tokens**, **713 distinct signs**. ICIT `000` is a missing-data placeholder,
not a sign; it is gated before analysis and never a prediction candidate.

Reading order: stored physical left-to-right; as-stored for direction `L/R`, reversed for
`R/L`. One documented deviation from upstream: we reverse only unambiguous `R/L` and flag
all other directions `reading_order_known=False`.

Licensing: the raw corpus (upstream GPL-3.0) is not redistributed; derived sign-sequence
artifacts remain GPL-3.0 if redistributed; project code is Apache-2.0. Transparency
statement, not legal advice.

## 3. Grouped-partition audit (new)

All splits share one grouping implementation (`connected_groups` in `sampling.py`, the same
union-find code path as `split_records`): records are connected by artifact, inscription,
and exact-sequence identity, and whole components move together. `scripts/run_group_audit.py`
audits the components. Findings on the 4,624-span known-direction corpus:

- **2,262 connected components**; median component 5 tokens (p90 8, p95 11, p99 30).
- **The largest component (INDUS-0038) holds 3,645 tokens = 22.1% of all tokens** across
  1,575 spans, 1,459 inscriptions, 939 artifacts — created by **exact-sequence duplicate
  connectivity** chaining many artifacts into one component.
- It exceeds one 5-fold target (3,294 tokens), so grouped folds cannot be perfectly
  balanced. This explains why grouped test sets vary from ~2,300 to ~6,300 tokens: fold
  composition depends on where large components land.
- **Grouping is deliberately not weakened to balance folds**; doing so would leak linked
  duplicates across train and test.

## 4. Evaluation protocol

- **Grouped holdout splits** as above; overlap diagnostics confirm zero record, artifact,
  and sequence overlap in the headline evaluations.
- **Gap-split spans**; masked single-sign restoration; **OOV targets count as failures**.
- **Variability**: reported SDs describe split variability, NOT confidence intervals.
- **Null models**: within-span shuffles for entropy/segmentation; label permutations for
  cluster-role alignment.

### 4a. Grouped cross-fitted bigram-vs-trigram inference (replaces token-level test)

The earlier token-level permutation test was statistically invalid: tokens within artifacts
are dependent, and pooling over overlapping repeated splits can count the same record
multiple times. The corrected design (`scripts/run_ngram_inference.py`) uses deterministic
5-fold grouped cross-fitting:

1. Connected components are indivisible groups assigned to folds by seeded greedy
   balancing that places the **largest components first** (primary: token count;
   secondary: span count; seeded tie-breaking). Realized fold loads and the
   unavoidable imbalance from oversized components are reported.
2. MKN bigram and trigram are trained on four folds; every token of the held-out
   fold is scored under both models; each record receives **exactly one
   out-of-fold prediction**.
3. The paired difference is `d = log2 P_trigram - log2 P_bigram` (bits/token);
   positive favours the trigram. Token differences are aggregated **within each
   connected group** before inference.
4. **Primary estimand**: macro (unweighted) mean of group means — "does the
   trigram help a typical connected group?"
5. **Secondary estimand**: token-weighted mean — "does it help a typical token?"
   Large components dominate this one. The two answer different questions and are
   **never averaged together**.
6. The complete procedure is repeated under five predefined fold seeds. Each seed
   is an independent run; repeated predictions across seeds are **never pooled as
   independent observations**, and no seed is averaged into the primary result.
7. The largest connected component is reported on its own, inside the full
   estimate, and in an explicitly labelled sensitivity analysis excluding it. The
   full-data estimate remains primary; the large group is not dropped to improve
   significance.
8. Per-token scores are saved with inscription, artifact and group identifiers so
   every stratum can be audited back to the source records.
9. Exploratory breakdowns use clear length labels (`1–2`, `3–4`, `5–7`, `8+`) and
   per-token frequency bands defined from **each fold's training tokens only**.
   Artifact-type metadata is retained. Sample sizes are reported per stratum and
   small strata are descriptive only.

**Limits of the intervals.** The cluster-bootstrap intervals resample *fixed*
out-of-fold group scores. They capture group-level sampling variability but **not**
the uncertainty from retraining the models, and the two fitted models share training
data within a fold. They are not a substitute for whole-pipeline simulation, which
the calibration runner provides.

The group-level randomization p-value is retained as a **descriptive** statistic
only; see §4b.


### 4b. Three separate questions, and what the calibration does and does not show

The earlier draft of this section claimed that a 100% two-sided rejection rate in
the synthetic calibration proved the randomization test anti-conservative. **That
claim is withdrawn.** It conflated three different questions:

1. **Predictive difference** — do the fitted bigram and trigram differ in
   held-out performance, in either direction? Tested two-sided.
2. **Predictive improvement** — does the trigram improve held-out performance?
   Tested one-sided positive.
3. **Structural departure** — is the observed gain unusually large relative to a
   specified first-order generative null passed through the *complete* estimation
   pipeline? Tested by whole-pipeline simulation.

A first-order data generator can produce a genuine predictive **disadvantage** for
a more complex fitted model: a modified Kneser-Ney trigram carries more parameters
than the bigram, so with no genuine higher-order structure it genuinely loses
held-out log loss. A significant *negative* effect is therefore a real predictive
difference, not a false discovery of higher-order structure. Counting it as a
detection, as the earlier runner did, is a category error.

Rules now enforced by `scripts/run_power_analysis.py`:

- Every simulated dataset records the signed effect, its interval, the two-sided
  result, a positive-rejection indicator, a negative-rejection indicator, and the
  structural-test decision once that test is enabled.
- A rate is labelled a **false-positive rate only where the scenario satisfies the
  null hypothesis of that particular decision rule**. Valid nulls for the
  *improvement* rule are `unigram`, `markov`, `shuffled_real`, and
  `trigram_mixture` at `lam=0`. The *difference* column is a rejection rate and is
  **never** a false-positive rate in this grid.
- Position-conditioned, deterministic positional-slot, and within-inscription
  shuffled corpora are named **controls or alternatives**, not pure first-order
  nulls: they retain dependencies arising from length, position, or sign
  composition. The **slot generator is an alternative, not a null** — its measured
  positive-rejection rate is 1.000 with a *positive* mean effect, because position
  within a span is correlated with order and a fitted trigram genuinely extracts
  it. That column is **power**, not a false-positive rate.
- Monte Carlo p-values use the plus-one correction and never equal zero. Rates
  report their denominator and count failed and skipped simulations.

**The randomization p-value remains unusable, for the reason now stated correctly.**
Under zero higher-order structure the macro effect is systematically negative
(pre-correction 1× runs: mean −0.0626, range [−0.0917, −0.0246]), because the
fitted trigram loses. The sign-flip test assumes group differences are symmetric
about zero under the null; because the null is centred below zero, the test rejects
almost always and does not control the false-positive rate for the improvement
question. It is retained only as a descriptive statistic and is not cited as
evidence.

**The earlier "null band" is also withdrawn as confirmatory evidence.** The band
[−0.0917, −0.0246] was the observed minimum and maximum of **four** simulations per
cell. A minimum and maximum from four draws is not a calibrated null band, and the
observed +0.0279 was never legitimately "above the entire band" on that basis. The
structural question is now answered only by the dedicated structural test, which
generates corpora from a fitted first-order null, refits both models, re-runs the
entire evaluation pipeline for every synthetic corpus, and compares the observed
positive effect with the resulting null distribution. Threshold selection and
false-positive evaluation use **disjoint** simulation sets.

**Nor is the test claimed to be valid on its face.** Its assumptions and its
cross-validation dependence still require examination: the null is fitted to finite
data, generated corpora have no artifact or duplicate structure, and the p-value is
conditional on the fitted null model. It is not proof of a linguistic mechanism.

**Calibration results** (112 replicates at 1× size, 6 per cell):

| Cell | mean effect | difference rule | improvement rule | FP-valid |
| --- | --- | --- | --- | --- |
| unigram | −0.0612 | 1.000 | **0.000** | yes |
| markov (fitted null) | −0.0914 | 1.000 | **0.000** | yes |
| position_only | −0.0677 | 1.000 | **0.000** | yes |
| shuffled_real | −0.0383 | 1.000 | **0.000** | yes |
| trigram λ=0.00 | −0.0958 | 1.000 | **0.000** | yes |
| trigram λ=0.35 | −0.0147 | 0.500 | 0.000 | no |
| trigram λ=0.40 | +0.0065 | 0.167 | **0.333** | no |
| trigram λ=0.50 | +0.0579 | 1.000 | **1.000** | no |
| trigram λ=1.00 | +0.5360 | 1.000 | **1.000** | no |
| slot generator | +0.0105 | 1.000 | 1.000 | **no** |

Readings that follow directly:

- The **difference rule rejects in 100% of cells, including every zero-effect cell**.
  It measures "the fitted models differ", not "higher-order structure exists", and
  is therefore not a discovery criterion.
- The **improvement rule is well behaved**: 0/30 false positives across the five
  valid null cells, with power rising 0.000 → 0.333 → 1.000 as λ goes 0.35 → 0.50.
  The observed real-corpus effect sits between the λ = 0.40 and λ = 0.50 cells.
- **Structural test**: observed real-corpus effect **+0.0241** against the fitted
  first-order null (mean −0.0914, 95% threshold −0.0862); **0 of 40** null draws
  reached it → **p = 0.0244**, structural departure supported at α = 0.05. The
  rule's false-positive rate on the independent evaluation half was **0.10 (2/20)** —
  imprecise at that sample size, and consistent with a mildly liberal rule.

This is the evidence that keeps the higher-order claim provisional rather than
established: the test is calibrated at n = 20 per half, not at the scale needed to
resolve a λ ≈ 0.43 effect, and it is conditional on an imperfectly reproducing null.


## 5. Corrected sensitivity terminology and diagnostics

- **Sequence-order, not transcription.** The 2x2x2 sensitivity matrix
  (`scripts/run_sensitivity.py`) varies **sequence-order processing**
  (`reading_order_normalized` vs `physical_as_stored` — two ways of ordering the SAME
  transcription), direction inclusion, and completeness filtering. It does **not** compare
  independent transcription traditions. Headline result is robust across all 8 cells:
  context beats frequency by +17.31 to +22.24 pp and position by +15.03 to +19.31 pp,
  10/0/0 everywhere.
- **Direction diagnostics** (`scripts/run_direction_diagnostics.py`) investigate the
  as-stored vs reading-order-normalized gap. On **identical union-grouped partitions**
  the as-stored advantage is about **1.2 pp** (0.3193 vs 0.3076 top-1), smaller than
  the 2.7–3.1 pp quoted from non-aligned splits:
  - All orientation variants share the **same union-derived group keys**, so neither
    ordering can leak an equivalent sequence across the split.
  - Reversal is reported under **three boundary models**: `asymmetric` (default,
    mean delta −0.0134), `none` (+0.0094), `symmetric` (−0.0046). **No model is
    exactly reversal-invariant**, including `symmetric`. Equalizing the edge terms
    removes the *boundary* asymmetry, but the context model's left term is
    `P(w | prev)` while its right term is `P(next | w)`; these are transpose-related
    and coincide only under detailed balance. The edge-symmetric models do show a
    smaller delta than the default. Interior masks remain exactly mirror-symmetric
    (unit-tested).
  - **Cross-direction transfer is reported with OOV separated.** At *matched* OOV
    (both 0.1815), `normalized_LR_to_RL` scores 0.0697 on shared-vocabulary targets
    against `stored_LR_to_RL` at 0.2808. The earlier claim that vocabulary coverage
    explains the gap is **refuted**: the ordering effect persists at equal OOV.
  - **Position classes are reported separately**: singleton 0.0536, first 0.1604,
    interior 0.3476, last 0.3485 (the previously missing last-position output).
  - Stratification: the gap concentrates in short spans and complete records;
    several sites reverse sign. No single explanation is established; the pipeline
    default is unchanged and is not switched because another order predicts better.
- **Fully nested model selection.** `run_transformer.py` and `run_embeddings.py` select
  configurations per OUTER split: an inner grouped fit/validation split is carved from the
  outer TRAIN partition only; all configurations are scored on inner validation; the
  selected configuration (and stopping epoch) is refit on the full outer train and evaluated
  once on the outer test. No outer-test record enters selection or early stopping
  (unit-tested). The transformer vocabulary is built from fitting data only; unseen signs
  are `<UNK>` context tokens; `<PAD>/<MASK>/<UNK>` are never candidates; OOV test targets
  remain failures and are reported.
- **Same-family transcription sensitivity.** `scripts/run_transcription_sensitivity.py`
  builds an **explicit artifact/inscription-level pair table before gap splitting**.
  Only CISI with exactly one unambiguous record on each side enter the primary
  comparison; multiple primary records are never merged because their sequences
  agree, and one external inscription is never copied onto several primary spans.
  The external side does not inherit the primary's boundary-completeness flags,
  because the external source does not supply them. Two grouping protocols are
  reported, both assigning identical folds to the same matched artifacts:

  | Protocol | Sequences crossing folds per seed | Primary context top-1 | External context top-1 |
  | --- | --- | --- | --- |
  | `artifact_only` | **56–74** | 0.3959 | 0.3834 |
  | `artifact_plus_union_duplicates` | **0** | **0.3016** | **0.3018** |

  Record accounting: **1,841** matched inscriptions, **369** held out per seed,
  **17,342** prediction events, after excluding 534 ambiguous-primary and 1,668
  unmatched-primary CISI. The earlier ~39.6% figure came from artifact-only
  grouping, which leaks 56–74 duplicate sequences across the split and inflates the
  score. Under the strict protocol both transcriptions agree almost exactly and
  context still beats frequency (+19.6 pp) and position (+15.4 pp). Agreement
  between these ICIT-derived sources is NOT independent inter-annotator agreement.
  Raw external sequences are not redistributed.

## 6. Models compared

- **Frequency baseline**; **position baseline** (training-only buckets).
- **Witten-Bell bigram context model** (the reference), **MKN n-grams** (n=1-5).
- **PPMI/SVD** and **skip-gram** embeddings; **tiny masked-sign transformer** — both under
  fully nested selection.

## 7. Claims hierarchy

### Primary supported finding

Immediate context improves sign prediction beyond frequency and position under the
evaluated corpus and preprocessing choices: +17 to +22 pp over frequency and +15 to +19 pp
over position, positive in every tested cell and partition convention.

### Secondary claim — provisional, now supported by a calibrated test

A trigram contains a small amount of predictive information beyond a bigram.
On the full-data estimate (primary fold seed 0) the macro group effect is
**+0.02702 bits/token** (95% CI [+0.01048, +0.04395]), stable across five fold
seeds (range [+0.02592, +0.03147]). The token-weighted effect is **+0.03623** with
95% CI [−0.01173, +0.09211], which **includes zero**, because a few very large
groups dominate token weighting. The two estimands answer different questions and
are never averaged together.

The group-level randomization p-value is **descriptive only and is not cited**
(§4b). The earlier claim that the observed effect sat above a "calibrated null
band" is withdrawn: that band was the min/max of four simulations.

**What now supports the claim** is the structural test (§4b), not a p-value:

- The corrected **improvement rule has a 0.000 false-positive rate across all five
  valid null cells** (unigram, markov, position_only, shuffled_real, trigram λ=0;
  0/30 replicates).
- The **structural test rejects**: the observed real-corpus effect (+0.0241) exceeds
  the fitted first-order null distribution, with 0 of 40 null draws reaching it →
  **p = 0.0244**. The rule's false-positive rate on the independent evaluation half
  was 0.10 (2/20), imprecise at that sample size.
- The difference rule rejects in 100% of cells including zero-effect ones, so it is
  **not** a discovery criterion and is not used as one.

**Limits that keep this provisional.** The p-value is conditional on a null fitted
to finite data that preserves only count, length, vocabulary and first-order
transition structure; generated corpora have no artifact or duplicate structure,
which the real corpus does. Power at the observed effect is modest (λ = 0.40 cell
detects 33% of the time). Larger calibration runs and the remaining corpus sizes
are required before promotion.

### Negative result — learned and compact models

The tested transformer and embedding models do not outperform the bigram under
fully nested evaluation at this corpus size: transformer **23.65% ± 2.87** versus
bigram **29.21% ± 1.87** restoration top-1, losing on 10 of 10 outer splits
(−5.56 pp); embeddings ~9% versus 29.2%. The earlier non-nested transformer figure
of 19.4% is superseded by the nested result.

**Compact structural models are also a negative result** under two comparable
budgets (all models on the same capped subset; all models on the full outer-training
partition, with the HMM capped and saying so):

| Budget | bigram | position exact | position relative | exact+complete | HMM |
| --- | --- | --- | --- | --- | --- |
| matched (cap 600 records) | **7.06** | 8.49 | 8.90 | 8.42 | 8.16 |
| full | **5.97** | 7.51 | 7.29 | 7.56 | 8.09 |

Evaluation takes full records, so artifact identity and grouping survive the runner;
HMM state count and position smoothing strength are selected on inner **grouped**
splits. All models share a training-only vocabulary policy with `<UNK>` always
present, so one unseen sign never discards a sequence's known-token contributions;
OOV rates are reported per model (0.130 matched, 0.020 full). The states remain an
economical latent-state description of positional sequence structure and are not
words, phrases, grammatical roles, or semantic classes.


### Exploratory only

Regional variation; motif variation; sign clusters; segmentation; individual sign
associations; candidate sign variants; per-stratum direction diagnostics.

### Explicit non-claims

- No decipherment. No semantic readings. No identification of language.
- No demonstrated words, morphemes, affixes, or phrases.
- No proof that the system encodes natural language.

## 8. Threats to validity and limitations

- **Likely corpus-size limitation.** ~18k tokens over 713 signs is far below the data
  regime where learned sequence models are competitive; the bigram's dominance is a
  statement about this corpus, not about model classes in general. Phase 8 power analysis
  (`scripts/run_power_analysis.py`) quantifies what the pipeline can detect at this size.
- **Transcription consistency, not agreement.** Same-family comparisons measure pipeline
  consistency within one transcription family. No trusted machine-readable ICIT<->Parpola
  concordance exists, so sign-level cross-tradition agreement is not measurable here.
- **Artificial masks.** Restoration masks are synthetic single-sign deletions, not verified
  archaeological restorations.
- **Component-size imbalance.** The largest connected component (22.1% of tokens) limits
  fold balance; grouped inference inherits this granularity.
- **No linguistic-unit claims.** Entropy differences, pair rankings, and segmenter
  agreement measure statistical structure only. Clusters are not functional groups.