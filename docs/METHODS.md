# Arthānveṣaṇa — Retrospective Methods and Claims Specification

Retrospective methods and claims specification for the computational study in this
repository. This document freezes the protocol, terminology, and claims boundary for
subsequent untouched experiments; it was written AFTER the initial analyses, so it is a
retrospective specification, not a preregistration. All numbers are reproduced from the
reports under `outputs/` and the conventions in `README.md` and `data/PROVENANCE.md`; no
method is described here that the pipeline does not implement.

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

1. Connected components are indivisible groups assigned to folds by seeded greedy balancing
   (primary: token count; secondary: span count).
2. MKN bigram and trigram are trained on four folds; every token of the held-out fold is
   scored under both models; each record receives **exactly one out-of-fold prediction**.
3. Token differences `d = log2 P_trigram - log2 P_bigram` are aggregated **within each
   connected group** before inference.
4. **Primary estimand**: macro (unweighted) mean of group means, with a group-level
   sign-flip randomization test (20,000 permutations, `p = (exceedances+1)/(permutations+1)`,
   never 0) and a 95% cluster bootstrap over groups (10,000 replicates).
5. **Secondary estimand** (reported as secondary): token-weighted mean, uncertainty by
   resampling whole groups.

Result on this corpus: **macro group effect +0.0279 bits/token, 95% CI [+0.0112, +0.0446]**
(2,262 groups, zero fold-leakage). Secondary token-weighted +0.0440 bits/token with 95% CI
[−0.0177, +0.1187] — **including zero**, because a few very large groups dominate token
weighting. The corrected effect is roughly half the invalid token-level estimate (+0.065).

### 4b. The randomization p-value is NOT usable (false-positive calibration)

The synthetic calibration (`scripts/run_power_analysis.py`) falsified the inferential use of
the group-level randomization test. Over 32 simulated corpora at 1x size, **every cell —
including all 16 zero-higher-order-effect corpora — returned the floor p-value 0.001**, and
under zero higher-order structure the macro effect was systematically **negative**
(mean −0.0626, range [−0.0917, −0.0246]) rather than centred on zero.

Cause: a modified Kneser-Ney trigram carries more parameters than the bigram, so with no
genuine higher-order structure it *loses* held-out log loss. The sign-flip test assumes group
differences are symmetric about zero under the null; because the null is centred near −0.06,
the test rejects almost always and does not control the false-positive rate.

Consequences, applied throughout this document:

- **Do not cite the randomization p-value (0.0016) as evidence.** It is anti-conservative.
- The **point estimate remains informative**, but it must be read **against the calibrated
  null band** instead. Under zero higher-order structure the macro effect lies in
  **[−0.0917, −0.0246]** (1x size); the observed **+0.0279 sits above that entire band**, which
  is the defensible basis for a small positive higher-order component.
- Detection rate is 1.000 in every cell, so it measures "the test always rejects", not power.
  Redesigning the test (e.g. a null calibrated on the bigram-vs-trigram parameter penalty, or
  a paired comparison against matched null corpora) is required before any p-value is quoted.

## 5. Corrected sensitivity terminology and diagnostics

- **Sequence-order, not transcription.** The 2x2x2 sensitivity matrix
  (`scripts/run_sensitivity.py`) varies **sequence-order processing**
  (`reading_order_normalized` vs `physical_as_stored` — two ways of ordering the SAME
  transcription), direction inclusion, and completeness filtering. It does **not** compare
  independent transcription traditions. Headline result is robust across all 8 cells:
  context beats frequency by +17.31 to +22.24 pp and position by +15.03 to +19.31 pp,
  10/0/0 everywhere.
- **Direction diagnostics** (`scripts/run_direction_diagnostics.py`) investigate the
  ~2.7-3.1 pp as-stored advantage:
  - Global reversal with start/end flags swapped is nearly aggregate-invariant
    (mean top-1 delta -0.0028), but the boundary probe confirms a real asymmetry: the
    span START uses the `<S>` start distribution while the END gets a uniform
    no-evidence term (`restore._mask_distributions`). Interior masks are exactly
    mirror-symmetric (unit-tested).
  - Cross-direction transfer is poor under high OOV (normalized-direction training
    5.7-5.9% top-1) and much better as-stored/reversed (22.9-28.4%), reflecting
    vocabulary coverage rather than order alone.
  - Stratification: the normalized-minus-stored gap concentrates in short spans
    (len<=3: -0.0246) and complete records; several sites (Kalibangan +0.0295,
    Dholavira +0.0147) reverse sign. No single explanation is established; the
    pipeline default is unchanged.
- **Fully nested model selection.** `run_transformer.py` and `run_embeddings.py` select
  configurations per OUTER split: an inner grouped fit/validation split is carved from the
  outer TRAIN partition only; all configurations are scored on inner validation; the
  selected configuration (and stopping epoch) is refit on the full outer train and evaluated
  once on the outer test. No outer-test record enters selection or early stopping
  (unit-tested). The transformer vocabulary is built from fitting data only; unseen signs
  are `<UNK>` context tokens; `<PAD>/<MASK>/<UNK>` are never candidates; OOV test targets
  remain failures and are reported.
- **Same-family transcription sensitivity.** `scripts/run_transcription_sensitivity.py`
  matches the primary corpus to the external ICIT transcription by CISI (one-to-one,
  exclusions reported), builds IDENTICAL artifact-level test sets from stable catalog ids,
  and evaluates frequency/position/context under both transcriptions. Agreement between
  these ICIT-derived sources is NOT independent inter-annotator agreement. Raw external
  sequences are not redistributed.

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

### Secondary claim, pending corrected inference

A trigram may contain a small amount of predictive information beyond a bigram. The evidence
is **the point estimate against the calibrated null band**, NOT a p-value: the macro group
effect is +0.028 bits/token with 95% CI [+0.0112, +0.0446], and synthetic calibration places
the zero-higher-order-effect macro effect in [−0.0917, −0.0246], so the observed value sits
above the entire null band. The group-level randomization p-value (0.0016) is
**anti-conservative and is not cited** (see §4b); the token-weighted interval includes zero.
Do not promote this claim until the test is redesigned and the result replicates on
alternative fold seeds and corpus variants.

### Null calibration (new)

`scripts/run_power_analysis.py` is now a required part of the evidence chain: no higher-order
claim may be reported without a matched zero-effect null band from the same pipeline.

### Negative result

The tested transformer and embedding models do not outperform the bigram under fully
nested evaluation at this corpus size (earlier non-nested runs: transformer 19.4%,
embeddings ~9% vs bigram 29.2%; nested results regenerate into `outputs/transformer` and
`outputs/embeddings`).

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