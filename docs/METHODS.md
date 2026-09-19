# Arthānveṣaṇa — Methods and Claims

Preregistered-style statement of design, evaluation protocol, and claims boundary for the
computational study in this repository. All numbers below are reproduced from the reports under
`outputs/` and the conventions documented in `README.md` and `data/PROVENANCE.md`; no method is
described here that the pipeline does not implement.

## 1. Objective and scope

The objective is the **structural characterization of the Sindhu-Sarasvatī (Indus) sign
sequences**: distributional, sequential, and positional regularities, measured against explicit
null models. This study is **not a decipherment**. No sign-to-phoneme, sign-to-morpheme, or
sign-to-meaning mapping is proposed, tested, or reported anywhere in the pipeline. Models assign
probability mass to sign identities only; metrics such as top-1 restoration accuracy describe how
well a model predicts a masked *sign code*, not a reading of the inscription. Every task operates
on the ICIT numeric sign encoding exactly as delivered by the upstream corpus.

## 2. Data

The analysis corpus is pinned, not floating: `sanitized_corpus.json` from upstream repository
`ShaktiOSindia/indus-sign-regimes-deposit`, commit `e48b3ec1e90f368079f5126790613cded6f6f56c`
(2026-08-12), SHA-256 `345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68`,
re-verified by `scripts/build_corpus.py` on every rebuild. Published properties: **5,704
catalogued inscriptions** (5,536 analyzable after excluding 168 all-`000` rows), **18,065 sign
tokens**, **713 distinct signs**, 77 sites, 32 artefact types. The ICIT code `000` is a
missing-data placeholder, not a sign; it is removed ("gated") before analysis and is never a
prediction candidate.

Reading order follows upstream `chr_lib` conventions: symbols are stored physically
left-to-right; sequences are used as stored for direction `L/R` and reversed for `R/L`. One
documented deviation from upstream is adopted deliberately: upstream reverses stored order for
*every* non-`L/R` direction (`-`, `NR`, `BUS`, `SYM`, `T/B`), whereas this project reverses only
unambiguous `R/L` and flags the remainder with `reading_order_known=False`, excluding them from
direction-sensitive analyses (positional extremes use only the 3,154 known-direction records
with both ends complete).

Licensing is kept separable by construction: the raw corpus (upstream **GPL-3.0**) is not
redistributed in this repository; derived artifacts that encode sign sequences remain GPL-3.0 if
redistributed; this project's code and analysis are **Apache-2.0** and run on any
user-supplied corpus. Aggregate statistics are treated as facts/measurements, citable with
attribution. This is a transparency statement, not legal advice.

## 3. Evaluation protocol (pre-registered-style)

All restoration experiments share one protocol, fixed in code before model comparison:

- **Grouped holdout splits.** Records are connected by artifact, inscription, and exact-sequence
  identity (union-find over those keys in `split_records`) and whole connected groups are
  assigned to train or test (80/20, seeded). Duplicate sequences can therefore never straddle the
  split, and per-field train/test overlap counts are reported as leakage diagnostics.
- **Gap-split spans.** Known-direction records are split at missing signs (`000`); models are fit
  and evaluated on contiguous observed spans. Artificial span starts are model boundaries, not
  inferred inscription boundaries; a start-incomplete span's first position is predicted from the
  unigram prior, not from a spurious start token.
- **Shared task.** Masked **single-sign restoration**: every observed test position is masked in
  turn; all models rank the training sign vocabulary for the same masked positions.
  Out-of-vocabulary targets **count as failures** (reciprocal rank zero) rather than being
  excluded; mean OOV target fractions are 2.0–3.0% across protocols.
- **Variability reporting.** Each comparison runs over 10 seeded grouped splits. Reported means ±
  SD describe **split variability, not confidence intervals** over independent experiments;
  repeated overlapping splits are dependent by construction.
- **Null models.** Within-span shuffles (20 replicates, preserving lengths and unigram counts)
  provide the order null for entropy and segmentation; label permutations provide the null for
  cluster–role alignment.

Evaluation is additionally run under four partition conventions (readable spans, complete
inscriptions, unique sequences, complete unique sequences) and as leave-one-site-out for sites
with ≥100 eligible inscriptions, with linked artifacts and duplicate groups purged from training.

## 4. Models compared

- **Frequency baseline**: training sign counts, no context or position.
- **Position baseline**: training-only buckets (span length, zero-based index, completeness
  flags); unseen buckets fall back to global frequency.
- **Witten-Bell bigram context model** (`restore.py`): the reference ceiling. Masked positions
  are scored by multiplying forward and backward bigram pass distributions over the available
  left/right context.
- **Modified Kneser–Ney n-grams** (n = 1–5), evaluated on held-out perplexity.
- **PPMI/SVD embeddings** (0.75 context smoothing, truncated SVD) and **skip-gram** (gensim
  Word2Vec, sg=1, negative=5, fixed seeds), scored for restoration by reconstructed association
  with observed in-window neighbors; also clustered (k-means sweep, hierarchical) and tested
  against permutation nulls.
- **Tiny masked-sign transformer** (1–2 layers, dim 32–64, ≤146,779 parameters), mask-one-sign
  objective, early stopping on a grouped validation split, identical splits and masked positions
  as the bigram.

## 5. Confirmatory claims

These are supported by the paired, multi-seed grouped-split design (paired differences with
wins/ties/losses reported per split):

1. **Context beats frequency and position.** On gap-split spans, the bigram reaches top-1
   29.21% ± 1.87% versus 10.82% (frequency) and 12.84% (position); mean paired differences are
   +18.39 and +16.38 percentage points, **10/0/0 in the bigram's favor**, and the ordering holds
   under all four partition conventions (top-1 28.92–29.56%, always 10/0/0).
2. **The bigram beats learned representations at this corpus size.** PPMI/SVD (8.66% ± 1.20%)
   and skip-gram (9.15% ± 0.65%) lose all 10 paired splits (0/0/10); the transformer reaches
   19.36% ± 2.13% with ~113k parameters and also loses all 10 splits (0/0/10, mean paired
   difference −9.86 points). Smoothing upgrades do not close the gap: MKN bigram perplexity 63.06
   versus 73.66 for Witten-Bell, and higher-order MKN plateaus at ~59.8.
3. **Sign order is structured above shuffle nulls.** Observed conditional entropy
   H(s2|s1) = 3.339 bits versus 4.681 ± 0.012 bits under 20 within-span shuffles (unigram
   entropy 6.796 bits is shuffle-invariant by construction). Positional bias is real and
   reported descriptively (e.g. signs 545/027/201 beginning-biased at 1.00; 167/161 end-biased
   at 1.00), without any affix interpretation.

## 6. Exploratory observations (hypothesis-generating only)

- **Motif stratification.** On the metadata sidecar (41.6% corpus coverage; motif known for
  1,514 inscriptions), single-split restoration differs across motif groups (bull 31.85%,
  other-known 27.11%, unknown 26.38% context top-1, with differing OOV rates). These are
  descriptive subsets with high split noise, not independent experiments.
- **Cluster–role alignment.** Embedding clusters sit **at** the permutation null (k-means purity
  0.630 versus null mean 0.624, p = 0.180 PPMI / p = 0.135 skip-gram); cross-method ARI is 0.067
  and split-stability ARI is 0.109–0.204. No stable functional grouping is claimed.
- **Regional variation.** Leave-one-site-out context top-1 ranges from 22.17% (Harappa) to
  31.55% (Lothal); site differences confound vocabulary coverage, training size, and artifact
  mix, and **do not establish dialects or languages**.
- **Transcription audit signals.** Same-family agreement (exact sequence 0.7095 over 2,375
  inscriptions) is dominated by a small set of edge codes (700, 033, 032, 034) dropped by one
  source — an editorial-convention difference, not random noise.

## 7. Threats to validity and non-claims

- **Corpus size ceiling.** ~18k tokens over 713 signs is far below the data regime where learned
  sequence models are competitive; the bigram's dominance is a statement about this corpus, not
  about model classes in general.
- **Transcription consistency, not agreement.** The 0.992 direction cross-check and same-family
  sequence agreement measure pipeline consistency **within one transcription family**; they are
  not inter-annotator agreement. Against the independent mayig CISI hand-transcription, only
  catalog overlap (179/179 seals) and length agreement (~0.84–0.86) are computable — **no trusted
  machine-readable ICIT↔Parpola concordance exists**, so sign-level cross-tradition agreement is
  not measurable here.
- **Artificial masks.** Restoration masks are synthetic single-sign deletions within observed
  spans, not verified archaeological restorations; observed-span edges are not genuine
  inscription edges where boundaries are damaged.
- **No linguistic-unit claims.** Entropy differences, pair rankings (530 pairs at exploratory
  BH q ≤ 0.05), and segmenter agreement (boundary F1 0.443 real vs 0.416 shuffled) measure
  statistical structure and algorithmic consistency; none of them establishes morphemes, words,
  affixes, or that the signs encode language at all.
- **Clusters are not functional groups** (Section 6); cluster bootstrap ranges describe
  resampling variability conditional on a split, not bias-corrected uncertainty.
