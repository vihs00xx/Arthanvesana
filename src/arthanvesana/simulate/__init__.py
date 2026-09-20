"""Grouped cross-fitting pipeline shared by the inference runner and the
power analysis. Every synthetic corpus goes through the same path as the real
corpus: connected components -> fold assignment -> MKN cross-fitting ->
group-level aggregation."""
from arthanvesana.simulate.generators import (  # noqa: F401
    empirical_profile,
    gen_hmm_slots,
    gen_markov,
    gen_position_only,
    gen_shuffled_real,
    gen_trigram_mixture,
    gen_unigram,
)
from arthanvesana.simulate.pipeline import (  # noqa: F401
    crossfit_effect,
    records_from_seqs,
)
