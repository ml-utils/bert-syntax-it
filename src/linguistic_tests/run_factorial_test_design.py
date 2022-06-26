from linguistic_tests.lm_utils import assert_almost_equal
from linguistic_tests.lm_utils import SprouseSentencesOrder
from scipy.stats import chi2


# todo: move to lm_utils
# todo: move to lm_utils

# todo: parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file


def get_pvalue_with_likelihood_ratio_test(full_model_ll, reduced_model_ll):
    likelihood_ratio = 2 * (reduced_model_ll - full_model_ll)
    p = chi2.sf(likelihood_ratio, 1)  # L2 has 1 DoF more than L1
    return p


def _get_dd_score(sentences_scores, sentences_ordering=SprouseSentencesOrder):
    a_short_nonisland_idx = sentences_ordering.SHORT_NONISLAND
    b_long_nonisland = sentences_ordering.LONG_NONISLAND
    c_short_island = sentences_ordering.SHORT_ISLAND
    d_long_island = sentences_ordering.LONG_ISLAND
    example_lenght_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[b_long_nonisland]
    )
    example_structure_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[c_short_island]
    )
    example_total_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[d_long_island]
    )
    example_island_effect_with_lp = example_total_effect_with_lp - (
        example_lenght_effect_with_lp + example_structure_effect_with_lp
    )
    example_dd_with_lp = example_structure_effect_with_lp - (
        sentences_scores[b_long_nonisland] - sentences_scores[d_long_island]
    )
    example_dd_with_lp *= -1
    assert_almost_equal(example_island_effect_with_lp, example_dd_with_lp)
    return example_dd_with_lp


# def plot_all_phenomena(phenomena_names, lp_avg_scores):
#     for idx, phenomenon in enumerate(phenomena_names):
#         plot_results(phenomenon, lp_avg_scores[idx], ScoringMeasures.LP.name)
