from unittest import TestCase

import pytest
from linguistic_tests.compute_model_score import get_unparsed_example_scores
from linguistic_tests.file_utils import get_pickle_filename
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.plots_and_prints import _print_example
from linguistic_tests.plots_and_prints import _print_testset_results
from linguistic_tests.plots_and_prints import plot_results
from linguistic_tests.run_factorial_test_design import _get_dd_score
from linguistic_tests.run_factorial_test_design import main
from linguistic_tests.run_factorial_test_design import score_factorial_testsets
from linguistic_tests.run_minimal_pairs_test_design import score_factorial_testset
from linguistic_tests.testset import save_scored_testsets
from tqdm import tqdm


class TestRunSprouseTests(TestCase):
    @pytest.mark.skip("todo")
    def test_main(self):
        main()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_plot_results(self):
        plot_results()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_sprouse_tests(self):
        score_factorial_testsets()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_score_sprouse_testset(self):
        score_factorial_testset()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_save_scored_testsets(self):
        save_scored_testsets()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_testset_results(self):
        _print_testset_results()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_pickle_filename(self):
        get_pickle_filename()
        raise NotImplementedError

    def test_get_dd_score(self):
        sentences_scores = [1.1, 0.9, 0.8, -0.5]
        dd_score = _get_dd_score(sentences_scores)
        assert dd_score > 1

        sentences_scores = [1, 0.3, 1, -0.7]
        dd_score = _get_dd_score(sentences_scores)
        assert dd_score >= 1

        sentences_scores = [0.6, 1, -0.6, -0.8]
        dd_score = _get_dd_score(sentences_scores)
        assert dd_score > 0.5

        sentences_scores = [0.6, 0.3, 0.1, -1.1]
        dd_score = _get_dd_score(sentences_scores)
        assert dd_score > 0.7

    def test_sentences_ordering(self):
        assert SprouseSentencesOrder.SHORT_NONISLAND == 0
        assert SprouseSentencesOrder.LONG_NONISLAND == 1
        assert SprouseSentencesOrder.SHORT_ISLAND == 2
        assert SprouseSentencesOrder.LONG_ISLAND == 3

    # former deprecated method
    # todo: use in int/unit tests to comprare outcome with newer version of this method
    def legacy_run_sprouse_test_helper(
        model_type,
        model,
        tokenizer,
        device,
        testset,
        examples_in_sprouse_format=True,
        sentence_ordering=SprouseSentencesOrder,
        max_examples=50,
        verbose=False,
    ):
        sent_ids = []
        sentences_per_example = 4

        if len(testset["sentences"]) > 50:
            testset["sentences"] = testset["sentences"][:max_examples]

        examples_count = len(testset["sentences"])
        lp_short_nonisland_average = 0
        lp_long_nonisland_avg = 0
        lp_short_island_avg = 0
        lp_long_island_avg = 0
        penlp_short_nonisland_average = 0
        DDs_with_lp = []
        DDs_with_pen_lp = []

        for example_idx, example_data in enumerate(tqdm(testset["sentences"])):

            (
                lps,
                pen_lps,
                lls,
                penlls,
                pen_sentence_log_weights,
                sentence_log_weights,
                sentences,
            ) = get_unparsed_example_scores(
                device,
                example_data,
                model,
                model_type,
                sent_ids,
                tokenizer,
                sentences_per_example,
                sprouse_format=examples_in_sprouse_format,
            )
            if verbose:
                _print_example(sentences, sentence_ordering)

            DDs_with_lp.append(_get_dd_score(lps, sentence_ordering))
            DDs_with_pen_lp.append(_get_dd_score(pen_lps, sentence_ordering))
            lp_short_nonisland_average += lps[sentence_ordering.SHORT_NONISLAND]
            lp_long_nonisland_avg += lps[sentence_ordering.LONG_NONISLAND]
            lp_short_island_avg += lps[sentence_ordering.SHORT_ISLAND]
            lp_long_island_avg += lps[sentence_ordering.LONG_ISLAND]
            # todo: do also for penlp, ll and penll
            penlp_short_nonisland_average += pen_lps[0]

        lp_short_nonisland_average /= examples_count
        lp_long_nonisland_avg /= examples_count
        lp_short_island_avg /= examples_count
        lp_long_island_avg /= examples_count
        penlp_short_nonisland_average /= examples_count
        lp_averages = [
            lp_short_nonisland_average,
            lp_long_nonisland_avg,
            lp_short_island_avg,
            lp_long_island_avg,
        ]

        correc_count_DD_lp = len([x for x in DDs_with_lp if x > 0])
        accuracy_DD_lp = correc_count_DD_lp / len(DDs_with_lp)
        print(f"accuracy with DDs_with_lp: {accuracy_DD_lp}")
        correc_count_DD_penlp = len([x for x in DDs_with_pen_lp if x > 0])
        accuracy_DD_penlp = correc_count_DD_penlp / len(DDs_with_lp)
        print(f"accuracy with DDs_with_penlp: {accuracy_DD_penlp}")

        # todo: also for ll, penll

        # todo: also return penlp, ll, penll
        return lp_averages
