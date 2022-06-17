from unittest import TestCase

import pytest
from linguistic_tests.run_sprouse_tests import get_dd_score
from linguistic_tests.run_sprouse_tests import get_pickle_filename
from linguistic_tests.run_sprouse_tests import main
from linguistic_tests.run_sprouse_tests import plot_all_phenomena
from linguistic_tests.run_sprouse_tests import plot_results
from linguistic_tests.run_sprouse_tests import print_testset_results
from linguistic_tests.run_sprouse_tests import run_sprouse_test_helper
from linguistic_tests.run_sprouse_tests import save_scored_testsets
from linguistic_tests.run_sprouse_tests import score_sprouse_testset
from linguistic_tests.run_sprouse_tests import score_sprouse_testsets


class TestRunSprouseTests(TestCase):
    @pytest.mark.skip("todo")
    def test_main(self):
        main()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_plot_all_phenomena(self):
        plot_all_phenomena()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_plot_results(self):
        plot_results()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_sprouse_test_helper(self):
        run_sprouse_test_helper()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_sprouse_tests(self):
        score_sprouse_testsets()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_score_sprouse_testset(self):
        score_sprouse_testset()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_save_scored_testsets(self):
        save_scored_testsets()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_testset_results(self):
        print_testset_results()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_pickle_filename(self):
        get_pickle_filename()
        raise NotImplementedError

    def test_get_dd_score(self):
        sentences_scores = [1.1, 0.9, 0.8, -0.5]
        dd_score = get_dd_score(sentences_scores)
        assert dd_score > 1

        sentences_scores = [1, 0.3, 1, -0.7]
        dd_score = get_dd_score(sentences_scores)
        assert dd_score >= 1

        sentences_scores = [0.6, 1, -0.6, -0.8]
        dd_score = get_dd_score(sentences_scores)
        assert dd_score > 0.5

        sentences_scores = [0.6, 0.3, 0.1, -1.1]
        dd_score = get_dd_score(sentences_scores)
        assert dd_score > 0.7

    def test_sentences_ordering(self):
        from linguistic_tests.lm_utils import SprouseSentencesOrder

        assert SprouseSentencesOrder.SHORT_NONISLAND == 0
        assert SprouseSentencesOrder.LONG_NONISLAND == 1
        assert SprouseSentencesOrder.SHORT_ISLAND == 2
        assert SprouseSentencesOrder.LONG_ISLAND == 3
