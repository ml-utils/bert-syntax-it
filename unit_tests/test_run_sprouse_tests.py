from unittest import TestCase

import pytest
from linguistic_tests.run_sprouse_tests import create_test_jsonl_files_tests
from linguistic_tests.run_sprouse_tests import get_dd_score
from linguistic_tests.run_sprouse_tests import get_sentence_from_row
from linguistic_tests.run_sprouse_tests import main
from linguistic_tests.run_sprouse_tests import plot_all_phenomena
from linguistic_tests.run_sprouse_tests import plot_results
from linguistic_tests.run_sprouse_tests import read_sentences_item
from linguistic_tests.run_sprouse_tests import run_sprouse_test
from linguistic_tests.run_sprouse_tests import run_sprouse_test_helper
from linguistic_tests.run_sprouse_tests import run_sprouse_tests
from linguistic_tests.run_sprouse_tests import write_sentence_item
from linguistic_tests.run_sprouse_tests import write_sentence_pair


class TestRunSprouseTests(TestCase):
    @pytest.mark.skip("todo")
    def test_create_test_jsonl_files_tests(self):
        create_test_jsonl_files_tests()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_from_row(self):
        get_sentence_from_row()
        raise NotImplementedError

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
    def test_read_sentences_item(self):
        read_sentences_item()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_sprouse_test(self):
        run_sprouse_test()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_sprouse_test_helper(self):
        run_sprouse_test_helper()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_sprouse_tests(self):
        run_sprouse_tests()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_write_sentence_item(self):
        write_sentence_item()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_write_sentence_pair(self):
        write_sentence_pair()
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
