from unittest import TestCase

import pytest
from linguistic_tests.run_sprouse_tests import create_test_jsonl_files_tests
from linguistic_tests.run_sprouse_tests import get_out_dir
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
    def test_get_out_dir(self):
        get_out_dir()
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
