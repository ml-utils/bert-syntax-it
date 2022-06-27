from unittest import TestCase

import pytest

from src.linguistic_tests.notebook import interactive_mode
from src.linguistic_tests.notebook import main
from src.linguistic_tests.plots_and_prints import _print_example
from src.linguistic_tests.plots_and_prints import get_perc
from src.linguistic_tests.plots_and_prints import print_detailed_sentence_info
from src.linguistic_tests.run_test_design import run_tests_goldberg
from src.linguistic_tests.run_test_design import run_tests_lau_et_al


class TestNotebook(TestCase):
    @pytest.mark.skip("todo")
    def test_run_tests_goldberg(self):
        run_tests_goldberg()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_lau_et_al(self):
        run_tests_lau_et_al()
        raise NotImplementedError

    def test_get_perc(self):
        self.assertEqual("10.0 %", get_perc(10, 100))
        self.assertEqual("0.1 %", get_perc(1, 1000))

    @pytest.mark.skip("todo")
    def test_interactive_mode(self):
        interactive_mode()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_main(self):
        main()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_detailed_sentence_info(self):
        print_detailed_sentence_info()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_example(self):
        _print_example()
        raise NotImplementedError
