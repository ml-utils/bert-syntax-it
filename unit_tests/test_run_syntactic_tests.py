from unittest import TestCase

import pytest
from linguistic_tests.run_minimal_pairs_test_design import run_blimp_en
from linguistic_tests.run_minimal_pairs_test_design import run_tests_for_model_type


class TestRunSyntacticTests(TestCase):
    @pytest.mark.skip("todo")
    def test_run_blimp_en(self):
        run_blimp_en()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_for_model_type(self):
        run_tests_for_model_type()
        raise NotImplementedError
