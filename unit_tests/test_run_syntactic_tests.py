from unittest import TestCase

import pytest
from linguistic_tests.run_test_design import (
    rescore_testsets_and_save_pickles,
)
from linguistic_tests.run_test_design import run_tests_for_model_type


class TestRunSyntacticTests(TestCase):
    @pytest.mark.skip("todo")
    def test_rescore_testsets_and_save_pickles(self):
        rescore_testsets_and_save_pickles()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_for_model_type(self):
        run_tests_for_model_type()
        raise NotImplementedError
