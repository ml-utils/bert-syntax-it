from unittest import TestCase

import pytest
from linguistic_tests.run_syntactic_tests import run_blimp_en
from linguistic_tests.run_syntactic_tests import run_tests_for_model_type
from linguistic_tests.run_syntactic_tests import run_tests_it


class TestRunSyntacticTests(TestCase):
    @pytest.mark.skip("todo")
    def test_run_blimp_en(self):
        run_blimp_en()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_it(self):
        run_tests_it()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_for_model_type(self):
        run_tests_for_model_type()
        raise NotImplementedError
