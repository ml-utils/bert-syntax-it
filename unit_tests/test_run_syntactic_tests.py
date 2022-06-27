from unittest import TestCase

import pytest

from src.linguistic_tests.run_test_design import (
    rescore_testsets_and_save_pickles,
)


class TestRunSyntacticTests(TestCase):
    @pytest.mark.skip("todo")
    def test_rescore_testsets_and_save_pickles(self):
        rescore_testsets_and_save_pickles()
        raise NotImplementedError
