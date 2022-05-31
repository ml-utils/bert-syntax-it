from random import random
from unittest import TestCase

import pytest
from linguistic_tests.compute_model_score import count_accurate_in_example
from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.compute_model_score import get_sentence_score_JHLau
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import reduce_to_log_product
from linguistic_tests.compute_model_score import run_testset
from numpy import log


class TestComputeModelScore(TestCase):
    @pytest.mark.skip("todo")
    def test_count_accurate_in_example(self):
        count_accurate_in_example()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_example_scores(self):
        get_example_scores()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_score_JHLau(self):
        get_sentence_score_JHLau()
        raise NotImplementedError

    def test_perc(self):
        self.assertEqual(perc(10, 20), 50)
        self.assertEqual(perc(1, 100), 1)

    def test_reduce_to_log_product(self):

        a = random()
        b = random()
        actual = reduce_to_log_product([a, b])
        expected = log(a) + log(b)

        self.assertEqual(actual, expected)

    @pytest.mark.skip("todo")
    def test_run_testset(self):
        run_testset()
        raise NotImplementedError
