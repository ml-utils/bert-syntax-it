from random import random
from unittest import TestCase
from numpy import log
from compute_model_score import reduce_to_log_product


class TestComputeModelScore(TestCase):
    def test_reduce_to_log_product(self):

        a = random()
        b = random()
        actual = reduce_to_log_product([a, b])
        expected = log(a) + log(b)

        self.assertEqual(actual, expected)

