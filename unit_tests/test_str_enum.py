from unittest import TestCase

from src.linguistic_tests.testset import ScoringMeasures


class TestStrEnum(TestCase):
    def test_scoring_measures(self):

        scoring_measure = ScoringMeasures.PenLP
        assert str(scoring_measure) == "PenLP"
