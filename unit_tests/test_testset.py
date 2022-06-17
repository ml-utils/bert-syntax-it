from unittest import TestCase

from linguistic_tests.testset import ScoringMeasures


class TestTestset(TestCase):
    def test_ScoringMeasures_eq(self):

        assert "LP" == ScoringMeasures.LP
        assert "LP" == ScoringMeasures.LP.name
        assert ScoringMeasures.LP == ScoringMeasures.LP.name
        assert "PenLP" == ScoringMeasures.PenLP
        assert "PenLP" == ScoringMeasures.PenLP.name
        assert ScoringMeasures.PenLP == ScoringMeasures.PenLP.name
