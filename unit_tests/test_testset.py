from unittest import TestCase

import numpy as np
import pandas as pd
from linguistic_tests.testset import ScoringMeasures


class TestTestset(TestCase):
    def test_ScoringMeasures_eq(self):

        assert "LP" == ScoringMeasures.LP
        assert "LP" == ScoringMeasures.LP.name
        assert ScoringMeasures.LP == ScoringMeasures.LP.name
        assert "PenLP" == ScoringMeasures.PenLP
        assert "PenLP" == ScoringMeasures.PenLP.name
        assert ScoringMeasures.PenLP == ScoringMeasures.PenLP.name

    def test_likert_scale_conversion(self):
        scores_stype1 = [1.1, 3.3, 5.5, 7.7, 9.9]
        scores_stype2 = [2.2, 4.4, 6.6, 8.8, 10.1]
        all_scores = scores_stype1 + scores_stype2

        # <class 'pandas.core.arrays.categorical.Categorical'>, type(likert_bins)=<class 'numpy.ndarray'>
        all_scores_to_likert: pd.core.arrays.categorical.Categorical
        all_scores_to_likert, likert_bins = pd.cut(
            all_scores,
            bins=7,
            labels=np.arange(start=1, stop=8),
            retbins=True,
        )
        print("\n")
        print(f"{sorted(all_scores)=}")
        print(f"{sorted(all_scores_to_likert.to_numpy())=}")
        print(  # f"{type(all_scores_to_likert.get_values())=},"
            f"{type(all_scores_to_likert)=}, {type(likert_bins)=}"
        )
        print(f"{sorted(all_scores_to_likert)=}")
        print(f"{likert_bins=}")
        scores_stype1_to_likert = pd.cut(
            scores_stype1,
            bins=likert_bins,
            labels=np.arange(start=1, stop=8),
        )
        print(f"{scores_stype1_to_likert=}")
