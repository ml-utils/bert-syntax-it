from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.stats import zmap

from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.testset import ERROR_LP
from src.linguistic_tests.testset import Example
from src.linguistic_tests.testset import ScoringMeasures
from src.linguistic_tests.testset import Sentence
from src.linguistic_tests.testset import TestSet
from src.linguistic_tests.testset import TypedSentence


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
        print(f"{sorted(all_scores)}")
        print(f"{sorted(all_scores_to_likert.to_numpy())}")
        print(  # f"{type(all_scores_to_likert.get_values())},"
            f"{type(all_scores_to_likert)}, {type(likert_bins)}"
        )
        print(f"{sorted(all_scores_to_likert)}")
        print(f"{likert_bins}")
        scores_stype1_to_likert = pd.cut(
            scores_stype1,
            bins=likert_bins,
            labels=np.arange(start=1, stop=8),
        )
        print(f"{scores_stype1_to_likert}")

        # AttributeError: 'Categorical' object has no attribute 'values'
        # scores_stype1_to_likert = scores_stype1_to_likert.values
        # all_scores_to_likert = all_scores_to_likert.values

        # nb a categorical is a series: pd.Series(["a", "b", "c", "a"], dtype="category")
        # AttributeError: 'Categorical' object has no attribute 'apply'
        # scores_stype1_to_likert[['cc']]
        # scores_stype1_to_likert = scores_stype1_to_likert.apply(
        #     lambda col: pd.Categorical(col).codes)
        # all_scores_to_likert = all_scores_to_likert.apply(
        #     lambda col: pd.Categorical(col).codes)

        scores_stype1_to_likert = np.asarray(scores_stype1_to_likert)
        all_scores_to_likert = np.asarray(all_scores_to_likert)
        print(f"{scores_stype1_to_likert}")
        print(f"{all_scores_to_likert}")

        zscores_likert_categorical = zmap(
            scores_stype1_to_likert,
            all_scores_to_likert,
        )
        # zscores_likert: List[float] = zscores_likert_categorical.values
        # zscores_likert2 = zscores_likert_categorical[['cc']].apply(lambda col: pd.Categorical(col).codes)

        print(f"{zscores_likert_categorical}")

    def test_get_testset_sentence_types(self):
        testset = get_basic_testset()
        stypes = testset.get_sentence_types()
        assert 2 == len(stypes)
        assert SentenceNames.SHORT_NONISLAND in stypes
        assert SentenceNames.LONG_ISLAND in stypes

    def test_get_scoring_measures(self):
        testset = get_basic_testset()
        scoring_measures = testset.get_scoring_measures()
        assert 1 == len(scoring_measures)
        assert ScoringMeasures.LP in scoring_measures

    def test_get_acceptable_sentence_types(self):
        testset = get_basic_testset()
        acc_stypes = testset.get_acceptable_sentence_types()
        assert 1 == len(acc_stypes)
        assert SentenceNames.SHORT_NONISLAND in acc_stypes
        assert SentenceNames.LONG_ISLAND not in acc_stypes
        assert SentenceNames.SENTENCE_BAD not in acc_stypes

    def test_example_get_type_of_unacceptable_sentence(self):
        example = get_basic_example()
        unacc_stype = example.get_type_of_unacceptable_sentence()
        assert SentenceNames.LONG_ISLAND == unacc_stype

    def test_example_get_sentence_types(self):
        example = get_basic_example()
        stypes = example.get_sentence_types()
        assert 2 == len(stypes)
        assert SentenceNames.SHORT_NONISLAND in stypes
        assert SentenceNames.LONG_ISLAND in stypes

    def test_sentence_get_score(self):
        sent = get_basic_sentence("The pen is on the table")
        assert ERROR_LP != sent.get_score(ScoringMeasures.LP)

    def test_example_get_score_diff(self):
        example = get_basic_example()
        scoring_measure = ScoringMeasures.LP
        diff = example.get_score_diff(
            scoring_measure, SentenceNames.SHORT_NONISLAND, SentenceNames.LONG_ISLAND
        )
        assert 0 == diff

    def test_example_is_scored_accurately_for(self):
        example = get_basic_example()
        scoring_measure = ScoringMeasures.LP
        is_accurate = example.is_scored_accurately_for(
            scoring_measure, SentenceNames.SHORT_NONISLAND
        )
        assert not is_accurate


def get_basic_sentence(txt=""):
    sent = Sentence(txt)
    sent.lp_softmax = -5
    return sent


def get_basic_example():

    typed_senteces = [
        TypedSentence(
            SentenceNames.SHORT_NONISLAND, get_basic_sentence("The pen is on the table")
        ),
        TypedSentence(
            SentenceNames.LONG_ISLAND, get_basic_sentence("The is pen on the table")
        ),
    ]
    example = Example(typed_senteces)
    return example


def get_basic_testset():

    testset = TestSet(
        linguistic_phenomenon="wh",
        model_descr="bert-base-uncased",
        dataset_source=DataSources.SPROUSE,
        experimental_design=ExperimentalDesigns.MINIMAL_PAIRS,
        examples=[get_basic_example()],
        scoring_measures=[ScoringMeasures.LP],
    )
    return testset
