import os
from unittest import TestCase

import pytest
from linguistic_tests.file_utils import parse_testsets
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.run_factorial_test_design import get_testset_params
from linguistic_tests.run_factorial_test_design import score_sprouse_testsets
from linguistic_tests.testset import Example
from linguistic_tests.testset import load_testset_from_pickle
from linguistic_tests.testset import parse_testset
from linguistic_tests.testset import Sentence
from linguistic_tests.testset import TestSet
from linguistic_tests.testset import TypedSentence

from int_tests.int_tests_utils import get_test_data_dir


class TestTestset(TestCase):

    # todo: patch with mock so a unit test does not load files from disk
    def test_parse_testset_sprouse(self):

        p = get_test_data_dir() / "sprouse"
        testset_dir_path = str(p)
        filename = "mini_wh_adjunct_island" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="sprouse")
        examples_list = testset["sentences"]
        scoring_measures = [
            ScoringMeasures.LP,
            ScoringMeasures.PenLP,
            ScoringMeasures.LL,
            ScoringMeasures.PLL,
        ]
        dataset_source = "Sprouse paper"
        parsed_testset = parse_testset(
            filename,
            ModelTypes.BERT,
            dataset_source,
            "some_model",
            examples_list,
            "sprouse",
            scoring_measures,
        )

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 4
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    # todo: patch with mock so a unit test does not load files from disk
    def test_parse_testset_blimp(self):
        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)
        filename = "mini_wh_island" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="json_lines")
        examples_list = testset["sentences"]
        scoring_measures = [
            ScoringMeasures.LP,
            ScoringMeasures.PenLP,
            ScoringMeasures.LL,
            ScoringMeasures.PLL,
        ]
        dataset_source = "Blimp paper"
        parsed_testset = parse_testset(
            filename,
            ModelTypes.BERT,
            dataset_source,
            "some_model",
            examples_list,
            "blimp",
            scoring_measures,
        )

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 2
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    # todo: patch with mock so a unit test does not load files from disk
    def test_parse_testset_custom_it(self):
        p = get_test_data_dir() / "custom_it"
        testset_dir_path = str(p)
        filename = "mini_wh_adjunct_islands" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="blimp")
        examples_list = testset["sentences"]
        scoring_measures = [
            ScoringMeasures.LP,
            ScoringMeasures.PenLP,
            ScoringMeasures.LL,
            ScoringMeasures.PLL,
        ]
        dataset_source = "Madeddu"
        parsed_testset = parse_testset(
            filename,
            ModelTypes.BERT,
            dataset_source,
            "some_model",
            examples_list,
            "sprouse",
            scoring_measures,
        )

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 4
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    def test_get_testset_sentence_types(self):
        testset = self.get_basic_testset()
        stypes = testset.get_sentence_types()
        assert 2 == len(stypes)
        assert SentenceNames.SHORT_NONISLAND in stypes
        assert SentenceNames.LONG_ISLAND in stypes

    def test_get_scoring_measures(self):
        testset = self.get_basic_testset()
        scoring_measures = testset.get_scoring_measures()
        assert 1 == len(scoring_measures)
        assert ScoringMeasures.LP in scoring_measures

    def test_get_acceptable_sentence_types(self):
        testset = self.get_basic_testset()
        acc_stypes = testset.get_acceptable_sentence_types()
        assert 1 == len(acc_stypes)
        assert SentenceNames.SHORT_NONISLAND in acc_stypes
        assert SentenceNames.LONG_ISLAND not in acc_stypes

    def test_example_get_sentence_types(self):
        example = self.get_basic_example()
        stypes = example.get_sentence_types()
        assert 2 == len(stypes)
        assert SentenceNames.SHORT_NONISLAND in stypes
        assert SentenceNames.LONG_ISLAND in stypes

    @staticmethod
    def get_basic_example():
        typed_senteces = [
            TypedSentence(
                SentenceNames.SHORT_NONISLAND, Sentence("The pen is on the table")
            ),
            TypedSentence(
                SentenceNames.LONG_ISLAND, Sentence("The is pen on the table")
            ),
        ]
        example = Example(typed_senteces)
        return example

    @staticmethod
    def get_basic_testset():

        testset = TestSet(
            linguistic_phenomenon="wh",
            model_descr="bert",
            dataset_source="sprouse",
            examples=[TestTestset.get_basic_example()],
            scoring_measures=[ScoringMeasures.LP],
            model_type=ModelTypes.BERT,
        )
        return testset


@pytest.mark.enable_socket
def test_serialization(tmp_path):
    p = get_test_data_dir() / "custom_it"
    testset_dir_path = str(p)

    # filename = "mini_wh_adjunct_islands" + ".jsonl"
    # filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
    # testset = load_testset_data(filepath, examples_format="blimp")
    # examples_list = testset["sentences"]
    # parsed_testset = parse_testset(filename, "some_model", examples_list, "sprouse")

    # todo: use a real model to score the testset
    model_type = ModelTypes.BERT
    model_name = "dbmdz/bert-base-italian-xxl-cased"
    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
    phenomena = [
        "mini_wh_adjunct_island",
    ]

    tests_subdir = "sprouse/"
    p = get_test_data_dir() / tests_subdir
    testset_dir_path = str(p)
    _, _, dataset_source = get_testset_params(tests_subdir)
    scoring_measures = [
        ScoringMeasures.LP,
        ScoringMeasures.PenLP,
        ScoringMeasures.LL,
        ScoringMeasures.PLL,
    ]
    parsed_testsets = parse_testsets(
        testset_dir_path,
        phenomena,
        dataset_source,
        "sprouse",
        "sprouse",
        model_name,
        model_type,
        scoring_measures,
        max_examples=1000,
    )
    scored_testsets = score_sprouse_testsets(
        model_type,
        model,
        tokenizer,
        DEVICES.CPU,
        parsed_testsets,
    )

    tmp_file = tmp_path / "tmpfile.pickle"
    scored_testsets[0].save_to_pickle(tmp_file)

    testset_fromfile = load_testset_from_pickle(tmp_file)

    # todo: assert that it contains the relevant fields/properties
    assert testset_fromfile.avg_DD_lp == scored_testsets[0].avg_DD_lp
    assert (
        testset_fromfile.linguistic_phenomenon
        == scored_testsets[0].linguistic_phenomenon
    )
    assert testset_fromfile == scored_testsets[0]
