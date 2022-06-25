import os
from unittest import TestCase

import pytest
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.run_factorial_test_design import get_testset_params
from linguistic_tests.run_factorial_test_design import score_factorial_testsets
from linguistic_tests.testset import DataSources
from linguistic_tests.testset import ExperimentalDesigns
from linguistic_tests.testset import load_testset_from_pickle
from linguistic_tests.testset import parse_testset
from linguistic_tests.testset import parse_testsets

from int_tests.int_tests_utils import get_test_data_dir


class TestTestset(TestCase):

    # todo: patch with mock to make a unit test that does not load files from disk
    def test_parse_testset_sprouse(self):

        p = get_test_data_dir() / "sprouse"
        testset_dir_path = str(p)
        filename = "mini_wh_adjunct_island" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="sprouse")
        examples_list = testset["sentences"]
        model_name = "bert-base-uncased"
        scoring_measures = [
            ScoringMeasures.LP,
            ScoringMeasures.PenLP,
            ScoringMeasures.LL,
            ScoringMeasures.PLL,
        ]
        dataset_source = DataSources.SPROUSE
        parsed_testset = parse_testset(
            filename,
            model_name,
            dataset_source,
            examples_list,
            ExperimentalDesigns.FACTORIAL,
            scoring_measures,
        )

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 4
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    # todo: patch with mock to make a unit test that does not load files from disk
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
        dataset_source = DataSources.BLIMP_EN
        parsed_testset = parse_testset(
            filename,
            "bert-base-uncased",
            dataset_source,
            examples_list,
            ExperimentalDesigns.MINIMAL_PAIRS,
            scoring_measures,
        )

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 2
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    # todo: patch with mock to make a unit test that does not load files from disk
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
        dataset_source = DataSources.MADEDDU
        parsed_testset = parse_testset(
            filename,
            "bert-base-uncased",
            dataset_source,
            examples_list,
            ExperimentalDesigns.FACTORIAL,
            scoring_measures,
        )

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 4
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0


@pytest.mark.enable_socket
def test_serialization(tmp_path):
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

    experimental_design = ExperimentalDesigns.FACTORIAL
    # p = get_test_data_dir() / "custom_it"
    # testset_dir_path = str(p)
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
        experimental_design,
        model_name,
        scoring_measures,
        max_examples=1000,
    )
    scored_testsets = score_factorial_testsets(
        model_type, model, tokenizer, DEVICES.CPU, parsed_testsets, experimental_design
    )

    # todo: workaround for known issue:
    from linguistic_tests.lm_utils import SentenceNames

    scored_testsets[0].avg_zscores_of_likerts_by_measure_and_by_stype[
        ScoringMeasures.LP
    ][SentenceNames.SHORT_ISLAND] = 0.6

    tmp_file = tmp_path / "tmpfile.pickle"
    scored_testsets[0].save_to_pickle(tmp_file)

    testset_fromfile = load_testset_from_pickle(
        tmp_file, expected_experimental_design=experimental_design
    )

    # todo: assert that it contains the relevant fields/properties
    assert testset_fromfile.avg_DD_lp == scored_testsets[0].avg_DD_lp
    assert (
        testset_fromfile.linguistic_phenomenon
        == scored_testsets[0].linguistic_phenomenon
    )
    assert testset_fromfile == scored_testsets[0]
