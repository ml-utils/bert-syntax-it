import os
from unittest import TestCase
from unittest.mock import patch

import pytest
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.file_utils import parse_testsets
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.run_factorial_test_design import get_testset_params
from linguistic_tests.run_factorial_test_design import score_sprouse_testsets
from linguistic_tests.run_minimal_pairs_test_design import run_blimp_en
from linguistic_tests.testset import ERROR_LP
from matplotlib import pyplot as plt

from int_tests.int_tests_utils import get_test_data_dir


class TestRunTestSets(TestCase):
    # todo: also run these tests mocking the models (no remote calls) and with
    #  string variables instead of reading from files.
    #  (for mock models: save a model file with 1 layer and few nodes?)
    #  also move the mini test samples from the actual test data folder
    #  to the integration tests folder
    #  make output and models folder as optional parameters, to use them in
    #  unit tests

    @pytest.mark.enable_socket
    def test_run_blimp_en_tests(self):
        testset_filenames = ["mini_wh_island"]
        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)
        examples_format = "json_lines"
        run_blimp_en(
            model_type=ModelTypes.BERT,
            model_name="bert-base-uncased",
            dataset_source="Blimp paper",
            testset_filenames=testset_filenames,
            testset_dir_path=testset_dir_path,
            examples_format=examples_format,
        )

    @pytest.mark.enable_socket
    @patch.object(plt, "show")
    def test_run_sprouse_tests(self, mock1):
        assert plt.show is mock1

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

        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if model_type in BERT_LIKE_MODEL_TYPES:
            scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
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

        for testset in scored_testsets:
            assert testset.avg_DD_lp != ERROR_LP
            assert testset.avg_DD_penlp != ERROR_LP
            assert testset.avg_DD_ll != ERROR_LP
            assert testset.avg_DD_penll != ERROR_LP

    @pytest.mark.slow
    @pytest.mark.enable_socket
    def test_run_syntactic_it_tests(self):
        model_type = ModelTypes.BERT
        testset_files = [
            "mini_wh_adjunct_islands.jsonl",
        ]
        p = get_test_data_dir() / "custom_it"
        testset_dir_path = str(p)

        self.run_syntactic_tests_it_legacy_impl(
            model_type,
            testset_filenames=testset_files,
            testset_dir_path=testset_dir_path,
        )

    @staticmethod
    def run_syntactic_tests_it_legacy_impl(
        model_type, testset_filenames=None, testset_dir_path=None
    ):
        if model_type == ModelTypes.GPT:
            model_name = "GroNLP/gpt2-small-italian"
        if model_type == ModelTypes.GEPPETTO:
            model_name = "LorenzoDeMattei/GePpeTto"
        elif model_type == ModelTypes.BERT:
            model_name = "bert-base-uncased"  # NB bert large uncased is about 1GB
            model_name = str(get_models_dir() / "bert-base-italian-uncased")
            model_name = str(get_models_dir() / "bert-base-italian-cased/")
            model_name = str(get_models_dir() / "bert-base-italian-xxl-cased")
            model_name = "dbmdz/bert-base-italian-cased"
            model_name = "dbmdz/bert-base-italian-xxl-cased"
            # model_name = # str(get_models_dir() / "gilberto-uncased-from-camembert.tar.gz")
            # eval_suite = 'it'
        elif model_type == ModelTypes.GILBERTO:
            model_name = "idb-ita/gilberto-uncased-from-camembert"

        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
        if testset_dir_path is None:
            p = (
                get_syntactic_tests_dir() / "syntactic_tests_it"
            )  # "./outputs/syntactic_tests_it/"
            testset_dir_path = str(p)
        if testset_filenames is None:
            testset_filenames = [  # 'variations_tests.jsonl'
                "wh_adjunct_islands.jsonl",
                "wh_complex_np_islands.jsonl",
                "wh_subject_islands.jsonl",
                "wh_whether_island.jsonl",
            ]
        sentences_per_example = 3
        for test_file in testset_filenames:
            filepath = os.path.join(testset_dir_path, test_file)
            print_orange(f"running test {filepath}")
            testset_data = load_testset_data(filepath)

            if model_type in [
                ModelTypes.BERT,
                ModelTypes.GILBERTO,
                ModelTypes.ROBERTA,
            ]:
                # run_testset(testsets_dir, test_file, model, tokenizer,
                # score_based_on=sentence_score_bases.SOFTMAX)
                run_testset(
                    model_type,
                    model,
                    tokenizer,
                    DEVICES.CPU,
                    testset_data,
                    sentences_per_example,
                )
            elif model_type in [ModelTypes.GPT, ModelTypes.GEPPETTO]:
                run_testset(
                    model_type,
                    model,
                    tokenizer,
                    DEVICES.CPU,
                    testset_data,
                    sentences_per_example,
                )

    @staticmethod
    def profile_slowdowns():
        import cProfile
        import pstats
        import os
        import tqdm
        import json

        model_type = ModelTypes.ROBERTA  # ModelTypes.GPT  #
        model_name = "roberta-large"  # "roberta-base" #"gpt2-medium"
        # "gpt2-large"  # 'gpt2' #  "bert-large-uncased"
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)

        testset_filename = "mini_wh_island.jsonl"
        testset_filepath = os.path.join(testset_dir_path, testset_filename)

        print(f"loading testset file {testset_filepath}..")
        with open(testset_filepath, "r") as json_file:
            json_list = list(json_file)
        print("testset loaded.")

        examples = []
        for json_str in tqdm(json_list):
            example = json.loads(json_str)

            sentence_good = example[SentenceNames.SENTENCE_GOOD]
            sentence_bad = example[SentenceNames.SENTENCE_BAD]
            examples.append(
                {
                    SentenceNames.SENTENCE_GOOD: sentence_good,
                    SentenceNames.SENTENCE_BAD: sentence_bad,
                    SentenceNames.SENTENCE_GOOD_2ND: "",
                }
            )
        testset = {"sentences": examples}
        sentences_per_example = 2

        with cProfile.Profile() as pr:
            run_testset(
                model_type,
                model,
                tokenizer,
                DEVICES.CPU,
                testset,
                sentences_per_example,
            )

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
