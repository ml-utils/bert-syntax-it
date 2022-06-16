from unittest import TestCase
from unittest.mock import patch

import pytest
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.run_sprouse_tests import run_sprouse_tests
from linguistic_tests.run_syntactic_tests import run_blimp_en
from linguistic_tests.run_syntactic_tests import run_tests_it
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
        testset_filenames = ["mini_wh_island.jsonl"]
        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)
        run_blimp_en(
            model_type=model_types.BERT,
            model_name="bert-base-uncased",
            testset_filenames=testset_filenames,
            testset_dir_path=testset_dir_path,
        )

    @pytest.mark.enable_socket
    @patch.object(plt, "show")
    def test_run_sprouse_tests(self, mock1):
        assert plt.show is mock1

        model_type = model_types.BERT
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
        phenomena = [
            "mini_wh_adjunct_island",
        ]
        p = get_test_data_dir() / "sprouse"
        testset_dir_path = str(p)
        scored_testsets = run_sprouse_tests(
            model_type,
            model,
            tokenizer,
            DEVICES.CPU,
            phenomena_root_filenames=phenomena,
            testset_dir_path=testset_dir_path,
        )

        for testset in scored_testsets:
            assert testset.avg_DD_lp != -200

    @pytest.mark.slow
    @pytest.mark.enable_socket
    def test_run_syntactic_it_tests(self):
        model_type = model_types.BERT
        testset_files = [
            "mini_wh_adjunct_islands.jsonl",
        ]
        p = get_test_data_dir() / "custom_it"
        testset_dir_path = str(p)

        run_tests_it(
            model_type,
            testset_filenames=testset_files,
            testset_dir_path=testset_dir_path,
        )

    @staticmethod
    def profile_slowdowns():
        import cProfile
        import pstats
        import os
        import tqdm
        import json

        model_type = model_types.ROBERTA  # model_types.GPT  #
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
