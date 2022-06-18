import os
from unittest import TestCase
from unittest.mock import patch

import pytest
from linguistic_tests.compute_model_score import logistic2
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
from linguistic_tests.run_sprouse_tests import score_sprouse_testsets
from linguistic_tests.run_syntactic_tests import run_blimp_en
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
        p = get_test_data_dir() / "sprouse"
        testset_dir_path = str(p)
        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if model_type in BERT_LIKE_MODEL_TYPES:
            scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
        parsed_testsets = parse_testsets(
            testset_dir_path,
            phenomena,
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
            assert testset.avg_DD_lp != -200
            assert testset.avg_DD_penlp != -200
            assert testset.avg_DD_ll != -200
            assert testset.avg_DD_penll != -200

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

    @pytest.mark.skip("todo")
    def test_model_outputs(self):
        # plot the whole nd array of a model logits output
        # sort the values (np.sort) and use the index for the x-axis

        # todo: load a model
        # tokenize a sentence and pass it as input
        raise NotImplementedError

    def plot_logistic2(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # 100 linearly spaced numbers
        x = np.linspace(-30, 30, 1000)

        # the function, which is y = x^2 here
        y = logistic2(x)

        # setting the axes at the centre
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.spines['left'].set_position('center')
        # ax.spines['bottom'].set_position('zero')
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')

        plt.xlim((-20, 20))

        # plot the function
        plt.plot(x, y, label="default")

        y_k_05 = logistic2(x, k=0.5)
        plt.plot(x, y_k_05, label="y_k_05")

        y_k_025 = logistic2(x, k=0.25)
        plt.plot(x, y_k_025, label="y_k_025")

        y_k_010 = logistic2(x, k=0.1)
        plt.plot(x, y_k_010, label="y_k_010")

        # show the plot

        plt.legend()
        plt.show()
