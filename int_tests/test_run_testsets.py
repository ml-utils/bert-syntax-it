import logging
import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt
from matplotlib.pyplot import show

import src.linguistic_tests
from int_tests.int_tests_utils import get_test_data_dir
from src.linguistic_tests import file_utils
from src.linguistic_tests import plots_and_prints
from src.linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import get_models_dir
from src.linguistic_tests.lm_utils import get_results_dir
from src.linguistic_tests.lm_utils import get_syntactic_tests_dir
from src.linguistic_tests.lm_utils import get_testset_params
from src.linguistic_tests.lm_utils import load_model
from src.linguistic_tests.lm_utils import load_testset_data
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import print_orange
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.plots_and_prints import print_accuracies
from src.linguistic_tests.run_test_design import rescore_testsets_and_save_pickles
from src.linguistic_tests.run_test_design import run_test_design
from src.linguistic_tests.run_test_design import score_factorial_testsets
from src.linguistic_tests.testset import ERROR_LP
from src.linguistic_tests.testset import parse_testsets
from unit_tests.test_compute_model_score import get_unparsed_testset_scores


class TestRunTestSets(TestCase):
    # todo: also run these tests mocking the models (no remote calls) and with
    #  string variables instead of reading from files.
    #  (for mock models: save a model file with 1 layer and few nodes?)
    #  also move the mini test samples from the actual test data folder
    #  to the integration tests folder
    #  make output and models folder as optional parameters, to use them in
    #  unit tests

    @patch.object(src.linguistic_tests.run_test_design, get_testset_params.__name__)
    @pytest.mark.enable_socket
    def test_run_test_design_sprouse_bert(self, mock_get_testset_params):

        mock_get_testset_params.return_value = (
            [
                "mini_wh_adjunct_island",
                "mini_wh_complex_np",
                "mini_wh_subject_island",
                "mini_wh_whether_island",
            ],
            "sprouse",
            DataSources.SPROUSE,
            ExperimentalDesigns.FACTORIAL,
        )

        with patch.object(plots_and_prints.plt, show.__name__) as mock_plt_show:
            assert plt.show is mock_plt_show

            self._test_run_test_design_helper(
                mock_get_testset_params,
                tests_subdir="sprouse/",
                model_type=ModelTypes.BERT,
                model_name="dbmdz/bert-base-italian-xxl-cased",
                max_examples=8,
                rescore=True,
            )

    @patch.object(src.linguistic_tests.run_test_design, get_testset_params.__name__)
    @pytest.mark.enable_socket
    def test_run_test_design_blimp_gpt(self, mock_get_testset_params):

        mock_get_testset_params.return_value = (
            ["mini_wh_island"],
            "blimp",
            DataSources.BLIMP_EN,
            ExperimentalDesigns.MINIMAL_PAIRS,
        )

        self._test_run_test_design_helper(
            mock_get_testset_params,
            tests_subdir="blimp/",
            model_type=ModelTypes.GPT,
            model_name="gpt2",
            max_examples=5,
            rescore=True,
        )

    def _test_run_test_design_helper(
        self,
        mock_get_testset_params,
        tests_subdir: str,
        model_type: ModelTypes,
        model_name: str,
        max_examples: int,
        rescore,
    ):
        assert (
            src.linguistic_tests.run_test_design.get_testset_params
            is mock_get_testset_params
        )

        with patch("argparse.ArgumentParser"):  # as mock:
            with patch.object(
                file_utils,
                file_utils._parse_arguments.__name__,
                return_value=MagicMock(),
            ):  # as mock_method
                with tempfile.TemporaryDirectory() as tmpdirname:
                    print("created temporary directory", tmpdirname)

                    with patch.object(
                        src.linguistic_tests.file_utils, get_results_dir.__name__
                    ) as mock_get_results_dir:

                        assert (
                            src.linguistic_tests.file_utils.get_results_dir
                            is mock_get_results_dir
                        )
                        mock_get_results_dir.return_value = Path(tmpdirname)

                        with patch.object(
                            plots_and_prints, get_results_dir.__name__
                        ) as mock_get_results_dir2:
                            assert (
                                plots_and_prints.get_results_dir
                                is mock_get_results_dir2
                            )
                            mock_get_results_dir2.return_value = Path(tmpdirname)

                            with patch.object(
                                src.linguistic_tests.run_test_design,
                                get_syntactic_tests_dir.__name__,
                            ) as mock_get_syntactic_tests_dir:

                                assert (
                                    src.linguistic_tests.run_test_design.get_syntactic_tests_dir
                                    is mock_get_syntactic_tests_dir
                                )
                                mock_get_syntactic_tests_dir.return_value = (
                                    get_test_data_dir()
                                )

                                run_test_design(
                                    model_name=model_name,
                                    model_type=model_type,
                                    tests_subdir=tests_subdir,
                                    max_examples=max_examples,
                                    device=DEVICES.CPU,
                                    rescore=rescore,
                                    log_level=logging.WARNING,
                                )

    @pytest.mark.enable_socket
    def test_rescore_testsets_and_save_pickles(self):
        testset_filenames = ["mini_wh_island"]
        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)
        examples_format = "json_lines"
        rescore_testsets_and_save_pickles(
            model_type=ModelTypes.BERT,
            model_name="bert-base-uncased",
            testset_dir_path=testset_dir_path,
            testsets_root_filenames=testset_filenames,
            dataset_source=DataSources.BLIMP_EN,
            experimental_design=ExperimentalDesigns.MINIMAL_PAIRS,
            device=DEVICES.CPU,
            examples_format=examples_format,
        )

    @pytest.mark.enable_socket
    @patch.object(plt, show.__name__)  # "show"
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
        _, _, dataset_source, experimental_design = get_testset_params(tests_subdir)

        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if model_type in BERT_LIKE_MODEL_TYPES:
            scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]

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
            model_type,
            model,
            tokenizer,
            DEVICES.CPU,
            parsed_testsets,
            experimental_design,
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
                get_syntactic_tests_dir() / "mdd2"  # "syntactic_tests_it"
            )  # "./outputs/syntactic_tests_it/"
            testset_dir_path = str(p)
        if testset_filenames is None:
            testset_filenames = [  # 'variations_tests.jsonl'
                "wh_adjunct_islands.jsonl",
                "wh_complex_np_islands.jsonl",
                "wh_subject_islands.jsonl",
                "mini_wh_whether_island.jsonl",
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
                ModelTypes.GPT,
                ModelTypes.GEPPETTO,
            ]:
                # run_testset(testsets_dir, test_file, model, tokenizer,
                # score_based_on=sentence_score_bases.SOFTMAX)
                (
                    correct_lps_1st_sentence,
                    correct_pen_lps_1st_sentence,
                    correct_lps_2nd_sentence,
                    correct_pen_lps_2nd_sentence,
                    correct_lls_1st_sentence,
                    correct_pen_lls_1st_sentence,
                    correct_lls_2nd_sentence,
                    correct_pen_lls_2nd_sentence,
                ) = get_unparsed_testset_scores(
                    model_type,
                    model,
                    tokenizer,
                    DEVICES.CPU,
                    testset_data,
                    sentences_per_example,
                )

                examples_count = len(testset_data["sentences"])
                print_accuracies(
                    examples_count,
                    model_type,
                    correct_lps_1st_sentence,
                    correct_pen_lps_1st_sentence,
                    correct_lps_2nd_sentence,
                    correct_pen_lps_2nd_sentence,
                    correct_lls_1st_sentence,
                    correct_pen_lls_1st_sentence,
                    correct_lls_2nd_sentence,
                    correct_pen_lls_2nd_sentence,
                )

    @staticmethod
    def profile_slowdowns():
        import cProfile
        import pstats
        import os

        # import tqdm
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
        for json_str in json_list:
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
            get_unparsed_testset_scores(
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
