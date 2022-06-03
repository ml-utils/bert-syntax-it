from unittest import TestCase
from unittest.mock import patch

import pytest
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import model_types
from linguistic_tests.notebook import run_blimp_en
from linguistic_tests.notebook import run_tests_it
from linguistic_tests.run_sprouse_tests import run_sprouse_tests
from matplotlib import pyplot as plt


class TestRunTestSets(TestCase):
    # todo: also run these tests mocking the models (no remote calls) and with
    #  string variables instead of reading from files.
    #  (for mock models: save a model file with 1 layer and few nodes?)
    #  also move the mini test samples from the actual test data folder
    #  to the integration tests folder
    #  make output and models folder as optional parameters, to use them in
    #  unit tests

    @pytest.mark.enable_socket
    def test_run_blimp_en_tessts(self):
        testset_filenames = ["mini_wh_island.jsonl"]
        run_blimp_en(
            model_type=model_types.BERT,
            model_name="bert-base-uncased",
            testset_filenames=testset_filenames,
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
        run_sprouse_tests(
            model_type, model, tokenizer, DEVICES.CPU, phenomena=phenomena
        )

    @pytest.mark.slow
    @pytest.mark.enable_socket
    def test_run_syntactic_it_tests(self):
        model_type = model_types.BERT
        testset_files = [
            "mini_wh_adjunct_islands.jsonl",
        ]
        run_tests_it(model_type, testset_files=testset_files)
