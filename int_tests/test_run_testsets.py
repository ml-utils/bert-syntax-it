from unittest import TestCase

import pytest
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import model_types
from linguistic_tests.notebook import run_blimp_en
from linguistic_tests.notebook import run_tests_it
from linguistic_tests.run_sprouse_tests import run_sprouse_tests


class TestRunTestSets(TestCase):
    @pytest.mark.enable_socket
    def test_run_blimp_en_tessts(self):
        run_blimp_en()

    @pytest.mark.enable_socket
    def test_run_sprouse_tests(self):
        model_type = model_types.BERT
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
        run_sprouse_tests(model_type, model, tokenizer, DEVICES.CPU)

    @pytest.mark.slow
    @pytest.mark.enable_socket
    def test_run_syntactic_it_tests(self):
        # todo: make code parametric by model and testsets, do to pass minimal
        #  data during tests
        model_type = model_types.BERT
        run_tests_it(model_type)
