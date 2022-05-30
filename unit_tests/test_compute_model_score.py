from random import random
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from linguistic_tests.compute_model_score import count_accurate_in_example
from linguistic_tests.compute_model_score import DEVICES
from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.compute_model_score import get_penalty_term
from linguistic_tests.compute_model_score import get_sentence_score_JHLau
from linguistic_tests.compute_model_score import load_model
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import reduce_to_log_product
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import model_types
from numpy import log
from transformers import BertForMaskedLM as BRT_M
from transformers import BertTokenizer as BRT_T
from transformers import CamembertForMaskedLM as CM_M
from transformers import CamembertTokenizer as CM_T
from transformers import GPT2LMHeadModel as GPT_M
from transformers import GPT2Tokenizer as GPT_T
from transformers import RobertaForMaskedLM as RB_M
from transformers import RobertaTokenizer as RB_T


class TestComputeModelScore(TestCase):
    @pytest.mark.skip("todo")
    def test_count_accurate_in_example(self):
        count_accurate_in_example()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_example_scores(self):
        get_example_scores()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_penalty_term(self):
        get_penalty_term()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_score_JHLau(self):
        get_sentence_score_JHLau()
        raise NotImplementedError

    @patch.object(CM_T, "from_pretrained", return_value=Mock(spec=CM_T))
    @patch.object(CM_M, "from_pretrained", return_value=Mock(spec=CM_M))
    @patch.object(RB_T, "from_pretrained", return_value=Mock(spec=RB_T))
    @patch.object(RB_M, "from_pretrained", return_value=Mock(spec=RB_M))
    @patch.object(GPT_M, "from_pretrained", return_value=Mock(spec=GPT_M))
    @patch.object(GPT_T, "from_pretrained", return_value=Mock(spec=GPT_T))
    @patch.object(BRT_M, "from_pretrained", return_value=Mock(spec=BRT_M))
    @patch.object(BRT_T, "from_pretrained", return_value=Mock(spec=BRT_T))
    def test_load_model(self, mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8):

        for mock in [mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8]:
            assert mock in [
                CM_T.from_pretrained,
                CM_M.from_pretrained,
                RB_T.from_pretrained,
                RB_M.from_pretrained,
                GPT_M.from_pretrained,
                GPT_T.from_pretrained,
                BRT_M.from_pretrained,
                BRT_T.from_pretrained,
            ]

        self.__test_load_model_helper(
            model_types.BERT, "bert-base-uncased", BRT_M, BRT_T
        )
        self.__test_load_model_helper(model_types.GPT, "gpt2", GPT_M, GPT_T)
        self.__test_load_model_helper(
            model_types.GEPPETTO, "LorenzoDeMattei/GePpeTto", GPT_M, GPT_T
        )
        self.__test_load_model_helper(model_types.ROBERTA, "roberta-base", RB_M, RB_T)
        self.__test_load_model_helper(
            model_types.GILBERTO, "idb-ita/gilberto-uncased-from-camembert", CM_M, CM_T
        )

    @staticmethod
    def __test_load_model_helper(
        model_type, model_name, expected_model_class, expected_tokenizer_class
    ):
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
        assert isinstance(model, expected_model_class)
        assert isinstance(tokenizer, expected_tokenizer_class)

    @pytest.mark.skip("todo")
    def test_perc(self):
        perc()
        raise NotImplementedError

    def test_reduce_to_log_product(self):

        a = random()
        b = random()
        actual = reduce_to_log_product([a, b])
        expected = log(a) + log(b)

        self.assertEqual(actual, expected)

    @pytest.mark.skip("todo")
    def test_run_testset(self):
        run_testset()
        raise NotImplementedError
