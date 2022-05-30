from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from linguistic_tests.lm_utils import get_pen_score
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from transformers import BertForMaskedLM as BRT_M
from transformers import BertTokenizer as BRT_T
from transformers import GPT2LMHeadModel as GPT_M
from transformers import GPT2Tokenizer as GPT_T


class TestLMUtils(TestCase):
    @pytest.mark.skip("todo")
    def test_get_pen_score(self):
        get_pen_score()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentences_from_example(self):
        get_sentences_from_example()
        raise NotImplementedError

    @patch.object(GPT_M, "from_pretrained", return_value=Mock(spec=GPT_M))
    @patch.object(GPT_T, "from_pretrained", return_value=Mock(spec=GPT_T))
    @patch.object(BRT_M, "from_pretrained", return_value=Mock(spec=BRT_M))
    @patch.object(BRT_T, "from_pretrained", return_value=Mock(spec=BRT_T))
    def test_load_model_and_tokenizer(self, mock1, mock2, mock3, mock4):

        assert GPT_M.from_pretrained is mock4
        assert GPT_T.from_pretrained is mock3
        assert BRT_M.from_pretrained is mock2
        assert BRT_T.from_pretrained is mock1

        bert_name = "bert-base-uncased"
        bert, b_tokenizer = load_model_and_tokenizer(model_types.BERT, bert_name)
        assert isinstance(bert, BRT_M)
        assert isinstance(b_tokenizer, BRT_T)

        gpt2_name = "gpt2"
        gpt2, g_tokenizer = load_model_and_tokenizer(model_types.GPT, gpt2_name)
        assert isinstance(gpt2, GPT_M)
        assert isinstance(g_tokenizer, GPT_T)

        for model_type in [
            model_types.ROBERTA,
            model_types.GILBERTO,
            model_types.GEPPETTO,
        ]:
            with self.assertRaises(SystemExit):
                load_model_and_tokenizer(model_type, "")

    @pytest.mark.skip("todo")
    def test_load_testset_data(self):
        load_testset_data()
        raise NotImplementedError
