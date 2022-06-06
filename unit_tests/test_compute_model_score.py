from random import random
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
import torch
from linguistic_tests.compute_model_score import count_accurate_in_example
from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.compute_model_score import get_sentence_score_JHLau
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import reduce_to_log_product
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import model_types
from numpy import log
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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
    def test_get_sentence_score_JHLau(self):
        # todo, mock:
        #
        # model_types.ROBERTA
        # RobertaTokenizer convert_tokens_to_ids
        # RobertaForMaskedLM model(tokens_tensor, token_type_ids=segment_tensor)
        #
        # model_types.GILBERTO
        # CamembertTokenizer convert_tokens_to_ids
        # CamembertForMaskedLM model(tokens_tensor, token_type_ids=segment_tensor)

        get_sentence_score_JHLau()
        raise NotImplementedError

    def test_get_sentence_score_JHLau_empty(self):
        actual_score = get_sentence_score_JHLau(None, None, None, [], None)
        assert actual_score == (-200, None)

    def test_get_sentence_score_JHLau_gpt(self):
        vocab_size = 1000
        sentence_tokens = [
            "Chi",
            "ĠÃ¨",
            "Ġpartito",
            "Ġper",
            "ĠParigi",
            "Ġdopo",
            "Ġaver",
            "Ġfatto",
            "Ġle",
            "Ġvali",
            "gie",
            "?",
        ]
        sentence_ids = [
            9176,
            350,
            3691,
            306,
            4892,
            815,
            1456,
            857,
            337,
            3097,
            3298,
            31,
        ]
        lm_logits = torch.rand(1, len(sentence_ids), vocab_size)
        loss = torch.tensor(4.3)

        mock_gpt2_t = Mock(spec=GPT2Tokenizer)
        mock_gpt2_t.convert_tokens_to_ids.return_value = sentence_ids
        mock_gpt2_t.bos_token_id = 0
        mock_gpt2_m_return_value = CausalLMOutputWithCrossAttentions(
            loss=loss, logits=lm_logits
        )
        mock_gpt2_m = MagicMock(
            spec=GPT2LMHeadModel, return_value=mock_gpt2_m_return_value
        )

        actual_score = get_sentence_score_JHLau(
            model_types.GPT, mock_gpt2_m, mock_gpt2_t, sentence_tokens, DEVICES.CPU
        )
        assert actual_score != 0
        assert actual_score != -200

        mock_gpt2_t.convert_tokens_to_ids.assert_called_once()
        mock_gpt2_m.assert_called_once()

    def test_get_sentence_score_JHLau_bert(self):
        vocab_size = 1000
        sentence_tokens = [
            "Chi",
            "è",
            "partito",
            "per",
            "Parigi",
            "dopo",
            "aver",
            "fatto",
            "le",
            "valigie",
            "?",
        ]
        """tokenize_masked = [
            "[CLS]",
            "[MASK]",
            "è",
            "partito",
            "per",
            "Parigi",
            "dopo",
            "aver",
            "fatto",
            "le",
            "valigie",
            "?",
            "[SEP]",
        ]"""
        indexed_tokens = [
            102,
            104,
            198,
            5524,
            156,
            3984,
            693,
            1019,
            552,
            199,
            27527,
            3098,
            103,
        ]
        """tokens_tensor = torch.randint(
            low=100,
            high=vocab_size - 1,
            size=(len(sentence_tokens) - 1, len(sentence_tokens) + 1),
        )  # 11x13
        segment_tensor = torch.zeros(
            (len(sentence_tokens) - 1, len(sentence_tokens) + 1)
        )  # 11x13 all zeroes"""
        lm_logits = torch.rand(
            size=(len(indexed_tokens) - 1, len(indexed_tokens) + 1, vocab_size)
        )
        loss = None
        from transformers.models.bert.modeling_bert import MaskedLMOutput
        from transformers import BertForMaskedLM

        mock_bert_t = Mock(spec=BertTokenizer)
        mock_bert_t.convert_tokens_to_ids.return_value = indexed_tokens
        mock_bert_t.bos_token_id = 0
        mock_bert_m_return_value = MaskedLMOutput(loss=loss, logits=lm_logits)
        mock_bert_m = MagicMock(
            spec=BertForMaskedLM, return_value=mock_bert_m_return_value
        )

        actual_score = get_sentence_score_JHLau(
            model_types.BERT, mock_bert_m, mock_bert_t, sentence_tokens, DEVICES.CPU
        )

        # todo: more checks on the returned values
        assert actual_score != 0
        assert actual_score != -200

        # todo: specify how many times called
        mock_bert_t.convert_tokens_to_ids.assert_called()
        mock_bert_m.assert_called_once()

    def test_get_sentence_score_JHLau_unrecognized(self):
        unrecognizable_model_type = 10
        with self.assertRaises(ValueError):
            _ = get_sentence_score_JHLau(
                unrecognizable_model_type, None, None, [""], None
            )

    def test_perc(self):
        self.assertEqual(perc(10, 20), 50)
        self.assertEqual(perc(1, 100), 1)

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
