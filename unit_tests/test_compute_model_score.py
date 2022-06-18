from random import random
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch
from linguistic_tests import compute_model_score
from linguistic_tests.compute_model_score import count_accurate_in_example
from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.compute_model_score import get_sentence_score_JHLau
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import reduce_to_log_product
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import ModelTypes
from numpy import log
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import CamembertForMaskedLM
from transformers import CamembertTokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.bert.modeling_bert import MaskedLMOutput


class TestComputeModelScore(TestCase):
    @pytest.mark.skip("todo")
    def test_count_accurate_in_example(self):
        count_accurate_in_example()
        raise NotImplementedError

    # @pytest.mark.skip("todo")
    def test_get_example_scores(self):

        # todo: test for bert (logits) and gpt (softmax only)

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
        mock_bert_t = Mock(spec=BertTokenizer)
        mock_bert_t.tokenize.return_value = sentence_tokens
        model = None
        sent_ids = []

        example_data = None
        sentences_per_example = None
        sentences = [
            "Chi è partito per Parigi dopo aver fatto le valigie?",
            # 'Che cosa Gianni è partito per Parigi dopo aver fatto?',
            # 'Dopo aver fatto cosa, Gianni è partito per Parigi?'
        ]
        lp = -8.3
        log_logistic = (
            -200
        )  # todo: replace with actual value returned by a bert model for this sentence
        token_weights = np.random.random_sample(
            size=len(sentence_tokens)
        )  # list, len(sentence_tokens), min 11 max 21 (floats)
        with patch.object(
            compute_model_score, "get_sentences_from_example", return_value=sentences
        ) as _:
            with patch.object(
                compute_model_score,
                "get_sentence_score_JHLau",
                return_value=(lp, log_logistic, token_weights),
            ) as _:
                # don't mock: get_penalty_term
                # sentence_log_weight = -3.0
                # mock: reduce_to_log_product.return_value = -3.0
                self.get_example_scores_helper(
                    ModelTypes.BERT,
                    model,
                    mock_bert_t,
                    sent_ids,
                    example_data,
                    sentences_per_example,
                    BertForMaskedLM,
                    MaskedLMOutput,
                    BertTokenizer,
                )

    def get_example_scores_helper(
        self,
        model_type,
        model,
        tokenizer,
        sent_ids,
        example_data,
        sentences_per_example,
        model_class,
        model_output_class,
        tok_class,
    ):
        (
            lps,
            pen_lps,
            lls,
            penlls,
            pen_sentence_log_weights,
            sentence_log_weights,
            sentences,
        ) = get_example_scores(
            DEVICES.CPU,
            example_data,
            model,
            model_type,
            sent_ids,
            tokenizer,
            sentences_per_example,
        )
        print(f"\n{lps}")
        print(pen_lps)
        print(pen_sentence_log_weights)
        print(sentence_log_weights)
        print(sentences)

    def test_get_sentence_score_JHLau_empty(self):
        actual_score = get_sentence_score_JHLau(None, None, None, [], None)
        assert actual_score == (-200, None, None)

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
            ModelTypes.GPT, mock_gpt2_m, mock_gpt2_t, sentence_tokens, DEVICES.CPU
        )
        assert len(actual_score) == 3
        assert actual_score[0] != 0
        assert actual_score[0] != -200

        mock_gpt2_t.convert_tokens_to_ids.assert_called_once()
        mock_gpt2_m.assert_called_once()

    def test_get_sentence_score_JHLau_bert(self):
        self.get_sentence_score_JHLau_bert_helper(
            ModelTypes.BERT, BertForMaskedLM, MaskedLMOutput, BertTokenizer
        )
        self.get_sentence_score_JHLau_bert_helper(
            ModelTypes.ROBERTA, RobertaForMaskedLM, MaskedLMOutput, RobertaTokenizer
        )
        self.get_sentence_score_JHLau_bert_helper(
            ModelTypes.GILBERTO,
            CamembertForMaskedLM,
            MaskedLMOutput,
            CamembertTokenizer,
        )

    def get_sentence_score_JHLau_bert_helper(
        self, model_type, model_class, model_output_class, tok_class
    ):
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

        mock_bert_t = Mock(spec=tok_class)
        mock_bert_t.convert_tokens_to_ids.return_value = indexed_tokens
        mock_bert_t.bos_token_id = 0
        mock_bert_m_return_value = model_output_class(loss=loss, logits=lm_logits)
        mock_bert_m = MagicMock(spec=model_class, return_value=mock_bert_m_return_value)

        actual_score = get_sentence_score_JHLau(
            model_type, mock_bert_m, mock_bert_t, sentence_tokens, DEVICES.CPU
        )

        # todo: more checks on the returned values
        assert len(actual_score) == 3
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

    def test_run_testset(self):

        example_sentences = [
            "Chi è partito per Parigi dopo aver fatto le valigie?",
            "Che cosa Gianni è partito per Parigi dopo aver fatto?",
            "Dopo aver fatto cosa, Gianni è partito per Parigi?",
        ]
        testset = {
            "sentences": {
                0: example_sentences[0],
                1: example_sentences[1],
                2: example_sentences[2],
            }
        }
        mocked_model_score = (
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            example_sentences,
        )
        model = None
        tokenizer = None
        sentences_per_example = len(testset["sentences"])
        with patch.object(
            compute_model_score, "get_example_scores", return_value=mocked_model_score
        ) as _:
            run_testset(
                ModelTypes.BERT,
                model,
                tokenizer,
                DEVICES.CPU,
                testset,
                sentences_per_example,
            )
