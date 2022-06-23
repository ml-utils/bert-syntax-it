import logging
from collections import namedtuple
from random import random
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from linguistic_tests import compute_model_score
from linguistic_tests import lm_utils
from linguistic_tests.compute_model_score import count_accurate_in_example
from linguistic_tests.compute_model_score import get_sentence_acceptability_score
from linguistic_tests.compute_model_score import get_unparsed_example_scores
from linguistic_tests.compute_model_score import logistic2
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import reduce_to_log_product
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_penalty_term
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.run_minimal_pairs_test_design import get_unparsed_testset_scores
from linguistic_tests.testset import ERROR_LP
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

from unit_tests.test_lm_utils import get_basic_example_data_dict


class TestComputeModelScore(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        # logging.getLogger().setLevel(logging.DEBUG)

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

        example_data = get_basic_example_data_dict()
        sentences_per_example = None
        sentences = [
            "Chi è partito per Parigi dopo aver fatto le valigie?",
            # 'Che cosa Gianni è partito per Parigi dopo aver fatto?',
            # 'Dopo aver fatto cosa, Gianni è partito per Parigi?'
        ]
        lp_softmax = -8.3
        lp_logistic = ERROR_LP  # todo: replace with actual value returned by a bert model for this sentence
        with patch.object(
            lm_utils,
            get_sentences_from_example.__name__,
            return_value=sentences,  # "get_sentences_from_example"
        ) as _:
            with patch.object(
                compute_model_score,
                get_sentence_acceptability_score.__name__,  # "get_sentence_acceptability_score"
                return_value=(lp_softmax, lp_logistic),
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
                )

    def get_example_scores_helper(
        self,
        model_type,
        model,
        tokenizer,
        sent_ids,
        example_data,
        sentences_per_example,
    ):
        (lps, pen_lps, lls, penlls, sentences,) = get_unparsed_example_scores(
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
        print(sentences)

    def test_get_sentence_score_JHLau_empty(self):
        actual_score = get_sentence_acceptability_score(None, None, None, [], None)
        assert actual_score == (ERROR_LP, None)

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

        actual_score = get_sentence_acceptability_score(
            ModelTypes.GPT, mock_gpt2_m, mock_gpt2_t, sentence_tokens, DEVICES.CPU
        )
        assert len(actual_score) == 2
        assert actual_score[0] != 0
        assert actual_score[0] != ERROR_LP

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

        actual_score = get_sentence_acceptability_score(
            model_type, mock_bert_m, mock_bert_t, sentence_tokens, DEVICES.CPU
        )

        # todo: more checks on the returned values
        assert len(actual_score) == 2
        assert actual_score[0] != 0
        assert actual_score[0] != ERROR_LP

        # todo: specify how many times called
        mock_bert_t.convert_tokens_to_ids.assert_called()
        mock_bert_m.assert_called_once()

    def test_get_sentence_score_JHLau_unrecognized(self):
        unrecognizable_model_type = 10
        with self.assertRaises(ValueError):
            _ = get_sentence_acceptability_score(
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
        # logging.basicConfig(level=logging.DEBUG)
        example_sentences = [
            "Chi è partito per Parigi dopo aver fatto le valigie?",
            "Che cosa Gianni è partito per Parigi dopo aver fatto?",
            "Dopo aver fatto cosa, Gianni è partito per Parigi?",
            "Dopo aver fatto cosa, Gianni è partito per Parigi?",
        ]
        testset = {
            "sentences": [
                {
                    SentenceNames.SHORT_NONISLAND: example_sentences[0],
                    SentenceNames.LONG_ISLAND: example_sentences[1],
                    SentenceNames.LONG_NONISLAND: example_sentences[2],
                    SentenceNames.SHORT_ISLAND: example_sentences[3],
                }
            ]
        }
        mocked_model_score = (
            [0.1, 0.2, 0.3, 0.3],
            [0.1, 0.2, 0.3, 0.3],
            [0.1, 0.2, 0.3, 0.3],
            [0.1, 0.2, 0.3, 0.3],
            example_sentences,
        )
        model = None
        tokenizer = Mock()
        tokenizer.tokenize = Mock(return_value=[0, 1, 2, 3, 4, 5])
        sentences_per_example = len(testset["sentences"])
        with patch.object(
            compute_model_score,
            get_unparsed_example_scores.__name__,  # "get_unparsed_example_scores"
            return_value=mocked_model_score,
        ) as _:
            with patch.object(
                compute_model_score,
                get_sentence_acceptability_score.__name__,
                return_value=(ERROR_LP, ERROR_LP),
            ):
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
                    ModelTypes.BERT,
                    model,
                    tokenizer,
                    DEVICES.CPU,
                    testset,
                    sentences_per_example,
                )

                assert 0 == correct_lps_1st_sentence
                assert 0 == correct_pen_lps_1st_sentence
                assert 0 == correct_lps_2nd_sentence
                assert 0 == correct_pen_lps_2nd_sentence
                assert 0 == correct_lls_1st_sentence
                assert 0 == correct_pen_lls_1st_sentence
                assert 0 == correct_lls_2nd_sentence
                assert 0 == correct_pen_lls_2nd_sentence

    def test_get_unparsed_example_scores(self):
        example_data = get_basic_example_data_dict()
        tokenizer = Mock()
        tokenizer.tokenize = Mock(return_value=[])
        sent_ids = []
        model, model_type, sentences_per_example = (
            None,
            None,
            None,
        )

        with patch.object(
            compute_model_score,
            get_sentence_acceptability_score.__name__,  # "get_sentence_acceptability_score",
            return_value=(ERROR_LP, ERROR_LP),  # (lp_softmax, lp_logistic)
        ) as _:
            # todo: also patch tokenizer.tokenize(sentence)
            return_values_1 = get_unparsed_example_scores_legacy_impl(
                DEVICES.CPU,
                example_data,
                model,
                model_type,
                sent_ids,
                tokenizer,
                sentences_per_example,
                sprouse_format=False,
            )

            return_values_2 = get_unparsed_example_scores(
                DEVICES.CPU,
                example_data,
                model,
                model_type,
                sent_ids,
                tokenizer,
                sentences_per_example,
            )

        # todo: check that the equality comparison is recursive (elements of
        #  the lists of the tuples)
        assert return_values_1 == return_values_2, (
            f"The two return values are different: "
            f"\n{return_values_1=} "
            f"\n{return_values_2=}"
        )

    def test_logistic2(self):
        assert logistic2(20) > 0.99
        assert logistic2(-20) < 0.01


def get_unparsed_example_scores_legacy_impl(
    device,
    example_data: dict,
    model,
    model_type,
    sent_ids: List[int],
    tokenizer,
    sentences_per_example,
    sprouse_format=False,
):
    sentences = get_sentences_from_example(
        example_data, sentences_per_example, sprouse_format=sprouse_format
    )

    lps = []
    pen_lps = []
    lls = []
    penlls = []

    for sent_id, sentence in enumerate(sentences):
        sentence_tokens = tokenizer.tokenize(sentence)  # , return_tensors='pt'
        # if len(sentence_tokens) == 0:
        #     logging.warning(f"Warning: lenght 0 for {sentence=} from {example_data=}")
        text_len = len(sentence_tokens)

        # nb: this is patched to avoid any model/tokenizer calls
        lp_softmax, lp_logistic = get_sentence_acceptability_score(
            model_type, model, tokenizer, sentence_tokens, device
        )

        # acceptability measures by sentence idx
        penalty = get_penalty_term(text_len)
        lps.append(lp_softmax)
        # mean_lps.append(lp / text_len)
        pen_lps.append(lp_softmax / penalty)
        sent_ids.append(sent_id)
        if model_type in BERT_LIKE_MODEL_TYPES:
            lls.append(lp_logistic)
            penlls.append(lp_logistic / penalty)

    ScoreResults = namedtuple("ScoreResults", "lps pen_lps lls penlls sentences_txts")
    return ScoreResults(lps, pen_lps, lls, penlls, sentences)
