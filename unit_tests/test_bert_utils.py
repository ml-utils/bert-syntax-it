import os
import random
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from numpy import ndarray
from scipy.special import softmax
from tqdm import tqdm
from transformers import BertForMaskedLM as BRT_M
from transformers import BertTokenizer as BRT_T
from transformers import BertTokenizerFast
from transformers.modeling_outputs import MaskedLMOutput

import src.linguistic_tests
from src.linguistic_tests import bert_utils
from src.linguistic_tests.bert_utils import analize_example
from src.linguistic_tests.bert_utils import analize_sentence
from src.linguistic_tests.bert_utils import bert_get_logprobs_softmax
from src.linguistic_tests.bert_utils import check_unknown_words
from src.linguistic_tests.bert_utils import convert_ids_to_tokens
from src.linguistic_tests.bert_utils import count_split_words_in_sentence
from src.linguistic_tests.bert_utils import estimate_sentence_probability
from src.linguistic_tests.bert_utils import estimate_sentence_probability_from_text
from src.linguistic_tests.bert_utils import get_bert_output_single_masking
from src.linguistic_tests.bert_utils import get_score_descr
from src.linguistic_tests.bert_utils import get_sentence_probs_from_word_ids
from src.linguistic_tests.bert_utils import get_sentence_scores
from src.linguistic_tests.bert_utils import get_topk
from src.linguistic_tests.bert_utils import get_topk_tokens_from_bert_output
from src.linguistic_tests.bert_utils import print_orange
from src.linguistic_tests.bert_utils import tokenize_sentence
from src.linguistic_tests.compute_model_score import get_sentence_acceptability_score
from src.linguistic_tests.lm_utils import get_sentences_from_example
from src.linguistic_tests.lm_utils import load_testset_data
from src.linguistic_tests.lm_utils import ModelTypes

# from src.linguistic_tests.bert_utils import get_topk


CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


class TestBertUtils(TestCase):
    def create_patch(self, name):
        patcher = patch(name)
        thing = patcher.start()
        self.addCleanup(patcher.stop)
        return thing

    @pytest.mark.skip("todo")
    def test__check_unk_and_num_tokens(self):
        bert_utils.__check_unk_and_num_tokens()
        raise NotImplementedError

    @pytest.mark.skip(reason="todo")
    def test_get_acceptability_diffs(self):
        # TODO: test method before change with option to score method (softmax or logits)
        bert_utils.__get_acceptability_diffs()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test__get_example_estimates(self):
        bert_utils.__get_example_estimates()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test__get_example_tokens_and_oov_counts(self):
        bert_utils.__get_example_tokens_and_oov_counts()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test__get_scores_from_word_ids(self):
        bert_utils.__get_scores_from_word_ids()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_aanalize_example(self):
        analize_example()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_analize_sentence(self):
        analize_sentence()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_bert_get_logprobs(self):
        bert_get_logprobs_softmax()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_check_unknown_words(self):
        check_unknown_words()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_convert_ids_to_tokens(self):
        convert_ids_to_tokens()
        raise NotImplementedError

    def test_count_split_words_in_sentence(self):
        sentence_tokens = ["hair", "##dress", "##er", "##s"]
        assert count_split_words_in_sentence(sentence_tokens) == 1
        sentence_tokens = ["these", "are", "hair", "##dress", "##er", "##s"]
        assert count_split_words_in_sentence(sentence_tokens) == 1
        sentence_tokens = [
            "there",
            "are",
            "##n't",
            "any",
            "hair",
            "##dress",
            "##er",
            "##s",
        ]
        assert count_split_words_in_sentence(sentence_tokens) == 2

    @pytest.mark.skip("todo")
    def test_estimate_sentence_probability(self):
        estimate_sentence_probability()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_estimate_sentence_probability_from_text(self):
        estimate_sentence_probability_from_text()
        raise NotImplementedError

    def test_get_bert_output(self):

        # tokenizer_m = Mock(spec=BertTokenizerFast)
        output_m = Mock(spec=MaskedLMOutput)
        model_m = Mock(spec=BRT_M, return_value=output_m)
        # sentence = "Ha detto che il libro di ***mask*** ha 300 pagine."
        # tokens_list = ["He", "said", "the", "[MASK]", "book", "has", "300", "pages"]
        sentence_ids = [CLS_ID, 555, 556, 557, MASK_ID, 558, 559, 560, 561, SEP_ID]
        sentence_tokens_count = len(sentence_ids)
        masked_word_idx = 4

        # bert output is a MaskedLMOutput
        vocab_size = 1000
        # why there is one value for each token in the sentence, instead of just for the masked word(s)?
        # check if for the non masked words they are all zeros
        logits = torch.rand(1, sentence_tokens_count, vocab_size)
        output_m.logits = logits

        (
            logits_word_predictions,
            softmax_probabilities_word_predictions,
            logistic_probabilities_word_predictions,
        ) = get_bert_output_single_masking(model_m, sentence_ids, masked_word_idx)

        assert isinstance(logits_word_predictions, torch.Tensor)
        # assert res.shape == (k,)
        assert logits_word_predictions.shape == (vocab_size,)
        assert softmax_probabilities_word_predictions.shape == (vocab_size,)
        assert logistic_probabilities_word_predictions.shape == (vocab_size,)

    @pytest.mark.skip("todo")
    def test_get_score_descr(self):
        get_score_descr()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_probs_from_word_ids(self):
        get_sentence_probs_from_word_ids()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_scores(self):
        get_sentence_scores()
        raise NotImplementedError

    # @pytest.mark.parametrize("k", [5])
    def test_get_top_k(self):  # k
        k = 5
        vocab_size = 1000
        tokens = [
            "[CLS]",
            "He",
            "said",
            "the",
            "[MASK]",
            "book",
            "has",
            "300",
            "pages",
            "[SEP]",
        ]

        mask_ix = 4

        tokenizer_m = Mock(spec=BertTokenizerFast)
        tokenizer_m.convert_tokens_to_ids.return_value = [
            CLS_ID,
            555,
            556,
            557,
            MASK_ID,
            558,
            559,
            560,
            561,
            SEP_ID,
        ]
        output_m = Mock(spec=MaskedLMOutput)
        output_m.logits = torch.rand(1, len(tokens), vocab_size)
        model_m = Mock(spec=BRT_M, return_value=output_m)
        # model_m.return_value.logits = logits

        sentence_ids = tokenizer_m.convert_tokens_to_ids(tokens)

        (
            logits_word_predictions,
            softmax_probabilities_word_predictions,
            logistic_probabilities_word_predictions,
        ) = (
            torch.rand(vocab_size),
            torch.rand(vocab_size),
            torch.rand(vocab_size),
        )
        mock_get_bert_output = self.create_patch(
            "src.linguistic_tests.bert_utils.get_bert_output_single_masking"
        )
        mock_get_bert_output.return_value = (
            logits_word_predictions,
            softmax_probabilities_word_predictions,
            logistic_probabilities_word_predictions,
        )
        assert (
            src.linguistic_tests.bert_utils.get_bert_output_single_masking
            is mock_get_bert_output
        )

        topk_tokens_m = random.sample(range(0, vocab_size - 1), k)
        topk_probs_m = torch.rand(k)
        mock_get_topk_tokens_from_bert_output = self.create_patch(
            "src.linguistic_tests.bert_utils.get_topk_tokens_from_bert_output"
        )
        mock_get_topk_tokens_from_bert_output.return_value = (
            topk_tokens_m,
            topk_probs_m,
        )

        topk_tokens, topk_probs_softmax, topk_probs_logistic = get_topk(
            model_m, tokenizer_m, sentence_ids, mask_ix, k=k
        )

        assert isinstance(topk_tokens, list)
        assert len(topk_tokens) == k
        assert isinstance(topk_probs_softmax, torch.Tensor)
        assert topk_probs_softmax.shape == (k,)
        assert isinstance(topk_probs_logistic, torch.Tensor)
        assert topk_probs_logistic.shape == (k,)

    @pytest.mark.skip("todo")
    def test_get_topk_tokens_from_bert_output(self):
        get_topk_tokens_from_bert_output()
        raise NotImplementedError

    def test_tokenize_sentence(self):
        sequence = "He said the ***mask*** book has 300 pages."
        # vocab_size = 1000

        # data = {"input_ids": torch.tensor([[CLS_ID, 555, MASK_ID, SEP_ID]])}
        # be = BatchEncoding(data=data)  # it's like a python dictionary
        tokens_list = ["He", "said", "the", "[MASK]", "book", "has", "300", "pages"]
        tokenizer_return_values = [
            ["He", "said", "the"],
            ["book", "has", "300", "pages"],
        ]
        tokenizer_m = Mock(spec=BRT_T)
        tokenizer_m.tokenize.side_effect = tokenizer_return_values
        tokens, target_idx = tokenize_sentence(tokenizer_m, sequence)  # masked_word_idx
        # tokenizer.tokenize(pre) returns a list of strings
        # tokenized_target = tokenizer.tokenize(target)
        # tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        assert isinstance(tokens, list)
        all(isinstance(x, str) for x in tokens)
        assert len(tokens) == len(tokens_list) + 2

    @pytest.mark.skip(reason="todo: avoid loading large transformers model")
    def test_get_bert_sentence_score(self):
        # sentence1 = "Gianni ha detto che il manuale di linguistia ha duecento pagine."

        bert_model_name = "../models/bert-base-italian-xxl-cased/"
        bert_model = BRT_M.from_pretrained(bert_model_name)  #
        # bert_model_compare = BertForMaskedLM_Compare.from_pretrained(bert_model_name)
        bert_tokenizer = BRT_T.from_pretrained(
            bert_model_name,
            do_lower_case=(True if "uncased" in bert_model_name else False),
        )
        # bert_tokenizer_compare = BertTokenizer_Compare.from_pretrained(bert_model_name,
        #                                               do_lower_case=(True if "uncased" in bert_model_name else False))
        # bert_tokenized_sentence = bert_tokenizer.tokenize(sentence1)
        # bert_text_len = len(bert_tokenized_sentence)

        # gpt_model_name = (
        #     "../models/GroNLP-gpt2-small-italian"  # "GroNLP/gpt2-small-italian"
        # )
        # gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        # gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)

        testsets_dir = "../outputs/syntactic_tests_it/"
        testset_filename = "wh_adjunct_islands.jsonl"  # 'wh_complex_np_islands.jsonl', 'wh_subject_islands.jsonl', 'wh_whether_island.jsonl', 'variations_tests.jsonl'

        testset_filepath = os.path.join(testsets_dir, testset_filename)
        testset_examples = (load_testset_data(testset_filepath))["sentences"]

        MAX_EXAMPLES = 5
        for example_idx, example in tqdm(
            enumerate(testset_examples), total=min(len(testset_examples), MAX_EXAMPLES)
        ):  # or enumerate(tqdm(testset_examples))
            if example_idx >= MAX_EXAMPLES:
                break
            for sentence_idx, sentence in enumerate(
                get_sentences_from_example(example, 2)
            ):
                (bert_sentence_lp_actual, _,) = estimate_sentence_probability_from_text(
                    bert_model, bert_tokenizer, sentence
                )
                bert_tokenized_sentence = bert_tokenizer.tokenize(sentence)
                # gpt_text_len = len(gpt_tokenized_sentence)
                bert_sentence_lp_expected, _, _ = get_sentence_acceptability_score(
                    ModelTypes.BERT,
                    bert_model,
                    bert_tokenizer,
                    bert_tokenized_sentence,
                    device=None,
                )
                with self.subTest(example=example_idx, sentence=sentence_idx):
                    # self.assertEqual(bert_sentence_lp_expected, bert_sentence_lp_actual)
                    self.assertAlmostEqual(
                        bert_sentence_lp_expected, bert_sentence_lp_actual, 4
                    )

                # gpt_sentence_lp_expected, _ = get_sentence_score_JHLau(ModelTypes.GPT, gpt_model, gpt_tokenizer,
                #                                                    gpt_tokenized_sentence, device=None)


def print_tensor_ids_as_tokens(tens: torch.Tensor, tokenizer: BRT_T, msg):
    # fixme: in calling tests, patch tokenizer.ids_to_tokens()
    print_orange("print_tensor_ids_as_tokens")
    tens = torch.squeeze(tens)
    nparr = tens.numpy()
    df = pd.DataFrame(nparr)

    print(f"tens size: {tens.size()}, nparr shape: {nparr.shape}, df shape: {df.shape}")

    df = df.applymap(lambda x: tokenizer.ids_to_tokens[x])
    print(df)
    print(f"{msg} ({df.shape}): {df}")
    res_topk_tokens = nparr.apply_(lambda x: tokenizer.ids_to_tokens[x])
    print(f"{msg} ({res_topk_tokens.size()}): {res_topk_tokens}")


def test_bert_output():
    vocab_size = 1000
    tokens = [
        "[CLS]",
        "He",
        "said",
        "the",
        "[MASK]",
        "book",
        "has",
        "300",
        "pages",
        "[SEP]",
    ]
    tokens_count = len(tokens)
    # mask_ix = 4
    sentence_ids = [
        CLS_ID,
        555,
        556,
        557,
        MASK_ID,
        558,
        559,
        560,
        561,
        SEP_ID,
    ]
    # tokenizer_m = Mock(spec=BertTokenizerFast)
    # tokenizer_m.convert_tokens_to_ids.return_value = sentence_ids

    input_tensor = torch.LongTensor(sentence_ids)
    batch_size = 1
    input_tensor_batch = input_tensor.unsqueeze(0)  # equivalent to [input_tensor]

    output_m = Mock(spec=MaskedLMOutput)

    output_m.logits = torch.rand(batch_size, len(tokens), vocab_size)
    model_m = Mock(spec=BRT_M, return_value=output_m)

    model_output = model_m(input_tensor_batch)
    logits_whole_batch = (
        model_output.logits
    )  # (batch_size=1, sequence_length, config.vocab_size))

    logits_sentence = torch.squeeze(
        logits_whole_batch
    )  # (sequence_length, config.vocab_size))

    # todo, check: the softmax should be done individually for each tensor/array logits_sentence_softmax[i]
    # where i is the index of a token in the sentence
    # and why we also have a dimension with one entry for each token in the sentence, rather than just
    # for the masked token? possible answer: because there might be more than one masked token (ie 15% of
    # sentence); but then, should't the ..logits corresponding to the non-masked token be 0?
    LAST_DIMENSION_IDX = -1
    logits_sentence_softmax = softmax(logits_sentence.detach(), axis=LAST_DIMENSION_IDX)
    logits_sentence_normalized = torch.div(
        logits_sentence.detach(), torch.sum(logits_sentence.detach())
    )

    assert isinstance(input_tensor_batch, torch.Tensor)
    assert isinstance(logits_sentence, torch.Tensor)
    assert isinstance(logits_whole_batch, torch.Tensor)
    assert isinstance(
        logits_sentence_softmax, (torch.Tensor, ndarray)
    ), f"type(logits_sentence_softmax)={type(logits_sentence_softmax)}"
    assert isinstance(logits_sentence_normalized, torch.Tensor)

    assert input_tensor_batch.shape == (batch_size, tokens_count)
    assert logits_whole_batch.shape == (batch_size, tokens_count, vocab_size)

    assert logits_sentence.shape == (tokens_count, vocab_size)
    assert logits_sentence_softmax.shape == (tokens_count, vocab_size)
    assert logits_sentence_normalized.shape == (tokens_count, vocab_size)
