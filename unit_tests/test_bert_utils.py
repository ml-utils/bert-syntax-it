import os
import random
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from linguistic_tests.bert_utils import estimate_sentence_probability_from_text
from linguistic_tests.bert_utils import get_bert_output
from linguistic_tests.bert_utils import get_bert_output2
from linguistic_tests.bert_utils import print_orange
from linguistic_tests.bert_utils import tokenize_sentence
from linguistic_tests.compute_model_score import get_sentence_score_JHLau
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from scipy.special import softmax
from tqdm import tqdm
from transformers import BertForMaskedLM as BRT_M
from transformers import BertTokenizer as BRT_T
from transformers import BertTokenizerFast
from transformers import GPT2LMHeadModel as GPT_M
from transformers import GPT2Tokenizer as GPT_T
from transformers.modeling_outputs import MaskedLMOutput

import src
from src.linguistic_tests.bert_utils import get_topk

CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


def test_get_bert_output():

    tokenizer_m = Mock(spec=BertTokenizerFast)
    output_m = Mock(spec=MaskedLMOutput)
    model_m = Mock(spec=BRT_M, return_value=output_m)
    # sentence = "Ha detto che il libro di ***mask*** ha 300 pagine."
    # tokens_list = ["He", "said", "the", "[MASK]", "book", "has", "300", "pages"]
    sentence_ids = [CLS_ID, 555, 556, 557, MASK_ID, 558, 559, 560, 561, SEP_ID]
    masked_word_idx = 4

    # bert output is a MaskedLMOutput
    vocab_size = 1000
    logits = torch.rand(1, len(sentence_ids), vocab_size)
    output_m.logits = logits

    (res, res_softmax, res_normalized, logits_shifted_above_zero) = get_bert_output(
        model_m, tokenizer_m, sentence_ids, masked_word_idx
    )

    assert isinstance(res, torch.Tensor)
    # assert res.shape == (k,)


@pytest.mark.skip(reason="todo: avoid loading large transformers model")
def test_get_bert_output2():
    model_m = Mock(spec=BRT_M)
    tokenizer_m = Mock(spec=BertTokenizerFast, mask_token_id=MASK_ID)
    sentence_ids = [CLS_ID, 555, 556, 557, MASK_ID, 558, 559, 560, 561, SEP_ID]
    masked_word_idx = 4

    res, res_softmax, res_normalized = get_bert_output2(
        model_m, tokenizer_m, sentence_ids, masked_word_idx, verbose=True
    )
    k = 5
    _, res_topk_ids = torch.topk(res, k)
    _, res_softmax_topk_ids = torch.topk(res_softmax, k)
    _, res_normalized_topk_ids = torch.topk(res_normalized, k)

    print(f"res_topk_ids ({res_topk_ids.size()}): {res_topk_ids}")
    print(
        f"res_softmax_topk_ids ({res_softmax_topk_ids.size()}): {res_softmax_topk_ids}"
    )
    print(
        f"res_normalized_topk_ids ({res_normalized_topk_ids.size()}): {res_normalized_topk_ids}"
    )

    print_tensor_ids_as_tokens(res_topk_ids, tokenizer_m, "res_topk_tokens")
    print_tensor_ids_as_tokens(
        res_softmax_topk_ids, tokenizer_m, "res_softmax_topk_tokens"
    )
    print_tensor_ids_as_tokens(
        res_normalized_topk_ids, tokenizer_m, "res_normalized_topk_tokens"
    )

    # self.assertListEqual(res_topk_ids, res_softmax_topk_ids)
    # self.assertListEqual(res_topk_ids, res_normalized_topk_ids)


def test_tokenize_sentence():
    sequence = "He said the ***mask*** book has 300 pages."
    # vocab_size = 1000

    # data = {"input_ids": torch.tensor([[CLS_ID, 555, MASK_ID, SEP_ID]])}
    # be = BatchEncoding(data=data)  # it's like a python dictionary
    tokens_list = ["He", "said", "the", "[MASK]", "book", "has", "300", "pages"]
    tokenizer_return_values = [["He", "said", "the"], ["book", "has", "300", "pages"]]
    tokenizer_m = Mock(spec=BRT_T)
    tokenizer_m.tokenize.side_effect = tokenizer_return_values
    tokens, target_idx = tokenize_sentence(tokenizer_m, sequence)  # masked_word_idx
    # tokenizer.tokenize(pre) returns a list of strings
    # tokenized_target = tokenizer.tokenize(target)
    # tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert isinstance(tokens, list)
    all(isinstance(x, str) for x in tokens)
    assert len(tokens) == len(tokens_list) + 2


class TestBertUtils(TestCase):
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

        res_m, res_softmax_m, res_normalized_m = (
            torch.rand(vocab_size),
            torch.rand(vocab_size),
            torch.rand(vocab_size),
        )
        mock_get_bert_output = self.create_patch(
            "src.linguistic_tests.bert_utils.get_bert_output"
        )
        mock_get_bert_output.return_value = (
            res_m,
            res_softmax_m,
            res_normalized_m,
            None,
        )
        assert src.linguistic_tests.bert_utils.get_bert_output is mock_get_bert_output

        topk_tokens_m = random.sample(range(0, vocab_size - 1), k)
        topk_probs_m = torch.rand(k)
        mock_get_topk_tokens_from_bert_output = self.create_patch(
            "src.linguistic_tests.bert_utils.get_topk_tokens_from_bert_output"
        )
        mock_get_topk_tokens_from_bert_output.return_value = (
            topk_tokens_m,
            topk_probs_m,
        )

        topk_tokens, topk_probs, topk_probs_nonsoftmax = get_topk(
            model_m, tokenizer_m, sentence_ids, mask_ix, k=k
        )

        assert isinstance(topk_tokens, list)
        assert len(topk_tokens) == k
        assert isinstance(topk_probs, torch.Tensor)
        assert topk_probs.shape == (k,)
        assert isinstance(topk_probs_nonsoftmax, torch.return_types.topk)
        assert topk_probs_nonsoftmax.values.shape == (k,)

    def create_patch(self, name):
        patcher = patch(name)
        thing = patcher.start()
        self.addCleanup(patcher.stop)
        return thing

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
                bert_sentence_lp_expected, _ = get_sentence_score_JHLau(
                    model_types.BERT,
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

                # gpt_sentence_lp_expected, _ = get_sentence_score_JHLau(model_types.GPT, gpt_model, gpt_tokenizer,
                #                                                    gpt_tokenized_sentence, device=None)


@pytest.mark.skip(reason="todo")
def test_get_acceptability_diffs():
    # TODO: test method before change with option to score method (softmax or logits)
    return 0


def print_tensor_ids_as_tokens(tens: torch.Tensor, tokenizer: BRT_T, msg):
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

    tens = torch.LongTensor(sentence_ids).unsqueeze(0)

    output_m = Mock(spec=MaskedLMOutput)
    output_m.logits = torch.rand(1, len(tokens), vocab_size)
    model_m = Mock(spec=BRT_M, return_value=output_m)

    res_unsliced = model_m(tens).logits

    res_sequeezed = torch.squeeze(res_unsliced)
    res = res_sequeezed

    res_softmax = softmax(res.detach(), -1)
    res_normalized = torch.div(res.detach(), torch.sum(res.detach()))

    assert isinstance(tens, torch.Tensor)
    assert isinstance(res, torch.Tensor)
    assert isinstance(res_unsliced, torch.Tensor)
    assert isinstance(res_softmax, torch.Tensor)
    assert isinstance(res_normalized, torch.Tensor)

    assert tens.shape == (1, tokens_count)
    assert res_unsliced.shape == (1, tokens_count, vocab_size)

    assert res.shape == (tokens_count, vocab_size)
    assert res_softmax.shape == (tokens_count, vocab_size)
    assert res_normalized.shape == (tokens_count, vocab_size)
