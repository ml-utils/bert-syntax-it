import os
from unittest import TestCase
from unittest.mock import Mock

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
from transformers import BatchEncoding
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import BertTokenizerFast
from transformers.modeling_outputs import MaskedLMOutput

from src.linguistic_tests.bert_utils import get_topk

# from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
# from torch import softmax

CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


@pytest.mark.skip(reason="WIP")
@pytest.mark.parametrize("k", [5])
def test_get_top_k(k):
    # sequence = "Hello [MASK]"
    vocab_size = 1000

    data = {"input_ids": torch.tensor([[CLS_ID, 555, MASK_ID, SEP_ID]])}
    be = BatchEncoding(data=data)

    logits = torch.rand(1, 4, vocab_size)

    tokenizer_m = Mock(spec=BertTokenizer, return_value=be, mask_token_id=MASK_ID)
    model_m = Mock(spec=BertForMaskedLM)
    model_m.return_value.logits = logits
    # batch_enc = tokenizer_m(sequence, return_tensors="pt")
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
    # sentence_ids = batch_enc["input_ids"]
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
    mask_ix = 4
    sentence_ids = tokenizer_m.convert_tokens_to_ids(tokens)
    res = get_topk(model_m, tokenizer_m, sentence_ids, mask_ix, k=k)

    assert isinstance(res, torch.Tensor)
    assert res.shape == (k,)


def test_get_bert_output():

    tokenizer_m = Mock(spec=BertTokenizerFast)
    output_m = Mock(spec=MaskedLMOutput)
    model_m = Mock(spec=BertForMaskedLM, return_value=output_m)
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
    model_m = Mock(spec=BertForMaskedLM)
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


@pytest.mark.skip(reason="todo: avoid loading large transformers model")
def test_load_model_and_tokenizer():
    # todo
    model_name = "bert-base-uncased"
    # eval_suite = "it"
    bert, tokenizer = load_model_and_tokenizer(
        model_types.BERT, model_name, do_lower_case=False
    )
    print(f"{type(bert)=}, {type(tokenizer)=}")  # BertForMaskedLM, PreTrainedTokenizer


def test_tokenize_sentence():
    sequence = "He said the ***mask*** book has 300 pages."
    # vocab_size = 1000

    # data = {"input_ids": torch.tensor([[CLS_ID, 555, MASK_ID, SEP_ID]])}
    # be = BatchEncoding(data=data)  # it's like a python dictionary
    tokens_list = ["He", "said", "the", "[MASK]", "book", "has", "300", "pages"]
    tokenizer_return_values = [["He", "said", "the"], ["book", "has", "300", "pages"]]
    tokenizer_m = Mock(
        spec=BertTokenizer,  # BertTokenizerFast
        # return_value=tokens_list,
        # mask_token_id=MASK_ID
        side_effect=mock_tokenize,
    )
    tokenizer_m.tokenize.side_effect = tokenizer_return_values
    tokens, target_idx = tokenize_sentence(tokenizer_m, sequence)  # masked_word_idx
    # tokenizer.tokenize(pre) returns a list of strings
    # tokenized_target = tokenizer.tokenize(target)
    # tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert isinstance(tokens, list)
    all(isinstance(x, str) for x in tokens)
    assert len(tokens) == len(tokens_list) + 2


def mock_tokenize(str):
    return ["book", "has", "300", "pages"]


class TestBertUtils(TestCase):
    @pytest.mark.skip(reason="TODO, not implemented")
    def test_load_model_and_tokenizer(self):
        return 0

    @pytest.mark.skip(reason="todo: avoid loading large transformers model")
    def test_get_bert_sentence_score(self):
        # sentence1 = "Gianni ha detto che il manuale di linguistia ha duecento pagine."

        bert_model_name = "../models/bert-base-italian-xxl-cased/"
        bert_model = BertForMaskedLM.from_pretrained(bert_model_name)  #
        # bert_model_compare = BertForMaskedLM_Compare.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(
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


def test_get_acceptability_diffs():
    # TODO: test method before change with option to score method (softmax or logits)
    return 0


def print_tensor_ids_as_tokens(tens: torch.Tensor, tokenizer: BertTokenizer, msg):
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


@pytest.mark.skip(reason="todo: avoid loading large transformers model")
def test_bert_output(bert, sentence_ids, verbose=False):
    tens = torch.LongTensor(sentence_ids).unsqueeze(0)

    res_unsliced = bert(tens)
    res_sequeezed = torch.squeeze(res_unsliced)
    res = res_sequeezed

    res_softmax = softmax(res.detach(), -1)
    res_normalized = torch.div(res.detach(), torch.sum(res.detach()))
    if verbose:
        print(f"tens size {tens.size()}")
        print(f"res_unsliced size {res_unsliced.size()}")
        print(f"res size {res.size()}")
        print(f"res_softmax size {res_softmax.size()}")
        print(f"res_normalized size {res_normalized.size()}")

        print(f"res_unsliced {res_unsliced}")
        print(f"res_unsliced[0] {res_unsliced[0]}")
        print(f"res_unsliced[0][0] {res_unsliced[0][0]}")
        print(f"res_unsliced[0][12] {res_unsliced[0][12]}")
        print(f"res_unsliced[0][12][0] {res_unsliced[0][12][0]}")
        print(f"res_unsliced[0][12][32101] {res_unsliced[0][12][32101]}")
        print(f"res {res}")
        print(f"res[0] {res[0]}")
        print(f"res[12] {res[12]}")
        print(f"res[12][0] {res[12][0]}")
        print(f"res[12][32101] {res[12][32101]}")
