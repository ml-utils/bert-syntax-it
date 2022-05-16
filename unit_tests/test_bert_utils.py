from unittest import TestCase

import numpy
import torch
from pytorch_pretrained_bert import BertTokenizer
import pandas as pd
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch import softmax

import bert_utils, notebook
from lm_utils import model_types


class TestBertUtils(TestCase):
    def test_get_bert_output(self):
        model_name = f'models/bert-base-italian-xxl-cased/'
        eval_suite = 'it'
        bert, tokenizer = notebook.load_model_and_tokenizer(model_types.BERT, model_name, do_lower_case=False)

        sentence = "Ha detto che il libro di ***mask*** ha 300 pagine."
        tokens, masked_word_idx = bert_utils.tokenize_sentence(tokenizer, sentence)
        sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
        res, res_softmax, res_normalized = bert_utils.get_bert_output2(bert, tokenizer, sentence_ids, masked_word_idx,
                                                                      verbose=True)
        k = 5
        _, res_topk_ids = torch.topk(res, k)
        _, res_softmax_topk_ids = torch.topk(res_softmax, k)
        _, res_normalized_topk_ids = torch.topk(res_normalized, k)

        print(f'res_topk_ids ({res_topk_ids.size()}): {res_topk_ids}')
        print(f'res_softmax_topk_ids ({res_softmax_topk_ids.size()}): {res_softmax_topk_ids}')
        print(f'res_normalized_topk_ids ({res_normalized_topk_ids.size()}): {res_normalized_topk_ids}')

        print_tensor_ids_as_tokens(res_topk_ids, tokenizer, 'res_topk_tokens')
        print_tensor_ids_as_tokens(res_softmax_topk_ids, tokenizer, 'res_softmax_topk_tokens')
        print_tensor_ids_as_tokens(res_normalized_topk_ids, tokenizer, 'res_normalized_topk_tokens')

        self.assertListEqual(res_topk_ids, res_softmax_topk_ids)
        self.assertListEqual(res_topk_ids, res_normalized_topk_ids)


def test_get_acceptability_diffs():
    # TODO: test method before change with option to score method (softmax or logits)
    return 0


def print_tensor_ids_as_tokens(tens: torch.Tensor, tokenizer: BertTokenizer, msg):
    bert_utils.print_orange('print_tensor_ids_as_tokens')
    tens = torch.squeeze(tens)
    nparr = tens.numpy()
    df = pd.DataFrame(nparr)

    print(f'tens size: {tens.size()}, nparr shape: {nparr.shape}, df shape: {df.shape}')

    df = df.applymap(lambda x: tokenizer.ids_to_tokens[x])
    print(df)
    print(f'{msg} ({df.shape}): {df}')
    res_topk_tokens = nparr.apply_(lambda x: tokenizer.ids_to_tokens[x])
    print(f'{msg} ({res_topk_tokens.size()}): {res_topk_tokens}')


def test_bert_output(bert: BertPreTrainedModel, sentence_ids, verbose=False):
    tens = torch.LongTensor(sentence_ids).unsqueeze(0)

    res_unsliced = bert(tens)
    res_sequeezed = torch.squeeze(res_unsliced)
    res = res_sequeezed

    res_softmax = softmax(res.detach(), -1)
    res_normalized = torch.div(res.detach(), torch.sum(res.detach()))
    if verbose:
        print(f'tens size {tens.size()}')
        print(f'res_unsliced size {res_unsliced.size()}')
        print(f'res size {res.size()}')
        print(f'res_softmax size {res_softmax.size()}')
        print(f'res_normalized size {res_normalized.size()}')

        print(f'res_unsliced {res_unsliced}')
        print(f'res_unsliced[0] {res_unsliced[0]}')
        print(f'res_unsliced[0][0] {res_unsliced[0][0]}')
        print(f'res_unsliced[0][12] {res_unsliced[0][12]}')
        print(f'res_unsliced[0][12][0] {res_unsliced[0][12][0]}')
        print(f'res_unsliced[0][12][32101] {res_unsliced[0][12][32101]}')
        print(f'res {res}')
        print(f'res[0] {res[0]}')
        print(f'res[12] {res[12]}')
        print(f'res[12][0] {res[12][0]}')
        print(f'res[12][32101] {res[12][32101]}')