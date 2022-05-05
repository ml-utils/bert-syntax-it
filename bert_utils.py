import json
import torch
from pytorch_pretrained_bert import BertForMaskedLM, tokenization
from torch.nn.functional import softmax
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import numpy as np


def load_testset_data(file_path):
    with open(file_path, 'r') as json_file:
        #json_list = list(json_file)
        testset_data = json.load(json_file)

        #for i in data:
        #    print(i)

    return testset_data


def analize_sentence(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence: str):
    """
    :param bert:
    :param tokenizer:
    :param sentence: a string in which the word to mask is delimited with asteriscs: ***word***
    :return: topk_tokens, topk_probs
    """
    tokens, target_idx = tokenize_sentence(tokenizer, sentence)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    return get_topk(bert, tokenizer, sentence_ids, target_idx)


def get_topk(bert: BertPreTrainedModel, tokenizer: BertTokenizer,
             sentence_ids, masked_word_idx):
    res, res_softmax = get_bert_output(bert, tokenizer, sentence_ids, masked_word_idx)
    return get_topk_tokens_from_bert_output(res_softmax, tokenizer)


def get_topk_tokens_from_bert_output(res_softmax, tokenizer, k=5):
    topk_probs, topk_ids = torch.topk(res_softmax, k)  # todo: get ids/indexes, not their probability value
    topk_tokens = convert_ids_to_tokens(tokenizer, topk_ids)
    return topk_tokens, topk_probs


def get_bert_output(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence_ids, masked_word_idx):
    # todo: check that masked_word_idx remains correct when some words are split (and therefore there are more tokens than words)
    tens = torch.LongTensor(sentence_ids).unsqueeze(0)

    res_unsliced = bert(tens)
    res=res_unsliced[0, masked_word_idx]

    # todo: produce probabilities not with softmax (not using an exponential, to avoiding the maximization of top results),
    #  then compare these probailities with the softmax ones, expecially for ungrammatical sentences
    res_softmax = softmax(res.detach(), -1)
    #RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    return res, res_softmax


def convert_ids_to_tokens(tokenizer, ids):
    """Converts a sequence of ids in wordpiece tokens using the vocab."""
    tokens = []
    for i in ids:
        if torch.is_tensor(i):
            i = i.item()
        # print(f"id: {i}, type: {type(i)}")
        try:
            tokens.append(tokenizer.ids_to_tokens[i])
        except:
            print(f"Unable to find id {i} {type(i)} in the vocabulary")

    return tokens


def estimate_sentence_probability(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence_ids: list[int]):
    # iterate for each word, mask it and get the probability
    # sum the logs of the probabilities

    MASK_ID = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    CLS_ID = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

    sentence_prob_estimate = 0
    for masked_index in range(len(sentence_ids)):
        masked_sentence_ids = sentence_ids.copy()
        masked_word_id = masked_sentence_ids[masked_index]
        masked_sentence_ids[masked_index] = MASK_ID
        masked_sentence_ids = [CLS_ID] + masked_sentence_ids + [SEP_ID]
        print(f'testing {convert_ids_to_tokens(tokenizer, [masked_word_id])} '
              f'at {masked_index+1} in sentence {convert_ids_to_tokens(tokenizer, masked_sentence_ids)}')
        probability_of_this_masking = get_sentence_probs_from_word_ids(bert, tokenizer, masked_sentence_ids,
                                                                     [masked_word_id], masked_index+1)
        sentence_prob_estimate += np.log(probability_of_this_masking[0])

    # todo: also alternative method with formula from paper balanced on sentence lenght

    return np.exp(sentence_prob_estimate)


def estimate_sentence_probability_from_text(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence: str):
    tokens = tokenizer.tokenize(sentence)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    return estimate_sentence_probability(bert, tokenizer, sentence_ids)


def get_probs_for_words(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sent, w1, w2):
    tokens, masked_word_idx = tokenize_sentence(tokenizer, sent)

    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    try:
        masked_words_ids=tokenizer.convert_tokens_to_ids([w1,w2])
    except KeyError:
        print("skipping",w1,w2,"bad wins")
        return None

    return get_sentence_probs_from_word_ids(bert, tokenizer, sentence_ids, masked_words_ids, masked_word_idx)


def get_sentence_probs_from_word_ids(bert: BertPreTrainedModel, tokenizer: BertTokenizer,
                                     sentence_ids, masked_words_ids, masked_word_idx):
    res, res_softmax = get_bert_output(bert, tokenizer, sentence_ids, masked_word_idx)

    topk_tokens, top_probs = get_topk_tokens_from_bert_output(res_softmax, tokenizer, k=10)

    scores = res[masked_words_ids]
    probs = [float(x) for x in scores]
    scores_softmax = res_softmax[masked_words_ids]
    probs_softmax = [float(x) for x in scores_softmax]

    return probs_softmax  # probs


def tokenize_sentence(tokenizer: BertTokenizer, sent: str):
    print(f'sent: {sent}')
    pre,target,post=sent.split('***')
    print(f'pre: {pre}, target: {target}, post: {post}')
    if 'mask' in target.lower():
        target=['[MASK]']
    else:
        target=tokenizer.tokenize(target)

    # todo, fixme: the vocabulary of the pretrained model from Bostrom & Durrett (2020) does not have entries for CLS, UNK
    # fixme: tokenizer.tokenize(pre), does not recognize the words
    tokens = ['[CLS]']+tokenizer.tokenize(pre)  # tokens = tokenizer.tokenize(pre)
    target_idx=len(tokens)
    print(f'target_idx: {target_idx}')
    tokens += target + tokenizer.tokenize(post)+['[SEP]']
    print(f'tokens {tokens}')
    return tokens, target_idx