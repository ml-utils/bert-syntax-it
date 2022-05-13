import json
import math

import torch
from pytorch_pretrained_bert import BertForMaskedLM, tokenization
from torch.nn.functional import softmax
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import numpy as np


GOOD_SENTENCE_1_IDX = 0
SENTENCE_BAD_EXTRACTION_IDX = 1
GOOD_SENTENCE_2_IDX = 2
UNK_TOKEN = '[UNK]'


def load_testset_data(file_path):
    with open(file_path, mode='r', encoding="utf-8") as json_file:
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

    # todo: print tokp for all maskings of the sentence

    return get_topk(bert, tokenizer, sentence_ids, target_idx, k=5)


def get_sentences_from_example(example : dict):
    by_sentence_variant_name = False

    if by_sentence_variant_name:
        sentence_names_wh_wheter_islands = ['sentence_good_no_extraction', 'sentence_bad_extraction',
                          'sentence_good_extraction_resumption', 'sentence_good_extraction_as_subject']
        sentence_names_wh_complex_np_islands = ['sentence_good_no_extraction', 'sentence_bad_extraction',
                                            'sentence_good_no_island', 'sentence_good_no_island_as_subject']
        sentence_names = sentence_names_wh_complex_np_islands
        sentences = []
        for sentence_name in sentence_names:
            sentences.append(example[sentence_name])
    else:
        sentences = list(example.values())[0:3]

    return sentences


def analize_example(bert: BertPreTrainedModel, tokenizer: BertTokenizer, example_idx: int, example,
                    score_based_on='softmax'):
    """
    :param bert:
    :param tokenizer:
    :param example: 2-4 sentences of an example
    :return:
    """

    sentences = get_sentences_from_example(example)
    tokens_by_sentence, oov_counts = __get_example_tokens_and_oov_counts(tokenizer, sentences)
    __check_unk_and_num_tokens(example_idx, sentences, tokens_by_sentence)

    sentence_probability_estimates, logitis_by_sentence, sentences_estimates_normalized_logitis \
        = __get_example_estimates(bert, tokenizer, sentences, tokens_by_sentence)

    base_sentence_less_acceptable, second_sentence_less_acceptable, \
    acceptability_diff_base_sentence, acceptability_diff_second_sentence, \
           penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence, \
           logitis_normalized_bad_sentence, logitis_normalized_base_sentence, logitis_normalized_2nd_good_sentence \
        = __get_acceptability_diffs(bert, tokenizer, sentence_probability_estimates, sentences_estimates_normalized_logitis,
                                    example_idx, oov_counts, sentences, tokens_by_sentence)

    return base_sentence_less_acceptable, second_sentence_less_acceptable, \
           acceptability_diff_base_sentence, acceptability_diff_second_sentence, \
           penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence, \
           logitis_normalized_bad_sentence, logitis_normalized_base_sentence, logitis_normalized_2nd_good_sentence, \
           oov_counts
    # todo: check if the lp (log prob) calculation is the same as in the perper Lau et al. 2020) for estimating sentence probability
    # check also if it reproduces the results from the paper (for english bert), and greenberg, and others
    # check differences btw english and italian bert
    # check diff btw bert base and large
    # ..check diff btw case and uncased.. (NB some problems where due to the open file not in utf format)
    # ..checks for gpt ..
    # analise the good sentences with low probability: for each masking, see the topk
    # ..
    # todo: analize limitations of sentence acceptability estimates with Bert:
    # series of sentences, show how using a different vocabulary choice (more or less frequent), changes the
    # score/acceptability of the sentence, making it more or less likely/acceptable than a previous one,
    # regardless of grammaticality
    # ..
    # new test sets exploring what sentence changes sways acceptability estimates
    #
    # todo: skip examples that don't have at least 3 sentences
    # ..


def __get_example_estimates(bert, tokenizer, sentences, tokens_by_sentence):
    sentence_probability_estimates = []
    logitis_by_sentence = []
    max_logitis = 0
    for sentence_idx, sentence in enumerate(sentences):
        if sentence is not None and len(sentence) > 0:
            # prob = estimate_sentence_probability_from_text(bert, tokenizer, sentence)
            pen_lp, logitis_nonnegative = get_sentence_scores(bert, tokenizer, sentence,
                                                              tokens_by_sentence[sentence_idx])
            sentence_probability_estimates.append(pen_lp)
            logitis_by_sentence.append(logitis_nonnegative)
            if max(logitis_nonnegative) > max_logitis:
                max_logitis = max(logitis_nonnegative)
    normalized_logitis_by_sentence = []
    sentences_estimates_normalized_logitis = []
    for sentence_idx, sentence_logitis in enumerate(logitis_by_sentence):
        normalized_logitis_by_sentence.append([word_logitis / max_logitis for word_logitis in sentence_logitis])
        this_sentence_estimate_normalized_logitis = 0
        for word_logitis in normalized_logitis_by_sentence[sentence_idx]:
            # do math.log of each word score and add to the total
            this_sentence_estimate_normalized_logitis += math.log(word_logitis)
        sentences_estimates_normalized_logitis.append(this_sentence_estimate_normalized_logitis)

    return sentence_probability_estimates, logitis_by_sentence, sentences_estimates_normalized_logitis


def __check_unk_and_num_tokens(example_idx, sentences, tokens_by_sentence):
    for sentence_idx, sentence_tokens in enumerate(tokens_by_sentence):
        if UNK_TOKEN in sentence_tokens:
            print_red(f'this sentence {sentence_idx} ({sentences[sentence_idx]}) has at least an UNK token: '
                      f'{sentences[sentence_idx]}')

    # the ungrammatical sentence must not be shorter than the other three sentences
    sentence_bad_tokens_count = len(tokens_by_sentence[SENTENCE_BAD_EXTRACTION_IDX])
    for sentence_idx, sentence_tokens in enumerate(tokens_by_sentence):
        if len(sentence_tokens) < sentence_bad_tokens_count:
            print(f'example {example_idx}:  sentence {sentence_idx} ({sentences[sentence_idx]}) has less tokens '
                  f'({len(sentence_tokens)}) '
                  f'than the bad sentence ({sentence_bad_tokens_count})')
        if len(sentence_tokens) == 0:
            print(sentences[SENTENCE_BAD_EXTRACTION_IDX])


def __get_example_tokens_and_oov_counts(tokenizer, sentences):
    tokens_by_sentence = []
    oov_counts = []
    for sentence in sentences:
        if sentence is not None:
            tokens = tokenizer.tokenize(sentence)
            tokens_by_sentence.append(tokens)
            oov_counts.append(count_split_words_in_sentence(tokens))
    return tokens_by_sentence, oov_counts

def __get_acceptability_diffs(bert, tokenizer, sentence_probability_estimates, sentences_estimates_normalized_logitis,
                              example_idx, oov_counts, sentences, tokens_by_sentence):
    penLP_bad_sentence = sentence_probability_estimates[SENTENCE_BAD_EXTRACTION_IDX]
    penLP_base_sentence = sentence_probability_estimates[GOOD_SENTENCE_1_IDX]
    logitis_normalized_bad_sentence = sentences_estimates_normalized_logitis[SENTENCE_BAD_EXTRACTION_IDX]
    logitis_normalized_base_sentence = sentences_estimates_normalized_logitis[GOOD_SENTENCE_1_IDX]

    penLP_2nd_good_sentence = None
    logitis_normalized_2nd_good_sentence = None
    if len(sentence_probability_estimates) > 2:
        penLP_2nd_good_sentence = sentence_probability_estimates[GOOD_SENTENCE_2_IDX]
        logitis_normalized_2nd_good_sentence = sentences_estimates_normalized_logitis[GOOD_SENTENCE_2_IDX]

    base_sentence_less_acceptable = False
    second_sentence_less_acceptable = False
    acceptability_diff_base_sentence = 0
    acceptability_diff_second_sentence = 0
    for sentence_idx, sentence_prob in enumerate(sentence_probability_estimates):
        if sentence_idx == 0:
            acceptability_diff_base_sentence = sentence_prob - penLP_bad_sentence
        elif sentence_idx == 2:
            acceptability_diff_second_sentence = sentence_prob - penLP_bad_sentence

        if sentence_prob < penLP_bad_sentence:
            print_orange(f'\nexample {example_idx} (oov_count: {oov_counts}): '
                         f'sentence {sentence_idx} ({sentences[sentence_idx]}, '
                         f'has less PenLP ({sentence_prob:.1f}) '
                         f'than the bad sentence ({penLP_bad_sentence:.1f}) ({sentences[SENTENCE_BAD_EXTRACTION_IDX]})')
            sentence_ids = tokenizer.convert_tokens_to_ids(tokens_by_sentence[sentence_idx])
            estimate_sentence_probability(bert, tokenizer, sentence_ids, verbose = True)
            if sentence_idx == 0:
                base_sentence_less_acceptable = True
            elif sentence_idx == 2:
                second_sentence_less_acceptable = True

    return base_sentence_less_acceptable, second_sentence_less_acceptable, \
           acceptability_diff_base_sentence, acceptability_diff_second_sentence, \
           penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence, \
           logitis_normalized_bad_sentence, logitis_normalized_base_sentence, logitis_normalized_2nd_good_sentence


def count_split_words_in_sentence(sentence_tokens):
    split_words_in_sentence = 0  ## count how many ##tokens there are, subtract from total
    token_of_previously_counted_split_word = False
    for token in sentence_tokens:
        if not token.startswith('##'):
            token_of_previously_counted_split_word = False
        elif not token_of_previously_counted_split_word:
            split_words_in_sentence += 1
            token_of_previously_counted_split_word = True
    return split_words_in_sentence


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def red_txt(txt: str):
    return f'{bcolors.RED}{txt}{bcolors.ENDC}'


def print_red(txt: str):
    print_in_color(txt, bcolors.RED)


def print_orange(txt: str):
    print_in_color(txt, bcolors.WARNING)


def print_in_color(txt, color: bcolors):
    print(f'{color}{txt}{bcolors.ENDC}')


def generate_text_with_bert(bert: BertPreTrainedModel, tokenizer: BertTokenizer, starting_word = 'Il'):
    # convert tokens to ids, append mask to the end
    # get topk with k = 1
    # problem: mask prediction is assumed to result in a complete sentence, not one that would be further extended.
    # Could use more than 1 masked word at the end (albeit in training it whas 15 % of the total sentence).
    return 0


def get_sentence_scores(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence, sentence_tokens):

    text_len = len(sentence_tokens)
    lp, logitis_nonnegative = estimate_sentence_probability_from_text(bert, tokenizer, sentence)
    # lp = bert_get_logprobs(sentence_tokens, bert, tokenizer)
    # model_score(tokenize_input, tokenize_context, bert, tokenizer, device, args)
    penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
    pen_lp = lp / penalty
    return pen_lp, logitis_nonnegative


def check_if_word_in_vocabulary():
    return 0


def check_unknown_words(tokenizer: BertTokenizer):
    # NB: the uncased model also strips any accent markers from words. Use bert cased.

    words_to_check = ['è']
    print_token_info(tokenizer, UNK_TOKEN)

    unk_tokens = tokenizer.tokenize(UNK_TOKEN)
    unk_id = tokenizer.convert_tokens_to_ids(unk_tokens)
    print(f'token {UNK_TOKEN} ({unk_tokens}) has ids {unk_id}')
    print_token_info(tokenizer, 'riparata')
    print_token_info(tokenizer, 'non')
    print_token_info(tokenizer, 'che')
    print_token_info(tokenizer, 'Chi')

    words_ids = tokenizer.convert_tokens_to_ids(words_to_check)
    print(words_ids)
    recognized_tokens = convert_ids_to_tokens(tokenizer, words_ids)
    print(recognized_tokens)


def print_token_info(tokenizer: BertTokenizer, word_to_check: str):
    tokens = tokenizer.tokenize(word_to_check)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f'word {word_to_check} hs tokens {tokens} and ids {ids}')

    if len(tokens) > 1:
        is_word_in_vocab = word_to_check in tokenizer.vocab
        print(f'is_word_in_vocab: {is_word_in_vocab}, vocab size: {len(tokenizer.vocab)}')


def get_topk(bert: BertPreTrainedModel, tokenizer: BertTokenizer,
             sentence_ids, masked_word_idx, k=5):
    res, res_softmax, res_normalized, logitis_shifted_above_zero \
        = get_bert_output(bert, tokenizer, sentence_ids, masked_word_idx)
    topk_probs_nonsoftmax = torch.topk(res_normalized, k)
    topk_tokens, topk_probs = get_topk_tokens_from_bert_output(res_softmax, tokenizer, k)
    return topk_tokens, topk_probs, topk_probs_nonsoftmax


def get_topk_tokens_from_bert_output(res_softmax, tokenizer, k=5):
    topk_probs, topk_ids = torch.topk(res_softmax, k)  # todo: get ids/indexes, not their probability value
    topk_tokens = convert_ids_to_tokens(tokenizer, topk_ids)
    return topk_tokens, topk_probs


def get_bert_output2(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence_ids, masked_word_idx, verbose=False):
    tens = torch.LongTensor(sentence_ids).unsqueeze(0)

    res_unsliced = bert(tens)
    res_sequeezed = torch.squeeze(res_unsliced)
    res_sliced = res_sequeezed[masked_word_idx]
    res = res_sliced
    res_softmax = softmax(res.detach(), -1)
    logitis_min_abs = torch.abs(torch.min(res.detach()))
    logitis_shifted_above_zero = torch.add(res.detach(), logitis_min_abs)
    logitis_sum = torch.sum(logitis_shifted_above_zero)
    res_normalized = torch.div(logitis_shifted_above_zero, logitis_sum)
    res_normalized_sum = torch.sum(res_normalized)
    if verbose:
        print(f'res size {res.size()} {res}')
        print(f'res_softmax size {res_softmax.size()} {res_softmax}')
        print(f'res_normalized size {res_normalized.size()} {res_normalized}')
        print(f'logitis_max {torch.max(res)}, logitis_min_abs {logitis_min_abs}')

        k = 5
        _, res_topk_ids = torch.topk(res, k)
        _, res_softmax_topk_ids = torch.topk(res_softmax, k)
        res_normalized_topk_probs, res_normalized_topk_ids = torch.topk(res_normalized, k+200)

        print(f'res_topk_ids {res_topk_ids}')
        print(f'res_softmax_topk_ids {res_softmax_topk_ids}')
        print(f'res_normalized_topk_ids {res_normalized_topk_ids}')
        print(f'logitis_sum {logitis_sum}, res_normalized_sum {res_normalized_sum}')

        print(f'res_normalized_topk_probs {res_normalized_topk_probs}')
        print(f'normalized probs from topids by softmax: ', [res_normalized[top_id_from_softmax] for top_id_from_softmax in res_softmax_topk_ids])


    # topk res[mask_idx]

    res_sliced = torch.select(res, 0, masked_word_idx)
    print(f'res_sliced size {res_sliced.size()} {res_sliced}')
    # print(f'res[masked_word_idx] size {res[masked_word_idx].size()} {res[masked_word_idx]}')

    return res, res_softmax, res_normalized


def get_bert_output(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence_ids, masked_word_idx, verbose=False):
    # todo: check that masked_word_idx remains correct when some words are split (and therefore there are more tokens than words)
    tens = torch.LongTensor(sentence_ids).unsqueeze(0)

    res_unsliced = bert(tens)
    res = res_unsliced[0, masked_word_idx]

    # todo: produce probabilities not with softmax (not using an exponential, to avoiding the maximization of top results),
    #  then compare these probailities with the softmax ones, expecially for ungrammatical sentences
    res_softmax = softmax(res.detach(), -1)

    logitis_min_abs = torch.abs(torch.min(res.detach()))
    logitis_shifted_above_zero = torch.add(res.detach(), logitis_min_abs)
    logitis_sum = torch.sum(logitis_shifted_above_zero)
    res_normalized = torch.div(logitis_shifted_above_zero, logitis_sum)

    if verbose:
        print(f'tens size {tens.size()}')
        print(f'res_unsliced size {res_unsliced.size()}')
        print(f'res size {res.size()}')
        print(f'res_softmax size {res_softmax.size()}')
        print(f'res_normalized size {res_normalized.size()}')

    #RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    return res, res_softmax, res_normalized, logitis_shifted_above_zero


def convert_ids_to_tokens(tokenizer: BertTokenizer, ids):
    """Converts a sequence of ids in wordpiece tokens using the vocab."""
    tokens = []
    for i in ids:
        if torch.is_tensor(i):
            if torch.numel(i) > 1:
                print_orange(f'Warning: tensor has more than one item: {i.size()}')
            i = i.item()
        # print(f"id: {i}, type: {type(i)}")
        try:
            tokens.append(tokenizer.ids_to_tokens[i])
        except:
            print(f"Unable to find id {i} {type(i)} in the vocabulary")

    return tokens


def estimate_sentence_probability(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence_ids: list[int],
                                  verbose: bool = False):
    # iterate for each word, mask it and get the probability
    # sum the logs of the probabilities

    MASK_ID = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    CLS_ID = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

    sentence_prob_estimate = 0
    words_logitis_nonnegative = []
    for masked_index in range(len(sentence_ids)):
        masked_sentence_ids = sentence_ids.copy()
        masked_word_id = masked_sentence_ids[masked_index]
        masked_sentence_ids[masked_index] = MASK_ID
        masked_sentence_ids = [CLS_ID] + masked_sentence_ids + [SEP_ID]

        probability_of_words_predictions_in_this_masking, topk_tokens, logitis_nonnegative \
            = get_sentence_probs_from_word_ids(bert, tokenizer, masked_sentence_ids,
                                               [masked_word_id], masked_index+1)
        probability_of_this_masking = probability_of_words_predictions_in_this_masking[0]  # we specified only one word

        # todo: unit test to avoid regressions in the output
        # store, for each word masking, a value in an array, to be normalized later with the max logitis of the 2 sentences to compare
        # (or even return it and postpone the calculation later, doind a max over all examples, to have comparable measures)
        # three arrays, one for each score measure: with softmax, and normalized logitis, and normalized logitis as probs
        words_logitis_nonnegative.append(logitis_nonnegative[0])
        sentence_prob_estimate += np.log(probability_of_this_masking)
        if verbose:
            print(f'testing {convert_ids_to_tokens(tokenizer, [masked_word_id])} '
                  f'at {masked_index+1} in sentence {convert_ids_to_tokens(tokenizer, masked_sentence_ids)}, '
                  f'topk_tokens: {topk_tokens}')
            # todo: also print the topk for the masking prediction

    # todo: also alternative method with formula from paper balanced on sentence lenght

    log_prob = sentence_prob_estimate
    return log_prob, words_logitis_nonnegative  # np.exp(log_prob)


def bert_get_logprobs(tokenize_input, model, tokenizer):
    batched_indexed_tokens = []
    batched_segment_ids = []

    tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]


    for i in range(len(tokenize_input)):
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_index = i + 1
        tokenize_masked = tokenize_combined.copy()
        tokenize_masked[masked_index] = '[MASK]'
        # unidir bert
        # for j in range(masked_index, len(tokenize_combined)-1):
        #    tokenize_masked[j] = '[MASK]'

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segment_ids = [0] * len(tokenize_masked)

        batched_indexed_tokens.append(indexed_tokens)
        batched_segment_ids.append(segment_ids)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(batched_indexed_tokens)
    segment_tensor = torch.tensor(batched_segment_ids)

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segment_tensor)
        predictions = outputs[0]

    # go through each word and sum their logprobs
    lp = 0.0
    for i in range(len(tokenize_input)):
        masked_index = i + 1
        predicted_score = predictions[i, masked_index]
        predicted_prob = softmax(predicted_score)  # softmax(predicted_score.cpu().numpy())
        lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])

    return lp


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

    probs_for_words, topk_tokens, _ = get_sentence_probs_from_word_ids(bert, tokenizer, sentence_ids,
                                                                    masked_words_ids, masked_word_idx)
    return probs_for_words


def get_sentence_probs_from_word_ids(bert: BertPreTrainedModel, tokenizer: BertTokenizer,
                                     sentence_ids, masked_words_ids, masked_word_idx, scorebase = 'softmax'):
    """

    :param bert:
    :param tokenizer:
    :param sentence_ids:
    :param masked_words_ids:
    :param masked_word_idx:
    :param scorebase: default softmax. Possible values: softmax, logitis, logitis_nonnegative,
    normalized_prob, normalized
    :return:
    """
    logitis, res_softmax, res_normalized, logitis_shifted_above_zero \
        = get_bert_output(bert, tokenizer, sentence_ids, masked_word_idx)

    topk_tokens, top_probs = get_topk_tokens_from_bert_output(res_softmax, tokenizer, k=10)

    # todo: implement for logitis_nonnegative
    # needs the max value among the two sentences to compare

    if scorebase == 'softmax':
        scores = __get_scores_from_word_ids(res_softmax, masked_words_ids)
    elif scorebase == 'logitis':
        scores = __get_scores_from_word_ids(logitis,masked_words_ids)
    else:
        raise Exception('Error, no scorebase defined.')

    return scores, topk_tokens, __get_scores_from_word_ids(logitis_shifted_above_zero, masked_words_ids)


def __get_scores_from_word_ids(scores, word_ids_mask_predictions):
    scores = scores[word_ids_mask_predictions]
    return [float(x) for x in scores]


def tokenize_sentence(tokenizer: BertTokenizer, sent: str):
    print(f'sent: {sent}')
    pre,target,post=sent.split('***')
    print(f'pre: {pre}, target: {target}, post: {post}')
    if 'mask' in target.lower():
        target=['[MASK]']
    else:
        target = tokenizer.tokenize(target)

    # todo, fixme: the vocabulary of the pretrained model from Bostrom & Durrett (2020) does not have entries for CLS, UNK
    # fixme: tokenizer.tokenize(pre), does not recognize the words
    tokens = ['[CLS]']+tokenizer.tokenize(pre)  # tokens = tokenizer.tokenize(pre)
    target_idx=len(tokens)
    print(f'target_idx: {target_idx}')
    tokens += target + tokenizer.tokenize(post)+['[SEP]']
    print(f'tokens {tokens}')
    return tokens, target_idx