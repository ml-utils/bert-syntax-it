from functools import reduce

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModel, AutoTokenizer  # pytorch_transformers

from transformers import BertTokenizer
from transformers import BertForMaskedLM  # BertModel as BertForMaskedLM  #
from transformers import RobertaTokenizer, RobertaForMaskedLM # RobertaModel

from scipy.special import softmax
import numpy as np

from lm_utils import model_types, get_sentences_from_example
from lm_utils import GOOD_SENTENCE_2_IDX, GOOD_SENTENCE_1_IDX, SENTENCE_BAD_IDX


class DEVICES:
    CPU = 'cpu'
    CUDA = 'cuda:X'


def load_model(model_type, model_name, device):
    # Load pre-trained model and tokenizer
    if model_type == model_types.GPT:
        print(f'loading model {model_name}..')
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f'model loaded. Loading tokenizer {model_name}..')
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print(f'tokenizer loaded.')
    elif model_type == model_types.BERT:
        print(f'loading model {model_name}..')
        model = BertForMaskedLM.from_pretrained(model_name)  # BertForMaskedLM.from_pretrained(model_name)
        print(f'model loaded. Loading tokenizer {model_name}..')
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                  do_lower_case=(True if "uncased" in model_name else False))
        print(f'tokenizer loaded.')
    elif model_type == model_types.ROBERTA:
        print(f'loading model {model_name}..')
        model = RobertaForMaskedLM.from_pretrained(model_name)
        print(f'model loaded. Loading tokenizer {model_name}..')
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        print(f'tokenizer loaded.')
    elif model_type == model_types.GILBERTO:
        print(f'loading model {model_name}..')
        model = AutoModel.from_pretrained(model_name)
        print(f'model loaded. Loading tokenizer {model_name}..')
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        print(f'tokenizer loaded.')
    else:
        return

    # put model to device (GPU/CPU)
    device = torch.device(device)
    model.to(device)

    # eval mode; no dropout
    model.eval()
    return model, tokenizer


def run_testset(model_type, model, tokenizer, device, testset):
    """
    Adapted from https://github.com/jhlau/acceptability-prediction-in-context/blob/master/code/compute_model_score.py
    :param model_type:
    :param model:
    :param tokenizer:
    :param device:
    :param testset:
    :return:
    """
    sent_ids = []

    correct_lps_1st_sentence = 0
    correct_pen_lps_1st_sentence = 0
    correct_lps_2nd_sentence = 0
    correct_pen_lps_2nd_sentence = 0
    correct_logweights_1st_sentence = 0
    correct_logweights_2nd_sentence = 0
    correct_pen_logweights_1st_sentence = 0
    correct_pen_logweights_2nd_sentence = 0
    for example_idx, example_data in enumerate(tqdm(testset['sentences'])):
        sentences = get_sentences_from_example(example_data)
        lps = []
        # mean_lps = []
        pen_lps = []
        sentence_log_weights = []
        pen_sentence_log_weights = []
        token_weights_by_sentence = []
        min_token_weight = 200
        max_token_weight = -200
        normalized_weights = []
        for sent_id, sentence in enumerate(sentences):
            sentence_tokens = tokenizer.tokenize(sentence)  # , return_tensors='pt'
            text_len = len(sentence_tokens)
            lp, token_weights = get_sentence_score_JHLau(model_type, model, tokenizer, sentence_tokens, device)
            if model_type in [model_types.BERT, model_types.ROBERTA]:
                min_token_weight = min(min(token_weights), min_token_weight)
                max_token_weight = max(max(token_weights), max_token_weight)
                token_weights_by_sentence.append(token_weights)
            # acceptability measures by sentence idx
            penalty = get_penalty_term(text_len)
            lps.append(lp)
            # mean_lps.append(lp / text_len)
            pen_lps.append(lp / penalty)
            sent_ids.append(sent_id)
        if model_type in [model_types.BERT, model_types.ROBERTA]:
            # normalize token weights
            max_token_weight -= min_token_weight  # normalize the max value
            for sentence_idx, token_weights_this_sentence in enumerate(token_weights_by_sentence):
                token_weights_by_sentence[sentence_idx] = [(x-min_token_weight)/max_token_weight for x in token_weights_this_sentence]
                sentence_log_weight = reduce_to_log_product(token_weights_by_sentence[sentence_idx])
                sentence_log_weights.append(sentence_log_weight)
                text_lenght = len(token_weights_by_sentence[sentence_idx])
                penalty = get_penalty_term(text_lenght)
                pen_sentence_log_weights.append(sentence_log_weight / penalty)
        if lps[GOOD_SENTENCE_1_IDX] > lps[SENTENCE_BAD_IDX]:
            correct_lps_1st_sentence += 1
        if pen_lps[GOOD_SENTENCE_1_IDX] > pen_lps[SENTENCE_BAD_IDX]:
            correct_pen_lps_1st_sentence += 1
        if model_type in [model_types.BERT, model_types.ROBERTA]:
            if sentence_log_weights[GOOD_SENTENCE_1_IDX] > sentence_log_weights[SENTENCE_BAD_IDX]:
                correct_logweights_1st_sentence += 1
            if pen_sentence_log_weights[GOOD_SENTENCE_1_IDX] > pen_sentence_log_weights[SENTENCE_BAD_IDX]:
                correct_pen_logweights_1st_sentence += 1
        if len(sentences) > 2:
            if lps[GOOD_SENTENCE_2_IDX] > lps[SENTENCE_BAD_IDX]:
                correct_lps_2nd_sentence += 1
            if pen_lps[GOOD_SENTENCE_2_IDX] > pen_lps[SENTENCE_BAD_IDX]:
                correct_pen_lps_2nd_sentence += 1
            if model_type in [model_types.BERT, model_types.ROBERTA]:
                if sentence_log_weights[GOOD_SENTENCE_2_IDX] > sentence_log_weights[SENTENCE_BAD_IDX]:
                    correct_logweights_2nd_sentence += 1
                if pen_sentence_log_weights[GOOD_SENTENCE_2_IDX] > pen_sentence_log_weights[SENTENCE_BAD_IDX]:
                    correct_pen_logweights_2nd_sentence += 1

    examples_count = len(testset['sentences'])
    print(f'test results report:')
    print(f'acc. correct_lps_1st_sentence: {perc(correct_lps_1st_sentence, examples_count):.1f} %')
    print(f'acc. correct_pen_lps_1st_sentence: {perc(correct_pen_lps_1st_sentence, examples_count):.1f} %')
    print(f'acc. correct_lps_2nd_sentence: {perc(correct_lps_2nd_sentence, examples_count):.1f} %')
    print(f'acc. correct_pen_lps_2nd_sentence: {perc(correct_pen_lps_2nd_sentence, examples_count):.1f} %')

    if model_type in [model_types.BERT, model_types.ROBERTA]:
        print(f'acc. correct_logweights_1st_sentence: {perc(correct_logweights_1st_sentence, examples_count):.1f} %')
        print(f'acc. correct_pen_logweights_1st_sentence: {perc(correct_pen_logweights_1st_sentence, examples_count):.1f} %')
        print(f'acc. correct_logweights_2nd_sentence: {perc(correct_logweights_2nd_sentence, examples_count):.1f} %')
        print(f'acc. correct_pen_logweights_2nd_sentence: {perc(correct_pen_logweights_2nd_sentence, examples_count):.1f} %')


def reduce_to_log_product(seq):
    return reduce((lambda x, y: x + np.log(y)), seq, 0)


def count_accurate_in_example(scores_by_sentence):
    correct_1st_sentence_comparison = 0
    if scores_by_sentence[GOOD_SENTENCE_1_IDX] > scores_by_sentence[SENTENCE_BAD_IDX]:
        correct_1st_sentence_comparison = 1

    correct_2nd_sentence_comparison = 0
    if len(scores_by_sentence) > 2:
        if scores_by_sentence[GOOD_SENTENCE_2_IDX] > scores_by_sentence[SENTENCE_BAD_IDX]:
            correct_2nd_sentence_comparison = 1

    return correct_1st_sentence_comparison, correct_2nd_sentence_comparison

def get_penalty_term(text_lenght):
    return (5 + text_lenght) ** 0.8 / (5 + 1) ** 0.8


def perc(value, total):
    return 100 * (value / total)


# nb, for bert it uses softmax
def get_sentence_score_JHLau(model_type: model_types, model, tokenizer, sentence_tokens, device):

    if len(sentence_tokens) == 0:
        return -200, None

    if model_type == model_types.GPT:

        # not use context variant:
        #prepend the sentence with <|endoftext|> token, so that the loss is computed correctly
        tensor_input = torch.tensor([[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(sentence_tokens)], device=device)
        labels = torch.tensor([[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(sentence_tokens)], device=device)
        labels[:,:1] = -1
        loss = model(tensor_input, labels=tensor_input)
        return float(loss[0]) * -1.0 * len(sentence_tokens), None

    elif model_type in [model_types.BERT, model_types.ROBERTA]:  # , model_types.GILBERTO

        batched_indexed_tokens = []
        batched_segment_ids = []

        # not use_context variant:
        tokenize_combined = ["[CLS]"] + sentence_tokens + ["[SEP]"]

        for i in range(len(sentence_tokens)):
            # Mask a token that we will try to predict back with `BertForMaskedLM`
            masked_index = i + 1 + 0  # not use_context variant
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
        tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)
        segment_tensor = torch.tensor(batched_segment_ids, device=device)
        # print(f'sentence_tokens: {sentence_tokens}')
        # Predict all tokens
        with torch.no_grad():
            # print(f'type(model): {type(model)}')
            outputs = model(tokens_tensor, token_type_ids=segment_tensor)
            # when using bert-large-uncased and transformers BertModel:
            # type(outputs) : <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>
            # fields: attentions, cross attentions, hidden states, last hidden state, past key values, pooler output
            # nb fun(**arg): take a dictionary of key-value pairs and unpack it into keyword arguments in a function call.
            # when using bert-large-uncased and transformers BertForMaskedLM:
            # type(outputs) : MaskedLMOutput
            predictions = outputs[0]

        # go through each word and sum their logprobs
        lp = 0.0
        #     logits_min_abs = torch.abs(torch.min(res.detach()))
        #     logits_shifted_above_zero = torch.add(res.detach(), logits_min_abs)
        #     logits_sum = torch.sum(logits_shifted_above_zero)
        #     res_normalized = torch.div(logits_shifted_above_zero, logits_sum)
        tokens_scores = []
        for i in range(len(sentence_tokens)):
            masked_index = i + 1 + 0  # not use_context variant
            predicted_score = predictions[i, masked_index]
            token_score = predicted_score[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]]
            tokens_scores.append(float(token_score))
            predicted_prob = softmax(predicted_score.cpu().numpy())
            lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])
        return lp, tokens_scores
    else:
        print(f'Error: unrecognized model type {model_type}')
