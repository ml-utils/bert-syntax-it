from functools import reduce
from typing import List

import numpy as np
import torch
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import get_penalty_term
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import sent_idx
from linguistic_tests.testset import Example
from linguistic_tests.testset import TestSet
from scipy.special import expit as logistic
from scipy.special import softmax
from tqdm import tqdm


def score_dataclass_testset(
    model_type: ModelTypes, model, tokenizer, device, testset: TestSet
):
    for example_idx, example in enumerate(tqdm(testset.examples)):
        score_example(
            device,
            example,
            model,
            model_type,
            tokenizer,
        )

    # todo, fixme: some scoring measures are calculated only for Bert-like (bidirectional) models, where
    #  the score is just an approximation of the acceptability
    #  if model_type in [ModelTypes.BERT, ModelTypes.ROBERTA, ModelTypes.GILBERTO]:
    for scoring_measure in testset.accuracy_per_score_type_per_sentence_type.keys():
        for (
            stype_acceptable_sentence
        ) in testset.accuracy_per_score_type_per_sentence_type[scoring_measure].keys():
            accurate_count = 0
            for example_idx, example in enumerate(tqdm(testset.examples)):
                if example.is_scored_accurately(
                    scoring_measure, stype_acceptable_sentence
                ):
                    accurate_count += 1
            accuracy = accurate_count / len(testset.examples)
            testset.accuracy_per_score_type_per_sentence_type[scoring_measure][
                stype_acceptable_sentence
            ] = accuracy

    return testset


def print_accuracy_scores(testset: TestSet):
    print(f"test results report, {testset.linguistic_phenomenon}:")
    for scoring_measure in testset.accuracy_per_score_type_per_sentence_type.keys():
        # print(f"scores with {scoring_measure}")
        for (
            stype_acceptable_sentence
        ) in testset.accuracy_per_score_type_per_sentence_type[scoring_measure].keys():
            # fixme: 0 values for accuracy base on logistic scoring measure
            accuracy = testset.accuracy_per_score_type_per_sentence_type[
                scoring_measure
            ][stype_acceptable_sentence]
            print(
                f"Accuracy with {scoring_measure.name} for {stype_acceptable_sentence.name}: {accuracy:%} "
            )


# todo: mark as deprecated, move to test section to use as comparison for outcome of new method
def run_testset(
    model_type: ModelTypes, model, tokenizer, device, testset, sentences_per_example
):
    """
    Adapted from https://github.com/jhlau/acceptability-prediction-in-context/
    blob/master/code/compute_model_score.py
    :param model_type:
    :param model:
    :param tokenizer:
    :param device:
    :param testset:
    :return:
    """
    sent_ids: List[int] = []

    correct_lps_1st_sentence = 0
    correct_pen_lps_1st_sentence = 0
    correct_lps_2nd_sentence = 0
    correct_pen_lps_2nd_sentence = 0
    correct_logweights_1st_sentence = 0
    correct_logweights_2nd_sentence = 0
    correct_pen_logweights_1st_sentence = 0
    correct_pen_logweights_2nd_sentence = 0
    for example_idx, example_data in enumerate(tqdm(testset["sentences"])):
        (
            lps,
            pen_lps,
            lls,
            penlls,
            pen_sentence_log_weights,
            sentence_log_weights,
            sentences,
        ) = get_example_scores(
            device,
            example_data,
            model,
            model_type,
            sent_ids,
            tokenizer,
            sentences_per_example,
        )
        if lps[sent_idx.GOOD_1] > lps[sent_idx.BAD]:
            correct_lps_1st_sentence += 1
        if pen_lps[sent_idx.GOOD_1] > pen_lps[sent_idx.BAD]:
            correct_pen_lps_1st_sentence += 1
        if model_type in BERT_LIKE_MODEL_TYPES:
            if (
                sentence_log_weights[sent_idx.GOOD_1]
                > sentence_log_weights[sent_idx.BAD]
            ):
                correct_logweights_1st_sentence += 1
            if (
                pen_sentence_log_weights[sent_idx.GOOD_1]
                > pen_sentence_log_weights[sent_idx.BAD]
            ):
                correct_pen_logweights_1st_sentence += 1
        if len(sentences) > 2:
            if lps[sent_idx.GOOD_2] > lps[sent_idx.BAD]:
                correct_lps_2nd_sentence += 1
            if pen_lps[sent_idx.GOOD_2] > pen_lps[sent_idx.BAD]:
                correct_pen_lps_2nd_sentence += 1
            if model_type in BERT_LIKE_MODEL_TYPES:
                if (
                    sentence_log_weights[sent_idx.GOOD_2]
                    > sentence_log_weights[sent_idx.BAD]
                ):
                    correct_logweights_2nd_sentence += 1
                if (
                    pen_sentence_log_weights[sent_idx.GOOD_2]
                    > pen_sentence_log_weights[sent_idx.BAD]
                ):
                    correct_pen_logweights_2nd_sentence += 1

    examples_count = len(testset["sentences"])
    print_accuracies(
        examples_count,
        model_type,
        correct_lps_1st_sentence,
        correct_pen_lps_1st_sentence,
        correct_lps_2nd_sentence,
        correct_pen_lps_2nd_sentence,
        correct_logweights_1st_sentence,
        correct_pen_logweights_1st_sentence,
        correct_logweights_2nd_sentence,
        correct_pen_logweights_2nd_sentence,
    )

    return (
        correct_lps_1st_sentence,
        correct_pen_lps_1st_sentence,
        correct_lps_2nd_sentence,
        correct_pen_lps_2nd_sentence,
        correct_logweights_1st_sentence,
        correct_pen_logweights_1st_sentence,
        correct_logweights_2nd_sentence,
        correct_pen_logweights_2nd_sentence,
    )


def print_accuracies(
    examples_count,
    model_type,
    correct_lps_1st_sentence,
    correct_pen_lps_1st_sentence,
    correct_lps_2nd_sentence,
    correct_pen_lps_2nd_sentence,
    correct_logweights_1st_sentence=None,
    correct_pen_logweights_1st_sentence=None,
    correct_logweights_2nd_sentence=None,
    correct_pen_logweights_2nd_sentence=None,
):
    print("test results report:")
    print(
        f"acc. correct_lps_1st_sentence: {perc(correct_lps_1st_sentence, examples_count):.1f} %"
    )
    print(
        f"acc. correct_pen_lps_1st_sentence: {perc(correct_pen_lps_1st_sentence, examples_count):.1f} %"
    )
    print(
        f"acc. correct_lps_2nd_sentence: {perc(correct_lps_2nd_sentence, examples_count):.1f} %"
    )
    print(
        f"acc. correct_pen_lps_2nd_sentence: {perc(correct_pen_lps_2nd_sentence, examples_count):.1f} %"
    )

    if model_type in BERT_LIKE_MODEL_TYPES:
        print(
            f"acc. correct_logweights_1st_sentence: {perc(correct_logweights_1st_sentence, examples_count):.1f} %"
        )
        print(
            f"acc. correct_pen_logweights_1st_sentence: {perc(correct_pen_logweights_1st_sentence, examples_count):.1f} %"
        )
        print(
            f"acc. correct_logweights_2nd_sentence: {perc(correct_logweights_2nd_sentence, examples_count):.1f} %"
        )
        print(
            f"acc. correct_pen_logweights_2nd_sentence: {perc(correct_pen_logweights_2nd_sentence, examples_count):.1f} %"
        )


def score_example(
    device,
    example: Example,
    model,
    model_type,
    tokenizer,
):
    for _idx, typed_sentence in enumerate(example.sentences):
        sentence = typed_sentence.sent
        sentence.tokens = tokenizer.tokenize(sentence.txt)  # , return_tensors='pt'
        if len(sentence.tokens) == 0:
            print(f"Warning: lenght 0 for {sentence=} from {example=}")
        text_len = len(sentence.tokens)
        lp, log_logistic, token_weights = get_sentence_score_JHLau(
            model_type, model, tokenizer, sentence.tokens, device
        )

        sentence.lp = lp
        penalty = get_penalty_term(text_len)
        sentence.pen_lp = lp / penalty
        if model_type in BERT_LIKE_MODEL_TYPES:
            example.min_token_weight = min(min(token_weights), example.min_token_weight)
            example.max_token_weight = max(max(token_weights), example.max_token_weight)
            sentence.token_weights = token_weights
            sentence.log_logistic = log_logistic
            sentence.pen_log_logistic = log_logistic / penalty

    if model_type in BERT_LIKE_MODEL_TYPES:
        # normalize token weights
        example.max_token_weight -= example.min_token_weight  # normalize the max value
        for _idx, typed_sentence in enumerate(example.sentences):
            sentence = typed_sentence.sent
            # normalized the token weights
            sentence.token_weights = [
                (x - example.min_token_weight) / example.max_token_weight
                for x in sentence.token_weights
            ]
        for _idx, typed_sentence in enumerate(example.sentences):
            sentence = typed_sentence.sent
            sentence.sentence_log_weight = reduce_to_log_product(sentence.token_weights)
            text_lenght = len(sentence.tokens)
            penalty = get_penalty_term(text_lenght)
            sentence.pen_sentence_log_weight = sentence.sentence_log_weight / penalty

    return example


def get_example_scores(
    device,
    example_data,
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
    # mean_lps = []
    pen_lps = []
    lls = []
    penlls = []
    sentence_log_weights = []
    pen_sentence_log_weights = []
    token_weights_by_sentence = []
    min_token_weight = 200
    max_token_weight = -200
    # normalized_weights = []
    for sent_id, sentence in enumerate(sentences):
        sentence_tokens = tokenizer.tokenize(sentence)  # , return_tensors='pt'
        if len(sentence_tokens) == 0:
            print(f"Warning: lenght 0 for {sentence=} from {example_data=}")
        text_len = len(sentence_tokens)
        lp, log_logistic, token_weights = get_sentence_score_JHLau(
            model_type, model, tokenizer, sentence_tokens, device
        )
        if model_type in BERT_LIKE_MODEL_TYPES:
            min_token_weight = min(min(token_weights), min_token_weight)
            max_token_weight = max(max(token_weights), max_token_weight)
            token_weights_by_sentence.append(token_weights)
        # acceptability measures by sentence idx
        penalty = get_penalty_term(text_len)
        lps.append(lp)
        # mean_lps.append(lp / text_len)
        pen_lps.append(lp / penalty)
        sent_ids.append(sent_id)
    if model_type in BERT_LIKE_MODEL_TYPES:
        lls.append(log_logistic)
        penlls.append(log_logistic / penalty)

        # normalize token weights
        max_token_weight -= min_token_weight  # normalize the max value
        for sentence_idx, token_weights_this_sentence in enumerate(
            token_weights_by_sentence
        ):
            token_weights_by_sentence[sentence_idx] = [
                (x - min_token_weight) / max_token_weight
                for x in token_weights_this_sentence
            ]
            sentence_log_weight = reduce_to_log_product(
                token_weights_by_sentence[sentence_idx]
            )
            sentence_log_weights.append(sentence_log_weight)
            text_lenght = len(token_weights_by_sentence[sentence_idx])
            penalty = get_penalty_term(text_lenght)
            pen_sentence_log_weights.append(sentence_log_weight / penalty)
    return (
        lps,
        pen_lps,
        lls,
        penlls,
        pen_sentence_log_weights,
        sentence_log_weights,
        sentences,
    )


def reduce_to_log_product(seq):
    return reduce((lambda x, y: x + np.log(y)), seq, 0)  # fixme:
    # RuntimeWarning: divide by zero encountered in log


def count_accurate_in_example(scores_by_sentence):
    correct_1st_sentence_comparison = 0
    if scores_by_sentence[sent_idx.GOOD_1] > scores_by_sentence[sent_idx.BAD]:
        correct_1st_sentence_comparison = 1

    correct_2nd_sentence_comparison = 0
    if len(scores_by_sentence) > 2:
        if scores_by_sentence[sent_idx.GOOD_2] > scores_by_sentence[sent_idx.BAD]:
            correct_2nd_sentence_comparison = 1

    return correct_1st_sentence_comparison, correct_2nd_sentence_comparison


def perc(value, total):
    return 100 * (value / total)


# nb, for bert it uses softmax
def get_sentence_score_JHLau(model_type, model, tokenizer, sentence_tokens, device):
    """

    :param model_type:
    :param model:
    :param tokenizer:
    :param sentence_tokens:
    :param device:
    :return:
    A tuple (lp, tokens_scores), where lp is an acceptability estimate of the
    given sentence, while tokens_scores are the pre-softmax output of the model
    for each sentence obtained by masking one of its tokens. Tokens_scores is
    none for Gpt models, it is given only for Bert like mdels to allow for
    alternative estimates of the sentence acceptability.
    For Gpt models, lp is the output loss of the model for the given sentence,
    while for Bert models is an estimate obtained by summing the logs of the
    scores given by the model as a prediction for each of the tokens in the
    sentence when it is masked.
    """
    if len(sentence_tokens) == 0:
        return -200, None, None

    if model_type in [ModelTypes.GPT, ModelTypes.GEPPETTO]:

        # not use context variant:
        # prepend the sentence with <|endoftext|> token, so that the loss is
        # computed correctly
        sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        tensor_input = torch.tensor(
            [[tokenizer.bos_token_id] + sentence_ids], device=device
        )
        labels = torch.tensor([[tokenizer.bos_token_id] + sentence_ids], device=device)
        labels[:, :1] = -1
        loss = model(tensor_input, labels=tensor_input)
        return float(loss[0]) * -1.0 * len(sentence_tokens), None, None

    elif model_type in BERT_LIKE_MODEL_TYPES:  #

        batched_indexed_tokens = []
        batched_segment_ids = []

        # not use_context variant:
        tokenize_combined = ["[CLS]"] + sentence_tokens + ["[SEP]"]

        for i in range(len(sentence_tokens)):
            # Mask a token that we will try to predict back with
            # `BertForMaskedLM`
            masked_index = i + 1 + 0  # not use_context variant
            tokenize_masked = tokenize_combined.copy()
            tokenize_masked[masked_index] = "[MASK]"
            # unidir bert
            # for j in range(masked_index, len(tokenize_combined)-1):
            #    tokenize_masked[j] = '[MASK]'

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
            # Define sentence A and B indices associated to 1st and 2nd
            # sentences (see paper)
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
            # type(outputs) : <class 'transformers.modeling_outputs.
            # BaseModelOutputWithPoolingAndCrossAttentions'>
            # fields: attentions, cross attentions, hidden states, last hidden
            # state, past key values, pooler output nb fun(**arg): take a
            # dictionary of key-value pairs and unpack it into keyword
            # arguments in a function call. when using bert-large-uncased and
            # transformers BertForMaskedLM: type(outputs) : MaskedLMOutput
            predictions = outputs[0]

        # go through each word and sum their logprobs
        lp = 0.0
        log_logistic = 0.0
        #     logits_min_abs = torch.abs(torch.min(res.detach()))
        #     logits_shifted_above_zero = torch.add(res.detach(),
        #     logits_min_abs)
        #     logits_sum = torch.sum(logits_shifted_above_zero)
        #     res_normalized = torch.div(logits_shifted_above_zero, logits_sum)
        tokens_scores = []
        for i in range(len(sentence_tokens)):
            masked_index = i + 1 + 0  # not use_context variant
            predicted_score = predictions[i, masked_index]
            token_score = predicted_score[
                tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]
            ]
            tokens_scores.append(float(token_score))
            predicted_prob = softmax(predicted_score.cpu().numpy())

            logistic_score = logistic(predicted_score.cpu().numpy())
            masked_word_id = tokenizer.convert_tokens_to_ids(
                [tokenize_combined[masked_index]]
            )[0]
            lp += np.log(predicted_prob[masked_word_id])
            log_logistic += np.log(logistic_score[masked_word_id])
        return lp, log_logistic, tokens_scores
    else:
        raise ValueError(f"Error: unrecognized model type {model_type}")
