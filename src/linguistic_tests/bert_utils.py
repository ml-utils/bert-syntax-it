import math

import numpy as np
import torch
from linguistic_tests.compute_model_score import logistic3
from linguistic_tests.lm_utils import get_pen_score
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import sent_idx
from linguistic_tests.lm_utils import sentence_score_bases
from linguistic_tests.lm_utils import special_tokens
from scipy.special import softmax
from transformers import BertForMaskedLM
from transformers import BertForMaskedLM as BertPreTrainedModel
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel
from transformers import RobertaForMaskedLM
from transformers import RobertaModel
from transformers.modeling_outputs import MaskedLMOutput


def analize_sentence(
    bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence: str
):
    """
    :param bert:
    :param tokenizer:
    :param sentence: a string in which the word to mask is delimited with
    asteriscs: ***word***
    :return: topk_tokens, topk_probs
    """
    tokens, target_idx = tokenize_sentence(tokenizer, sentence)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)

    # todo: print tokp for all maskings of the sentence

    return get_topk(bert, tokenizer, sentence_ids, target_idx, k=5)


def analize_example(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    example_idx: int,
    example,
    sentences_per_example,
    score_based_on,  # =sentence_score_bases.SOFTMAX,
):
    """
    :param bert:
    :param tokenizer:
    :param example: 2-4 sentences of an example
    :return:
    """

    sentences = get_sentences_from_example(example, sentences_per_example)
    tokens_by_sentence, oov_counts = __get_example_tokens_and_oov_counts(
        tokenizer, sentences
    )

    (
        penLP_by_sentence,
        words_logits_by_sentence,
        normalized_logits_by_sentence,
    ) = __get_example_estimates(
        bert, tokenizer, sentences, tokens_by_sentence, score_based_on
    )

    (
        base_sentence_less_acceptable,
        second_sentence_less_acceptable,
        acceptability_diff_base_sentence,
        acceptability_diff_second_sentence,
        score_base_sentence,
        score_bad_sentence,
        score_2nd_good_sentence,
    ) = __get_acceptability_diffs(
        bert,
        tokenizer,
        penLP_by_sentence,
        normalized_logits_by_sentence,
        example_idx,
        oov_counts,
        sentences,
        tokens_by_sentence,
        score_based_on,
    )

    return (
        base_sentence_less_acceptable,
        second_sentence_less_acceptable,
        acceptability_diff_base_sentence,
        acceptability_diff_second_sentence,
        score_base_sentence,
        score_bad_sentence,
        score_2nd_good_sentence,
        oov_counts,
    )
    # todo: check if the lp (log prob) calculation is the same as in the paper
    #  Lau et al. 2020)
    #  for estimating sentence probability
    # check also if it reproduces the results from the paper (for english
    # bert), and greenberg, and others
    # check differences btw english and italian bert
    # check diff btw bert base and large
    # ..check diff btw case and uncased.. (NB some problems where due to the
    # open file not in utf format)
    # ..checks for gpt ..
    # analise the good sentences with low probability: for each masking, see
    # the topk
    # ..
    # todo: analize limitations of sentence acceptability estimates with Bert:
    # series of sentences, show how using a different vocabulary choice (more
    # or less frequent), changes the score/acceptability of the sentence,
    # making it more or less likely/acceptable than a previous one,
    # regardless of grammaticality
    # ..
    # new test sets exploring what sentence changes sways acceptability
    # estimates
    #
    # todo: skip examples that don't have at least 3 sentences
    # ..


def __get_example_estimates(bert, tokenizer, sentences, tokens_by_sentence, scorebase):
    penLP_by_sentence = []
    words_logits_by_sentence = []
    max_logits = 0
    for sentence_idx, sentence in enumerate(sentences):
        if sentence is not None and len(sentence) > 0:
            # prob = estimate_sentence_probability_from_text(bert, tokenizer,
            # sentence)
            pen_lp, logits_nonnegative_for_each_word = get_sentence_scores(
                bert,
                tokenizer,
                sentence,
                tokens_by_sentence[sentence_idx],
                scorebase,
            )
            penLP_by_sentence.append(pen_lp)
            if isinstance(bert, BertForMaskedLM):
                words_logits_by_sentence.append(logits_nonnegative_for_each_word)
                if max(logits_nonnegative_for_each_word) > max_logits:
                    max_logits = max(logits_nonnegative_for_each_word)

    sentences_estimates_normalized_logits = []
    if isinstance(bert, BertForMaskedLM):
        words_normalized_logits_by_sentence = []
        for sentence_idx, sentence_logits in enumerate(words_logits_by_sentence):
            words_normalized_logits_by_sentence.append(
                [word_logits / max_logits for word_logits in sentence_logits]
            )
            this_sentence_estimate_normalized_logits = 0
            # print(f'words_normalized_logits_by_sentence[sentence_idx]:
            # {words_normalized_logits_by_sentence[sentence_idx]}')
            for word_logits in words_normalized_logits_by_sentence[sentence_idx]:
                # do math.log of each word score and add to the total
                this_sentence_estimate_normalized_logits += math.log(word_logits)
            this_sentence_estimate_normalized_logits = get_pen_score(
                this_sentence_estimate_normalized_logits,
                len(tokens_by_sentence[sentence_idx]),
            )
            sentences_estimates_normalized_logits.append(
                this_sentence_estimate_normalized_logits
            )
    return (
        penLP_by_sentence,
        words_logits_by_sentence,
        sentences_estimates_normalized_logits,
    )


def preprocessing_checks_to_example(example_idx, sentences, tokens_by_sentence):
    __check_unk_and_num_tokens(example_idx, sentences, tokens_by_sentence)


def __check_unk_and_num_tokens(example_idx, sentences, tokens_by_sentence):
    for sentence_idx, sentence_tokens in enumerate(tokens_by_sentence):
        if special_tokens.UNK in sentence_tokens:
            print_red(
                f"this sentence {sentence_idx} ({sentences[sentence_idx]}) "
                f"has at least an UNK token: {sentences[sentence_idx]}"
            )

    # the ungrammatical sentence must not be shorter than the other 3 sentences
    sentence_bad_tokens_count = len(tokens_by_sentence[sent_idx.BAD])
    for sentence_idx, sentence_tokens in enumerate(tokens_by_sentence):
        if len(sentence_tokens) < sentence_bad_tokens_count:
            print(
                f"example {example_idx}:  sentence {sentence_idx} "
                f"({sentences[sentence_idx]}) has less tokens "
                f"({len(sentence_tokens)}) "
                f"than the bad sentence ({sentence_bad_tokens_count})"
            )
        if len(sentence_tokens) == 0:
            print(sentences[sent_idx.BAD])


def __get_example_tokens_and_oov_counts(tokenizer, sentences):
    tokens_by_sentence = []
    oov_counts = []
    for sentence in sentences:
        if sentence is not None:
            tokens = tokenizer.tokenize(sentence)
            tokens_by_sentence.append(tokens)
            oov_counts.append(count_split_words_in_sentence(tokens))
    return tokens_by_sentence, oov_counts


def get_score_descr(score_based_on):
    # todo, fixme: check why only the penalty versions of the scores is returned here

    if score_based_on == sentence_score_bases.SOFTMAX:
        return ScoringMeasures.PenLP.name
    elif score_based_on == sentence_score_bases.NORMALIZED_LOGITS:
        return ScoringMeasures.PenNormLogits
    elif score_based_on == sentence_score_bases.LOGISTIC_FUN:
        return ScoringMeasures.PLL
    else:
        return score_based_on


def __get_acceptability_diffs(
    model,
    tokenizer,
    penLP_by_sentence,
    normalized_logits_by_sentence,
    example_idx,
    oov_counts,
    sentences,
    tokens_by_sentence,
    score_based_on,  # =sentence_score_bases.SOFTMAX,
):
    if type(model) in [BertForMaskedLM, RobertaModel, RobertaForMaskedLM]:
        if score_based_on == sentence_score_bases.SOFTMAX:
            score_by_sentence = penLP_by_sentence
        elif score_based_on == sentence_score_bases.NORMALIZED_LOGITS:
            score_by_sentence = normalized_logits_by_sentence
        score_descr = get_score_descr(score_based_on)
    elif type(model) in [GPT2LMHeadModel]:
        score_by_sentence = penLP_by_sentence
    else:
        raise ValueError(f"Unrecognized model type: {type(model)}")

    score_bad_sentence = score_by_sentence[sent_idx.BAD]
    score_base_sentence = score_by_sentence[sent_idx.GOOD_1]
    score_2nd_good_sentence = None
    if len(score_by_sentence) > 2:
        score_2nd_good_sentence = score_by_sentence[sent_idx.GOOD_2]

    base_sentence_less_acceptable = False
    second_sentence_less_acceptable = False
    acceptability_diff_base_sentence = 0
    acceptability_diff_second_sentence = 0
    for sentence_idx, sentence_score in enumerate(score_by_sentence):
        if sentence_idx == sent_idx.GOOD_1:
            acceptability_diff_base_sentence = sentence_score - score_bad_sentence
        elif sentence_idx == sent_idx.GOOD_2:
            acceptability_diff_second_sentence = sentence_score - score_bad_sentence

        if sentence_score < score_bad_sentence:
            print_orange(
                f"\nexample {example_idx} (oov_count: {oov_counts}): "
                f"sentence {sentence_idx} ({sentences[sentence_idx]}, "
                f"has less {score_descr} ({sentence_score:.1f}) "
                f"than the bad sentence ({score_bad_sentence:.1f}) "
                f"({sentences[sent_idx.BAD]})"
            )
            sentence_ids = tokenizer.convert_tokens_to_ids(
                tokens_by_sentence[sentence_idx]
            )
            estimate_sentence_probability(
                model, tokenizer, sentence_ids, score_based_on, verbose=True
            )
            if sentence_idx == 0:
                base_sentence_less_acceptable = True
            elif sentence_idx == 2:
                second_sentence_less_acceptable = True

    return (
        base_sentence_less_acceptable,
        second_sentence_less_acceptable,
        acceptability_diff_base_sentence,
        acceptability_diff_second_sentence,
        score_base_sentence,
        score_bad_sentence,
        score_2nd_good_sentence,
    )


def count_split_words_in_sentence(sentence_tokens):
    split_words_in_sentence = (
        0  # count how many ##tokens there are, subtract from total
    )
    token_of_previously_counted_split_word = False
    for token in sentence_tokens:
        if not token.startswith("##"):
            token_of_previously_counted_split_word = False
        elif not token_of_previously_counted_split_word:
            split_words_in_sentence += 1
            token_of_previously_counted_split_word = True
    return split_words_in_sentence


def generate_text_with_bert(
    bert: BertPreTrainedModel, tokenizer: BertTokenizer, starting_word="Il"
):
    # convert tokens to ids, append mask to the end
    # get topk with k = 1
    # problem: mask prediction is assumed to result in a complete sentence,
    # not one that would be further extended.
    # Could use more than 1 masked word at the end (albeit in training it was
    # 15 % of the total sentence).
    return 0


def get_sentence_scores(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    sentence,
    sentence_tokens,
    scorebase,
):

    text_len = len(sentence_tokens)
    lp, logits_nonnegative = estimate_sentence_probability_from_text(
        bert,
        tokenizer,
        sentence,
        scorebase,
    )
    # lp = bert_get_logprobs(sentence_tokens, bert, tokenizer)
    # model_score(tokenize_input, tokenize_context, bert, tokenizer, device,
    # args)

    pen_lp = get_pen_score(lp, text_len)
    return pen_lp, logits_nonnegative


def check_unknown_words(tokenizer: BertTokenizer):
    # NB: the uncased model also strips any accent markers from words.
    # Use bert cased.

    words_to_check = ["è"]
    print_token_info(tokenizer, special_tokens.UNK)

    unk_tokens = tokenizer.tokenize(special_tokens.UNK)
    unk_id = tokenizer.convert_tokens_to_ids(unk_tokens)
    print(f"token {special_tokens.UNK} ({unk_tokens}) has ids {unk_id}")
    print_token_info(tokenizer, "riparata")
    print_token_info(tokenizer, "non")
    print_token_info(tokenizer, "che")
    print_token_info(tokenizer, "Chi")

    words_ids = tokenizer.convert_tokens_to_ids(words_to_check)
    print(words_ids)
    recognized_tokens = convert_ids_to_tokens(tokenizer, words_ids)
    print(recognized_tokens)


def print_token_info(tokenizer: BertTokenizer, word_to_check: str):
    tokens = tokenizer.tokenize(word_to_check)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"word {word_to_check} hs tokens {tokens} and ids {ids}")

    if len(tokens) > 1:
        is_word_in_vocab = word_to_check in tokenizer.vocab
        print(
            f"is_word_in_vocab: {is_word_in_vocab}, vocab size: "
            f"{len(tokenizer.vocab)}"
        )


def get_topk(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    sentence_ids,
    masked_word_idx,
    k=5,
):
    (
        res,
        res_softmax,
        res_logistic,
        res_normalized,
        logits_shifted_above_zero,
    ) = get_bert_output_single_masking(bert, sentence_ids, masked_word_idx)
    topk_probs_nonsoftmax = torch.topk(res_normalized, k)
    topk_tokens, topk_probs = get_topk_tokens_from_bert_output(
        res_softmax, tokenizer, k
    )
    return topk_tokens, topk_probs, topk_probs_nonsoftmax


def get_topk_tokens_from_bert_output(res_softmax, tokenizer, k=5):
    topk_probs, topk_ids = torch.topk(
        res_softmax, k
    )  # todo: get ids/indexes, not their probability value
    topk_tokens = convert_ids_to_tokens(tokenizer, topk_ids)
    return topk_tokens, topk_probs


def get_bert_output_single_masking(
    bert: BertPreTrainedModel,
    sentence_ids,
    masked_word_idx,
    verbose=False,
):
    """
        # gets bert output for a single sentence in which only the token at
        # @masked_word_idx is masked
    :param bert:
    :param sentence_ids:
    :param masked_word_idx:
    :param verbose:
    :return:
    """

    # from the docs:
    # https: // huggingface.co / docs / transformers / main_classes / output
    # The outputs object is a SequenceClassifierOutput, as we can see in the
    # documentation of that class below, it means it has an optional loss, a
    # logits an optional hidden_states and an optional attentions attribute.
    # Here we have the loss since we passed along labels, but we don’t have
    # hidden_states and attentions because we didn’t pass
    # output_hidden_states=True or output_attentions=True.

    # todo: check that masked_word_idx remains correct when some words are
    #  split (and therefore there are more tokens than words)
    sentence_ids_as_tensor = torch.LongTensor(sentence_ids)
    # equivalent to batch = [sentence_ids_as_tensor] (batch of 1 element)
    sentence_ids_in_tensor_batch = sentence_ids_as_tensor.unsqueeze(0)

    # <class 'transformers.modeling_outputs.MaskedLMOutput'>
    bert_out = bert(sentence_ids_in_tensor_batch)

    # NB: in CausalLMOutput, logits has shape:
    # (batch_size, sequence_length, config.vocab_size))
    # Represents the prediction scores of the language modeling head
    # (scores for each vocabulary token before SoftMax).
    # https://huggingface.co/docs/transformers/main_classes/output
    if isinstance(bert_out, MaskedLMOutput):
        logits_whole_batch = bert_out.logits
    else:
        logits_whole_batch = bert_out

    # print(f'masked_word_idx: {masked_word_idx}, {type(res_unsliced)=}')
    # masked_word_idx: 1, type(res_unsliced):
    # masked_word_idx: 5, type(res_unsliced): <class 'torch.Tensor'>

    # <class 'transformers.modeling_outputs.CausalLMOutputWithCrossAttentions'>
    first_element_in_batch_idx = 0
    # going from shape (batch_size, sequence_length, config.vocab_size))
    # to shape = (vocab_size)
    logits_masked_word_predictions = logits_whole_batch[
        first_element_in_batch_idx, masked_word_idx
    ]

    # todo: produce probabilities not with softmax (not using an exponential,
    #  to avoiding the maximization of top results),
    #  then compare these probailities with the softmax ones, expecially for
    #  ungrammatical sentences
    detached_logits_masked_word_predictions = logits_masked_word_predictions.detach()
    LAST_DIMENSION_IDX = -1
    softmax_on_logits_masked_word_predictions = softmax(
        detached_logits_masked_word_predictions, axis=LAST_DIMENSION_IDX
    )
    logistic_values_on_logits_masked_word_predictions = logistic3(
        detached_logits_masked_word_predictions
    )

    logits_min = torch.min(detached_logits_masked_word_predictions)
    logits_shifted_from_zero = torch.subtract(
        detached_logits_masked_word_predictions, logits_min
    )
    logits_sum = torch.sum(logits_shifted_from_zero)
    logits_normalized = torch.div(logits_shifted_from_zero, logits_sum)

    if verbose:
        print(f"tens size {sentence_ids_in_tensor_batch.size()}")
        print(f"res_unsliced size {logits_whole_batch.size()}")
        print(f"res size {detached_logits_masked_word_predictions.size()}")
        print(f"res_softmax size {softmax_on_logits_masked_word_predictions.size()}")
        print(f"res_normalized size {logits_normalized.size()}")

    # RuntimeError: Can't call numpy() on Tensor that requires grad.
    # Use tensor.detach().numpy() instead.
    return (
        detached_logits_masked_word_predictions,
        softmax_on_logits_masked_word_predictions,
        logistic_values_on_logits_masked_word_predictions,
        logits_normalized,
        logits_shifted_from_zero,
    )


def convert_ids_to_tokens(tokenizer: BertTokenizer, ids):
    """Converts a sequence of ids in wordpiece tokens using the vocab."""

    if torch.is_tensor(ids[0]):
        ids_as_ints = []
        for i in ids:
            if torch.is_tensor(i):
                if torch.numel(i) > 1:
                    print_orange(
                        f"Warning: tensor has more than one item: " f"{i.size()}"
                    )
                ids_as_ints.append(i.item())
        ids = ids_as_ints

    if hasattr(tokenizer, "convert_ids_to_tokens") and callable(
        getattr(tokenizer, "convert_ids_to_tokens")
    ):
        return tokenizer.convert_ids_to_tokens(ids)
    else:
        tokens = []
        for i in ids:
            try:
                tokens.append(tokenizer.ids_to_tokens[i])
            except Exception as err:
                print(f"Unable to find id {i} {type(i)} in the vocabulary. {str(err)}")
        return tokens


def estimate_sentence_probability(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    sentence_ids: list[int],
    scorebase,  # =sentence_score_bases.SOFTMAX,
    verbose: bool = False,
):
    # iterate for each word, mask it and get the probability
    # sum the logs of the probabilities

    MASK_ID = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    sentence_prob_estimate = 0
    words_logits_nonnegative = []
    for masked_index in range(len(sentence_ids)):
        masked_sentence_ids = sentence_ids.copy()
        masked_word_id = masked_sentence_ids[masked_index]
        masked_sentence_ids[masked_index] = MASK_ID
        masked_sentence_ids = [CLS_ID] + masked_sentence_ids + [SEP_ID]

        (
            probability_of_words_predictions_in_this_masking,
            topk_tokens,
            logits_nonnegative,
        ) = get_sentence_probs_from_word_ids(
            bert,
            tokenizer,
            masked_sentence_ids,
            [masked_word_id],
            masked_index + 1,
            scorebase=scorebase,
        )
        probability_of_this_masking = probability_of_words_predictions_in_this_masking[
            0
        ]  # we specified only one word

        # todo: unit test to avoid regressions in the output
        # store, for each word masking, a value in an array, to be normalized
        # later with the max logits of the 2 sentences to compare
        # (or even return it and postpone the calculation later, doind a max
        # over all examples, to have comparable measures)
        # three arrays, one for each score measure: with softmax, and
        # normalized logits, and normalized logits as probs
        words_logits_nonnegative.append(logits_nonnegative[0])
        sentence_prob_estimate += np.log(probability_of_this_masking)
        if verbose:
            print(
                f"testing {convert_ids_to_tokens(tokenizer, [masked_word_id])}"
                f" at {masked_index+1} in sentence "
                f"{convert_ids_to_tokens(tokenizer, masked_sentence_ids)}, "
                f"topk_tokens: {topk_tokens}"
            )
            # todo: also print the topk for the masking prediction

    # todo: also alternative method with formula from paper balanced on
    #  sentence lenght

    log_prob = sentence_prob_estimate
    return log_prob, words_logits_nonnegative  # np.exp(log_prob)


def bert_get_logprobs(tokenize_input, model, tokenizer):
    batched_indexed_tokens = []
    batched_segment_ids = []

    tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]

    for i in range(len(tokenize_input)):
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_index = i + 1
        tokenize_masked = tokenize_combined.copy()
        tokenize_masked[masked_index] = "[MASK]"
        # unidir bert
        # for j in range(masked_index, len(tokenize_combined)-1):
        #    tokenize_masked[j] = '[MASK]'

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
        # Define sentence A and B indices associated to 1st and 2nd sentences
        # (see paper)
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
        predicted_prob = softmax(
            predicted_score
        )  # softmax(predicted_score.cpu().numpy())
        lp += np.log(
            predicted_prob[
                tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]
            ]
        )

    return lp


def estimate_sentence_probability_from_text(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    sentence: str,
    scorebase,  # =sentence_score_bases.SOFTMAX,
):
    tokens = tokenizer.tokenize(sentence)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    return estimate_sentence_probability(bert, tokenizer, sentence_ids, scorebase)


def get_probs_for_words(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    sent,
    w1,
    w2,
    scorebase,  # =sentence_score_bases.SOFTMAX,
):
    tokens, masked_word_idx = tokenize_sentence(tokenizer, sent)

    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    try:
        masked_words_ids = tokenizer.convert_tokens_to_ids([w1, w2])
    except KeyError:
        print("skipping", w1, w2, "bad wins")
        return None

    probs_for_words, topk_tokens, _ = get_sentence_probs_from_word_ids(
        bert,
        tokenizer,
        sentence_ids,
        masked_words_ids,
        masked_word_idx,
        scorebase=scorebase,
    )
    return probs_for_words


def get_sentence_probs_from_word_ids(
    bert: BertPreTrainedModel,
    tokenizer: BertTokenizer,
    sentence_ids,
    masked_words_ids,
    masked_word_idx,
    scorebase,
):
    """
    :param bert:
    :param tokenizer:
    :param sentence_ids:
    :param masked_words_ids:
    :param masked_word_idx:
    :param scorebase: default softmax. Possible values: softmax, logits,
    logits_nonnegative,
    normalized_prob, normalized
    :return:
    """
    # fixme: this should use the model call where each word is masked ..
    (
        logits_masked_word_predictions,
        softmax_on_logits_masked_word_predictions,
        logistic_values_on_logits_masked_word_predictions,
        logits_normalized,
        logits_shifted_above_zero,
    ) = get_bert_output_single_masking(bert, sentence_ids, masked_word_idx)

    # return (
    #     detached_logits_masked_word_predictions,
    #     softmax_on_logits_masked_word_predictions,
    #     logistic_values_on_logits_masked_word_predictions,
    #     logits_normalized,
    #     logits_shifted_from_zero,
    # )

    topk_tokens, top_probs = get_topk_tokens_from_bert_output(
        softmax_on_logits_masked_word_predictions, tokenizer, k=10
    )

    # todo: implement for logits_nonnegative
    # needs the max value among the two sentences to compare

    if scorebase == sentence_score_bases.SOFTMAX:
        res_to_use = softmax_on_logits_masked_word_predictions
    elif scorebase == sentence_score_bases.NORMALIZED_LOGITS:
        res_to_use = logits_masked_word_predictions
    elif scorebase == sentence_score_bases.LOGISTIC_FUN:
        res_to_use = logistic_values_on_logits_masked_word_predictions
    else:
        raise ValueError(f"Invalid scorebase defined: {scorebase}.")

    scores = __get_scores_from_word_ids(res_to_use, masked_words_ids)

    logits_nonnegative = __get_scores_from_word_ids(
        logits_shifted_above_zero, masked_words_ids
    )
    return scores, topk_tokens, logits_nonnegative


def __get_scores_from_word_ids(scores, word_ids_mask_predictions):
    scores = scores[word_ids_mask_predictions]
    return [float(x) for x in scores]


def tokenize_sentence(tokenizer: BertTokenizer, sent: str):
    print(f"sent: {sent}")
    pre, target, post = sent.split("***")
    print(f"pre: {pre}, target: {target}, post: {post}")

    if "mask" in [target.lower()]:  # todo:check, it was if "mask" in target.lower():
        tokenized_target = ["[MASK]"]

    else:

        tokenized_target = tokenizer.tokenize(target)

    # todo, fixme: the vocabulary of the pretrained model from Bostrom &
    #  Durrett (2020) does not have entries for CLS, UNK
    # fixme: tokenizer.tokenize(pre), does not recognize the words
    tokens = ["[CLS]"] + tokenizer.tokenize(pre)  # tokens
    # = tokenizer.tokenize(pre)
    target_idx = len(tokens)
    print(f"target_idx: {target_idx}")
    tokens += tokenized_target + tokenizer.tokenize(post) + ["[SEP]"]
    print(f"tokens {tokens}")
    return tokens, target_idx


def main():
    pass


if __name__ == "__main__":
    main()
