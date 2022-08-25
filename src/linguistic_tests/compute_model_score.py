import logging
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
from scipy.special import expit as logistic
from scipy.special import softmax
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import MaskedLMOutput

from .lm_utils import BERT_LIKE_MODEL_TYPES
from .lm_utils import get_penalty_term
from .lm_utils import ModelTypes
from .lm_utils import sent_idx
from .testset import ERROR_LP
from .testset import Example
from .testset import parse_example
from .testset import SPROUSE_SENTENCE_TYPES

# from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput


AcceptabilityScoreRes = namedtuple(
    "AcceptabilityScoreRes", ["lp_softmax", "lp_logistic"]
)


def score_example(
    device,
    example: Example,
    model,
    model_type: ModelTypes,
    tokenizer,
):
    for _idx, typed_sentence in enumerate(example.sentences):
        sentence = typed_sentence.sent
        sentence.tokens = tokenizer.tokenize(sentence.txt)  # , return_tensors='pt'
        if len(sentence.tokens) == 0:
            logging.warning(f"Warning: lenght 0 for {sentence} from {example}")
        text_len = len(sentence.tokens)
        lp_softmax, lp_logistic = get_sentence_acceptability_score(
            model_type, model, tokenizer, sentence.tokens, device
        )

        penalty = get_penalty_term(text_len)

        sentence.lp_softmax = lp_softmax
        logging.log(
            logging.NOTSET,
            f"Assigning field {sentence.pen_lp_softmax} with value {lp_softmax / penalty}: {sentence.txt}",
        )
        sentence.pen_lp_softmax = lp_softmax / penalty
        logging.log(logging.NOTSET, f"{sentence.pen_lp_softmax}")
        if model_type in BERT_LIKE_MODEL_TYPES:
            sentence.lp_logistic = lp_logistic
            sentence.pen_lp_logistic = lp_logistic / penalty

    return example


def get_unparsed_example_scores(
    device,
    example_data: dict,
    model,
    model_type,
    sent_ids,  # : List[int]
    tokenizer,
    sentences_per_example,
    sprouse_format=False,
    expected_sent_types=SPROUSE_SENTENCE_TYPES,
):
    parsed_example = parse_example(example_data, expected_sent_types)
    scored_example = score_example(
        device,
        parsed_example,
        model,
        model_type,
        tokenizer,
    )

    lps = []
    pen_lps = []
    lls = []
    penlls = []

    for stype in scored_example.sentences:
        sent = stype.sent
        lps.append(sent.lp_softmax)
        pen_lps.append(sent.pen_lp_softmax)
        if model_type in BERT_LIKE_MODEL_TYPES:
            lls.append(sent.lp_logistic)
            penlls.append(sent.pen_lp_logistic)

    sentences_txts = [stype.sent.txt for stype in scored_example.sentences]

    ScoreResults = namedtuple("ScoreResults", "lps pen_lps lls penlls sentences_txts")
    return ScoreResults(lps, pen_lps, lls, penlls, sentences_txts)


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


def logistic2(x, L=1, x0=0, k=1):
    r"""
    A logistic function.
    :param x:
    :param L: the curve's maximum value
    :param x0: the x value of the sigmoid's midpoint
    :param k: the logistic growth rate or steepness of the curve
    :return:
    """

    # with np.errstate(over='raise'):
    #    try:
    if L == 1 and x0 == 0 and k == 1:
        # use default logistic function from scipy
        logistic_values = logistic(x)
    else:
        exponent = -k * (x - x0)  # this gets overflows
        logistic_values = L / (1 + np.exp(exponent))
    #    except FloatingPointError as fl_err:
    # logging.warning(f"Warning for params: {k}, {x0}, {L}: {fl_err}")

    return logistic_values


def logistic3(x):
    return logistic2(x, k=0.25)


def logistic4(x):
    return logistic2(x, k=4)


def get_sentence_acceptability_score(
    model_type, model, tokenizer, sentence_tokens, device, verbose=False
):
    """
    Calculates a sentence acceptability score from a Gpt-like or a Bert-like model.
    For Bert-like system it's and estimate of the sentence acceptability as described in
    the paper Lau, J. H., Armendariz, C., Lappin, S., Purver, M., & Shu, C. (2020).
    How furiously can colorless green ideas sleep? sentence acceptability in context.
    Transactions of the Association for Computational Linguistics, 8, 296-310.
    Modified from https://github.com/ml-utils/acceptability-prediction-in-context/blob/master/code/compute_model_score.py
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
        logging.error(
            f"Warning, can't compute score of empty sentence: {sentence_tokens}"
        )
        return AcceptabilityScoreRes(lp_softmax=ERROR_LP, lp_logistic=None)

    if model_type in [ModelTypes.GPT, ModelTypes.GEPPETTO]:

        # not use context variant:
        # (nb "context" is a sequence of words preceding the main input sentence)
        # prepend the sentence with <|endoftext|> token, so that the loss is
        # computed correctly
        sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        sentence_ids_in_batch = [[tokenizer.bos_token_id] + sentence_ids]
        sentence_ids_in_batch_as_tensor = torch.tensor(
            sentence_ids_in_batch, device=device
        )

        batch_labels = torch.tensor(sentence_ids_in_batch, device=device)

        DO_NOT_COMPUTE_LOSS_OVER_THESE_TOKENS = -1
        # slicing assignment: assigning to all the rows (elements of batch), and only first colum (token in sentence)
        # basically is telling not to compute loss on the bos token
        batch_labels[:, :1] = DO_NOT_COMPUTE_LOSS_OVER_THESE_TOKENS
        # nb: labels should be the "correct" output tokens the model should return
        # nb: there is no masked token in this case
        model_output: CausalLMOutputWithCrossAttentions = model(
            # nb: this labels variable not actually used when "not using context"
            # todo, check: is this a copy&paste error?
            sentence_ids_in_batch_as_tensor,
            labels=sentence_ids_in_batch_as_tensor,
        )
        loss = model_output.loss  # in this case equivalent to model_output[0]
        return AcceptabilityScoreRes(
            lp_softmax=float(loss) * -1.0 * len(sentence_tokens), lp_logistic=None
        )

    elif model_type in BERT_LIKE_MODEL_TYPES:  #

        # gets bert output for a batch containing all variations of the sentence
        # in which only one word is masked

        batch_of_masked_sentences_ids = []
        batch_of_segment_ids = []

        # not use_context variant:
        sentence_tokens_with_specials = ["[CLS]"] + sentence_tokens + ["[SEP]"]

        for i in range(len(sentence_tokens)):
            # Mask a token that we will try to predict back with
            # `BertForMaskedLM`
            masked_token_index = i + 1 + 0  # not use_context variant, len(context) = 0
            sentence_tokens_with_mask = sentence_tokens_with_specials.copy()
            sentence_tokens_with_mask[masked_token_index] = "[MASK]"
            # unidir bert
            # for j in range(masked_index, len(tokenize_combined)-1):
            #    tokenize_masked[j] = '[MASK]'

            sentence_ids_with_mask = tokenizer.convert_tokens_to_ids(
                sentence_tokens_with_mask
            )
            # Define sentence A and B indices associated to 1st and 2nd
            # sentences (see paper)
            SENTENCE_A_IDX = 0
            segment_ids = [SENTENCE_A_IDX] * len(sentence_tokens_with_mask)

            batch_of_masked_sentences_ids.append(sentence_ids_with_mask)
            batch_of_segment_ids.append(segment_ids)

        sentences_batch_as_tens = torch.tensor(
            batch_of_masked_sentences_ids, device=device
        )
        segments_batch_as_tens = torch.tensor(batch_of_segment_ids, device=device)

        # Predict all tokens
        with torch.no_grad():

            model_output = model(
                sentences_batch_as_tens, token_type_ids=segments_batch_as_tens
            )

            # when using bert-large-uncased and transformers BertModel (not supported here):
            # type(outputs) : <class 'transformers.modeling_outputs.
            # BaseModelOutputWithPoolingAndCrossAttentions'>
            # fields: attentions, cross attentions, hidden states, last hidden
            # state, past key values, pooler output nb fun(**arg): take a
            # dictionary of key-value pairs and unpack it into keyword
            # arguments in a function call.
            # When using bert-large-uncased and transformers BertForMaskedLM:
            # type(outputs) : MaskedLMOutput

            if isinstance(model_output, MaskedLMOutput):
                predictions_logits_whole_batch = (
                    model_output.logits
                )  # = model_output[0]
            else:
                # for former/deprecated version of transformers
                predictions_logits_whole_batch = model_output

        # go through each word in the sentence and sum the logprobs of their predictions when masked
        lp_softmax = 0.0
        lp_logistic = 0.0
        for i in range(len(sentence_tokens)):
            masked_token_index = i + 1 + 0  # not use_context variant
            cuda_logits_this_masking = predictions_logits_whole_batch[
                i, masked_token_index
            ]

            logits = cuda_logits_this_masking.cpu().numpy()
            softmax_probabilities = softmax(logits)
            logistic_probabilities = logistic2(logits)

            masked_word_id = tokenizer.convert_tokens_to_ids(
                [sentence_tokens_with_specials[masked_token_index]]
            )[0]
            lp_softmax += np.log(softmax_probabilities[masked_word_id])
            lp_logistic += np.log(logistic_probabilities[masked_word_id])
            # verbose=True
            if verbose:
                print(
                    f"masked word  scores: "
                    f"{logits[masked_word_id]}, "
                    f"{logistic_probabilities[masked_word_id]}, "
                    f"{softmax_probabilities[masked_word_id]}, "
                    f"{np.log(logistic_probabilities[masked_word_id])}, "
                    f"{np.log(softmax_probabilities[masked_word_id])}, "
                    f"({sentence_tokens_with_specials[masked_token_index]})"
                )

        return AcceptabilityScoreRes(lp_softmax=lp_softmax, lp_logistic=lp_logistic)

    else:
        raise ValueError(f"Error: unrecognized model type {model_type}")
