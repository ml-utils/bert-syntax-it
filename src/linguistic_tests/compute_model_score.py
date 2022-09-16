import logging
from collections import namedtuple
from functools import reduce
from typing import List
from typing import Tuple

import numpy as np
import torch
from scipy.special import expit as logistic
from scipy.special import softmax
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import MaskedLMOutput

from .lm_utils import BERT_LIKE_MODEL_TYPES
from .lm_utils import DEVICES
from .lm_utils import get_penalty_term
from .lm_utils import ModelTypes
from .lm_utils import sent_idx
from .testset import ERROR_LP
from .testset import Example
from .testset import parse_example
from .testset import Sentence
from .testset import SPROUSE_SENTENCE_TYPES

# from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

score_per_masking_DEFAULT = None
AcceptabilityScoreRes = namedtuple(
    "AcceptabilityScoreRes",
    ["lp_softmax", "lp_logistic", "score_per_masking", "logistic_score_per_masking"],
)
BOS_TOKEN_COUNT = 1


def score_example(
    device,
    example: Example,
    model,
    model_type: ModelTypes,
    tokenizer,
):
    for _idx, typed_sentence in enumerate(example.sentences):
        sentence = typed_sentence.sent

        # todo: redundant, needed for gilbert: if model uncased, uncase the sentence text before tokenizing
        sentence.tokens = tokenizer.tokenize(sentence.txt)  # , return_tensors='pt'
        if len(sentence.tokens) == 0:
            logging.warning(f"Warning: lenght 0 for {sentence} from {example}")
        text_len = len(sentence.tokens)
        lp_softmax, lp_logistic, _, _ = get_sentence_acceptability_score(
            model_type, model, tokenizer, sentence, device
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
    model_type,
    model,
    tokenizer,
    sentence: Sentence,  # sentence_tokens: List[str],
    device: DEVICES,
    verbose=False,
    at_once_mask_all_tokens_of_a_word=False,
) -> AcceptabilityScoreRes:
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

    if len(sentence.tokens) == 0:
        logging.error(
            f"Warning, can't compute score of empty sentence: {sentence.tokens}"
        )
        return AcceptabilityScoreRes(
            lp_softmax=ERROR_LP,
            lp_logistic=None,
            score_per_masking=None,
            logistic_score_per_masking=None,
        )

    if model_type in [ModelTypes.GPT, ModelTypes.GEPPETTO]:

        # not use context variant:
        # (nb "context" is a sequence of words preceding the main input sentence)
        # prepend the sentence with <|endoftext|> token, so that the loss is
        # computed correctly
        sentence_ids = tokenizer.convert_tokens_to_ids(sentence.tokens)
        sentence_tokens_with_specials = [tokenizer.bos_token] + sentence.tokens
        # todo, fixme: is there a eos token missing?
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
        with torch.no_grad():
            model_output: CausalLMOutputWithCrossAttentions = model(
                # nb: this labels variable not actually used when "not using context"
                # todo, check: is this a copy&paste error?
                sentence_ids_in_batch_as_tensor,
                labels=sentence_ids_in_batch_as_tensor,
            )
            loss = model_output.loss  # in this case equivalent to model_output[0]
            predictions_logits_whole_batch = (
                model_output.logits
            )  # (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size))

        surprisals_per_token = get_surprisals_per_token_from_gpt2_logits(
            predictions_logits_whole_batch,
            sentence,
            sentence_tokens_with_specials,
            tokenizer,
        )

        return AcceptabilityScoreRes(
            lp_softmax=float(loss) * -1.0 * len(sentence.tokens),
            lp_logistic=None,
            score_per_masking=surprisals_per_token,
            logistic_score_per_masking=None,
        )

    elif model_type in BERT_LIKE_MODEL_TYPES:  #

        # gets bert output for a batch containing all variations of the sentence
        # in which only one word is masked

        batch_of_masked_sentences_ids = []
        batch_of_segment_ids = []

        # not use_context variant:
        sentence_tokens_with_specials = (
            [tokenizer.cls_token] + sentence.tokens + [tokenizer.sep_token]
        )

        if at_once_mask_all_tokens_of_a_word:
            # use the pretokenizer, obtain the list of pretokens to fill Sentence.pretokens

            # fixme: BertTokenizer from pretrained has no backend_tokenizer, no pretokenization possible
            # eg.: [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
            pretokenization = (
                tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence.txt)
            )

            sentence.pretokens = []
            current_token_idx = 0
            for pretoken, (_, _) in pretokenization:
                tokens_in_pretoken = tokenizer.tokenize(pretoken)
                last_token_idx = current_token_idx + len(tokens_in_pretoken)
                sentence.pretokens.append(
                    (pretoken, (current_token_idx, last_token_idx))
                )
                current_token_idx = last_token_idx
                # todo: check this with a unit test

        if at_once_mask_all_tokens_of_a_word:
            masking_configurations_count = len(sentence.pretokens)
        else:
            masking_configurations_count = len(sentence.tokens)

        for masking_configuration_idx in range(masking_configurations_count):
            sentence_tokens_with_mask = sentence_tokens_with_specials.copy()
            # Mask a token (or the tokens of a word/pretoken) that we will try to predict back with `BertForMaskedLM`
            if at_once_mask_all_tokens_of_a_word:
                # todo: test this in a unit test
                # keep track of correspondence btw masking_configuration_idx and token idx
                current_pretoken, (
                    pretoken_starts_at,
                    pretoken_ends_at,
                ) = sentence.pretokens[masking_configuration_idx]
                for token_idx in range(pretoken_starts_at, pretoken_ends_at):
                    masked_token_index = token_idx + BOS_TOKEN_COUNT
                    sentence_tokens_with_mask[masked_token_index] = tokenizer.mask_token
            else:
                masked_token_index = masking_configuration_idx + BOS_TOKEN_COUNT
                sentence_tokens_with_mask[masked_token_index] = tokenizer.mask_token

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

        (
            PseudoLogLikelihood,
            logistic_score_per_masking,
            lp_logistic,
            negative_token_log_probabilities,
        ) = calculate_PseudoLogLikelihood_from_model_output(
            predictions_logits_whole_batch,
            sentence,
            sentence_tokens_with_specials,
            tokenizer,
            verbose,
            at_once_mask_all_tokens_of_a_word,
        )

        return AcceptabilityScoreRes(
            lp_softmax=PseudoLogLikelihood,
            lp_logistic=lp_logistic,
            score_per_masking=negative_token_log_probabilities,
            logistic_score_per_masking=logistic_score_per_masking,
        )

    else:
        raise ValueError(f"Error: unrecognized model type {model_type}")


def get_surprisals_per_token_from_gpt2_logits(
    predictions_logits_whole_batch,
    sentence: Sentence,
    sentence_tokens_with_specials,
    tokenizer,
) -> List[float]:
    # logits shape: # (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size))
    negative_token_log_probabilities: List[float] = []
    masking_configurations_count = len(sentence.tokens)
    for masking_configuration_idx in range(masking_configurations_count):
        masked_token_index = masking_configuration_idx + BOS_TOKEN_COUNT
        cuda_logits_this_masking = predictions_logits_whole_batch[0, masked_token_index]

        _, _ = update_sentence_score_totals(
            0,
            0,
            cuda_logits_this_masking,
            masked_token_index,
            [],
            negative_token_log_probabilities,
            sentence_tokens_with_specials,
            tokenizer,
        )
    return negative_token_log_probabilities


def calculate_PseudoLogLikelihood_from_model_output(
    predictions_logits_whole_batch,
    sentence: Sentence,
    sentence_tokens_with_specials,
    tokenizer,
    verbose,
    at_once_mask_all_tokens_of_a_word=False,
) -> Tuple[float, List[float], float, List[float]]:

    # go through each word in the sentence and sum the logprobs of their predictions when masked
    PseudoLogLikelihood = 0.0
    PseudoLogLikelihood2_logistic_based = 0.0
    negative_token_log_probabilities: List[float] = []
    negative_token_log_logistic_pseudo_probabilities: List[float] = []

    if at_once_mask_all_tokens_of_a_word:
        masking_configurations_count = len(sentence.pretokens)
    else:
        masking_configurations_count = len(sentence.tokens)

    # if we are masking according to pretokens (words) we should iterate according to those instead of tokens (subwords)
    for masking_configuration_idx in range(masking_configurations_count):

        if at_once_mask_all_tokens_of_a_word:
            # todo: test this in a unit test
            # keep track of correspondence btw masking_configuration_idx and token idx
            current_pretoken, (
                pretoken_starts_at,
                pretoken_ends_at,
            ) = sentence.pretokens[masking_configuration_idx]
            # there are multiple masked_tokens for each pretoken/word
            for token_idx in range(pretoken_starts_at, pretoken_ends_at):

                # size of output logits: (batch_lenght, encoded_sentence_tokens_incl_specials, vocab)
                # batch_lenght = num of maskings = num of tokens or num of pretokens/words
                #   masking configuration index:
                #       in case of pretokens, a "masking configuration" is a sentence where a word/pretokens (all its tokens) are masked

                masked_token_index = token_idx + BOS_TOKEN_COUNT
                cuda_logits_this_masking = predictions_logits_whole_batch[
                    masking_configuration_idx, masked_token_index
                ]

                (
                    PseudoLogLikelihood,
                    PseudoLogLikelihood2_logistic_based,
                ) = update_sentence_score_totals(
                    PseudoLogLikelihood,
                    PseudoLogLikelihood2_logistic_based,
                    cuda_logits_this_masking,
                    masked_token_index,
                    negative_token_log_logistic_pseudo_probabilities,
                    negative_token_log_probabilities,
                    sentence_tokens_with_specials,
                    tokenizer,
                )
        else:
            masked_token_index = masking_configuration_idx + BOS_TOKEN_COUNT
            cuda_logits_this_masking = predictions_logits_whole_batch[
                masking_configuration_idx, masked_token_index
            ]

            (
                PseudoLogLikelihood,
                PseudoLogLikelihood2_logistic_based,
            ) = update_sentence_score_totals(
                PseudoLogLikelihood,
                PseudoLogLikelihood2_logistic_based,
                cuda_logits_this_masking,
                masked_token_index,
                negative_token_log_logistic_pseudo_probabilities,
                negative_token_log_probabilities,
                sentence_tokens_with_specials,
                tokenizer,
            )

        # if verbose:
        #     print(
        #         f"masked word  scores: "
        #         f"{logits[masked_token_id]}, "
        #         f"{logistic_pseudo_probabilities[masked_token_id]}, "
        #         f"{softmax_probabilities[masked_token_id]}, "
        #         f"{np.log(logistic_pseudo_probabilities[masked_token_id])}, "
        #         f"{np.log(softmax_probabilities[masked_token_id])}, "
        #         f"({sentence_tokens_with_specials[masked_token_index]})"
        #     )

    return (
        PseudoLogLikelihood,
        negative_token_log_logistic_pseudo_probabilities,
        PseudoLogLikelihood2_logistic_based,
        negative_token_log_probabilities,
    )


def update_sentence_score_totals(
    PseudoLogLikelihood,
    PseudoLogLikelihood2_logistic_based,
    cuda_logits_this_masking,
    masked_token_index,
    negative_token_log_logistic_pseudo_probabilities,
    negative_token_log_probabilities,
    sentence_tokens_with_specials,
    tokenizer,
):
    token_log_probability, token_log_logistic_pseudo_probability = get_token_log_scores(
        tokenizer,
        cuda_logits_this_masking,
        sentence_tokens_with_specials,
        masked_token_index,
    )

    (
        PseudoLogLikelihood,
        PseudoLogLikelihood2_logistic_based,
    ) = update_sentence_score_totals_helper(
        PseudoLogLikelihood,
        PseudoLogLikelihood2_logistic_based,
        negative_token_log_logistic_pseudo_probabilities,
        negative_token_log_probabilities,
        token_log_logistic_pseudo_probability,
        token_log_probability,
    )

    return PseudoLogLikelihood, PseudoLogLikelihood2_logistic_based


def update_sentence_score_totals_helper(
    PseudoLogLikelihood,
    PseudoLogLikelihood2_logistic_based,
    negative_token_log_logistic_pseudo_probabilities,
    negative_token_log_probabilities,
    token_log_logistic_pseudo_probability,
    token_log_probability,
):
    negative_token_log_probabilities.append(-token_log_probability)
    # PseudoLogLikelihood (PLL): the sum over the conditional log probabilities
    # log P_MLM (w_t | W_\t) of each sentence token (Salazar et al 2021 Masked Language Model Scoring)
    PseudoLogLikelihood += token_log_probability
    negative_token_log_logistic_pseudo_probabilities.append(
        -token_log_logistic_pseudo_probability
    )
    PseudoLogLikelihood2_logistic_based += token_log_logistic_pseudo_probability

    return PseudoLogLikelihood, PseudoLogLikelihood2_logistic_based


def get_token_log_scores(
    tokenizer,
    cuda_logits_single_masking,
    sentence_tokens_with_specials,
    masked_token_index,
):
    logits = cuda_logits_single_masking.cpu().numpy()
    softmax_probabilities = softmax(logits)
    logistic_pseudo_probabilities = logistic2(logits)

    masked_token_id = tokenizer.convert_tokens_to_ids(
        [sentence_tokens_with_specials[masked_token_index]]
    )[0]

    # this are the values that should be about the same for all masked words in the sentence
    # spikes should indicate ungrammatical/unfluent sentences
    token_log_probability = np.log(softmax_probabilities[masked_token_id])
    token_log_logistic_pseudo_probability = np.log(
        logistic_pseudo_probabilities[masked_token_id]
    )

    return token_log_probability, token_log_logistic_pseudo_probability
