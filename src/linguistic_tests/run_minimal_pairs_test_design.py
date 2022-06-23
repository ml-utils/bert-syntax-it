import logging
import os
import time
from typing import Dict
from typing import List

from linguistic_tests.bert_utils import estimate_sentence_probability
from linguistic_tests.compute_model_score import get_unparsed_example_scores
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.file_utils import parse_testsets
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import sent_idx
from linguistic_tests.testset import TestSet
from tqdm import tqdm


def run_tests_goldberg():
    # todo: use sentence acceptability estimates (PenLP e PenNL), and see
    #  results on goldberg testset
    # also for blimp testset with tests non intended for bert, compare with
    # the results on gpt and other models
    return 0


def run_tests_blimp():
    # todo
    return 0


def run_tests_lau_et_al():
    # todo: compare results with other models
    return 0


def print_detailed_sentence_info(bert, tokenizer, sentence_txt, scorebase):
    print_red(f"printing details for sentence {sentence_txt}")
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    estimate_sentence_probability(
        bert, tokenizer, sentence_ids, scorebase, verbose=True
    )


def run_blimp_it_island_effects():
    # todo: use the batch implementation of get model output

    pass


def run_blimp_en(
    model_type,
    model_name,
    dataset_source,
    testset_filenames,
    testset_dir_path,
    examples_format="blimp",
    max_examples=1000,
):
    sent_types_descr = "blimp"

    scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
    if model_type in BERT_LIKE_MODEL_TYPES:
        scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]

    parsed_testsets = parse_testsets(
        testset_dir_path,
        testset_filenames,
        dataset_source,
        examples_format,
        sent_types_descr,
        model_name,
        model_type,
        scoring_measures,
        max_examples=1000,
    )

    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
    for parsed_testset in parsed_testsets:
        print_orange(
            f"Scoring testset {parsed_testset.linguistic_phenomenon}, on {model_type=} {model_name=}"
        )
        parsed_testset.examples = parsed_testset.examples[0:max_examples]

        scored_testset = score_minimal_pairs_testset(
            model_type, model, tokenizer, DEVICES.CPU, parsed_testset
        )

        scored_testset.model_descr = model_name
        filename = f"{scored_testset.linguistic_phenomenon}.testset.pickle"
        if os.path.exists(filename):
            timestamp = time.strftime("%Y-%m-%d_h%Hm%Ms%S")
            filename = (
                f"{scored_testset.linguistic_phenomenon}-{timestamp}.testset.pickle"
            )
        scored_testset.save_to_pickle(filename)

    return parsed_testsets


def run_tests_for_model_type(model_type):
    print(f"model_type: {model_type}")
    # model_name, eval_suite = arg_parse()

    # todo: run on the following testsets (minimal pairs):
    # (use same pretrained models.. or comparable ones to those in the papers)
    # blimp: ..
    # golderg: ..
    # Lau et al: https://github.com/ml-utils/
    # acceptability-prediction-in-context/tree/
    # 0a274d1d9f70f389ddc6b6d796bd8f815833056c/code

    # run_syntactic_tests_it_legacy_impl(model_type)

    # if model_type == ModelTypes.GPT:
    #    print('importing gpt_tests..')
    #     from gpt_tests import main as main2
    #    print('imported.')
    #    main2()

    # run_eval(eval_suite, bert, tokenizer)
    # prob1 = estimate_sentence_probability_from_text(bert, tokenizer,
    # 'What is your name?')
    # prob2 = estimate_sentence_probability_from_text(bert, tokenizer,
    # 'What is name your?')
    # print(f'prob1: {prob1}, prob2: {prob2}')
    # eval_it(bert, tokenizer)
    # custom_eval("What is your name?", bert, tokenizer)


def score_minimal_pairs_testset(
    model_type: ModelTypes, model, tokenizer, device: DEVICES, testset: TestSet
):
    # assigning sentence scores
    for example_idx, example in enumerate(tqdm(testset.examples)):
        score_example(
            device,
            example,
            model,
            model_type,
            tokenizer,
        )

    # scoring accuracy rates
    for scoring_measure in testset.get_scoring_measures():
        for stype_acceptable_sentence in testset.get_acceptable_sentence_types():
            accurate_count = 0
            for example_idx, example in enumerate(testset.examples):
                if example.is_scored_accurately_for(
                    scoring_measure, stype_acceptable_sentence
                ):
                    accurate_count += 1
            accuracy = accurate_count / len(testset.examples)
            testset.accuracy_per_score_type_per_sentence_type[scoring_measure][
                stype_acceptable_sentence
            ] = accuracy

    return testset


# todo: mark as deprecated, move to test section to use as comparison for outcome of new method
def get_unparsed_testset_scores(
    model_type: ModelTypes,
    model,
    tokenizer,
    device,
    testset: Dict,
    sentences_per_example,
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
    # todo: parse testset and run score_testset

    sent_ids: List[int] = []

    correct_lps_1st_sentence = 0
    correct_pen_lps_1st_sentence = 0
    correct_lps_2nd_sentence = 0
    correct_pen_lps_2nd_sentence = 0
    correct_lls_1st_sentence = 0
    correct_pen_lls_1st_sentence = 0
    correct_lls_2nd_sentence = 0
    correct_pen_lls_2nd_sentence = 0
    logging.debug(f"\n{testset['sentences']=}")
    print(f"\n{type(testset['sentences'])=}, {testset['sentences']=}")
    for example_idx, example_data in enumerate(tqdm(testset["sentences"])):
        logging.debug(f"{example_idx=}, {example_data=}")
        print(f"{example_idx=}, {example_data=}")
        (lps, pen_lps, lls, penlls, sentences,) = get_unparsed_example_scores(
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
            if lls[sent_idx.GOOD_1] > lls[sent_idx.BAD]:
                correct_lls_1st_sentence += 1
            if penlls[sent_idx.GOOD_1] > penlls[sent_idx.BAD]:
                correct_pen_lls_1st_sentence += 1
        if len(sentences) > 2:
            if lps[sent_idx.GOOD_2] > lps[sent_idx.BAD]:
                correct_lps_2nd_sentence += 1
            if pen_lps[sent_idx.GOOD_2] > pen_lps[sent_idx.BAD]:
                correct_pen_lps_2nd_sentence += 1
            if model_type in BERT_LIKE_MODEL_TYPES:
                if lls[sent_idx.GOOD_2] > lls[sent_idx.BAD]:
                    correct_lls_2nd_sentence += 1
                if penlls[sent_idx.GOOD_2] > penlls[sent_idx.BAD]:
                    correct_pen_lls_2nd_sentence += 1

    return (
        correct_lps_1st_sentence,
        correct_pen_lps_1st_sentence,
        correct_lps_2nd_sentence,
        correct_pen_lps_2nd_sentence,
        correct_lls_1st_sentence,
        correct_pen_lls_1st_sentence,
        correct_lls_2nd_sentence,
        correct_pen_lls_2nd_sentence,
    )


def print_accuracy_scores(testset: TestSet):
    logging.info(f"test results report, {testset.linguistic_phenomenon}:")
    for scoring_measure in testset.accuracy_per_score_type_per_sentence_type.keys():
        logging.debug(f"scores with {scoring_measure}")
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
    logging.info("test results report:")
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


def main():
    # print_profession_nouns()
    # t_determiner_noun_agreement_1()
    pass


if __name__ == "__main__":
    main()
