import logging

from linguistic_tests.compute_model_score import get_unparsed_example_scores
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import sent_idx
from linguistic_tests.testset import parse_testsets
from linguistic_tests.testset import save_scored_testsets
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

        score_minimal_pairs_testset(  # scored_testset =
            model_type, model, tokenizer, DEVICES.CPU, parsed_testset
        )
    save_scored_testsets(parsed_testsets, model_name, dataset_source)

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
    testset: dict,
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

    sent_ids: list[int] = []

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


def main():
    # print_profession_nouns()
    # t_determiner_noun_agreement_1()
    pass


if __name__ == "__main__":
    main()
