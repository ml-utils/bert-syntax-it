import logging
from statistics import mean

import numpy as np
import pandas as pd
from linguistic_tests.compute_model_score import get_unparsed_example_scores
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DataSources
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import ExperimentalDesigns
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import get_testset_params
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import sent_idx
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.plots_and_prints import print_accuracy_scores
from linguistic_tests.testset import Example
from linguistic_tests.testset import get_dd_score_parametric
from linguistic_tests.testset import get_merged_score_across_testsets
from linguistic_tests.testset import load_testsets_from_pickles
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
    model_type: ModelTypes,
    model,
    tokenizer,
    device: DEVICES,
    testset: TestSet,
) -> TestSet:

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


def main_minimal_pairs(
    tests_subdir="blimp/from_blim_en/islands/",
    rescore=False,
    log_level=logging.INFO,
    max_examples=50,
):

    #     _setup_logging(log_level)
    #     args = _parse_arguments()

    # model_dir = str(get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch")

    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

    logging.info(f"Will run tests with models: {MODEL_TYPES_AND_NAMES_EN.values()}")

    (
        testsets_root_filenames,
        broader_test_type,
        dataset_source,
        experimental_design,
    ) = get_testset_params(tests_subdir)

    for model_name, model_type in MODEL_TYPES_AND_NAMES_EN.items():
        print_orange(f"Starting test session for {model_type=}, and {dataset_source=}")

        if rescore:
            rescore_testsets_and_save_pickles(
                model_type=model_type,
                model_name=model_name,
                testset_dir_path=testset_dir_path,
                testsets_root_filenames=testsets_root_filenames,
                dataset_source=dataset_source,
                experimental_design=experimental_design,
                examples_format="json_lines",
                max_examples=max_examples,
            )

        loaded_testsets = load_testsets_from_pickles(
            dataset_source,
            testsets_root_filenames,
            model_name,
            expected_experimental_design=experimental_design,
        )
        for scored_testset in loaded_testsets:
            print_accuracy_scores(scored_testset)


def score_factorial_testset(
    model_type: ModelTypes,
    model,
    tokenizer,
    device: DEVICES,
    testset: TestSet,
    experimental_design: ExperimentalDesigns,
) -> TestSet:

    # assigning sentence scores and testset accuracy rates
    score_minimal_pairs_testset(model_type, model, tokenizer, device, testset)
    if experimental_design == ExperimentalDesigns.MINIMAL_PAIRS:
        return testset

    # doing factorial design scores
    for example_idx, example in enumerate(testset.examples):
        (
            example.DD_with_lp,
            example.DD_with_penlp,
            example.DD_with_ll,
            example.DD_with_penll,
        ) = _get_example_dd_scores(example, model_type)
        if example.DD_with_lp > 0:
            testset.accuracy_by_DD_lp += 1 / len(testset.examples)
        if example.DD_with_penlp > 0:
            testset.accuracy_by_DD_penlp += 1 / len(testset.examples)
        if model_type in BERT_LIKE_MODEL_TYPES:
            if example.DD_with_ll > 0:
                testset.accuracy_by_DD_ll += 1 / len(testset.examples)
            if example.DD_with_penll > 0:
                testset.accuracy_by_DD_penll += 1 / len(testset.examples)

        for _idx, typed_sentence in enumerate(example.sentences):
            stype = typed_sentence.stype
            sentence = typed_sentence.sent

            testset.lp_average_by_sentence_type[stype] += sentence.lp_softmax
            testset.penlp_average_by_sentence_type[stype] += sentence.pen_lp_softmax
            if model_type in BERT_LIKE_MODEL_TYPES:
                testset.ll_average_by_sentence_type[stype] += sentence.lp_logistic
                testset.penll_average_by_sentence_type[
                    stype
                ] += sentence.pen_lp_logistic

    for stype in testset.get_sentence_types():
        testset.lp_average_by_sentence_type[stype] /= len(testset.examples)
        testset.penlp_average_by_sentence_type[stype] /= len(testset.examples)
        if model_type in BERT_LIKE_MODEL_TYPES:
            testset.ll_average_by_sentence_type[stype] /= len(testset.examples)
            testset.penll_average_by_sentence_type[stype] /= len(testset.examples)

    testset.avg_DD_lp = get_dd_score_parametric(
        a_short_nonisland_score=testset.lp_average_by_sentence_type[
            SentenceNames.SHORT_NONISLAND
        ],
        b_long_nonisland_score=testset.lp_average_by_sentence_type[
            SentenceNames.LONG_NONISLAND
        ],
        c_short_island_score=testset.lp_average_by_sentence_type[
            SentenceNames.SHORT_ISLAND
        ],
        d_long_island_score=testset.lp_average_by_sentence_type[
            SentenceNames.LONG_ISLAND
        ],
    )
    testset.avg_DD_penlp = get_dd_score_parametric(
        a_short_nonisland_score=testset.penlp_average_by_sentence_type[
            SentenceNames.SHORT_NONISLAND
        ],
        b_long_nonisland_score=testset.penlp_average_by_sentence_type[
            SentenceNames.LONG_NONISLAND
        ],
        c_short_island_score=testset.penlp_average_by_sentence_type[
            SentenceNames.SHORT_ISLAND
        ],
        d_long_island_score=testset.penlp_average_by_sentence_type[
            SentenceNames.LONG_ISLAND
        ],
    )
    if model_type in BERT_LIKE_MODEL_TYPES:
        testset.avg_DD_ll = get_dd_score_parametric(
            a_short_nonisland_score=testset.ll_average_by_sentence_type[
                SentenceNames.SHORT_NONISLAND
            ],
            b_long_nonisland_score=testset.ll_average_by_sentence_type[
                SentenceNames.LONG_NONISLAND
            ],
            c_short_island_score=testset.ll_average_by_sentence_type[
                SentenceNames.SHORT_ISLAND
            ],
            d_long_island_score=testset.ll_average_by_sentence_type[
                SentenceNames.LONG_ISLAND
            ],
        )
        testset.avg_DD_penll = get_dd_score_parametric(
            a_short_nonisland_score=testset.penll_average_by_sentence_type[
                SentenceNames.SHORT_NONISLAND
            ],
            b_long_nonisland_score=testset.penll_average_by_sentence_type[
                SentenceNames.LONG_NONISLAND
            ],
            c_short_island_score=testset.penll_average_by_sentence_type[
                SentenceNames.SHORT_ISLAND
            ],
            d_long_island_score=testset.penll_average_by_sentence_type[
                SentenceNames.LONG_ISLAND
            ],
        )

    return testset


def _get_example_dd_scores(example: Example, model_type: ModelTypes):

    example_dd_with_lp = _get_example_dd_score(example, ScoringMeasures.LP)
    example_dd_with_penlp = _get_example_dd_score(example, ScoringMeasures.PenLP)

    example_dd_with_ll, example_dd_with_pll = None, None
    if model_type in BERT_LIKE_MODEL_TYPES:
        example_dd_with_ll = _get_example_dd_score(example, ScoringMeasures.LL)
        example_dd_with_pll = _get_example_dd_score(example, ScoringMeasures.PLL)

    return (
        example_dd_with_lp,
        example_dd_with_penlp,
        example_dd_with_ll,
        example_dd_with_pll,
    )


def _get_example_dd_score(example: Example, score_name):
    for typed_sentence in example.sentences:
        stype = typed_sentence.stype
        sent = typed_sentence.sent
        if stype == SentenceNames.SHORT_NONISLAND:
            a_short_nonisland = sent
        elif stype == SentenceNames.LONG_NONISLAND:
            b_long_nonisland = sent
        elif stype == SentenceNames.SHORT_ISLAND:
            c_short_island = sent
        elif stype == SentenceNames.LONG_ISLAND:
            d_long_island = sent
        else:
            raise ValueError(f"Unexpected sentence type: {stype}")

    return get_dd_score_parametric(
        a_short_nonisland.get_score(score_name),
        b_long_nonisland.get_score(score_name),
        c_short_island.get_score(score_name),
        d_long_island.get_score(score_name),
    )


def score_factorial_testsets(
    model_type: ModelTypes,
    model,
    tokenizer,
    device: DEVICES,
    parsed_testsets: list[TestSet],
    experimental_design: ExperimentalDesigns,
) -> list[TestSet]:

    # todo: see activation levels in the model layers, try to identify several phenomena: clause segmentation,
    #  different constructs, long vs short dependencies, wh vs rc dependencies, islands vs non islands

    # todo: see if the pretrained models by Bostrom et al. 2020 perform better (on Sprouse and Blimp english test data )
    #  when they use more linguistically plausible subwords units for tokenization.

    scored_testsets = []
    for parsed_testset in parsed_testsets:
        logging.info(
            f"Scoring testset {parsed_testset.linguistic_phenomenon}, on {model_type=} {parsed_testset.model_descr}"
        )
        scored_testset = score_factorial_testset(
            model_type, model, tokenizer, device, parsed_testset, experimental_design
        )
        scored_testsets.append(scored_testset)

    if experimental_design is ExperimentalDesigns.FACTORIAL:
        _calculate_zscores_across_testsets(scored_testsets)

    return scored_testsets


def _calculate_zscores_across_testsets(scored_testsets: list[TestSet]):
    #  first get a reference for mean and sd:
    #  after the 4 testsets have been scored, merge the arrays of scores for
    #  the 4 phenomena in the testset, and for all 4 sentence types in the
    #  examples.
    scoring_measures = scored_testsets[0].get_scoring_measures()
    logging.debug(f"Calculating zscores for {scoring_measures=}")
    merged_scores_by_scoring_measure: dict[ScoringMeasures, list[float]] = dict()
    for scoring_measure in scoring_measures:

        merged_scores_by_scoring_measure[
            scoring_measure
        ] = get_merged_score_across_testsets(scoring_measure, scored_testsets)

        logging.debug(
            f"For {scoring_measure.name} got {len(merged_scores_by_scoring_measure[scoring_measure])} scores, "
            f"with min {min(merged_scores_by_scoring_measure[scoring_measure])}, "
            f"max {max(merged_scores_by_scoring_measure[scoring_measure])}, "
            f"and mean {mean(merged_scores_by_scoring_measure[scoring_measure])} "
        )

    likert_bins_by_scoring_measure = dict()  # : dict[ScoringMeasures, ..bins_type]
    merged_likert_scores_by_scoring_measure = dict()
    for scoring_measure in merged_scores_by_scoring_measure.keys():

        (
            likert_scores_merged,
            likert_bins,
        ) = pd.cut(  # todo: use pd.qcut instead of pd.cut?
            merged_scores_by_scoring_measure[scoring_measure],
            bins=7,
            labels=np.arange(start=1, stop=8),
            retbins=True,
            # right=False,
        )

        merged_likert_scores_by_scoring_measure[scoring_measure] = np.asarray(
            likert_scores_merged
        )
        likert_bins_by_scoring_measure[scoring_measure] = likert_bins
        logging.debug(f"{likert_scores_merged=}")

    for scoring_measure in merged_scores_by_scoring_measure.keys():
        for testset in scored_testsets:
            testset.set_avg_zscores_by_measure_and_by_stype(
                scoring_measure,
                merged_scores_by_scoring_measure[scoring_measure],
                merged_likert_scores_by_scoring_measure[scoring_measure],
                likert_bins_by_scoring_measure[scoring_measure],
            )

        logging.debug(
            f"{scoring_measure}: {testset.avg_zscores_by_measure_and_by_stype=}"
        )
        logging.debug(
            f"{scoring_measure} std errors: {testset.std_error_of_zscores_by_measure_and_by_stype=}"
        )


def rescore_testsets_and_save_pickles(
    model_type: ModelTypes,
    model_name: str,
    testset_dir_path: str,
    testsets_root_filenames: list[str],
    dataset_source: DataSources,
    experimental_design: ExperimentalDesigns,
    examples_format: str = "json_lines",
    max_examples=1000,
) -> list[TestSet]:

    scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
    if model_type in BERT_LIKE_MODEL_TYPES:
        scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
    logging.info(f"Running testsets from dir {testset_dir_path}")

    parsed_testsets = parse_testsets(
        testset_dir_path,
        testsets_root_filenames,
        dataset_source,
        examples_format,
        experimental_design,
        model_name,
        scoring_measures,
        max_examples=max_examples,
    )

    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

    scored_testsets = score_factorial_testsets(
        model_type,
        model,
        tokenizer,
        DEVICES.CPU,
        parsed_testsets,
        experimental_design,
    )
    save_scored_testsets(scored_testsets, model_name, dataset_source)

    return scored_testsets
