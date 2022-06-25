import logging
from statistics import mean

import numpy as np
import pandas as pd
from linguistic_tests.lm_utils import assert_almost_equal
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DataSources
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import ExperimentalDesigns
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import get_testset_params
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import MODEL_NAMES_IT
from linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_IT
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.plots_and_prints import _print_testset_results
from linguistic_tests.plots_and_prints import plot_testsets
from linguistic_tests.run_minimal_pairs_test_design import score_minimal_pairs_testset
from linguistic_tests.testset import Example
from linguistic_tests.testset import get_dd_score_parametric
from linguistic_tests.testset import get_merged_score_across_testsets
from linguistic_tests.testset import load_testsets_from_pickles
from linguistic_tests.testset import parse_testsets
from linguistic_tests.testset import save_scored_testsets
from linguistic_tests.testset import TestSet
from scipy.stats import chi2


# todo: move to lm_utils
# todo: move to lm_utils

# todo: parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file


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


def get_pvalue_with_likelihood_ratio_test(full_model_ll, reduced_model_ll):
    likelihood_ratio = 2 * (reduced_model_ll - full_model_ll)
    p = chi2.sf(likelihood_ratio, 1)  # L2 has 1 DoF more than L1
    return p


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


def _get_dd_score(sentences_scores, sentences_ordering=SprouseSentencesOrder):
    a_short_nonisland_idx = sentences_ordering.SHORT_NONISLAND
    b_long_nonisland = sentences_ordering.LONG_NONISLAND
    c_short_island = sentences_ordering.SHORT_ISLAND
    d_long_island = sentences_ordering.LONG_ISLAND
    example_lenght_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[b_long_nonisland]
    )
    example_structure_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[c_short_island]
    )
    example_total_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[d_long_island]
    )
    example_island_effect_with_lp = example_total_effect_with_lp - (
        example_lenght_effect_with_lp + example_structure_effect_with_lp
    )
    example_dd_with_lp = example_structure_effect_with_lp - (
        sentences_scores[b_long_nonisland] - sentences_scores[d_long_island]
    )
    example_dd_with_lp *= -1
    assert_almost_equal(example_island_effect_with_lp, example_dd_with_lp)
    return example_dd_with_lp


# def plot_all_phenomena(phenomena_names, lp_avg_scores):
#     for idx, phenomenon in enumerate(phenomena_names):
#         plot_results(phenomenon, lp_avg_scores[idx], ScoringMeasures.LP.name)


def load_and_plot_pickle(
    phenomena,
    model_name,
    dataset_source,
    model_type: ModelTypes,
    expected_experimental_design: ExperimentalDesigns,
    loaded_testsets=None,
):

    if loaded_testsets is None:
        loaded_testsets = load_testsets_from_pickles(
            dataset_source,
            phenomena,
            model_name,
            expected_experimental_design=expected_experimental_design,
        )

    plot_testsets(loaded_testsets, model_type)


def rescore_testsets_and_save_pickles(
    model_type: ModelTypes,
    testset_dir_path: str,
    testsets_root_filenames: list[str],
    dataset_source: DataSources,
    examples_format: str = "sprouse",  # "blimp", "json_lines", "sprouse"
    max_examples=1000,
):
    model_name = MODEL_NAMES_IT[model_type]
    experimental_design = ExperimentalDesigns.FACTORIAL

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
        model_type, model, tokenizer, DEVICES.CPU, parsed_testsets, experimental_design
    )
    save_scored_testsets(scored_testsets, model_name, dataset_source)


def _setup_logging(log_level):
    fmt = "[%(levelname)s] %(asctime)s - %(message)s"

    logging.getLogger("matplotlib.font_manager").disabled = True
    # stdout_handler = calcula.StreamHandler(sys.stdout)
    # root_logger = logging.getLogger()
    # root_logger.addFilter(NoFontMsgFilter())
    # root_logger.addFilter(NoStreamMsgFilter())

    logging.basicConfig(format=fmt, level=log_level)  #
    this_module_logger = logging.getLogger(__name__)
    this_module_logger.setLevel(log_level)


def _parse_arguments():

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_types",
        help=f"specify the models to run. { {i.name: i.value for i in ModelTypes} }",
        nargs="+",  # 1 or more values expected => creates a list
        type=int,
        choices=[i.value for i in MODEL_NAMES_IT.keys()],
        default=[i.value for i in MODEL_NAMES_IT.keys()],
    )
    arg_parser.add_argument(
        "--datasource",
        nargs="?",
        choices=["sprouse", "madeddu"],
    )
    # arg_parser.add_argument(
    #     "--rescore"
    # )
    args = arg_parser.parse_args()
    return args


def main(
    # todo: save accuracy results to csv file
    #  also another csv file with details on sentences scores
    #  and an option to load the report csv and print them in the command line
    tests_subdir="syntactic_tests_it/",  # tests_subdir="sprouse/"
    rescore=False,
    log_level=logging.INFO,
    max_examples=50,
):

    _setup_logging(log_level)
    args = _parse_arguments()

    # model_types_to_run = [
    #     ModelTypes(model_type_int) for model_type_int in args.model_types
    # ]
    if args.datasource == "sprouse":
        tests_subdir = "sprouse/"
    elif args.datasource == "madeddu":
        tests_subdir = "syntactic_tests_it/"
    # rescore =

    logging.info(f"Will run tests with models: {MODEL_TYPES_AND_NAMES_IT.values()}")

    # todo: also add command line option for tests subdir path
    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

    (
        testsets_root_filenames,
        broader_test_type,
        dataset_source,
        experimental_design,
    ) = get_testset_params(tests_subdir)

    for model_name, model_type in MODEL_TYPES_AND_NAMES_IT.items():
        print_orange(f"Starting test session for {model_type=}, and {dataset_source=}")
        # add score with logistic function (instead of softmax)

        # todo: check that accuracy values are scored and stored correctly
        #  (it seems they are scored twice and not shown when loading pickles)
        # save results to csv (for import in excel table)
        # autosave plots as *.png

        # create_test_jsonl_files_tests()

        if rescore:
            rescore_testsets_and_save_pickles(
                model_type,
                testset_dir_path,
                testsets_root_filenames,
                dataset_source,
                max_examples=max_examples,
            )

        loaded_testsets = load_testsets_from_pickles(
            dataset_source,
            testsets_root_filenames,
            MODEL_NAMES_IT[model_type],
            expected_experimental_design=experimental_design,
        )

        _print_testset_results(
            loaded_testsets, broader_test_type, model_type, testsets_root_filenames
        )

        load_and_plot_pickle(
            testsets_root_filenames,
            MODEL_NAMES_IT[model_type],
            dataset_source,
            model_type,
            expected_experimental_design=experimental_design,
            loaded_testsets=loaded_testsets,
        )
        print_orange(f"Finished test session for {model_type=}")

    # todo:
    # plot with 7+1x7 subplots of a testset (one subplot for each example)
    # nb: having the standard errors in the plots is already overcoming this,
    # showing the variance
