import logging
from statistics import mean

import numpy as np
import pandas as pd
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.file_utils import _parse_arguments
from linguistic_tests.file_utils import _setup_logging
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DataSources
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import ExperimentalDesigns
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import get_testset_params
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.plots_and_prints import _print_testset_results
from linguistic_tests.plots_and_prints import plot_testsets
from linguistic_tests.plots_and_prints import print_accuracy_scores
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
    raise NotImplementedError


def run_tests_lau_et_al():
    # todo: compare results with other models
    raise NotImplementedError


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
        ) = example.get_dd_scores(model_type)
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


def run_test_design(
    model_types_and_names: dict[str, ModelTypes],
    tests_subdir,
    max_examples,
    rescore=False,
    log_level=logging.INFO,
):

    _setup_logging(log_level)
    args = _parse_arguments()

    # model_types_to_run = [
    #     ModelTypes(model_type_int) for model_type_int in args.model_types
    # ]
    # todo: also add command line option for tests subdir path
    if args.datasource == "sprouse":
        tests_subdir = "sprouse/"
    elif args.datasource == "madeddu":
        tests_subdir = "syntactic_tests_it/"
    elif args.datasource == "blimp":
        tests_subdir = "blimp/from_blim_en/islands/"
    # model_dir = str(get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch")
    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

    logging.info(f"Will run tests with models: {model_types_and_names.values()}")

    (
        testsets_root_filenames,
        broader_test_type,
        dataset_source,
        experimental_design,
    ) = get_testset_params(tests_subdir)

    for model_name, model_type in model_types_and_names.items():
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

        if experimental_design == ExperimentalDesigns.FACTORIAL:

            # todo: add experimental_design param to work also with minimal pairs testsets
            _print_testset_results(
                loaded_testsets, broader_test_type, model_type, testsets_root_filenames
            )

            load_and_plot_pickle(
                testsets_root_filenames,
                model_name,
                dataset_source,
                model_type,
                expected_experimental_design=experimental_design,
                loaded_testsets=loaded_testsets,
            )
            # todo:
            # plot with 7+1x7 subplots of a testset (one subplot for each example)
            # nb: having the standard errors in the plots is already overcoming this,
            # showing the variance

        print_orange(f"Finished test session for {model_type=}")
