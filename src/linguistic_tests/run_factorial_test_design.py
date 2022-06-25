import logging

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
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.plots_and_prints import _print_testset_results
from linguistic_tests.plots_and_prints import plot_testsets
from linguistic_tests.plots_and_prints import print_accuracy_scores
from linguistic_tests.run_minimal_pairs_test_design import score_factorial_testsets
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


def get_pvalue_with_likelihood_ratio_test(full_model_ll, reduced_model_ll):
    likelihood_ratio = 2 * (reduced_model_ll - full_model_ll)
    p = chi2.sf(likelihood_ratio, 1)  # L2 has 1 DoF more than L1
    return p


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
    # todo: also add command line option for tests subdir path
    if args.datasource == "sprouse":
        tests_subdir = "sprouse/"
    elif args.datasource == "madeddu":
        tests_subdir = "syntactic_tests_it/"
    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

    logging.info(f"Will run tests with models: {MODEL_TYPES_AND_NAMES_IT.values()}")

    (
        testsets_root_filenames,
        broader_test_type,
        dataset_source,
        experimental_design,
    ) = get_testset_params(tests_subdir)

    for model_name, model_type in MODEL_TYPES_AND_NAMES_IT.items():
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
