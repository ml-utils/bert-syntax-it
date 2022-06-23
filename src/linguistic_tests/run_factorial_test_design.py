import logging
import os.path
import time
from statistics import mean
from typing import List

from linguistic_tests.compute_model_score import print_accuracy_scores
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.file_utils import parse_testsets
from linguistic_tests.lm_utils import assert_almost_equale
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_results_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.testset import Example
from linguistic_tests.testset import get_dd_score_parametric
from linguistic_tests.testset import get_merged_score_across_testsets
from linguistic_tests.testset import load_testset_from_pickle
from linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
from linguistic_tests.testset import TestSet
from matplotlib import pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm


model_names_it = {
    ModelTypes.GEPPETTO: "LorenzoDeMattei/GePpeTto",
    ModelTypes.BERT: "dbmdz/bert-base-italian-xxl-cased",
    ModelTypes.GILBERTO: "idb-ita/gilberto-uncased-from-camembert",
}  # ModelTypes.GPT # ModelTypes.ROBERTA  #

model_names_en = {
    ModelTypes.BERT: "bert-base-uncased",  # "bert-large-uncased"  #
    ModelTypes.GPT: "gpt2-large",
    ModelTypes.ROBERTA: "roberta-large",
}

sprouse_testsets_root_filenames = [  # 'rc_adjunct_island',
    # 'rc_complex_np', 'rc_subject_island', 'rc_wh_island', # fixme: rc_wh_island empty file
    "wh_adjunct_island",
    "wh_complex_np",
    "wh_subject_island",
    "wh_whether_island",
]
custom_it_island_testsets_root_filenames = [
    # "wh_adjunct_islands",
    # "wh_complex_np_islands",
    # "wh_whether_island",
    # "wh_subject_islands",
    "wh_whether_island",
    "wh_complex_np_islands",
    "wh_subject_islands",
    "wh_adjunct_islands",
]

# todo: parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file


def score_sprouse_testsets(
    model_type,
    model,
    tokenizer,
    device,
    parsed_testsets: List[TestSet],
) -> list[TestSet]:
    # todo: see activation levels in the model layers, try to identify several phenomena: clause segmentation,
    #  different constructs, long vs short dependencies, wh vs rc dependencies, islands vs non islands

    # todo: see if the pretrained models by Bostrom et al. 2020 perform better (on Sprouse and Blimp english test data )
    #  when they use more linguistically plausible subwords units for tokenization.

    scored_testsets = []
    for parsed_testset in parsed_testsets:

        scored_testset = score_factorial_testset(
            model_type,
            model,
            tokenizer,
            device,
            parsed_testset,
        )
        scored_testsets.append(scored_testset)

    # calculating the zscores
    #  first get a reference for mean and sd:
    #  after the 4 testsets have been scored, merge the arrays of scores for
    #  the 4 phenomena in the testset, and for all 4 sentence types in the
    #  examples.
    scoring_measures = scored_testsets[0].get_scoring_measures()
    logging.debug(f"Calculating zscores for {scoring_measures=}")
    merged_scores_by_scoring_measure: dict[ScoringMeasures, List[float]] = dict()
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

    for scoring_measure in merged_scores_by_scoring_measure.keys():
        for testset in scored_testsets:
            testset.set_avg_zscores_by_measure_and_by_stype(
                scoring_measure, merged_scores_by_scoring_measure[scoring_measure]
            )

        logging.debug(
            f"{scoring_measure}: {testset.avg_zscores_by_measure_and_by_stype=}"
        )
        logging.debug(
            f"{scoring_measure} std errors: {testset.std_error_of_zscores_by_measure_and_by_stype=}"
        )

    return scored_testsets


def get_test_session_descr(dataset_source, model_descr, score_name=""):
    session_descr = f"{dataset_source[:7]}_{model_descr}_{score_name}"
    session_descr = session_descr.replace(" ", "_").replace("/", "_")
    return session_descr


def plot_results(scored_testsets: list[TestSet], score_name, use_zscore=False):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 12.8))  # default figsize=(6.4, 4.8)

    window_title = get_test_session_descr(
        scored_testsets[0].dataset_source, scored_testsets[0].model_descr, score_name
    )

    fig.canvas.manager.set_window_title(window_title)
    axs_list = axs.reshape(-1)
    logging.debug(f"type axs_list: {type(axs_list)}, {len(axs_list)=}, {axs_list=}")

    preferred_axs_order = {"whether": 0, "complex": 1, "subject": 2, "adjunct": 3}
    for phenomenon_short_name, preferred_index in preferred_axs_order.items():
        if (
            phenomenon_short_name
            not in scored_testsets[preferred_index].linguistic_phenomenon
        ):
            for idx, testset in enumerate(scored_testsets):
                if phenomenon_short_name in testset.linguistic_phenomenon:
                    scored_testsets.remove(testset)
                    scored_testsets.insert(preferred_index, testset)
                    break

    set_xlabels = [False, False, True, True]
    for scored_testset, ax, set_xlabel in zip(scored_testsets, axs_list, set_xlabels):
        lines, labels = _plot_results_subplot(
            scored_testset,
            score_name,
            ax,
            use_zscore=use_zscore,
            set_xlabel=set_xlabel,
        )

    fig.suptitle(
        f"Model: {scored_testsets[0].model_descr}, "
        f"\n Dataset: {scored_testset.dataset_source}"
    )

    if use_zscore:
        zscore_note = "-zscores"
    else:
        zscore_note = ""

    saving_dir = str(get_results_dir())
    timestamp = time.strftime("%Y-%m-%d_h%Hm%Ms%S")
    filename = f"{window_title}{zscore_note}-{timestamp}.png"
    filepath = os.path.join(saving_dir, filename)

    print_orange(f"Saving plot to file {filepath} ..")
    plt.savefig(filepath)  # , dpi=300
    plt.figlegend(lines, labels)
    # fig.tight_layout()

    # plt.show()


def _plot_results_subplot(
    scored_testset: TestSet,
    scoring_measure,
    ax,
    use_zscore=False,
    set_xlabel=True,
):
    if use_zscore:
        DD_value = scored_testset.get_avg_DD_zscores(scoring_measure)

        y_values_ni = [
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.SHORT_NONISLAND
            ),
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.LONG_NONISLAND
            ),
        ]
        y_values_is = [
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.SHORT_ISLAND
            ),
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.LONG_ISLAND
            ),
        ]
    else:
        DD_value = scored_testset.get_avg_DD(scoring_measure)
        score_averages = scored_testset.get_avg_scores(scoring_measure)
        y_values_ni = [
            score_averages[SentenceNames.SHORT_NONISLAND],
            score_averages[SentenceNames.LONG_NONISLAND],
        ]
        y_values_is = [
            score_averages[SentenceNames.SHORT_ISLAND],
            score_averages[SentenceNames.LONG_ISLAND],
        ]

    logging.debug(f"{y_values_ni=}")
    logging.debug(f"{y_values_is=}")

    # todo: add p values
    # todo: add accuracy %

    x_values = ["SHORT", "LONG"]

    std_errors_ni = (
        scored_testset.get_std_errors_of_zscores_by_measure_and_sentence_structure(
            scoring_measure, SentenceNames.SHORT_NONISLAND
        )
    )
    std_errors_is = (
        scored_testset.get_std_errors_of_zscores_by_measure_and_sentence_structure(
            scoring_measure, SentenceNames.SHORT_ISLAND
        )
    )

    (non_island_line, _, _) = ax.errorbar(
        x_values, y_values_ni, yerr=std_errors_ni, capsize=5  # marker="o",
    ).lines
    (island_line, _, _) = ax.errorbar(
        x_values,
        y_values_is,
        linestyle="--",
        yerr=std_errors_is,
        capsize=5,  # marker="o",
    ).lines
    lines = [non_island_line, island_line]
    labels = ["Non-island structure", "Island structure"]

    ax.legend(title=f"DD = {DD_value:.2f}")

    if use_zscore:
        ax.set_ylabel(f"z-scores ({scoring_measure})")
        ax.set_ylim(ymin=-1.5, ymax=1.5)
    else:
        ax.set_ylabel(f"{scoring_measure} values")
    if set_xlabel:
        ax.set_xlabel("Dependency distance")
    # ax.set_aspect('equal', 'box')  # , 'box'
    ax.set_title(scored_testset.linguistic_phenomenon)

    return lines, labels


def get_pvalue_with_likelihood_ratio_test(full_model_ll, reduced_model_ll):
    likelihood_ratio = 2 * (reduced_model_ll - full_model_ll)
    p = chi2.sf(likelihood_ratio, 1)  # L2 has 1 DoF more than L1
    return p


def score_factorial_testset(
    model_type, model, tokenizer, device, testset: TestSet
) -> TestSet:
    # todo set scorebase param

    for example_idx, example in enumerate(tqdm(testset.examples)):
        score_example(
            device,
            example,
            model,
            model_type,
            tokenizer,
        )

    # scoring accuracy rates
    for scoring_measure in testset.accuracy_per_score_type_per_sentence_type.keys():
        for (
            stype_acceptable_sentence
        ) in testset.accuracy_per_score_type_per_sentence_type[scoring_measure].keys():
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

    # doing factorial design scores
    for example_idx, example in enumerate(testset.examples):
        (
            example.DD_with_lp,
            example.DD_with_penlp,
            example.dd_with_ll,
            example.dd_with_pll,
        ) = get_dd_scores_wdataclasses(example, model_type)
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


def print_example(example_data, sentence_ordering):
    logging.debug(f"sentence ordering is {type(sentence_ordering)}")
    print(
        f"\nSHORT_NONISLAND: {example_data[sentence_ordering.SHORT_NONISLAND]}"
        f"\nLONG_NONISLAND : {example_data[sentence_ordering.LONG_NONISLAND]}"
        f"\nSHORT_ISLAND : {example_data[sentence_ordering.SHORT_ISLAND]}"
        f"\nLONG_ISLAND : {example_data[sentence_ordering.LONG_ISLAND]}"
    )


def get_dd_scores_wdataclasses(example: Example, model_type: ModelTypes):

    example_dd_with_lp = get_example_dd_score(example, ScoringMeasures.LP)
    example_dd_with_penlp = get_example_dd_score(example, ScoringMeasures.PenLP)

    example_dd_with_ll, example_dd_with_pll = None, None
    if model_type in BERT_LIKE_MODEL_TYPES:
        example_dd_with_ll = get_example_dd_score(example, ScoringMeasures.LL)
        example_dd_with_pll = get_example_dd_score(example, ScoringMeasures.PLL)

    return (
        example_dd_with_lp,
        example_dd_with_penlp,
        example_dd_with_ll,
        example_dd_with_pll,
    )


def get_example_dd_score(example: Example, score_name):
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


def get_dd_score(sentences_scores, sentences_ordering=SprouseSentencesOrder):
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
    assert_almost_equale(example_island_effect_with_lp, example_dd_with_lp)
    return example_dd_with_lp


# def plot_all_phenomena(phenomena_names, lp_avg_scores):
#     for idx, phenomenon in enumerate(phenomena_names):
#         plot_results(phenomenon, lp_avg_scores[idx], ScoringMeasures.LP.name)


def save_scored_testsets(
    scored_testsets: List[TestSet], model_name: str, dataset_source: str
):
    for scored_testset in scored_testsets:
        scored_testset.model_descr = model_name
        filename = get_pickle_filename(
            dataset_source,
            scored_testset.linguistic_phenomenon,
            model_name,
        )

        scored_testset.save_to_pickle(filename)


def get_pickle_filename(
    dataset_source,
    linguistic_phenomenon,
    model_descr,
):
    # todo: filenames as pyplot filenames
    #  rename as get_pickle_filepath, ad results dir (same as pyplot images)

    filename_base = get_test_session_descr(dataset_source, model_descr)

    filename = f"{filename_base}_{linguistic_phenomenon}_.testset.pickle"
    return filename


def load_pickles(dataset_source, phenomena, model_name) -> list[TestSet]:
    # phenomena = [
    #     "custom-wh_whether_island",
    #     "custom-wh_complex_np_islands",
    #     "custom-wh_subject_islands",
    #     "custom-wh_adjunct_islands",
    # ]
    loaded_testsets = []
    for phenomenon in phenomena:
        filename = get_pickle_filename(dataset_source, phenomenon, model_name)
        loaded_testset = load_testset_from_pickle(filename)
        loaded_testsets.append(loaded_testset)

    return loaded_testsets


def load_and_plot_pickle(
    phenomena,
    model_name,
    dataset_source,
    model_type: ModelTypes,
    loaded_testsets=None,
):

    if loaded_testsets is None:
        loaded_testsets = load_pickles(dataset_source, phenomena, model_name)

    plot_testsets(loaded_testsets, model_type)


def plot_testsets(scored_testsets: List[TestSet], model_type: ModelTypes):

    plot_results(scored_testsets, ScoringMeasures.PenLP.name, use_zscore=True)
    plot_results(scored_testsets, ScoringMeasures.LP.name, use_zscore=True)

    # plot_results(scored_testsets, ScoringMeasures.LP.name)
    # plot_results(scored_testsets, ScoringMeasures.PenLP.name)

    if model_type in BERT_LIKE_MODEL_TYPES:
        # plot_results(scored_testsets, ScoringMeasures.LL.name)
        # plot_results(scored_testsets, ScoringMeasures.PLL.name)
        plot_results(scored_testsets, ScoringMeasures.LL.name, use_zscore=True)
        plot_results(scored_testsets, ScoringMeasures.PLL.name, use_zscore=True)


def print_sorted_sentences_to_check_spelling_errors2(
    score_descr, phenomena, model_name, dataset_source, loaded_testsets=None
):

    if loaded_testsets is None:
        loaded_testsets = load_pickles(dataset_source, phenomena, model_name)

    for testset in loaded_testsets:
        logging.info(
            f"printing for testset {testset.linguistic_phenomenon} calculated from {testset.model_descr}"
        )
        typed_sentences = testset.get_all_sentences_sorted_by_score(
            score_descr, reverse=False
        )
        print(f"{'idx':<5}" f"{score_descr:<10}" f"{'stype':<20}" f"{'txt':<85}")
        for idx, tsent in enumerate(typed_sentences):
            print(
                f"{idx : <5}"
                f"{tsent.sent.get_score(score_descr) : <10.2f}"
                f"{tsent.stype : <20}"
                f"{tsent.sent.txt : <85}"
            )


def print_sorted_sentences_to_check_spelling_errors(
    score_descr, phenomena, model_name, dataset_source, loaded_testsets=None
):
    logging.info("printing sorted_sentences_to_check_spelling_errors")

    if loaded_testsets is None:
        loaded_testsets = load_pickles(dataset_source, phenomena, model_name)

    for testset in loaded_testsets:
        logging.info(
            f"printing {score_descr} for testset {testset.linguistic_phenomenon} calculated from {testset.model_descr}"
        )
        for stype in SPROUSE_SENTENCE_TYPES:
            logging.info(f"printing for sentence type {stype}..")
            examples = testset.get_examples_sorted_by_sentence_type_and_score(
                stype, score_descr, reverse=False
            )
            print(f"{'idx':<5}" f"{score_descr:^10}" f"{'txt':<85}")
            for idx, example in enumerate(examples):
                print(
                    f"{idx : <5}"
                    f"{example[stype].get_score(score_descr) : <10.2f}"
                    f"{example[stype].txt : <85}"
                )


def print_examples_compare_diff(
    score_descr,
    sent_type1,
    sent_type2,
    phenomena,
    model_name,
    dataset_source,
    testsets=None,
):
    if testsets is None:
        testsets = load_pickles(dataset_source, phenomena, model_name)

    max_testsets = 4
    for testset in testsets[:max_testsets]:
        logging.info(
            f"printing testset for {testset.linguistic_phenomenon} from {testset.model_descr}"
        )
        print(
            f"comparing {sent_type1} and {sent_type2} ({testset.linguistic_phenomenon} from {testset.model_descr})"
        )
        examples = testset.get_examples_sorted_by_score_diff(
            score_descr, sent_type1, sent_type2, reverse=False
        )
        max_prints = 50
        print(
            f"{'diff':<8}"
            f"{'s1 '+score_descr:^10}"
            f"{'s1 txt (' + sent_type1 + ')':<95}"
            f"{'s2 '+score_descr:^10}"
            f"{'s2 txt (' + sent_type2 + ')':<5}"
        )
        for example in examples[0:max_prints]:
            print(
                f"{example.get_score_diff(score_descr, sent_type1, sent_type2) : <8.2f}"
                f"{example[sent_type1].get_score(score_descr) : ^10.2f}"
                f"{example[sent_type1].txt : <95}"
                f"{example[sent_type2].get_score(score_descr) :^10.2f}"
                f"{example[sent_type2].txt : <5}"
            )


def print_testset_results(
    scored_testsets: List[TestSet],
    dataset_source: str,
    model_type: ModelTypes,
    testsets_root_filenames: List[str],
):
    logging.info("Printing accuracy scores..")
    for scored_testset in scored_testsets:
        # todo: also print results in table format or csv for excel export or word doc report
        print_accuracy_scores(scored_testset)
        print(
            f"Testset accuracy with DDs_with_lp: {scored_testset.accuracy_by_DD_lp:%}"
        )
        print(
            f"Testset accuracy with DDs_with_penlp: {scored_testset.accuracy_by_DD_penlp:%}"
        )
        lp_averages = scored_testset.lp_average_by_sentence_type
        penlp_averages = scored_testset.penlp_average_by_sentence_type
        print(f"{lp_averages=}")
        print(f"{penlp_averages=}")

        if model_type in BERT_LIKE_MODEL_TYPES:
            print(
                f"Testset accuracy with DDs_with_ll: {scored_testset.accuracy_by_DD_ll:%}"
            )
            print(
                f"Testset accuracy with DDs_with_penll: {scored_testset.accuracy_by_DD_penll:%}"
            )
            ll_averages = scored_testset.ll_average_by_sentence_type
            penll_averages = scored_testset.penll_average_by_sentence_type
            print(f"{ll_averages=}")
            print(f"{penll_averages=}")

    score_descr = ScoringMeasures.PenLP.name

    print_sorted_sentences_to_check_spelling_errors2(
        score_descr,
        testsets_root_filenames,
        model_names_it[model_type],
        dataset_source,
        scored_testsets,
    )
    print_sorted_sentences_to_check_spelling_errors(
        score_descr,
        testsets_root_filenames,
        model_names_it[model_type],
        dataset_source,
        scored_testsets,
    )

    print_examples_compare_diff(
        score_descr,
        SentenceNames.SHORT_ISLAND,
        SentenceNames.LONG_ISLAND,
        testsets_root_filenames,
        model_names_it[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    print_examples_compare_diff(
        score_descr,
        SentenceNames.LONG_NONISLAND,
        SentenceNames.LONG_ISLAND,
        testsets_root_filenames,
        model_names_it[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    print_examples_compare_diff(
        score_descr,
        SentenceNames.SHORT_NONISLAND,
        SentenceNames.SHORT_ISLAND,
        testsets_root_filenames,
        model_names_it[model_type],
        dataset_source,
        testsets=scored_testsets,
    )


def rescore_testsets_and_save_pickles(
    model_type,
    testset_dir_path,
    testsets_root_filenames,
    dataset_source,
):
    model_name = model_names_it[model_type]
    examples_format = "sprouse"  # "blimp", "json_lines", "sprouse"
    sent_types_descr = "sprouse"  # "blimp" or "sprouse"
    # sentence_ordering = SprouseSentencesOrder  # BlimpSentencesOrder
    logging.info(f"Running testsets from dir {testset_dir_path}")
    scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
    if model_type in BERT_LIKE_MODEL_TYPES:
        scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
    parsed_testsets = parse_testsets(
        # todo: add scorebase var in testset class
        testset_dir_path,
        testsets_root_filenames,
        dataset_source,
        examples_format,
        sent_types_descr,
        model_name,
        model_type,
        scoring_measures,
        max_examples=1000,
    )

    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)

    scored_testsets = score_sprouse_testsets(
        model_type,
        model,
        tokenizer,
        device,
        parsed_testsets,
    )
    save_scored_testsets(scored_testsets, model_name, dataset_source)


def get_testset_params(tests_subdir):
    if tests_subdir == "syntactic_tests_it/":
        testsets_root_filenames = custom_it_island_testsets_root_filenames
        broader_test_type = "it_tests"
        dataset_source = "Madeddu (50 items per phenomenon)"
    elif tests_subdir == "sprouse/":
        testsets_root_filenames = sprouse_testsets_root_filenames
        broader_test_type = "sprouse"
        dataset_source = "Sprouse et al. 2016 (8 items per phenomenon)"
    else:
        raise ValueError(f"Invalid tests_subdir specified: {tests_subdir}")

    return testsets_root_filenames, broader_test_type, dataset_source


def main(
    tests_subdir="syntactic_tests_it/", rescore=False, log_level=logging.INFO
):  # tests_subdir="sprouse/"

    fmt = "[%(levelname)s] %(asctime)s - %(message)s"

    logging.getLogger("matplotlib.font_manager").disabled = True
    # stdout_handler = calcula.StreamHandler(sys.stdout)
    # root_logger = logging.getLogger()
    # root_logger.addFilter(NoFontMsgFilter())
    # root_logger.addFilter(NoStreamMsgFilter())

    logging.basicConfig(format=fmt, level=log_level)  #
    this_module_logger = logging.getLogger(__name__)
    this_module_logger.setLevel(log_level)

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_types",
        help=f"specify the models to run. { {i.name: i.value for i in ModelTypes} }",
        nargs="+",  # 0 or more values expected => creates a list
        type=int,
        choices=[i.value for i in model_names_it.keys()],
        default=[i.value for i in model_names_it.keys()],
    )
    # arg_parser.add_argument(
    #     "--rescore"
    # )
    args = arg_parser.parse_args()
    model_types_to_run = [
        ModelTypes(model_type_int) for model_type_int in args.model_types
    ]
    # rescore =
    logging.info(f"Will run tests with models: {model_types_to_run}")

    # todo: also add command line option for tests subdir path
    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

    testsets_root_filenames, broader_test_type, dataset_source = get_testset_params(
        tests_subdir
    )

    for model_type in model_types_to_run:
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
            )

        loaded_testsets = load_pickles(
            dataset_source,
            testsets_root_filenames,
            model_names_it[model_type],
        )

        print_testset_results(
            loaded_testsets, broader_test_type, model_type, testsets_root_filenames
        )

        load_and_plot_pickle(
            testsets_root_filenames,
            model_names_it[model_type],
            dataset_source,
            model_type,
            loaded_testsets=loaded_testsets,
        )
        print_orange(f"Finished test session for {model_type=}")

    # todo:
    # plot with 7+1x7 subplots of a testset (one subplot for each example)

    # todo:
    # normalize sentence/tokens scores for Bert models, to have scores comparables across a whole testset
    #
    # normalize results to a likert scale, for comparison with Sprouse et al 2016


if __name__ == "__main__":
    main(
        tests_subdir="syntactic_tests_it/", rescore=True, log_level=logging.DEBUG
    )  # tests_subdir="syntactic_tests_it/"  # tests_subdir="sprouse/"
