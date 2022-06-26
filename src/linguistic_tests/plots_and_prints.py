import logging
import os
import time
from typing import List

from matplotlib import pyplot as plt

from src.linguistic_tests.bert_utils import estimate_sentence_probability
from src.linguistic_tests.compute_model_score import perc
from src.linguistic_tests.lm_utils import _get_test_session_descr
from src.linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from src.linguistic_tests.lm_utils import get_results_dir
from src.linguistic_tests.lm_utils import MODEL_NAMES_IT
from src.linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import print_orange
from src.linguistic_tests.lm_utils import print_red
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
from src.linguistic_tests.testset import TestSet


def plot_results(
    scored_testsets: List[TestSet],
    score_name,
    use_zscore=False,
    likert=False,
    show_plot=False,
):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 12.8))  # default figsize=(6.4, 4.8)

    window_title = _get_test_session_descr(
        scored_testsets[0].dataset_source, scored_testsets[0].model_descr, score_name
    )

    fig.canvas.manager.set_window_title(window_title)
    axs_list = axs.reshape(-1)
    logging.debug(f"type axs_list: {type(axs_list)}, {len(axs_list)}, {axs_list}")

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
            likert=likert,
            set_xlabel=set_xlabel,
        )

    fig.suptitle(
        f"Model: {scored_testsets[0].model_descr}, "
        f"\n Dataset: {scored_testset.dataset_source} "
        f"({scored_testset.get_item_count_per_phenomenon()} items per phenomenon)"
    )

    zscore_note = ""
    if use_zscore:
        zscore_note = "-zscores"
    likert_note = ""
    if likert:
        likert_note = "-likert"

    saving_dir = str(get_results_dir())
    timestamp = time.strftime("%Y-%m-%d_h%Hm%Ms%S")
    filename = f"{window_title}{zscore_note}{likert_note}-{timestamp}.png"
    filepath = os.path.join(saving_dir, filename)

    plt.figlegend(lines, labels)
    # fig.tight_layout()

    print_orange(f"Saving plot to file {filepath} ..")
    plt.savefig(filepath)  # , dpi=300
    if show_plot:
        plt.show()


def _plot_results_subplot(
    scored_testset: TestSet,
    scoring_measure,
    ax,
    use_zscore=False,
    likert=False,  # only valid if also using z scores, on which the likerts are calculated
    set_xlabel=True,
):
    if use_zscore:
        DD_value = scored_testset.get_avg_DD_zscores(scoring_measure, likert=likert)

        y_values_ni = [
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.SHORT_NONISLAND, likert=likert
            ),
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.LONG_NONISLAND, likert=likert
            ),
        ]
        y_values_is = [
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.SHORT_ISLAND, likert=likert
            ),
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SentenceNames.LONG_ISLAND, likert=likert
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

    logging.debug(f"{y_values_ni}")
    logging.debug(f"{y_values_is}")

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

    if not use_zscore:
        ax.set_ylabel(f"{scoring_measure} values")
    else:
        ax.set_ylim(ymin=-1.5, ymax=1.5)
        if not likert:
            ax.set_ylabel(f"z-scores ({scoring_measure})")
        else:
            ax.set_ylabel(f"z-scores ({scoring_measure}, from 7-likerts)")

    if set_xlabel:
        ax.set_xlabel("Dependency distance")
    # ax.set_aspect('equal', 'box')  # , 'box'
    ax.set_title(scored_testset.linguistic_phenomenon)

    return lines, labels


def _print_example(example_data, sentence_ordering):
    logging.debug(f"sentence ordering is {type(sentence_ordering)}")
    print(
        f"\nSHORT_NONISLAND: {example_data[sentence_ordering.SHORT_NONISLAND]}"
        f"\nLONG_NONISLAND : {example_data[sentence_ordering.LONG_NONISLAND]}"
        f"\nSHORT_ISLAND : {example_data[sentence_ordering.SHORT_ISLAND]}"
        f"\nLONG_ISLAND : {example_data[sentence_ordering.LONG_ISLAND]}"
    )


def plot_testsets(
    scored_testsets: List[TestSet], model_type: ModelTypes, show_plot=False
):

    plot_results(
        scored_testsets,
        ScoringMeasures.PenLP.name,
        use_zscore=True,
        likert=True,
        show_plot=show_plot,
    )
    plot_results(
        scored_testsets,
        ScoringMeasures.LP.name,
        use_zscore=True,
        likert=True,
        show_plot=show_plot,
    )
    # plot_results(
    #     scored_testsets, ScoringMeasures.LP.name, use_zscore=True, likert=False
    # )
    # plot_results(
    #     scored_testsets, ScoringMeasures.PenLP.name, use_zscore=True, likert=False
    # )

    # plot_results(scored_testsets, ScoringMeasures.LP.name)  # without zscores
    # plot_results(scored_testsets, ScoringMeasures.PenLP.name)

    if model_type in BERT_LIKE_MODEL_TYPES:
        # plot_results(scored_testsets, ScoringMeasures.LL.name)
        # plot_results(scored_testsets, ScoringMeasures.PLL.name)
        plot_results(
            scored_testsets,
            ScoringMeasures.LL.name,
            use_zscore=True,
            likert=True,
            show_plot=show_plot,
        )
        plot_results(
            scored_testsets,
            ScoringMeasures.PLL.name,
            use_zscore=True,
            likert=True,
            show_plot=show_plot,
        )


def _print_sorted_sentences_to_check_spelling_errors2(
    score_descr,
    phenomena,
    model_name,
    dataset_source,
    loaded_testsets,
):

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


def _print_sorted_sentences_to_check_spelling_errors(
    score_descr,
    phenomena,
    model_name,
    dataset_source,
    loaded_testsets,
):
    logging.info("printing sorted_sentences_to_check_spelling_errors")

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


def _print_examples_compare_diff(
    score_descr,
    sent_type1,
    sent_type2,
    phenomena,
    model_name,
    dataset_source,
    testsets=None,
):

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


def _print_testset_results(
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
        print(f"{lp_averages}")
        print(f"{penlp_averages}")

        if model_type in BERT_LIKE_MODEL_TYPES:
            print(
                f"Testset accuracy with DDs_with_ll: {scored_testset.accuracy_by_DD_ll:%}"
            )
            print(
                f"Testset accuracy with DDs_with_penll: {scored_testset.accuracy_by_DD_penll:%}"
            )
            ll_averages = scored_testset.ll_average_by_sentence_type
            penll_averages = scored_testset.penll_average_by_sentence_type
            print(f"{ll_averages}")
            print(f"{penll_averages}")

    score_descr = ScoringMeasures.PenLP.name

    _print_sorted_sentences_to_check_spelling_errors2(
        score_descr,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        scored_testsets,
    )
    _print_sorted_sentences_to_check_spelling_errors(
        score_descr,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        scored_testsets,
    )

    _print_examples_compare_diff(
        score_descr,
        SentenceNames.SHORT_ISLAND,
        SentenceNames.LONG_ISLAND,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    _print_examples_compare_diff(
        score_descr,
        SentenceNames.LONG_NONISLAND,
        SentenceNames.LONG_ISLAND,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    _print_examples_compare_diff(
        score_descr,
        SentenceNames.SHORT_NONISLAND,
        SentenceNames.SHORT_ISLAND,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        testsets=scored_testsets,
    )


def print_detailed_sentence_info(bert, tokenizer, sentence_txt, scorebase):
    print_red(f"printing details for sentence {sentence_txt}")
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    estimate_sentence_probability(
        bert, tokenizer, sentence_ids, scorebase, verbose=True
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
                f"{testset.linguistic_phenomenon}: "
                f"Accuracy with {scoring_measure.name} "
                f"for {stype_acceptable_sentence.name}: {accuracy:%} "
                f"({testset.model_descr})"
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


def print_list_of_cached_models():
    from transformers.file_utils import get_cached_models

    print("getting list of cached models..")
    cached_models = get_cached_models()

    # print(f"{cached_models}")
    cached_models_names_urls = []
    cached_models_names_nosize_urls = []
    for model_info in cached_models:
        print(f"{model_info}")
        if model_info[2] != "no size":
            cached_models_names_urls.append(model_info[0])
        else:
            cached_models_names_nosize_urls.append(model_info[0])

    print(f"{len(cached_models_names_urls)}, {len(cached_models_names_nosize_urls)}")
    print("cached_models_names_urls:")
    for cached_models_name in cached_models_names_urls:
        print(cached_models_name)
    print("cached_models_names_nosize_urls:")
    for cached_models_name_nosize in cached_models_names_nosize_urls:
        print(cached_models_name_nosize)

    cached_models_names = []
    cached_models_names_nosize = []
    model_names_not_cached = []
    for model_name in MODEL_TYPES_AND_NAMES_EN.keys():
        # print(f"checking if {model_name} is in {cached_models_names_urls} or {cached_models_names_nosize_urls}.. ")
        found = False
        for cached_model_name_url in cached_models_names_urls:
            if model_name in cached_model_name_url:
                cached_models_names.append(model_name)
                found = True
                break
        if not found:
            for cached_models_name_nosize_url in cached_models_names_nosize_urls:
                if model_name in cached_models_name_nosize_url:
                    cached_models_names_nosize.append(model_name)
                    found = True
                    break
        if not found:
            model_names_not_cached.append(model_name)

    print("Cached model names en:")
    for model_name in cached_models_names:
        print(f"{model_name} cached")
    print("Cached model names, no size, en:")
    for model_name in cached_models_names_nosize:
        print(f"{model_name} cached but NO SIZE")
    print("Model names en NOT CACHED:")
    for model_name in model_names_not_cached:
        print(f"{model_name} NOT CACHED")

    return
