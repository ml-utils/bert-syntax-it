import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Union

from matplotlib import pyplot as plt

from src.linguistic_tests.bert_utils import estimate_sentence_probability
from src.linguistic_tests.compute_model_score import perc
from src.linguistic_tests.lm_utils import _get_test_session_descr
from src.linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import get_results_dir
from src.linguistic_tests.lm_utils import MODEL_NAMES_IT
from src.linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import print_orange
from src.linguistic_tests.lm_utils import print_red
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.lm_utils import SentenceNames as SN
from src.linguistic_tests.testset import Example
from src.linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
from src.linguistic_tests.testset import TestSet


MAX_EXAMPLES_PRINTS_PER_TESTSET = 99  # 50


def reorder_testsets_legacy(scored_testsets: List[TestSet]):
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


def reorder_testsets2(testsets: List[TestSet]) -> List[TestSet]:
    preferred_axs_order = {
        "wh_whether_island": 0,
        "rc_wh_island": 0,
        "complex": 1,
        "subject": 2,
        "adjunct": 3,
    }
    sorted_testsets = []
    testsets_by_wanted_position: Dict[int, TestSet] = dict()
    for testset in testsets:
        successfully_positioned = False
        for phenomenon_shortname in preferred_axs_order.keys():
            if phenomenon_shortname in testset.linguistic_phenomenon:
                wanted_position = preferred_axs_order[phenomenon_shortname]
                # check no testset already at current key
                if wanted_position in testsets_by_wanted_position:
                    logging.error(
                        f"Error while sorting testsets before plotting, "
                        f"at position {wanted_position} "
                        f"there is already testset for {testset.linguistic_phenomenon}"
                    )
                testsets_by_wanted_position[wanted_position] = testset
                successfully_positioned = True
                break
        if not successfully_positioned:
            logging.error(
                f"Could not position testset for {testset.linguistic_phenomenon}, while sorting before plotting."
            )
    for position in sorted(testsets_by_wanted_position.keys()):
        sorted_testsets.append(testsets_by_wanted_position[position])

    logging.debug(
        f"final testsets sorting: {[testset.linguistic_phenomenon for testset in sorted_testsets]}"
    )
    return sorted_testsets


def plot_single_testset_results(
    scored_testsets: List[TestSet],
    score_name,
    use_zscore=False,
    likert=False,
    show_plot=False,
    save_plot=False,
):
    if not show_plot and not save_plot:
        return

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 12.8))  # default figsize=(6.4, 4.8)

    window_title = _get_test_session_descr(
        scored_testsets[0].dataset_source,
        scored_testsets[0].linguistic_phenomenon,
        scored_testsets[0].model_descr,
        score_name,
    )

    fig.canvas.manager.set_window_title(window_title)
    axs_list = axs.reshape(-1)
    logging_debug(f"type axs_list: {type(axs_list)}, {len(axs_list)}, {axs_list}")

    scored_testsets = reorder_testsets2(scored_testsets)

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

    plt.figlegend(lines, labels)
    # fig.tight_layout()

    if save_plot:
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
        print_orange(f"Saving plot to file {filepath} ..")
        plt.savefig(filepath)  # , dpi=300
    if show_plot:
        plt.show()


def do_extended_testset_plot(
    scoring_measure: ScoringMeasures,
    testset: TestSet,
    show_plot=False,
):
    """
    Plots the lines for each item of a testset
    :param scoring_measure:
    :param testset:
    :return:
    """

    # alternative:
    # plot with 7+1x7 subplots of a testset (one subplot for each example)
    # nb: having the standard errors in the plots is already overcoming this,
    # showing the variance

    if not show_plot:
        return

    fig, ax = plt.subplots(1)
    for example in testset.examples:
        # logging_info(f"example={example.__str__(scoring_measure=scoring_measure)}")
        y_values_ni = [
            example.get_score(scoring_measure, SN.SHORT_NONISLAND),
            example.get_score(scoring_measure, SN.LONG_NONISLAND),
        ]
        y_values_is = [
            example.get_score(scoring_measure, SN.SHORT_ISLAND),
            example.get_score(scoring_measure, SN.LONG_ISLAND),
        ]
        x_values = ["SHORT", "LONG"]
        # logging_info(f"ploting: y_values_ni={y_values_ni}, y_values_is={y_values_is}")
        (non_island_line,) = ax.plot(x_values, y_values_ni, color="blue")
        (island_line,) = ax.plot(x_values, y_values_is, color="orange", linestyle="--")
        lines = [non_island_line, island_line]
        labels = ["Non-island structure", "Island structure"]

    ax.set_ylabel(f"{scoring_measure} values")  # if not use_zscore
    ax.set_xlabel("Dependency distance")  # if set_xlabel:

    # ax.set_aspect('equal', 'box')  # , 'box'
    ax.set_title(testset.linguistic_phenomenon)

    fig.suptitle(
        f"Model: {testset.model_descr}, "
        f"\n Dataset: {testset.dataset_source} "
        f"({testset.get_item_count_per_phenomenon()} items per phenomenon)"
    )

    plt.figlegend(lines, labels)

    plt.show()

    # problem: the likert scores and the zscores are not stored, only their
    # averages


def _plot_results_subplot(
    scored_testset: TestSet,
    scoring_measure: ScoringMeasures,
    ax,
    use_zscore=False,
    likert=False,  # only valid if also using z scores, on which the likerts are calculated
    set_xlabel=True,
):
    if use_zscore:
        DD_value = scored_testset.get_avg_DD_zscores(scoring_measure, likert=likert)

        y_values_ni = [
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SN.SHORT_NONISLAND, likert=likert
            ),
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SN.LONG_NONISLAND, likert=likert
            ),
        ]
        y_values_is = [
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SN.SHORT_ISLAND, likert=likert
            ),
            scored_testset.get_avg_zscores_by_measure_and_by_stype(
                scoring_measure, SN.LONG_ISLAND, likert=likert
            ),
        ]
    else:
        DD_value = scored_testset.get_avg_DD(scoring_measure)
        score_averages = scored_testset.get_avg_scores(scoring_measure)
        y_values_ni = [
            score_averages[SN.SHORT_NONISLAND],
            score_averages[SN.LONG_NONISLAND],
        ]
        y_values_is = [
            score_averages[SN.SHORT_ISLAND],
            score_averages[SN.LONG_ISLAND],
        ]

    logging_debug(f"{y_values_ni}")
    logging_debug(f"{y_values_is}")

    # todo: add p values
    # todo: add accuracy %

    x_values = ["SHORT", "LONG"]

    std_errors_ni = (
        scored_testset.get_std_errors_of_zscores_by_measure_and_sentence_structure(
            scoring_measure, SN.SHORT_NONISLAND
        )
    )
    std_errors_is = (
        scored_testset.get_std_errors_of_zscores_by_measure_and_sentence_structure(
            scoring_measure, SN.SHORT_ISLAND
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
    logging_debug(f"sentence ordering is {type(sentence_ordering)}")
    print(
        f"\nSHORT_NONISLAND: {example_data[sentence_ordering.SHORT_NONISLAND]}"
        f"\nLONG_NONISLAND : {example_data[sentence_ordering.LONG_NONISLAND]}"
        f"\nSHORT_ISLAND : {example_data[sentence_ordering.SHORT_ISLAND]}"
        f"\nLONG_ISLAND : {example_data[sentence_ordering.LONG_ISLAND]}"
    )


def plot_testsets(
    scored_testsets: List[TestSet],
    model_type: ModelTypes,
    show_plot=False,
    save_plot=False,
):
    scoring_measures_to_plot = [ScoringMeasures.PenLP.name, ScoringMeasures.LP.name]
    if model_type in BERT_LIKE_MODEL_TYPES:
        scoring_measures_to_plot += [ScoringMeasures.LL.name, ScoringMeasures.PLL.name]

    for scoring_measure in scoring_measures_to_plot:
        plot_single_testset_results(
            scored_testsets,
            scoring_measure,
            use_zscore=True,
            likert=True,
            show_plot=show_plot,
            save_plot=save_plot,
        )


def log_and_print(logging_level: int, msg, also_print=True):

    logging.log(logging_level, msg)

    if also_print:
        # because logging actually is delayed/buffered, and gets out of synch with
        # regular prints. Both logging and printing both until a fix is found.
        print_orange(msg)

    sys.stdout.flush()


def logging_info(msg, also_print=True):
    log_and_print(logging.INFO, msg, also_print=also_print)


def logging_debug(msg, also_print=True):
    log_and_print(logging.DEBUG, msg, also_print=also_print)


def _print_sorted_sentences_to_check_spelling_errors2(
    score_descr: ScoringMeasures,
    phenomena,
    model_name,
    dataset_source,
    loaded_testsets,
):

    for testset in loaded_testsets:
        logging_info(
            f"printing for testset {testset.linguistic_phenomenon} "
            f"calculated from {testset.model_descr} "
            f"(dataset_source={dataset_source})"
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
    score_descr: ScoringMeasures,
    phenomena,
    model_name,
    dataset_source,
    loaded_testsets,
):
    logging_info("printing sorted_sentences_to_check_spelling_errors")

    for testset in loaded_testsets:
        logging_info(
            f"printing {score_descr} for testset {testset.linguistic_phenomenon} calculated from {testset.model_descr}"
        )
        for stype in SPROUSE_SENTENCE_TYPES:
            logging_info(f"printing for sentence type {stype}..")
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


def excel_output(
    scored_testsets_by_datasource: Dict[DataSources, List[TestSet]],
    # model_descr: str,
):
    # todo: extend input dict also by dependency type (wh, rc)

    # todo: dd score based on likerts and zscores (save in the testset the likert and zscores for all the sentences)
    #  it's also more comparable among phenomena

    # todo
    #  excel file formatting: columns width, wrap text, ..
    #  file name from model descr, merge in one sheet multiple datasources of the same 4 phenomena
    #  ..

    import pandas as pd

    # import xlsxwriter

    # pip install openpyxl, add to requirements

    # todo: add accuracy scores sheet

    # todo: filename: models, and datasources?
    # excel files with prefix like pickle filenames (but one for multiple testsets/phenomena)

    # df_items_comparisons
    # file content: comparing examples (one row per example)
    # filename/sheet name: datasource, model (add these to the columns?)

    # todo: possibly save excel data for multiple scoring measures
    factorial_scoring_measure = ScoringMeasures.PenLP

    data_for_excel_examples_comparison = _excel_output_helper_examples_comparison(
        factorial_scoring_measure,
        scored_testsets_by_datasource,
    )
    data_for_excel_accuracy_scores = _excel_output_helper_accuracy(
        scored_testsets_by_datasource,
    )

    data_for_excel = [
        data_for_excel_examples_comparison,
        data_for_excel_accuracy_scores,
    ]

    sample_testset = list(scored_testsets_by_datasource.values())[0][0]
    _excel_output_helper_write_file(
        data_for_excel,
        sample_testset.model_descr,
        factorial_scoring_measure,
        pd,
    )


def _excel_output_helper_examples_comparison(
    scoring_measure: ScoringMeasures,
    scored_testsets_by_datasource,
):

    # data_for_dataframe: key=column_name, value: list_of_values
    data_for_dataframe: Dict[str, List[Union[str, float, bool]]] = dict()
    DATASOURCE_COL = "datasource"
    LINGUISTIC_PHENOMENON_COL = "linguistic_phenomenon"
    DD_SCORE_COLUMN_NAME = f"DD_{scoring_measure}"
    LENGHT_EFFECT_COL = "Lenght effect"
    STRUCTURE_EFFECT_COL = "Structure effect"
    TOTAL_EFFECT_COL = "Total effect"
    SCORING_MEASURE_COL = "Scoring measure"

    # initialize the columns
    number_column_names = [
        DD_SCORE_COLUMN_NAME,
        LENGHT_EFFECT_COL,
        STRUCTURE_EFFECT_COL,
        TOTAL_EFFECT_COL,
    ]

    column_names = [
        DATASOURCE_COL,
        LINGUISTIC_PHENOMENON_COL,
        SCORING_MEASURE_COL,
        DD_SCORE_COLUMN_NAME,
        LENGHT_EFFECT_COL,
        STRUCTURE_EFFECT_COL,
        TOTAL_EFFECT_COL,
    ]

    # todo: shorten col names
    sample_testset = list(scored_testsets_by_datasource.values())[0][0]
    for stype in sample_testset.get_sentence_types():

        STYPE_SCORE_COL = f"{stype.name}_score"
        column_names.append(STYPE_SCORE_COL)
        number_column_names.append(STYPE_SCORE_COL)

        if stype in sample_testset.get_acceptable_sentence_types():
            column_names.append(f"is_{stype.name}_scored_accurately")

        column_names.append(stype.name)

    for colum_name in column_names:
        data_for_dataframe[colum_name] = []

    # fill the data
    for datasource in scored_testsets_by_datasource:
        for scored_testset in scored_testsets_by_datasource[datasource]:
            for example in scored_testset.examples:
                _excel_output_helper_fill_example_data(
                    example,
                    datasource,
                    scored_testset,
                    scoring_measure,
                    data_for_dataframe,
                    DATASOURCE_COL,
                    LINGUISTIC_PHENOMENON_COL,
                    SCORING_MEASURE_COL,
                    DD_SCORE_COLUMN_NAME,
                    LENGHT_EFFECT_COL,
                    STRUCTURE_EFFECT_COL,
                    TOTAL_EFFECT_COL,
                )

    examples_comparison_data_for_excel = DataForExcel(
        column_names,
        number_column_names,
        data_for_dataframe,
        sheet_name=f"examples_comparison_{scoring_measure}",
    )

    return examples_comparison_data_for_excel


def _excel_output_helper_accuracy(
    scored_testsets_by_datasource: Dict[str, List[TestSet]]
):

    # each dataframe in a separate excel sheet

    # df_accuracy_scores (one row per phenomena)
    # file name (or additional columns): model name, datasource
    DATASOURCE_COL = "datasource"
    LINGUISTIC_PHENOMENON_COL = "linguistic_phenomenon"
    SCORING_MEASURE_COL = "Scoring measure"
    SENTENCE_TYPE_COL = "Sentence type"
    ACCURACY_COL = "Accuracy"
    SAMPLE_ACCEPTABLE_SENTENCE_COL = "Sample"
    SAMPLE_UNACCEPTABLE_SENTENCE_COL = "Unacceptable sample"
    column_names = [
        DATASOURCE_COL,
        LINGUISTIC_PHENOMENON_COL,
        SCORING_MEASURE_COL,
        SENTENCE_TYPE_COL,
        ACCURACY_COL,
        SAMPLE_ACCEPTABLE_SENTENCE_COL,
        SAMPLE_UNACCEPTABLE_SENTENCE_COL,
    ]

    number_column_names = [
        ACCURACY_COL,
    ]
    print(f"number_column_names={number_column_names}")
    # first_testsset = list(scored_testsets_by_datasource.values())[0][0]

    # data_for_dataframe: key=column_name, value: list_of_values
    data_for_dataframe: Dict[str, List[Union[str, float, bool]]] = dict()
    for colum_name in column_names:
        data_for_dataframe[colum_name] = []

    # sheet name: accuracy by stype (pairs comparison)

    for datasource in scored_testsets_by_datasource:
        for testset in scored_testsets_by_datasource[datasource]:
            _excel_output_helper_fill_accuracy_data(
                datasource,
                testset,
                data_for_dataframe,
                ACCURACY_COL,
                DATASOURCE_COL,
                LINGUISTIC_PHENOMENON_COL,
                SCORING_MEASURE_COL,
                SENTENCE_TYPE_COL,
                SAMPLE_ACCEPTABLE_SENTENCE_COL,
                SAMPLE_UNACCEPTABLE_SENTENCE_COL,
            )

    accuracy_data_for_excel = DataForExcel(
        column_names,
        number_column_names,
        data_for_dataframe,
        sheet_name="Accuracy scores",
    )

    return accuracy_data_for_excel

    # todo: save to excel file

    # sheet: examples factorial accuracy based on DD score (Testset accuracy with DDs_with_ ..+)
    # columns:
    # scoring_measure, phenomenon, dataset_source, accuracy perc
    # scored_testset.accuracy_by_DD_lp


@dataclass
class DataForExcel:
    column_names: List[str]
    number_column_names: List[str]
    data_for_dataframe: Dict[str, List[Union[str, float, bool]]]
    sheet_name: str


def _excel_output_helper_write_file(
    data_for_excel_sheets: List[DataForExcel],
    model_descr,
    factorial_scoring_measure: ScoringMeasures,
    pd,
):

    # todo: get results dir
    model_descr = model_descr.replace(" ", "_").replace("/", "_")

    filename = f"{model_descr}.xlsx"
    logging.info(f"Writing excel output to {filename}")

    with pd.ExcelWriter(filename) as writer:

        for data_for_excel_sheet in data_for_excel_sheets:

            data_for_dataframe = data_for_excel_sheet.data_for_dataframe
            column_names = data_for_excel_sheet.column_names
            number_column_names = data_for_excel_sheet.number_column_names
            sheet_name = data_for_excel_sheet.sheet_name

            df = pd.DataFrame.from_dict(data_for_dataframe)
            logging.debug(f"Dataframe lenght: {len(df)}")

            include_the_index_of_each_row = False
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                columns=column_names,
                index=include_the_index_of_each_row,
            )

            # column formatting
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            num_format = workbook.add_format({"num_format": "0.00"})
            for number_column_name in number_column_names:
                col_idx = df.columns.get_loc(number_column_name)
                # col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)

                worksheet.set_column(
                    first_col=col_idx,
                    last_col=col_idx,
                    width=None,
                    cell_format=num_format,
                )  # Adds formatting to column C

            # Given a dict of dataframes, for example:
            # dfs = {'gadgets': df_gadgets, 'widgets': df_widgets}
            #  for sheetname, df in dfs.items():  # loop through `dict` of dataframes

            # column width adjusting
            EXTRA_SPACE = 1
            numerical_columns_width = 6
            for idx, col in enumerate(df):  # loop through all columns
                if str(col) in number_column_names:
                    col_width = numerical_columns_width
                else:
                    series = df[col]
                    len_of_largest_item = series.astype(str).map(len).max()
                    col_width = (
                        max(
                            (
                                len_of_largest_item,
                                0,
                                # len(str(series.name))  # len of column name/header
                            )
                        )
                        + EXTRA_SPACE
                    )  # adding a little extra space
                worksheet.set_column(idx, idx, width=col_width)  # set column width

        writer.save()


def _excel_output_helper_fill_accuracy_data(
    datasource: DataSources,
    testset: TestSet,
    data_for_dataframe: Dict[str, List[Union[str, float, bool]]],
    ACCURACY_COL,
    DATASOURCE_COL,
    LINGUISTIC_PHENOMENON_COL,
    SCORING_MEASURE_COL,
    SENTENCE_TYPE_COL,
    SAMPLE_ACCEPTABLE_SENTENCE_COL,
    SAMPLE_UNACCEPTABLE_SENTENCE_COL,
):
    # fill the data

    datasource_descr = datasource[:8]

    for scoring_measure in testset.get_scoring_measures():
        for stype_acceptable in testset.get_acceptable_sentence_types():
            # fixme, check: 0 values for accuracy based on logistic scoring measure
            accuracy = testset.accuracy_per_score_type_per_sentence_type[
                scoring_measure
            ][stype_acceptable]
            data_for_dataframe[ACCURACY_COL].append(f"{accuracy:.2%}")

            data_for_dataframe[DATASOURCE_COL].append(datasource_descr)
            data_for_dataframe[LINGUISTIC_PHENOMENON_COL].append(
                testset.linguistic_phenomenon
            )
            data_for_dataframe[SCORING_MEASURE_COL].append(scoring_measure)
            data_for_dataframe[SENTENCE_TYPE_COL].append(stype_acceptable)

            sample_example: Example = testset.examples[0]
            stype_unacceptable = sample_example.get_type_of_unacceptable_sentence()
            data_for_dataframe[SAMPLE_ACCEPTABLE_SENTENCE_COL].append(
                sample_example[stype_acceptable].txt
            )
            data_for_dataframe[SAMPLE_UNACCEPTABLE_SENTENCE_COL].append(
                sample_example[stype_unacceptable].txt
            )


def _excel_output_helper_fill_example_data(
    example: Example,
    datasource: DataSources,
    scored_testset: TestSet,
    scoring_measure: ScoringMeasures,
    data_for_dataframe: Dict[str, List[Union[str, float, bool]]],
    DATASOURCE_COL: str,
    LINGUISTIC_PHENOMENON_COL: str,
    SCORING_MEASURE_COL: str,
    DD_SCORE_COLUMN_NAME: str,
    LENGHT_EFFECT_COL: str,
    STRUCTURE_EFFECT_COL: str,
    TOTAL_EFFECT_COL: str,
):
    """
    Fills the data corresponding to one row in the final excel file.
    :param example:
    :param datasource:
    :param scored_testset:
    :param scoring_measure:
    :param data_for_dataframe:
    :param DATASOURCE_COL:
    :param LINGUISTIC_PHENOMENON_COL:
    :param SCORING_MEASURE_COL:
    :param DD_SCORE_COLUMN_NAME:
    :param LENGHT_EFFECT_COL:
    :param STRUCTURE_EFFECT_COL:
    :param TOTAL_EFFECT_COL:
    :return:
    """
    # columns: phenomena,
    # dd score, lenght/structure/total effects,
    # scoring measure,
    # 4 cols for txt of short/long island/nonisland,
    # score for the 4 sentences
    datasource_descr = datasource[:8]
    data_for_dataframe[DATASOURCE_COL].append(datasource_descr)
    data_for_dataframe[LINGUISTIC_PHENOMENON_COL].append(
        scored_testset.linguistic_phenomenon
    )
    data_for_dataframe[SCORING_MEASURE_COL].append(scoring_measure)

    # todo: get dd score based on likert and zscores ..
    data_for_dataframe[DD_SCORE_COLUMN_NAME].append(
        example.get_dd_score(scoring_measure)
    )
    data_for_dataframe[LENGHT_EFFECT_COL].append(
        example.get_lenght_effect(scoring_measure)
    )
    data_for_dataframe[STRUCTURE_EFFECT_COL].append(
        example.get_structure_effect(scoring_measure)
    )
    data_for_dataframe[TOTAL_EFFECT_COL].append(
        example.get_total_effect(scoring_measure)
    )

    for stype in example.get_sentence_types():
        data_for_dataframe[f"{stype.name}_score"].append(
            example[stype].get_score(scoring_measure)
        )
        # todo: for acceptable stypes, add boolean column telling if it's scored accurately
        if stype != example.get_type_of_unacceptable_sentence():
            is_scored_accurately = example.is_scored_accurately_for(
                scoring_measure, stype
            )
            data_for_dataframe[f"is_{stype.name}_scored_accurately"].append(
                is_scored_accurately
            )
        data_for_dataframe[stype.name].append(example[stype].txt)


def _print_compare__examples_by_DD_score_helper(
    examples: List[Example],
    scoring_measure: ScoringMeasures,
    shorter_form=False,
):
    if shorter_form:
        max_chars = 20
    else:
        max_chars = 70

    print(
        f"{'DD':<6}"
        f"{'len_eff.':<8}"
        f"{'struct.':<8}"
        f"{'total':<8}"
        f"{'s1 ' + str(scoring_measure):<10}"
        f"{SN.SHORT_NONISLAND.name[:max_chars]:<{max_chars}}"
        f"{'s2 ' + str(scoring_measure):<10}"
        f"{SN.LONG_NONISLAND.name[:max_chars]:<{max_chars}}"
        f"{'s3 ' + str(scoring_measure):<10}"
        f"{SN.SHORT_ISLAND.name[:max_chars]:<{max_chars}}"
        f"{'s4 ' + str(scoring_measure):<10}"
        f"{SN.LONG_ISLAND.name[:max_chars]:<5}"
    )
    for example in examples[0:MAX_EXAMPLES_PRINTS_PER_TESTSET]:
        print(
            f"{example.get_dd_score(scoring_measure) : <6.2f}"
            f"{example.get_lenght_effect(scoring_measure) : <8.2f}"
            f"{example.get_structure_effect(scoring_measure) : <8.2f}"
            f"{example.get_total_effect(scoring_measure) : <8.2f}"
            f"{_prt_score(SN.SHORT_NONISLAND, example, scoring_measure, max_chars)}"
            f"{_prt_score(SN.LONG_NONISLAND, example, scoring_measure, max_chars)}"
            f"{_prt_score(SN.SHORT_ISLAND, example, scoring_measure, max_chars)}"
            f"{_prt_score(SN.LONG_ISLAND, example, scoring_measure, max_chars=5)}"
        )


def _prt_score(
    stype: SN, example: Example, scoring_measure: ScoringMeasures, max_chars
):
    if stype == example.get_type_of_unacceptable_sentence():
        accuracy_mark = "_"
    elif example.is_scored_accurately_for(scoring_measure, stype):
        accuracy_mark = "✓"
    else:
        accuracy_mark = "x"

    return (
        f" {accuracy_mark : <2}"
        f"{example[stype].get_score(scoring_measure) : <8.2f}"
        f"{example[stype].txt[:max_chars] : <{max_chars}}"
    )


def _print_compare__examples_by_DD_score(
    scoring_measure: ScoringMeasures,
    testset: TestSet,
):
    print(
        f"comparing examples by DD score based on {scoring_measure}  "
        f"({testset.linguistic_phenomenon} from {testset.model_descr}, "
        f"dataset_source={testset.dataset_source})"
    )
    examples_sorted_by_dd_score = testset.get_examples_sorted_by_DD_score(
        scoring_measure
    )
    _print_compare__examples_by_DD_score_helper(
        examples_sorted_by_dd_score, scoring_measure
    )


def _print_examples_compare_diff(
    scoring_measure: ScoringMeasures,
    sent_type1: SN,
    sent_type2: SN,
    phenomena: List[str],
    model_name: str,
    dataset_source: str,
    testsets: List[TestSet],
):
    if not all(
        item in testsets[0].get_sentence_types() for item in [sent_type1, sent_type2]
    ):
        logging.info(
            f"Skipping, {sent_type1} or {sent_type2} are not in {testsets[0].get_sentence_types()}"
        )
        return

    max_testsets = 4

    for testset in testsets[:max_testsets]:
        logging_info(
            f"printing testset for {testset.linguistic_phenomenon} "
            f"from {testset.model_descr} (dataset_source={dataset_source})"
        )

        print(
            f"comparing {sent_type1} and {sent_type2} "
            f"({testset.linguistic_phenomenon} from {testset.model_descr}) "
            f"dataset_source={testset.dataset_source}"
        )
        examples_sorted_by_score_diff = testset.get_examples_sorted_by_score_diff(
            scoring_measure, sent_type1, sent_type2, reverse=False
        )

        print(
            f"{'diff':<8}"
            f"{'s1 ' + scoring_measure:^10}"
            f"{'s1 txt (' + sent_type1 + ')':<70}"
            f"{'s2 ' + scoring_measure:^10}"
            f"{'s2 txt (' + sent_type2 + ')':<5}"
        )
        for example in examples_sorted_by_score_diff[0:MAX_EXAMPLES_PRINTS_PER_TESTSET]:
            print(
                f"{example.get_score_diff(scoring_measure, sent_type1, sent_type2) : <8.2f}"
                f"{example[sent_type1].get_score(scoring_measure) : ^10.2f}"
                f"{example[sent_type1].txt : <70}"
                f"{example[sent_type2].get_score(scoring_measure) :^10.2f}"
                f"{example[sent_type2].txt : <5}"
            )


def _print_testset_results(
    scored_testsets: List[TestSet],
    dataset_source: str,
    model_type: ModelTypes,
    testsets_root_filenames: List[str],
):
    scoring_measure = ScoringMeasures.PenLP

    logging_info("Printing accuracy scores..")
    for scored_testset in scored_testsets:
        # todo: also print results in table format or csv for excel export or word doc report
        print_accuracy_scores(scored_testset)
        print(
            f"Testset accuracy with DDs_with_lp: {scored_testset.accuracy_by_DD_lp:.2%} ({scored_testset.linguistic_phenomenon})"
        )
        print(
            f"Testset accuracy with DDs_with_penlp: {scored_testset.accuracy_by_DD_penlp:.2%} ({scored_testset.linguistic_phenomenon})"
        )
        lp_averages = scored_testset.lp_average_by_sentence_type
        penlp_averages = scored_testset.penlp_average_by_sentence_type
        logging.info(f"lp_averages: {lp_averages}")
        logging.info(f"penlp_averages: {penlp_averages}")

        if model_type in BERT_LIKE_MODEL_TYPES:
            print(
                f"Testset accuracy with DDs_with_ll: {scored_testset.accuracy_by_DD_ll:.2%} ({scored_testset.linguistic_phenomenon})"
            )
            print(
                f"Testset accuracy with DDs_with_penll: {scored_testset.accuracy_by_DD_penll:.2%} ({scored_testset.linguistic_phenomenon})"
            )
            ll_averages = scored_testset.ll_average_by_sentence_type
            penll_averages = scored_testset.penll_average_by_sentence_type
            logging.info(f"{ll_averages}")
            logging.info(f"{penll_averages}")

        _print_compare__examples_by_DD_score(scoring_measure, scored_testset)

    if scored_testsets[0].experimental_design in [
        ExperimentalDesigns.FACTORIAL,
        ExperimentalDesigns.MINIMAL_PAIRS,
    ]:
        _print_sorted_sentences_to_check_spelling_errors2(
            scoring_measure,
            testsets_root_filenames,
            MODEL_NAMES_IT[model_type],
            dataset_source,
            scored_testsets,
        )
        _print_sorted_sentences_to_check_spelling_errors(
            scoring_measure,
            testsets_root_filenames,
            MODEL_NAMES_IT[model_type],
            dataset_source,
            scored_testsets,
        )

    _print_examples_compare_diff(
        scoring_measure,
        SN.SHORT_NONISLAND,
        SN.LONG_NONISLAND,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    _print_examples_compare_diff(
        scoring_measure,
        SN.SHORT_ISLAND,
        SN.LONG_ISLAND,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    _print_examples_compare_diff(
        scoring_measure,
        SN.LONG_NONISLAND,
        SN.LONG_ISLAND,
        testsets_root_filenames,
        MODEL_NAMES_IT[model_type],
        dataset_source,
        testsets=scored_testsets,
    )
    _print_examples_compare_diff(
        scoring_measure,
        SN.SHORT_NONISLAND,
        SN.SHORT_ISLAND,
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
    logging_info(f"test results report, {testset.linguistic_phenomenon}:")
    for scoring_measure in testset.get_scoring_measures():
        logging_debug(f"scores with {scoring_measure}")
        for stype_acceptable_sentence in testset.get_acceptable_sentence_types():
            # fixme: 0 values for accuracy base on logistic scoring measure
            accuracy = testset.accuracy_per_score_type_per_sentence_type[
                scoring_measure
            ][stype_acceptable_sentence]

            print(
                f"{testset.linguistic_phenomenon}: "
                f"Accuracy with {scoring_measure} "
                f"for {stype_acceptable_sentence.name}: {accuracy:.2%} "
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
    logging_info("test results report:")
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


def get_perc(value, total):
    return f"{perc(value, total):.1f} %"
