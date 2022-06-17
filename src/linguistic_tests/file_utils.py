import csv
import dataclasses
import json
import os.path
from typing import List

import pandas
from linguistic_tests.lm_utils import BlimpSentencesOrder
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.testset import parse_testset
from linguistic_tests.testset import TestSet
from tqdm import tqdm


def parse_testsets(
    testset_dir_path,
    testset_filenames,
    examples_format,
    sent_types_descr,
    model_name,
    max_examples,
) -> List[TestSet]:
    parsed_testsets = []
    for testset_filename in testset_filenames:
        testset_filepath = os.path.join(testset_dir_path, testset_filename + ".jsonl")
        print(f"Parsing testset {testset_filepath}")
        testset_dict = load_testset_data(
            testset_filepath, examples_format=examples_format
        )  # es.: "json_lines"
        examples_list = testset_dict["sentences"]
        phenomenon_name = get_file_root(testset_filename)
        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        parsed_testset = parse_testset(
            phenomenon_name,
            model_name,
            examples_list,
            sent_types_descr,  # "blimp" or "sprouse"
            scoring_measures,
            max_examples=max_examples,
        )

        parsed_testsets.append(parsed_testset)

    return parsed_testsets


def convert_testset_to_csv(
    dir_path,
    input_filename,
    examples_format="blimp",
    sentence_ordering=BlimpSentencesOrder,
):

    # load json file using lm utils, specify format/order of sentences
    # specify colum names
    # if save file already exists don't overwrite
    # save to csv

    out_filename = input_filename + ".csv"
    out_filepath = os.path.join(dir_path, out_filename)
    if os.path.exists(out_filepath):
        print(f"output file already exists: {out_filepath}, skipping")
        return
    input_filepath = os.path.join(dir_path, input_filename)

    header_v1_horizontal = [
        "Island structure and short distance dependency",
        "Island structure and long distance dependency (ungrammatical)",
        "Nonisland structure and long distance dependency",
        "Nonisland structure and short distance dependency",
    ]
    _ = [  # header_v2_vertical
        "Island Type"
        "Clause type"
        "dependency distance"
        "island structure"
        "item number"
        "sentence"
    ]
    print(f"converting file {input_filepath}..")
    testset_data = load_testset_data(input_filepath, examples_format=examples_format)
    if examples_format == "blimp":
        testset_data = testset_data["sentences"]

    with open(
        os.path.join(dir_path, out_filepath), "w", newline="", encoding="UTF8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(header_v1_horizontal)

        for example_data in tqdm(testset_data):
            sentences = get_sentences_from_example(
                example_data, sentences_per_example=4
            )

            row_data = [
                sentences[sentence_ordering.SHORT_ISLAND],
                sentences[sentence_ordering.LONG_ISLAND],
                sentences[sentence_ordering.LONG_NONISLAND],
                sentences[sentence_ordering.SHORT_NONISLAND],
            ]
            writer.writerow(row_data)

            # or
            # writer = csv.DictWriter(f, fieldnames=fieldnames)
            # writer.writeheader()
            # writer.writerows(rows)


def convert_files_to_csv():
    dir_path = str(get_syntactic_tests_dir() / "syntactic_tests_it/")
    input_filenames = [
        # "wh_adjunct_islands.jsonl",
        "wh_complex_np_islands.jsonl",
        "wh_whether_island.jsonl",
        "wh_subject_islands.jsonl",
    ]

    for filename in input_filenames:
        convert_testset_to_csv(
            dir_path,
            filename,
            examples_format="blimp",
            sentence_ordering=BlimpSentencesOrder,
        )


def change_files_sentence_order():
    dir_path = str(get_syntactic_tests_dir() / "syntactic_tests_it/")
    input_filenames = [
        "wh_adjunct_islands.jsonl",
        "wh_complex_np_islands.jsonl",
        "wh_whether_island.jsonl",
        "wh_subject_islands.jsonl",
    ]

    for input_filename in input_filenames:
        change_file_sentence_order(
            dir_path,
            input_filename,
            input_sentence_ordering=BlimpSentencesOrder,
        )


def change_file_sentence_order(
    dir_path,
    input_filename,
    input_sentence_ordering=BlimpSentencesOrder,
):
    out_filename = "custom-" + input_filename  # + "-ref.jsonl"
    outdir = str(get_syntactic_tests_dir() / "sprouse/")
    out_filepath = os.path.join(outdir, out_filename)
    if os.path.exists(out_filepath):
        print(f"output file already exists: {out_filepath}, skipping..")
        return
    print(f"preparing file {out_filepath}..")
    input_filepath = os.path.join(dir_path, input_filename)
    print(f"reading from file {input_filepath}..")
    examples = load_testset_data(
        input_filepath,
        examples_format="blimp",  # , input_sentence_ordering=BlimpSentencesOrder
    )
    if input_sentence_ordering == BlimpSentencesOrder:
        testset_data = examples["sentences"]

    with open(out_filepath, mode="a", encoding="utf-8") as f:
        for example_data in testset_data:
            reformatted_dict = __get_reformatted_example(
                example_data
            )  # , input_sentence_ordering
            json_string = (
                json.dumps(reformatted_dict, ensure_ascii=False) + "\n"  # , indent=4
            )  # .encode('utf8')
            f.write(json_string)


def __get_reformatted_example(example_data):  # , input_sentence_ordering
    return {
        SentenceNames.SHORT_NONISLAND: example_data[SentenceNames.SHORT_NONISLAND],
        SentenceNames.LONG_NONISLAND: example_data[SentenceNames.LONG_NONISLAND],
        SentenceNames.SHORT_ISLAND: example_data[SentenceNames.SHORT_ISLAND],
        SentenceNames.LONG_ISLAND: example_data[SentenceNames.LONG_ISLAND],
    }


def convert_sprouse_csv_to_jsonl():
    # open csv file
    # parse ..
    testset_filepath = str(
        get_syntactic_tests_dir() / "sprouse/Experiment 2 materials - Italian.csv"
    )
    # examples = []  # sentence pairs

    df = pandas.read_csv(testset_filepath, sep=";", header=0, encoding="utf-8")
    # print(df.head(2))
    # print(df.info())

    # rslt_df = df.loc[#(df['Clause type'] == 'RC') &
    #                    (df['Island Type'] == 'Adjunct island') & (df['item number'] == 1)]

    CLAUSE_TYPE = "Clause type"
    clause_types = set(df[CLAUSE_TYPE].tolist())
    # print(f'clause_types: {clause_types}')

    for clause_type in clause_types:

        current_clause_sentences = df.loc[(df[CLAUSE_TYPE] == clause_type)]
        ISLAND_TYPE = "Island Type"
        island_types = set(current_clause_sentences[ISLAND_TYPE].tolist())
        # print(f'island_types: {island_types}')

        for island_type in island_types:
            # print(f'current clause_type: {clause_type}, current island_type: {island_type}')
            current_phenomenon_sentences = current_clause_sentences.loc[
                (current_clause_sentences[ISLAND_TYPE] == island_type)
            ]
            phenomenon_name = (
                clause_type.lower() + "_" + island_type.replace(" ", "_").lower()
            )

            filename = phenomenon_name + ".jsonl"
            filepath = os.path.abspath(
                os.path.join(str(get_syntactic_tests_dir() / "sprouse/"), filename)
            )
            if os.path.exists(filepath):
                print(f"file already exists, skipping: {filepath}")
                continue
            else:
                print(f"writing phenomenon_name: {phenomenon_name}")

            ITEM_NUMBER = "item number"
            item_numbers = set(current_phenomenon_sentences[ITEM_NUMBER].tolist())
            # print(f'item_numbers: {item_numbers}')
            for item_number in item_numbers:
                current_item_sentences = current_phenomenon_sentences.loc[
                    (current_phenomenon_sentences[ITEM_NUMBER] == item_number)
                ]
                # 4 sentences for 3 pairs
                sentence_bad = __get_sentence_from_row(
                    "Long", "Island", current_item_sentences
                )
                # print(f'bad_sentence: {sentence_bad}, type(bad_sentence): {type(sentence_bad)}')
                good_sentence_long_nonisland = __get_sentence_from_row(
                    "Long", "non-island", current_item_sentences
                )
                good_sentence_short_nonisland = __get_sentence_from_row(
                    "Short", "non-island", current_item_sentences
                )
                good_sentence_short_island = __get_sentence_from_row(
                    "Short", "Island", current_item_sentences
                )

                with open(filepath, mode="a", encoding="utf-8") as f:
                    __write_sentence_item(
                        f,
                        sentence_bad,
                        good_sentence_long_nonisland,
                        good_sentence_short_nonisland,
                        good_sentence_short_island,
                    )


def __get_sentence_from_row(C1, C2, current_item_sentences):
    C1_col = "Condition 1"
    C2_col = "Condition 2"
    SENTENCE = "sentence"
    # print(current_item_sentences.info())

    # C1_values = set(current_item_sentences[C1_col].tolist())
    # print(f'C1_values: {C1_values}')
    # C2_values = set(current_item_sentences[C2_col].tolist())
    # print(f'C1_values: {C2_values}')
    # print(f'params C1: {C1}, C2: {C2}')
    single_row_df = current_item_sentences.loc[
        (current_item_sentences[C1_col] == C1) & (current_item_sentences[C2_col] == C2)
    ]
    # print(single_row_df.info())
    sentence = single_row_df.iloc[0][SENTENCE]
    return sentence


def __write_sentence_item(
    f,
    sentence_bad,
    good_sentence_long_nonisland,
    good_sentence_short_nonisland,
    good_sentence_short_island,
):
    sentence_item = {
        SentenceNames.SHORT_NONISLAND: good_sentence_short_nonisland,
        SentenceNames.LONG_NONISLAND: good_sentence_long_nonisland,
        SentenceNames.SHORT_ISLAND: good_sentence_short_island,
        SentenceNames.LONG_ISLAND: sentence_bad,
    }
    json_string = (
        json.dumps(sentence_item, ensure_ascii=False) + "\n"
    )  # .encode('utf8')
    f.write(json_string)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def save_dataclass_to_json(self):
    json.dumps(self, cls=EnhancedJSONEncoder)


def get_file_root(path_str):
    filename = os.path.basename(path_str)
    filenameroot_no_extension = os.path.splitext(filename)[0]
    return filenameroot_no_extension


def main():
    print("converting files..")
    change_files_sentence_order()
    # convert_sprouse_csv_to_jsonl()


if __name__ == "__main__":
    main()
