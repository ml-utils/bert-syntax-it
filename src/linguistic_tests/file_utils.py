import csv
import json
import os.path

import pandas
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.run_sprouse_tests import BlimpSentencesOrder
from tqdm import tqdm


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
        raise ValueError(f"output file already exists: {out_filepath}")
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
        # "wh_complex_np_islands.jsonl",
        # "wh_whether_island.jsonl",
        # "wh_subject_islands.jsonl",
    ]

    for input_filename in input_filenames:
        change_file_sentence_order(
            dir_path,
            input_filename,
            sentence_ordering=BlimpSentencesOrder,
        )


def change_file_sentence_order(
    dir_path,
    input_filename,
    in_sentence_ordering=BlimpSentencesOrder,
):

    out_filename = input_filename + "-ref.jsonl"
    out_filepath = os.path.join(dir_path, out_filename)
    if os.path.exists(out_filepath):
        raise ValueError(f"output file already exists: {out_filepath}")

    input_filepath = os.path.join(dir_path, input_filename)
    examples = load_testset_data(
        input_filepath, in_sentence_ordering=BlimpSentencesOrder
    )
    if in_sentence_ordering == BlimpSentencesOrder:
        testset_data = examples["sentences"]

    # todo: save to new json file
    print(len(testset_data))


def create_test_jsonl_files_tests():
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
                sentence_bad = get_sentence_from_row(
                    "Long", "Island", current_item_sentences
                )
                # print(f'bad_sentence: {sentence_bad}, type(bad_sentence): {type(sentence_bad)}')
                good_sentence_long_nonisland = get_sentence_from_row(
                    "Long", "non-island", current_item_sentences
                )
                good_sentence_short_nonisland = get_sentence_from_row(
                    "Short", "non-island", current_item_sentences
                )
                good_sentence_short_island = get_sentence_from_row(
                    "Short", "Island", current_item_sentences
                )

                with open(filepath, mode="a", encoding="utf-8") as f:
                    write_sentence_item(
                        f,
                        sentence_bad,
                        good_sentence_long_nonisland,
                        good_sentence_short_nonisland,
                        good_sentence_short_island,
                    )


def get_sentence_from_row(C1, C2, current_item_sentences):
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


def write_sentence_pair(f, sentence_bad, good_sentence, conditions):
    sentence_pair = {
        "sentence_good": good_sentence,
        "sentence_bad": sentence_bad,
        "conditions": conditions,
    }
    f.write(json.dumps(sentence_pair) + "\n")


def read_sentences_item(example):
    parsed = dict()

    return parsed


def write_sentence_item(
    f,
    sentence_bad,
    good_sentence_long_nonisland,
    good_sentence_short_nonisland,
    good_sentence_short_island,
):
    sentence_item = {
        "short_nonisland": good_sentence_short_nonisland,
        "long_nonisland": good_sentence_long_nonisland,
        "short_island": good_sentence_short_island,
        "long_island": sentence_bad,
    }
    json_string = (
        json.dumps(sentence_item, ensure_ascii=False) + "\n"
    )  # .encode('utf8')
    f.write(json_string)


def main():
    print("converting files..")
    change_files_sentence_order()


if __name__ == "__main__":
    main()
