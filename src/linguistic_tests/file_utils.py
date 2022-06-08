import csv
import os.path

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


def main():
    print("converting files..")
    change_files_sentence_order()


if __name__ == "__main__":
    main()
