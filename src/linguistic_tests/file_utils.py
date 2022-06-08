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


def convert_files():
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


def main():
    print("converting files..")
    convert_files()


if __name__ == "__main__":
    main()
