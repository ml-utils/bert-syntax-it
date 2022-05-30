import json
import os.path

import pandas
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.linguistic_tests.compute_model_score import DEVICES
from src.linguistic_tests.compute_model_score import get_example_scores
from src.linguistic_tests.compute_model_score import load_model
from src.linguistic_tests.lm_utils import model_types


# todo: parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file


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
        "sentence_bad": sentence_bad,
    }
    json_string = (
        json.dumps(sentence_item, ensure_ascii=False) + "\n"
    )  # .encode('utf8')
    f.write(json_string)


def create_test_jsonl_files_tests():
    # open csv file
    # parse ..
    testset_filepath = get_out_dir() + "sprouse/Experiment 2 materials - Italian.csv"
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
                os.path.join(get_out_dir() + "sprouse/", filename)
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


def run_sprouse_tests(model_type, model, tokenizer, device):
    # testset_filepath = get_out_dir() + "blimp/from_blim_en/islands/complex_NP_island.jsonl"  # wh_island.jsonl' # adjunct_island.jsonl'
    phenomena = [  # 'rc_adjunct_island',
        # 'rc_complex_np', 'rc_subject_island', 'rc_wh_island', # fixme: rc_wh_island empty file
        "wh_adjunct_island",
        "wh_complex_np",
        "wh_subject_island",
        "wh_whether_island",
    ]
    for phenomenon_name in phenomena:
        filename = phenomenon_name + ".jsonl"
        filepath = os.path.abspath(os.path.join(get_out_dir() + "sprouse/", filename))
        score_averages = run_sprouse_test(
            filepath, model_type, model, tokenizer, device
        )
        plot_results(phenomenon_name, score_averages, "lp")


def plot_results(phenomenon_name, score_averages, score_descr):

    # todo: plot values
    #     lp_averages = [lp_short_nonisland_average, lp_long_nonisland_avg,
    #                    lp_short_island_avg, lp_long_island_avg]

    # nonisland line
    short_nonisland_average = [0, score_averages[0]]
    long_nonisland_avg = [1, score_averages[1]]
    x_values = [short_nonisland_average[0], long_nonisland_avg[0]]
    y_values = [short_nonisland_average[1], long_nonisland_avg[1]]
    plt.plot(x_values, y_values)

    # island line
    short_island_avg = [0, score_averages[2]]
    long_island_avg = [1, score_averages[3]]
    x_values = [short_island_avg[0], long_island_avg[0]]
    y_values = [short_island_avg[1], long_island_avg[1]]
    plt.plot(x_values, y_values, linestyle="--")
    plt.title(phenomenon_name)
    plt.ylabel(f"{score_descr} values")
    plt.show()


def run_sprouse_test(filepath, model_type, model, tokenizer, device):
    print(f"loading testset file {filepath}..")
    with open(filepath, mode="r", encoding="utf-8") as json_file:
        json_list = list(json_file)
    print("testset loaded.")

    examples = []
    for json_str in tqdm(json_list):
        example = json.loads(json_str)
        # print(f"result: {example}")
        # print(isinstance(example, dict))
        # parsed_example = read_sentences_item(example)
        # sentence_good = example['sentence_good']
        # sentence_bad = example['sentence_bad']
        examples.append(
            example
        )  # {'sentence_good': sentence_good, 'sentence_bad': sentence_bad, 'sentence_good_2nd': ""})
    testset = {"sentences": examples}

    # run_testset(model_type, model, tokenizer, device, testset)
    lp_averages = run_sprouse_test_helper(model_type, model, tokenizer, device, testset)
    print(f"{lp_averages=}")
    return lp_averages


def run_sprouse_test_helper(model_type, model, tokenizer, device, testset):
    sent_ids = []
    sentences_per_example = 4
    examples_count = len(testset["sentences"])
    lp_short_nonisland_average = 0
    lp_long_nonisland_avg = 0
    lp_short_island_avg = 0
    lp_long_island_avg = 0
    penlp_short_nonisland_average = 0

    for example_idx, example_data in enumerate(tqdm(testset["sentences"])):
        (
            lps,
            pen_lps,
            pen_sentence_log_weights,
            sentence_log_weights,
            sentences,
        ) = get_example_scores(
            device,
            example_data,
            model,
            model_type,
            sent_ids,
            tokenizer,
            sentences_per_example,
            sprouse_format=True,
        )

        #     sentence_item = {'short_nonisland': good_sentence_short_nonisland,
        #                      'short_island': good_sentence_short_island,
        #                      'long_nonisland': good_sentence_long_nonisland,
        #                      'sentence_bad': sentence_bad}

        lp_short_nonisland_average += lps[0]
        lp_long_nonisland_avg += lps[1]
        lp_short_island_avg += lps[2]
        lp_long_island_avg += lps[3]
        penlp_short_nonisland_average += pen_lps[0]
    lp_short_nonisland_average /= examples_count
    lp_long_nonisland_avg /= examples_count
    lp_short_island_avg /= examples_count
    lp_long_island_avg /= examples_count
    penlp_short_nonisland_average /= examples_count
    lp_averages = [
        lp_short_nonisland_average,
        lp_long_nonisland_avg,
        lp_short_island_avg,
        lp_long_island_avg,
    ]
    return lp_averages


def main():
    # create_test_jsonl_files_tests()

    model_type = model_types.BERT  # model_types.GPT # model_types.ROBERTA  #
    model_name = "dbmdz/bert-base-italian-xxl-cased"  # "bert-base-uncased"  # "gpt2-large"  # "roberta-large" # "bert-large-uncased"  #
    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)
    run_sprouse_tests(model_type, model, tokenizer, device)


def get_out_dir():
    return "../../outputs/"


if __name__ == "__main__":
    main()


def plot_all_phenomena(phenomena_names, lp_avg_scores):
    for idx, phenomenon in enumerate(phenomena_names):
        plot_results(phenomenon, lp_avg_scores[idx], "lp")
