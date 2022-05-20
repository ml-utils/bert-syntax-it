import csv

# parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file
import json
import os.path

import pandas
from tqdm import tqdm

from compute_model_score import DEVICES, load_model, run_testset
from lm_utils import model_types


def get_sentence_from_row(C1, C2, current_item_sentences):
    C1_col = 'Condition 1'
    C2_col = 'Condition 2'
    SENTENCE = 'sentence'
    #print(current_item_sentences.info())

    C1_values = set(current_item_sentences[C1_col].tolist())
    #print(f'C1_values: {C1_values}')
    C2_values = set(current_item_sentences[C2_col].tolist())
    #print(f'C1_values: {C2_values}')
    #print(f'params C1: {C1}, C2: {C2}')
    single_row_df = (current_item_sentences.loc[(current_item_sentences[C1_col] == C1)
                                & (current_item_sentences[C2_col] == C2)])
    #print(single_row_df.info())
    return single_row_df.iloc[0][SENTENCE]


def write_sentence_pair(f, sentence_bad, good_sentence, conditions):
    sentence_pair = {'sentence_good': good_sentence, 'sentence_bad': sentence_bad,
                      'conditions': conditions}
    f.write(json.dumps(sentence_pair) + "\n")


def create_test_jsonl_files_tests(model, tokenizer, testfile):
    # open csv file
    # parse ..
    testset_filepath = './outputs/sprouse/Experiment 2 materials - Italian.csv'
    examples = []  # sentence pairs

    df = pandas.read_csv(testset_filepath, sep=';', header=0)
    #print(df.head(2))
    #print(df.info())

    # rslt_df = df.loc[#(df['Clause type'] == 'RC') &
    #                    (df['Island Type'] == 'Adjunct island') & (df['item number'] == 1)]

    CLAUSE_TYPE = 'Clause type'
    clause_types = set(df[CLAUSE_TYPE].tolist())
    # print(f'clause_types: {clause_types}')

    for clause_type in clause_types:

        current_clause_sentences = df.loc[(df[CLAUSE_TYPE] == clause_type)]
        ISLAND_TYPE = 'Island Type'
        island_types = set(current_clause_sentences[ISLAND_TYPE].tolist())
        # print(f'island_types: {island_types}')

        for island_type in island_types:
            # print(f'current clause_type: {clause_type}, current island_type: {island_type}')
            current_phenomenon_sentences = current_clause_sentences.loc[(current_clause_sentences[ISLAND_TYPE] == island_type)]
            phenomenon_name = clause_type.lower() + '_' + island_type.replace(' ', '_').lower()

            filename = phenomenon_name + '.jsonl'
            filepath = os.path.join('./outputs/sprouse/', filename)
            if os.path.exists(filepath):
                print(f'file already exists, skipping: {filepath}')
                continue
            else:
                print(f'writing phenomenon_name: {phenomenon_name}')

            ITEM_NUMBER = 'item number'
            item_numbers = set(current_phenomenon_sentences[ITEM_NUMBER].tolist())
            # print(f'item_numbers: {item_numbers}')
            for item_number in item_numbers:
                current_item_sentences = current_phenomenon_sentences.loc[(current_phenomenon_sentences[ITEM_NUMBER] == item_number)]
                # 4 sentences for 3 pairs
                sentence_bad = get_sentence_from_row('Long', 'Island', current_item_sentences)
                #print(f'bad_sentence: {sentence_bad}, type(bad_sentence): {type(sentence_bad)}')
                good_sentence_long_nonisland = get_sentence_from_row('Long', 'non-island', current_item_sentences)
                good_sentence_short_nonisland = get_sentence_from_row('Short', 'non-island', current_item_sentences)
                good_sentence_short_island = get_sentence_from_row('Short', 'Island', current_item_sentences)

                with open(filepath, 'a') as f:
                    write_sentence_pair(f, sentence_bad, good_sentence_long_nonisland, 'long_nonisland')
                    write_sentence_pair(f, sentence_bad, good_sentence_short_nonisland, 'short_nonisland')
                    write_sentence_pair(f, sentence_bad, good_sentence_short_island, 'short_island')


def run_sprouse_tests(model_type, model, tokenizer, device):
    testset_filepath = './outputs/blimp/from_blim_en/islands/complex_NP_island.jsonl'  # wh_island.jsonl' # adjunct_island.jsonl'
    phenomena = ['rc_adjunct_island',
                 'rc_complex_np', 'rc_subject_island', 'rc_wh_island',
                 'wh_adjunct_island', 'wh_complex_np', 'wh_subject_island', 'wh_whether_island']
    for phenomenon_name in phenomena:
        filename = phenomenon_name + '.jsonl'
        filepath = os.path.join('./outputs/sprouse/', filename)
        run_sprouse_test(filepath, model_type, model, tokenizer, device)


def run_sprouse_test(filepath, model_type, model, tokenizer, device):
    print(f'loading testset file {filepath}..')
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    print(f'testset loaded.')

    examples = []
    for json_str in tqdm(json_list):
        example = json.loads(json_str)
        # print(f"result: {example}")
        # print(isinstance(example, dict))
        sentence_good = example['sentence_good']
        sentence_bad = example['sentence_bad']
        examples.append({'sentence_good': sentence_good, 'sentence_bad': sentence_bad, 'sentence_good_2nd': ""})
    testset = {'sentences': examples}

    run_testset(model_type, model, tokenizer, device, testset)


if __name__ == "__main__":
    model_type = model_types.BERT # model_types.GPT # model_types.ROBERTA  #
    model_name = 'dbmdz/bert-base-italian-xxl-cased' # "bert-base-uncased"  # "gpt2-large"  # "roberta-large" # "bert-large-uncased"  #
    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)
    run_sprouse_tests(model_type, model, tokenizer, device)

