import json
import os

from linguistic_tests.bert_utils import estimate_sentence_probability
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import SentenceNames
from tqdm import tqdm


def run_agreement_tests():
    return 0


def get_masked_word_probability(bert, tokenizer):
    return 0


def run_tests_goldberg():
    # todo: use sentence acceptability estimates (PenLP e PenNL), and see
    #  results on goldberg testset
    # also for blimp testset with tests non intended for bert, compare with
    # the results on gpt and other models
    return 0


def run_tests_blimp():
    # todo
    return 0


def run_tests_lau_et_al():
    # todo
    return 0


def print_detailed_sentence_info(bert, tokenizer, sentence_txt):
    print_red(f"printing details for sentence {sentence_txt}")
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    estimate_sentence_probability(bert, tokenizer, sentence_ids, verbose=True)


def run_blimp_en(
    model_type=None, model_name=None, testset_filenames=None, testset_dir_path=None
):
    if model_type is None:
        model_type = model_types.ROBERTA  # model_types.GPT  #
        model_name = "roberta-large"  # "roberta-base" #"gpt2-medium"
        # "gpt2-large"  # 'gpt2'  #  "bert-large-uncased"
        # "bert-base-uncased"  #    'dbmdz/bert-base-italian-xxl-cased' #
    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

    if testset_filenames is None:
        testset_filenames = [
            "wh_island.jsonl",
            "adjunct_island.jsonl",
            "complex_NP_island.jsonl",
        ]
    if testset_dir_path is None:
        p = get_syntactic_tests_dir() / "blimp/from_blim_en/islands"
        testset_dir_path = str(p)

    for testset_filename in testset_filenames:
        testset_filepath = os.path.join(testset_dir_path, testset_filename)
        # './outputs/blimp/from_blim_en/islands/adjunct_island.jsonl'

        print(f"loading testset file {testset_filepath}..")
        with open(testset_filepath, "r") as json_file:
            json_list = list(json_file)
        print("testset loaded.")

        examples = []
        for json_str in tqdm(json_list):
            example = json.loads(json_str)
            # print(f"result: {example}")
            # print(isinstance(example, dict))
            sentence_good = example[SentenceNames.SENTENCE_GOOD]
            sentence_bad = example[SentenceNames.SENTENCE_BAD]
            examples.append(
                {
                    SentenceNames.SENTENCE_GOOD: sentence_good,
                    SentenceNames.SENTENCE_BAD: sentence_bad,
                    SentenceNames.SENTENCE_GOOD_2ND: "",
                }
            )
        testset = {"sentences": examples}
        sentences_per_example = 2
        run_testset(
            model_type, model, tokenizer, DEVICES.CPU, testset, sentences_per_example
        )


def run_tests_it(model_type, testset_filenames=None, testset_dir_path=None):
    if model_type == model_types.GPT:
        model_name = "GroNLP/gpt2-small-italian"
    if model_type == model_types.GEPPETTO:
        model_name = "LorenzoDeMattei/GePpeTto"
    elif model_type == model_types.BERT:
        model_name = "bert-base-uncased"  # NB bert large uncased is about 1GB
        model_name = str(get_models_dir() / "bert-base-italian-uncased")
        model_name = str(get_models_dir() / "bert-base-italian-cased/")
        model_name = str(get_models_dir() / "bert-base-italian-xxl-cased")
        model_name = "dbmdz/bert-base-italian-cased"
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        # model_name = # str(get_models_dir() / "gilberto-uncased-from-camembert.tar.gz")
        # eval_suite = 'it'
    elif model_type == model_types.GILBERTO:
        model_name = "idb-ita/gilberto-uncased-from-camembert"

    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
    if testset_dir_path is None:
        p = (
            get_syntactic_tests_dir() / "syntactic_tests_it"
        )  # "./outputs/syntactic_tests_it/"
        testset_dir_path = str(p)
    if testset_filenames is None:
        testset_filenames = [  # 'variations_tests.jsonl'
            "wh_adjunct_islands.jsonl",
            "wh_complex_np_islands.jsonl",
            "wh_subject_islands.jsonl",
            "wh_whether_island.jsonl",
        ]
    sentences_per_example = 3
    for test_file in testset_filenames:
        filepath = os.path.join(testset_dir_path, test_file)
        print_orange(f"running test {filepath}")
        testset_data = load_testset_data(filepath)

        if model_type in [model_types.BERT, model_types.GILBERTO, model_types.ROBERTA]:
            # run_testset(testsets_dir, test_file, model, tokenizer,
            # score_based_on=sentence_score_bases.SOFTMAX)
            run_testset(
                model_type,
                model,
                tokenizer,
                DEVICES.CPU,
                testset_data,
                sentences_per_example,
            )
        elif model_type in [model_types.GPT, model_types.GEPPETTO]:
            run_testset(
                model_type,
                model,
                tokenizer,
                DEVICES.CPU,
                testset_data,
                sentences_per_example,
            )


def run_tests_for_model_type(model_type):
    print("model_type: {model_type}")
    # model_name, eval_suite = arg_parse()

    # todo: run on the following testsets (minimal pairs):
    # (use same pretrained models.. or comparable ones to those in the papers)
    # blimp: ..
    # golderg: ..
    # Lau et al: https://github.com/ml-utils/
    # acceptability-prediction-in-context/tree/
    # 0a274d1d9f70f389ddc6b6d796bd8f815833056c/code

    run_tests_it(model_type)

    # if model_type == model_types.GPT:
    #    print('importing gpt_tests..')
    #     from gpt_tests import main as main2
    #    print('imported.')
    #    main2()

    # run_eval(eval_suite, bert, tokenizer)
    # prob1 = estimate_sentence_probability_from_text(bert, tokenizer,
    # 'What is your name?')
    # prob2 = estimate_sentence_probability_from_text(bert, tokenizer,
    # 'What is name your?')
    # print(f'prob1: {prob1}, prob2: {prob2}')
    # eval_it(bert, tokenizer)
    # custom_eval("What is your name?", bert, tokenizer)


def main():
    # print_profession_nouns()
    # t_determiner_noun_agreement_1()
    pass


if __name__ == "__main__":
    main()
