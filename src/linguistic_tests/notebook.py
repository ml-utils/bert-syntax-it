import json
import os.path
import sys

from linguistic_tests.bert_utils import analize_example
from linguistic_tests.bert_utils import estimate_sentence_probability
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import red_txt
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


def interactive_mode():
    print("interactive mode")

    # load model than wait for input sentences
    model_name = str(get_models_dir() / "bert-base-italian-xxl-cased")
    # eval_suite = 'it'
    bert, tokenizer = load_model_and_tokenizer(
        model_types.BERT, model_name, do_lower_case=False
    )

    print("model loaded, waiting for sentences..")

    # given two sentences, print PenLPs, and diff btw PenLPs
    end_program = False
    while not end_program:
        good_sentence = input("Enter first sentence (good): ")
        if good_sentence == "exit":
            return
        bad_sentence = input("Enter 2nd sentence (bad): ")

        example = {
            "good_sentence": good_sentence,
            "bad_sentence": bad_sentence,
            "good_sentence2": None,
        }
        sentences_per_example = 2
        (
            base_sentence_less_acceptable,
            second_sentence_less_acceptable,
            acceptability_diff_base_sentence,
            acceptability_diff_second_sentence,
            penLP_base_sentence,
            penLP_bad_sentence,
            penLP_2nd_good_sentence,
            logits_normalized_bad_sentence,
            logits_normalized_base_sentence,
            logits_normalized_2nd_good_sentence,
            oov_counts,
        ) = analize_example(bert, tokenizer, -1, example, sentences_per_example)
        diff_penLP = round(penLP_base_sentence - penLP_bad_sentence, 3)

        print_red("PenLP:")
        print(
            f"Diff {red_txt(diff_penLP)}, "
            f"good ({penLP_base_sentence:.1f}), "
            f"bad ({penLP_bad_sentence:.1f}): "
            f"{good_sentence} || {bad_sentence}"
        )

        # analize both sentences with topk for each masking
        if diff_penLP >= 0:
            print_detailed_sentence_info(bert, tokenizer, good_sentence)
            print_detailed_sentence_info(bert, tokenizer, bad_sentence)


def print_detailed_sentence_info(bert, tokenizer, sentence_txt):
    print_red(f"printing details for sentence {sentence_txt}")
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    estimate_sentence_probability(bert, tokenizer, sentence_ids, verbose=True)


# todo same gpt2 as in the paper, comparable bert

# "GPT-2-large with 36 layers and 774M parameters.10 The model is pretrained
# on Radford et al.’s WebText dataset,
# which contains 40GB of English text extracted from Web pages and filtered
# for quality." Estimated that WebText
# contains about 8B tokens.
#
# ..
# huggingface.co: gpt2-large (model detail info?)(n_layer": 36,)
# "The OpenAI team wanted to train this model on a corpus as large as possible.
# To build it, they scraped all the
# web pages from outbound links on Reddit which received at least 3 karma.
# Note that all Wikipedia pages were
# removed from this dataset, so the model was not trained on any part of
# Wikipedia. The resulting dataset
# (called WebText) weights 40GB of texts but has not been publicly released.
# You can find a list of the top 1,000
# domains present in WebText here."
# https://huggingface.co/tftransformers/gpt2-large
#
# vs bert-large-uncased https://huggingface.co/bert-large-uncased
# 336M parameters. "pretrained on BookCorpus, a dataset consisting of 11,038
# unpublished books and English Wikipedia
# (excluding lists, tables and headers)." trained  "for one million steps
# with a batch size of 256"
#
# vs https://huggingface.co/roberta-large
# training data: 160GB of text
#
# todo: load blimp testset,
#  run each file,
#   extract json lines, pair of sentences from each
#  print accuracy results, compare with those in the paper
# adjunct island, gpt2 expected accuracy 91%
# 100%|██████████| 1000/1000 [37:03<00:00,  2.22s/it]test results report:
# acc. correct_lps_1st_sentence: 90.2 %
# acc. correct_pen_lps_1st_sentence: 90.2 %
def run_blimp_en(model_type=None, model_name=None, testset_filenames=None):
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
            sentence_good = example["sentence_good"]
            sentence_bad = example["sentence_bad"]
            examples.append(
                {
                    "sentence_good": sentence_good,
                    "sentence_bad": sentence_bad,
                    "sentence_good_2nd": "",
                }
            )
        testset = {"sentences": examples}
        sentences_per_example = 2
        run_testset(
            model_type, model, tokenizer, DEVICES.CPU, testset, sentences_per_example
        )


def run_tests_it(model_type, testset_files=None):
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
    p = (
        get_syntactic_tests_dir() / "syntactic_tests_it"
    )  # "./outputs/syntactic_tests_it/"
    testsets_dir = str(p)
    if testset_files is None:
        testset_files = [  # 'variations_tests.jsonl'
            # "wh_adjunct_islands.jsonl",
            # "wh_complex_np_islands.jsonl",
            "wh_subject_islands.jsonl",
            "wh_whether_island.jsonl",
        ]
    sentences_per_example = 3
    for test_file in testset_files:
        filepath = os.path.join(testsets_dir, test_file)
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
    if len(sys.argv) > 1:
        interactive_mode()
    else:
        # run_blimp_en()
        # raise SystemExit
        # print('choosing model type ..')
        model_type = model_types.BERT
        run_tests_for_model_type(model_type)


if __name__ == "__main__":
    main()
    # profile_slowdowns()
