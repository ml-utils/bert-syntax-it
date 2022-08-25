import argparse
import logging
import sys
from typing import Dict
from typing import List

from src.linguistic_tests.bert_utils import analize_example
from src.linguistic_tests.compute_model_score import score_example
from src.linguistic_tests.file_utils import _setup_logging
from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import get_num_of_available_cuda_gpus
from src.linguistic_tests.lm_utils import get_testset_params
from src.linguistic_tests.lm_utils import load_model_and_tokenizer
from src.linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
from src.linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_IT
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import print_red
from src.linguistic_tests.lm_utils import red_txt
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.lm_utils import sentence_score_bases
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.plots_and_prints import (
    _print_compare__examples_by_DD_score_helper,
)
from src.linguistic_tests.plots_and_prints import excel_output
from src.linguistic_tests.plots_and_prints import print_detailed_sentence_info
from src.linguistic_tests.run_test_design import run_test_design
from src.linguistic_tests.testset import parse_example
from src.linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
from src.linguistic_tests.testset import TestSet


def get_interactive_mode_arg_parser():
    import argparse

    arg_parser = argparse.ArgumentParser(exit_on_error=False)

    # example: --cmd example --s_ni ".." --l_ni ".." --s_is ".." --l_is ".."
    arg_parser.add_argument("--cmd", type=str, choices=["compare2", "example", "exit"])
    arg_parser.add_argument("--s_ni", type=str)
    arg_parser.add_argument("--l_ni", type=str)
    arg_parser.add_argument("--s_is", type=str)
    arg_parser.add_argument("--l_is", type=str)

    return arg_parser


def interactive_mode(device=DEVICES.CPU):
    print("interactive mode")
    _setup_logging(log_level=logging.DEBUG)
    print_red(f"Using cuda device {device}")  # logging.info

    import shlex

    # todo: interactive mode in which a whole command (pair of sentences) is passed in a single line
    # (each time keeping the session line for the next command, because there is a script startup delay
    # params: type of comparison/checks (..), s1_txt, s2_txt (within quotation marks)
    # model (can pass a list of models to load at startup)
    #
    # options for type of comparison/checks:
    # print ..
    # plot ..

    # todo: test tokenization with sentencepiece, check no unknown
    # todo: check topk
    # todo: list special tokens

    # load model than wait for input sentences
    scorebase = sentence_score_bases.SOFTMAX  # LOGISTIC_FUN
    print(f"Scores are based on scorebase={scorebase.name}")

    model_types_and_names = {**MODEL_TYPES_AND_NAMES_EN, **MODEL_TYPES_AND_NAMES_IT}

    # local_model_names = [
    #     str(get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch"),
    #     str(get_models_dir() / "bostromkaj/uni_20k_ep20_pytorch"),
    # ]
    model_names = list(model_types_and_names.keys())
    model_choice_correct = False
    while not model_choice_correct:
        print("These are the models available, pick a choice:")
        for idx, model_name in enumerate(model_names):
            print(f"({idx}) {model_name}")

        model_choice = input("Enter a model number: ")
        try:
            model_name = model_names[int(model_choice)]
            model_choice_correct = True
        except (IndexError, ValueError) as err:
            print(f"{err}, {type(err)}")

    model_type = model_types_and_names[
        model_name
    ]  # ModelTypes.ROBERTA  # ModelTypes.BERT  #
    # eval_suite = 'it'
    model, tokenizer = load_model_and_tokenizer(
        model_type, model_name, device, do_lower_case=False
    )

    print("model loaded, waiting for commands..")
    parser = get_interactive_mode_arg_parser()

    # given two sentences, print PenLPs, and diff btw PenLPs
    end_program = False
    while not end_program:
        valid_args = False
        # argString = input("Enter command: ")
        argString = ""
        print("Enter command: ", end="")
        try:
            sentinel = ""  # iter terminates when receives ""
            argString = " ".join(iter(input, sentinel))
        except (TypeError, ValueError) as err:  # ..
            print(f"{err}")
        logging.debug(f"argString = {argString}")

        try:
            args = parser.parse_args(shlex.split(argString))
            valid_args = True
        except argparse.ArgumentError as err:
            print(f"{err}")

        if valid_args:
            if args.cmd == "exit":
                end_program = True
            elif args.cmd == "compare2":
                compare_two_sentences(args, model_type, model, tokenizer)
            elif args.cmd == "example":
                score_and_print_inline_example(
                    args, model_type, model, tokenizer, device
                )


def score_and_print_inline_example(
    args, model_type, model, tokenizer, device=DEVICES.CPU
):
    example_dict = {
        SentenceNames.SHORT_NONISLAND: args.s_ni,
        SentenceNames.LONG_NONISLAND: args.l_ni,
        SentenceNames.SHORT_ISLAND: args.s_is,
        SentenceNames.LONG_ISLAND: args.l_is,
    }
    example = parse_example(example_dict, SPROUSE_SENTENCE_TYPES)

    logging.debug("Scoring example..")
    score_example(
        device,
        example,
        model,
        model_type,
        tokenizer,
    )
    logging.debug("done.")

    _print_compare__examples_by_DD_score_helper(
        [example], ScoringMeasures.PenLP, shorter_form=True
    )


def compare_two_sentences(args, model, tokenizer):
    # good_sentence = input("Enter first sentence (good): ")
    # bad_sentence = input("Enter 2nd sentence (bad): ")
    good_sentence = args.good
    bad_sentence = args.bad

    for sentence in [good_sentence, bad_sentence]:
        tokens = tokenizer.tokenize(sentence)
        print(
            f"{sentence} \n "
            f"tokenized as: {tokens} \n "
            f"with ids: {tokenizer.convert_tokens_to_ids(tokens)}"
        )

    # todo: print unk token and id

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
        # logits_normalized_bad_sentence,
        # logits_normalized_base_sentence,
        # logits_normalized_2nd_good_sentence,
        oov_counts,
    ) = analize_example(
        model, tokenizer, -1, example, sentences_per_example, score_based_on=None
    )

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
        print_detailed_sentence_info(model, tokenizer, good_sentence, scorebase=None)
        print_detailed_sentence_info(model, tokenizer, bad_sentence, scorebase=None)


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


def main(
    device: DEVICES,
    rescore=False,
    log_level=logging.INFO,
    show_plot=False,
    save_plot=False,
):
    if len(sys.argv) > 1:
        interactive_mode()  # device
    else:
        print_red(f"Using cuda device {device}")  # logging.info
        # todo: save accuracy and other results to csv file (for import in excel table)
        #  also another csv file with details on sentences scores
        #  and an option to load the report csv and print them in the command line

        # from linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
        # from linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_IT

        # run_test_design(
        #     model_types_and_names={
        #         "roberta-large": ModelTypes.ROBERTA,
        #     },
        #     tests_subdir="blimp/from_blim_en/islands/",
        #     max_examples=1000,
        #     device=device,
        #     rescore=rescore,
        #     log_level=log_level,
        #     show_plot=show_plot,
        # )

        tests_subdirs = [
            "sprouse/",
            "syntactic_tests_it/",
            # "variations/",
        ]

        # gpt_it_model_type_and_name = {
        #             "LorenzoDeMattei/GePpeTto": ModelTypes.GEPPETTO,
        #         }
        model_types_and_names: Dict[
            str, ModelTypes
        ] = MODEL_TYPES_AND_NAMES_IT  # gpt_it_model_type_and_name
        logging.info(f"Will run tests with models: {model_types_and_names.keys()}")
        for model_name, model_type in model_types_and_names.items():

            scored_testsets_by_datasource: Dict[DataSources, List[TestSet]] = dict()

            for tests_subdir in tests_subdirs:
                loaded_testsets = run_test_design(
                    # todo: use only one model
                    #  loop models in the outer for (one excel file output for each model)
                    #  check consistency with output on plots and prints
                    # or could aggregate also scores for multiple models in one excel file
                    model_name=model_name,
                    model_type=model_type,
                    tests_subdir=tests_subdir,
                    max_examples=99,
                    device=device,
                    rescore=rescore,
                    log_level=log_level,
                    show_plot=show_plot,
                    save_plot=save_plot,
                )

                # write excel output
                _, _, dataset_source, experimental_design = get_testset_params(
                    tests_subdir
                )
                # if experimental_design in [ExperimentalDesigns.FACTORIAL]:
                scored_testsets_by_datasource[dataset_source] = loaded_testsets

            testsets_first_datasource = list(scored_testsets_by_datasource.values())[0]
            first_testsset = testsets_first_datasource[0]
            if first_testsset.experimental_design in [ExperimentalDesigns.FACTORIAL]:
                excel_output(scored_testsets_by_datasource)
            else:
                logging.info(
                    f"Skipping excel output, experimental_design is {first_testsset.experimental_design}"
                )


if __name__ == "__main__":

    print(f"Number of available cuda gpus: {get_num_of_available_cuda_gpus()}")
    if get_num_of_available_cuda_gpus() > 0:
        main_setting_device = DEVICES.CUDA_0
    else:
        main_setting_device = DEVICES.CPU

    main_setting_rescore = True
    main_setting_show_plot = True
    main_settings_save_plot = True
    main_setting_log_level = logging.DEBUG

    main(
        device=main_setting_device,
        rescore=main_setting_rescore,
        log_level=main_setting_log_level,
        show_plot=main_setting_show_plot,
        save_plot=main_settings_save_plot,
    )
