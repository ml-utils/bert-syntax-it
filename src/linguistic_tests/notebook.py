import logging
import sys

from linguistic_tests.bert_utils import analize_example
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import red_txt
from linguistic_tests.lm_utils import sentence_score_bases
from linguistic_tests.plots_and_prints import print_accuracy_scores
from linguistic_tests.plots_and_prints import print_detailed_sentence_info
from linguistic_tests.run_minimal_pairs_test_design import run_blimp_en
from linguistic_tests.testset import DataSources
from linguistic_tests.testset import ExperimentalDesigns
from linguistic_tests.testset import load_testsets_from_pickles


def interactive_mode():
    print("interactive mode")

    # todo: test tokenization with sentencepiece, check no unknown
    # todo: check topk
    # todo: list special tokens

    # load model than wait for input sentences
    scorebase = sentence_score_bases.LOGISTIC_FUN
    print(f"Scores are based on {scorebase=}")
    model_dir = str(
        # get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch"
        get_models_dir()
        / "bostromkaj/uni_20k_ep20_pytorch"
        # "bert-base-uncased",
        # str(get_models_dir() / "bert-base-italian-xxl-cased")
    )
    model_name = model_dir
    model_type = ModelTypes.ROBERTA  # ModelTypes.BERT  #
    # eval_suite = 'it'
    model, tokenizer = load_model_and_tokenizer(
        model_type, model_name, do_lower_case=False
    )

    print("model loaded, waiting for sentences..")

    # given two sentences, print PenLPs, and diff btw PenLPs
    end_program = False
    while not end_program:
        good_sentence = input("Enter first sentence (good): ")
        if good_sentence == "exit":
            return
        bad_sentence = input("Enter 2nd sentence (bad): ")

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
        ) = analize_example(model, tokenizer, -1, example, sentences_per_example)

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
            print_detailed_sentence_info(model, tokenizer, good_sentence, scorebase)
            print_detailed_sentence_info(model, tokenizer, bad_sentence, scorebase)


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
    tests_subdir="syntactic_tests_it/",  # tests_subdir="sprouse/"
    rescore=False,
    log_level=logging.INFO,
    max_examples=1000,
):
    if len(sys.argv) > 1:
        interactive_mode()
    else:

        testset_filenames = [
            "wh_island",  # .jsonl
            "adjunct_island",
            "complex_NP_island",
        ]
        # model_dir = str(get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch")
        p = get_syntactic_tests_dir() / "blimp/from_blim_en/islands"
        testset_dir_path = str(p)
        dataset_source = DataSources.BLIMP_EN
        experimental_design = ExperimentalDesigns.MINIMAL_PAIRS

        for model_name, model_type in MODEL_TYPES_AND_NAMES_EN.items():

            if rescore:
                # todo: switch to parse testset and run minimal pairs test design
                run_blimp_en(
                    model_type=model_type,
                    testset_dir_path=testset_dir_path,
                    testset_filenames=testset_filenames,
                    dataset_source=dataset_source,
                    examples_format="json_lines",
                    max_examples=max_examples,
                )

            loaded_testsets = load_testsets_from_pickles(
                dataset_source,
                testset_filenames,
                model_name,
                expected_experimental_design=experimental_design,
            )

            for scored_testset in loaded_testsets:
                print_accuracy_scores(scored_testset)

            # raise SystemExit
            # print('choosing model type ..')
            # 'dbmdz/bert-base-italian-xxl-cased' #
            # models_to_run = [
            #     ModelTypes.BERT,
            #     ModelTypes.GEPPETTO,
            #     ModelTypes.GPT,
            #     ModelTypes.GILBERTO,
            # ]
            # from linguistic_tests.run_syntactic_tests import run_tests_for_model_type
            # for model_type in models_to_run:
            #     run_tests_for_model_type(model_type)


if __name__ == "__main__":
    # print_list_of_cached_models()
    main(rescore=True, max_examples=5)
    # profile_slowdowns()
