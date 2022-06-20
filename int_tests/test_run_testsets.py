import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from linguistic_tests.compute_model_score import logistic2
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.file_utils import parse_testsets
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.run_sprouse_tests import get_testset_params
from linguistic_tests.run_sprouse_tests import score_sprouse_testsets
from linguistic_tests.run_syntactic_tests import run_blimp_en
from linguistic_tests.testset import TypedSentence
from matplotlib import pyplot as plt
from torch import no_grad
from torch import tensor

from int_tests.int_tests_utils import get_test_data_dir


class TestRunTestSets(TestCase):
    # todo: also run these tests mocking the models (no remote calls) and with
    #  string variables instead of reading from files.
    #  (for mock models: save a model file with 1 layer and few nodes?)
    #  also move the mini test samples from the actual test data folder
    #  to the integration tests folder
    #  make output and models folder as optional parameters, to use them in
    #  unit tests

    @pytest.mark.enable_socket
    def test_run_blimp_en_tests(self):
        testset_filenames = ["mini_wh_island"]
        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)
        examples_format = "json_lines"
        run_blimp_en(
            model_type=ModelTypes.BERT,
            model_name="bert-base-uncased",
            dataset_source="Blimp paper",
            testset_filenames=testset_filenames,
            testset_dir_path=testset_dir_path,
            examples_format=examples_format,
        )

    @pytest.mark.enable_socket
    @patch.object(plt, "show")
    def test_run_sprouse_tests(self, mock1):
        assert plt.show is mock1

        model_type = ModelTypes.BERT
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
        phenomena = [
            "mini_wh_adjunct_island",
        ]
        tests_subdir = "sprouse/"
        p = get_test_data_dir() / tests_subdir
        testset_dir_path = str(p)
        _, _, dataset_source = get_testset_params(tests_subdir)

        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if model_type in BERT_LIKE_MODEL_TYPES:
            scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
        parsed_testsets = parse_testsets(
            testset_dir_path,
            phenomena,
            dataset_source,
            "sprouse",
            "sprouse",
            model_name,
            model_type,
            scoring_measures,
            max_examples=1000,
        )

        scored_testsets = score_sprouse_testsets(
            model_type,
            model,
            tokenizer,
            DEVICES.CPU,
            parsed_testsets,
        )

        for testset in scored_testsets:
            assert testset.avg_DD_lp != -200
            assert testset.avg_DD_penlp != -200
            assert testset.avg_DD_ll != -200
            assert testset.avg_DD_penll != -200

    @pytest.mark.slow
    @pytest.mark.enable_socket
    def test_run_syntactic_it_tests(self):
        model_type = ModelTypes.BERT
        testset_files = [
            "mini_wh_adjunct_islands.jsonl",
        ]
        p = get_test_data_dir() / "custom_it"
        testset_dir_path = str(p)

        self.run_syntactic_tests_it_legacy_impl(
            model_type,
            testset_filenames=testset_files,
            testset_dir_path=testset_dir_path,
        )

    @staticmethod
    def profile_slowdowns():
        import cProfile
        import pstats
        import os
        import tqdm
        import json

        model_type = ModelTypes.ROBERTA  # ModelTypes.GPT  #
        model_name = "roberta-large"  # "roberta-base" #"gpt2-medium"
        # "gpt2-large"  # 'gpt2' #  "bert-large-uncased"
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)

        testset_filename = "mini_wh_island.jsonl"
        testset_filepath = os.path.join(testset_dir_path, testset_filename)

        print(f"loading testset file {testset_filepath}..")
        with open(testset_filepath, "r") as json_file:
            json_list = list(json_file)
        print("testset loaded.")

        examples = []
        for json_str in tqdm(json_list):
            example = json.loads(json_str)

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

        with cProfile.Profile() as pr:
            run_testset(
                model_type,
                model,
                tokenizer,
                DEVICES.CPU,
                testset,
                sentences_per_example,
            )

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

    @staticmethod
    def run_syntactic_tests_it_legacy_impl(
        model_type, testset_filenames=None, testset_dir_path=None
    ):
        if model_type == ModelTypes.GPT:
            model_name = "GroNLP/gpt2-small-italian"
        if model_type == ModelTypes.GEPPETTO:
            model_name = "LorenzoDeMattei/GePpeTto"
        elif model_type == ModelTypes.BERT:
            model_name = "bert-base-uncased"  # NB bert large uncased is about 1GB
            model_name = str(get_models_dir() / "bert-base-italian-uncased")
            model_name = str(get_models_dir() / "bert-base-italian-cased/")
            model_name = str(get_models_dir() / "bert-base-italian-xxl-cased")
            model_name = "dbmdz/bert-base-italian-cased"
            model_name = "dbmdz/bert-base-italian-xxl-cased"
            # model_name = # str(get_models_dir() / "gilberto-uncased-from-camembert.tar.gz")
            # eval_suite = 'it'
        elif model_type == ModelTypes.GILBERTO:
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

            if model_type in [
                ModelTypes.BERT,
                ModelTypes.GILBERTO,
                ModelTypes.ROBERTA,
            ]:
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
            elif model_type in [ModelTypes.GPT, ModelTypes.GEPPETTO]:
                run_testset(
                    model_type,
                    model,
                    tokenizer,
                    DEVICES.CPU,
                    testset_data,
                    sentences_per_example,
                )

    @pytest.mark.enable_socket
    @patch.object(plt, "show")
    def test_model_outputs(self, pyplot_show_mock=None):
        assert plt.show is pyplot_show_mock
        plot_span_of_Bert_output_logitis()

    @staticmethod
    def plot_logistic2():

        # 1000 linearly spaced numbers
        x = np.linspace(-20, 25, 1000)

        # the function, which is y = x^2 here
        y = logistic2(x)

        # setting the axes at the centre
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.spines['left'].set_position('center')
        # ax.spines['bottom'].set_position('zero')
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')

        # plt.xlim((-20, 20))

        # plot the function
        plt.plot(x, y, label="default")

        y_k_05 = logistic2(x, k=0.5)
        plt.plot(x, y_k_05, label="y_k_05")

        y_k_025 = logistic2(x, k=0.25)
        plt.plot(x, y_k_025, label="y_k_025")

        y_k_010 = logistic2(x, k=0.1)
        plt.plot(x, y_k_010, label="y_k_010")

        # show the plot

        plt.legend()
        plt.show()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def plot_span_of_Bert_output_logitis():

    model_type = ModelTypes.BERT
    model_name = "dbmdz/bert-base-italian-xxl-cased"
    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

    do_random_sentences = True
    do_real_sentences = True
    if do_real_sentences:
        phenomena = [
            "wh_adjunct_island",  # "mini_wh_adjunct_island",
        ]
        tests_subdir = "sprouse/"
        p = get_test_data_dir() / tests_subdir
        testset_dir_path = str(p)
        _, _, dataset_source = get_testset_params(tests_subdir)
        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if model_type in BERT_LIKE_MODEL_TYPES:
            scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
        parsed_testsets = parse_testsets(
            testset_dir_path,
            phenomena,
            dataset_source,
            "sprouse",
            "sprouse",
            model_name,
            model_type,
            scoring_measures,
            max_examples=1000,
        )
        examples_to_plot = parsed_testsets[0].examples  # examples[0:2]
        plot_span_of_Bert_output_logitis_helper(
            _plot_typed_sentence_scores, examples_to_plot, tokenizer, model, model_name
        )

    if do_random_sentences:
        example_dict = {"sentences": [0, 1, 2, 3]}
        example_adict = AttrDict(example_dict)
        examples_to_plot = [example_adict]
        plot_span_of_Bert_output_logitis_helper(
            _plot_scores_of_random_sentence,
            examples_to_plot,
            tokenizer,
            model,
            model_name,
        )


def plot_span_of_Bert_output_logitis_helper(
    plotting_fun, examples_to_plot, tokenizer, model, model_name
):
    print(f"there are {len(examples_to_plot)=}")
    for example in examples_to_plot:
        fig, axs = plt.subplots(2, 2)
        axs_list = axs.reshape(-1)

        for typed_sentence, ax in zip(example.sentences, axs_list):
            plotting_fun(typed_sentence, tokenizer, model, ax)

        # plt.legend()
        # plt.suptitle(f"Model: {scored_testsets[0].model_descr}")
        fig.subplots_adjust(
            left=0.05,  # the left side of the subplots of the figure
            bottom=0.05,  # the bottom of the subplots of the figure
            right=0.95,  # the right side of the subplots of the figure
            top=0.95,  # the top of the subplots of the figure
            wspace=0.120,  # the amount of width reserved for space between subplots,
            # expressed as a fraction of the average axis width
            hspace=0.2,  # the amount of height reserved for space between subplots,
            # expressed as a fraction of the average axis height
        )
        fig.suptitle(f"Model: {model_name}")  # plt.suptitle(title)
        plt.show()


def _plot_scores_of_random_sentence(_, tokenizer, model, ax):
    import random

    sentence_lenght = 10
    vocab_size = 32100
    random_ids = [random.randint(0, vocab_size) for _ in range(sentence_lenght)]

    sentence_tokens = tokenizer.convert_ids_to_tokens(random_ids)
    sentence_type_descr = "RAND"
    _plot_bert_sentence_scores(
        sentence_tokens, sentence_type_descr, ax, tokenizer, model, zoom=False
    )


def _plot_typed_sentence_scores(typed_sentence: TypedSentence, tokenizer, model, ax):

    sentence = typed_sentence.sent
    sentence.tokens = tokenizer.tokenize(sentence.txt)  # , return_tensors='pt'
    sentence_tokens = sentence.tokens
    sentence_type_descr = typed_sentence.stype.name
    _plot_bert_sentence_scores(
        sentence_tokens, sentence_type_descr, ax, tokenizer, model, zoom=False
    )


def _plot_bert_sentence_scores(
    sentence_tokens,
    sentence_type_descr,
    ax,
    tokenizer,
    model,
    zoom=True,
    verbose=False,
    plot_probabilities=True,
):
    batched_indexed_tokens = []
    batched_segment_ids = []
    device = DEVICES.CPU
    # not use_context variant:
    tokenize_combined = ["[CLS]"] + sentence_tokens + ["[SEP]"]
    for i in range(len(sentence_tokens)):
        # Mask a token that we will try to predict back with
        # `BertForMaskedLM`
        masked_token_index = i + 1 + 0  # not use_context variant
        tokenize_masked = tokenize_combined.copy()
        tokenize_masked[masked_token_index] = "[MASK]"
        # unidir bert
        # for j in range(masked_index, len(tokenize_combined)-1):
        #    tokenize_masked[j] = '[MASK]'

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
        # Define sentence A and B indices associated to 1st and 2nd
        # sentences (see paper)
        segment_ids = [0] * len(tokenize_masked)

        batched_indexed_tokens.append(indexed_tokens)
        batched_segment_ids.append(segment_ids)
    tokens_tensor = tensor(batched_indexed_tokens, device=device)
    segment_tensor = tensor(batched_segment_ids, device=device)

    with no_grad():
        model_output = model(tokens_tensor, token_type_ids=segment_tensor)
        predictions_logits_whole_batch = model_output.logits  # model_output[0]
        vocab_size = len(predictions_logits_whole_batch[0, 1].cpu().numpy())
        min_masking_rank = vocab_size - 1
        for i in range(len(sentence_tokens)):
            masked_token_index = i + 1 + 0  # not use_context variant
            predictions_scores_this_masking = predictions_logits_whole_batch[
                i, masked_token_index
            ]

            predicted_scores_numpy = predictions_scores_this_masking.cpu().numpy()
            sorted_output = np.sort(predicted_scores_numpy)

            if zoom:
                # slicing the output array to the last 2500 values
                top_values_to_plot = 2500
            else:
                top_values_to_plot = len(sorted_output)

            output_to_plot_y = sorted_output[-top_values_to_plot:]
            if plot_probabilities:
                output_to_plot_y = logistic2(output_to_plot_y)
            output_to_plot_x = range(
                len(sorted_output) - top_values_to_plot, len(sorted_output)
            )
            plotted_lines = ax.plot(
                output_to_plot_x,
                output_to_plot_y,
                label=f"{tokenize_combined[masked_token_index]}",
            )
            masked_token_id = tokenizer.convert_tokens_to_ids(
                [tokenize_combined[masked_token_index]]
            )[0]
            token_score = np.asscalar(predictions_scores_this_masking[masked_token_id])
            print(f"{type(token_score)=}, {token_score=}")
            np_where_result = np.where(sorted_output == token_score)

            if verbose:
                np_where_aq_result = np.where(np.isclose(sorted_output, token_score))
                # np_where_gt_result = np.where(sorted_output > token_score)
                # np_where_lt_result = np.where(sorted_output < token_score)
                print(f"{np_where_result=}, {len(np_where_result[0])=}, {token_score=}")
                print(f"{len(np_where_aq_result[0])=}")
                # print(f"{len(np_where_gt_result[0])=}")
                # print(f"{len(np_where_lt_result[0])=}")
                # print(f"{np_where_gt_result[0][0]=}")
            # nb: Tuple of arrays returned from np.where:  (array([..found_indexes], dtype=..),)
            if len(np_where_result[0]) > 1:
                masked_token_new_id = np.asscalar(np_where_result[0][0])
            else:
                masked_token_new_id = np.asscalar(np_where_result[0])
            min_masking_rank = min(masked_token_new_id, min_masking_rank)
            if not plot_probabilities:
                ax.axvline(x=masked_token_new_id, color=plotted_lines[0].get_color())

            # ax.xaxis.grid(which='minor')

            # todo:
            # count how many, in the bert output array (sorted_output), are above certain tresholds:
            print(f"{vocab_size=}, {len(sorted_output)=}, {sorted_output.shape=}")
            thresholds = [0, 5, 10, 15, 20]
            for threshold in thresholds:
                argwhere_result = np.argwhere(sorted_output > threshold)
                if verbose:
                    print(f"{argwhere_result.shape=}, {argwhere_result.size=}")
                if argwhere_result.size > 0:
                    idx = argwhere_result[0]
                    if verbose:
                        print(
                            f"Idx of first element above {threshold} "
                            f"is {idx} (marks the top {len(sorted_output) - idx}), "
                            f"with value {sorted_output[idx]}"
                        )
                else:
                    if verbose:
                        print(f"No element above {threshold}")

            # top k min value
            k_values = [5, 10, 20]
            for k in k_values:
                topk_idx = len(sorted_output) - k
                if verbose:
                    print(f"top {k=} min value {sorted_output[topk_idx]} ({topk_idx=})")
    # ax.legend(title=f"{typed_sentence.stype.name}")
    ax.set_title(f"{sentence_type_descr} ({min_masking_rank=})")
    # ax.set_ylabel("logitis")
    if zoom:
        ax.set_xlim(xmin=min_masking_rank - 5, xmax=vocab_size + 1)
        ax.set_ylim(ymin=3, ymax=22)
    ax.legend(loc="upper left")
    ax.grid(True)
