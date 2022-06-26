import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from torch import no_grad
from torch import tensor

from int_tests.int_tests_utils import get_test_data_dir
from src.linguistic_tests.compute_model_score import logistic2
from src.linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import get_testset_params
from src.linguistic_tests.lm_utils import load_model
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.testset import parse_testsets
from src.linguistic_tests.testset import TypedSentence


class DiagnosticsWithPlots:
    def plot_logistic2(self):

        # generate linearly spaced numbers start, stop, num
        x_range = np.linspace(start=-20, stop=25, num=1000)

        # _ = logistic2(x_range)  # y_default =

        k_values = [4, 2, 1, 0.5, 0.25, 0.1]
        for k in k_values:
            y_values = logistic2(x_range, k=k)
            plt.plot(x_range, y_values, label=f"{k}")

        # show the plot
        plt.legend()
        plt.show()

    def model_outputs(self):

        plot_span_of_Bert_output_logitis()


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
        _, _, dataset_source, experimental_design = get_testset_params(tests_subdir)
        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if model_type in BERT_LIKE_MODEL_TYPES:
            scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]

        parsed_testsets = parse_testsets(
            testset_dir_path,
            phenomena,
            dataset_source,
            "sprouse",
            experimental_design,
            model_name,
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
    print(f"there are {len(examples_to_plot)}")
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
            wspace=0.120,
            # the amount of width reserved for space between subplots,
            # expressed as a fraction of the average axis width
            hspace=0.2,
            # the amount of height reserved for space between subplots,
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
        sentence_tokens,
        sentence_type_descr,
        ax,
        tokenizer,
        model,
        zoom=True,  # zoom=False
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

            if plot_probabilities:
                output_to_plot_logistic = logistic2(sorted_output)
                output_to_plot_softmax = softmax(sorted_output)

            if zoom:
                # slicing the output array to the last 2500 values
                top_values_to_plot = 40  # 2500
                output_to_plot_logistic = output_to_plot_logistic[-top_values_to_plot:]
                output_to_plot_softmax = output_to_plot_softmax[-top_values_to_plot:]
            else:
                top_values_to_plot = len(sorted_output)

            output_to_plot_x = range(
                len(sorted_output) - top_values_to_plot, len(sorted_output)
            )
            # plotted_lines_logistic = ax.plot(
            #     output_to_plot_x,lo
            #     output_to_plot_logistic,
            #     label=f"{tokenize_combined[masked_token_index]}",
            # )
            print(
                f"Plotting {len(output_to_plot_softmax)} values: {output_to_plot_softmax}"
            )
            plotted_lines_softmax = ax.plot(
                output_to_plot_x,
                output_to_plot_softmax,
                # color='r',
            )
            ax.set_xticks(output_to_plot_x)
            plotted_lines = plotted_lines_softmax
            masked_token_id = tokenizer.convert_tokens_to_ids(
                [tokenize_combined[masked_token_index]]
            )[0]
            token_score = np.asscalar(predictions_scores_this_masking[masked_token_id])
            print(f"{type(token_score)}, {token_score}")
            np_where_result = np.where(sorted_output == token_score)

            if verbose:
                np_where_aq_result = np.where(np.isclose(sorted_output, token_score))
                # np_where_gt_result = np.where(sorted_output > token_score)
                # np_where_lt_result = np.where(sorted_output < token_score)
                print(f"{np_where_result}, {len(np_where_result[0])}, {token_score}")
                print(f"{len(np_where_aq_result[0])}")
                # print(f"{len(np_where_gt_result[0])}")
                # print(f"{len(np_where_lt_result[0])}")
                # print(f"{np_where_gt_result[0][0]}")
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
            print(f"{vocab_size}, {len(sorted_output)}, {sorted_output.shape}")
            thresholds = [0, 5, 10, 15, 20]
            for threshold in thresholds:
                argwhere_result = np.argwhere(sorted_output > threshold)
                if verbose:
                    print(f"{argwhere_result.shape}, {argwhere_result.size}")
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
                    print(f"top {k} min value {sorted_output[topk_idx]} ({topk_idx})")
    # ax.legend(title=f"{typed_sentence.stype.name}")
    ax.set_title(f"{sentence_type_descr} ({min_masking_rank})")
    # ax.set_ylabel("logitis")
    if zoom:
        ax.set_xlim(xmin=min_masking_rank - 5, xmax=vocab_size + 1)
        # ax.set_ylim(ymin=3, ymax=22)
    ax.legend(loc="upper left")
    ax.grid(True)
