import os
from typing import Dict
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.linguistic_tests.compute_model_score import get_sentence_acceptability_score
from src.linguistic_tests.file_utils import load_object_from_pickle
from src.linguistic_tests.file_utils import save_obj_to_pickle
from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import get_results_dir
from src.linguistic_tests.lm_utils import get_syntactic_tests_dir
from src.linguistic_tests.lm_utils import load_model
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.testset import Example
from src.linguistic_tests.testset import parse_testsets
from src.linguistic_tests.testset import parse_typed_sentence
from src.linguistic_tests.testset import TestSet
from src.linguistic_tests.testset import TypedSentence

matplotlib.rc("text", usetex=False)


class ProfileModelScoring:

    # todo: test model scoring as in paper ..
    # verify that with bert, all tokens of a sentence get more or less the same score
    # load all the sentences in the testset, the the first [0, n], show the scores in print
    # plot the average score ..

    # with the italian models: bert, roberta, gpt2 and our testset
    # with the english models: bert, roberta, gpt2, and ..blimp or sprouse testsets

    # @pytest.mark.enable_socket
    # def test_model_scores_numerical_properties(self):
    #    print_model_scores_numerical_properties()
    pass


def print_model_scores_numerical_properties_helper(
    testset_filenames,
    testset_dir_path,
    model_name,
    model_type,
    experimental_design,
    datasource,
    max_examples=9999,
):

    examples_format = "json_lines"
    device = DEVICES.CPU

    # rescore_testsets_and_save_pickles(...)

    parsed_testsets = parse_testsets(
        testset_dir_path,
        testset_filenames,
        datasource,
        examples_format,
        experimental_design,
        model_name,
        scoring_measures=[ScoringMeasures.LP, ScoringMeasures.PenLP],
        max_examples=max_examples,
    )

    model, tokenizer = load_model(model_type, model_name, device)

    # analyze_avg_of_sentences_of_same_lenght(parsed_testsets, tokenizer, model_type, model, device)
    compare_token_peaks_for_acceptability(
        parsed_testsets, tokenizer, model_type, model, device
    )


def get_example_min_acceptability_diff(e: Example):

    types_of_acceptable_sentences = e.get_types_of_acceptable_sentences()
    acceptable_scores = []
    for stype in types_of_acceptable_sentences:
        acceptable_scores.append(e.get_score(ScoringMeasures.PenLP, stype))
    min_acceptable_score = min(acceptable_scores)

    return (
        e.get_score(ScoringMeasures.PenLP, e.get_type_of_unacceptable_sentence())
        - min_acceptable_score
    )


def compare_token_peaks_for_acceptability(
    parsed_testsets: List[TestSet], tokenizer, model_type, model, device
):

    # plot sentences, tokens names on x axis,
    # staked vertically, same y axis tange
    # chooose which ones to plot ..
    # MAX_FIGURES = 4
    # figures_count = 0

    # todo, fixme: short and long dependency sentence forms seem to be inverted for non island structures
    #  (that is, long dependencies have been classified as short, and viceversa)
    #  (this affects DD score and accuracy reports in the wrong categories)

    # todo, experiment: mask more than one token in a row? (for OOV words)
    #  se the top prediction, and see how these tokens are scored in the spike plot
    #  actually, maybe for all OOV, that are split in multiple tokens, they should be all masked ..
    #  (otherwise it gives too many clues..)

    # note: the spikes are affected by ..consistency:
    # words that are in the same semantic field, reduce the ..entropy
    # es. sindacato and sciopero; if sciopero is missing, sindacato has high entropy
    # design island effects example that control for constitency: both sentence in a minimal pair should have the two "cueing" words

    # note: content, in vocabulary words reduce entropy when there is more than one.. a simple and correct sentence, but with just
    # one content (vs functional) word, receives high entropy
    # eg: Che cosa dici che il sindacato indi ##rà ?
    # eg.2: Che cosa dici che il socio avrebbe rinnovato? (content word, but vague "rinnovare"; it's not very ..linked to the other word)
    # question: therefore Blimp examples should have very high entropy, since they are ..semantically ..inconsistent
    # question: so content words are preponderant over functional words? they are preponderant over ..syntactic constructions violations..
    # ..as if in a bag of words scenario ..

    # eg.3: Di chi pensi che il film abbia catturato l'attenzione degli spettatori?

    # todo: run this comparisons with the blimp dataset
    model_descr = (parsed_testsets[0].model_descr).replace("/", "")
    filename = (
        f"all_examples_{model_descr}_{parsed_testsets[0].dataset_source}_tmp.pickle"
    )
    filepath = os.path.join(str(get_results_dir()), filename)
    if os.path.exists(filepath):
        all_examples = load_object_from_pickle(filename)
    else:
        all_examples = []
        for parsed_testset in parsed_testsets:
            for example_idx, example in enumerate(tqdm(parsed_testset.examples)):
                example.linguistic_phenomenon = parsed_testset.linguistic_phenomenon
                example.model_descr = parsed_testset.model_descr
                example.dataset_source = parsed_testset.dataset_source
                all_examples.append(example)
                for _idx, typed_sentence in enumerate(example.sentences):
                    sentence = typed_sentence.sent

                    sentence.tokens = tokenizer.tokenize(sentence.txt)
                    if len(sentence.tokens) == 0:
                        print("no tokens, skipping..")
                        continue

                    (
                        lp_softmax,
                        lp_logistic,
                        score_per_masking,
                        logistic_score_per_masking,
                    ) = get_sentence_acceptability_score(
                        model_type,
                        model,
                        tokenizer,
                        sentence,
                        device,
                        at_once_mask_all_tokens_of_a_word=False,
                    )
                    sentence.score_per_masking = score_per_masking
                    sentence.pen_lp_softmax = lp_softmax
        # save scored examples to avoid reloading each time
        save_obj_to_pickle(all_examples, filename)

    # use cuda
    # todo: separate for each phenomenon (open 1 window/figure for each phenomenon for each show)
    # sort by diff btw unacceptable and acceptable sentences

    all_examples = sorted(
        all_examples, reverse=True, key=get_example_min_acceptability_diff
    )

    # get_lenght_effect
    # get_structure_effect
    for example in all_examples:
        fig, axs = plt.subplots(4, 1, squeeze=True)
        for _idx, (typed_sentence, subplot_ax) in enumerate(
            zip(example.sentences, axs)
        ):
            sentence = typed_sentence.sent

            text_len = len(sentence.tokens)
            token_idxes = [x + 1 for x in range(text_len)]
            # todo: use subplots, one figure/window for each example (with 4 sentences)
            subplot_ax.plot(
                token_idxes,
                sentence.score_per_masking,
                # logistic_score_per_masking,  # score_per_masking,
                marker="o",
                linestyle="dashed",
                color="blue",
            )
            labels = [
                rf"{token.replace('Ġ','_').replace('#','_')}"
                for token in typed_sentence.sent.tokens
            ]
            subplot_ax.set_xticks(ticks=token_idxes, labels=labels)
            subplot_ax.set_ylim([0, 8])
            # subplot_ax.set_title(f"{typed_sentence.stype} | {sentence.pen_lp_softmax:.2f}")
            subplot_ax.annotate(
                rf"{typed_sentence.stype}, PLL: {sentence.pen_lp_softmax:.2f}",
                xy=(0.8, 0.9),
                xycoords="axes fraction",
            )
            # figures_count += 1
            # if figures_count == MAX_FIGURES:
            #     plt.show()
            #     figures_count = 0
            # plt.figure()
        surprisal_label = (
            r"token surprisal, $\displaystyle\ - log P_{MLM} (w_t | W_{\setminus t}) $"
        )
        fig.text(
            0.05,
            0.5,
            surprisal_label,
            va="center",
            rotation=90,
            fontdict={"size": 16},
            usetex=True,
        )
        # plt.ylabel(surprisal_label, horizontalalignment='right', verticalalignment ='top')
        # subplot_ax.set_ylabel(surprisal_label)
        fig.suptitle(
            rf"{example.linguistic_phenomenon} | {example.model_descr} | Testset: {example.dataset_source}"
            "\n"
            r"$PLL_{unacceptable} - min(PLL_{acceptable})$"
            rf" = {get_example_min_acceptability_diff(example):.2f}"
            "\n"
            r"(PLL : pseudo-log-likelihood, )",
            usetex=True,
        )
        plt.show()
    # plt.show()

    for parsed_testset in parsed_testsets:
        for example_idx, example in enumerate(tqdm(parsed_testset.examples)):
            pass


def analyze_avg_of_sentences_of_same_lenght(
    parsed_testsets, tokenizer, model_type, model, device
):
    all_acceptable_sentences = []
    all_unacceptable_sentences = []
    sentence_lenght_statistics: Dict[
        int, int
    ] = dict()  # sentence lenght, how many with that lenght
    for parsed_testset in parsed_testsets:
        for example_idx, example in enumerate(tqdm(parsed_testset.examples)):

            for _idx, typed_sentence in enumerate(example.sentences):

                if typed_sentence.stype is example.get_type_of_unacceptable_sentence():
                    all_unacceptable_sentences.append(typed_sentence)
                else:
                    all_acceptable_sentences.append(typed_sentence)

                sentence = typed_sentence.sent
                sentence.tokens = tokenizer.tokenize(
                    sentence.txt
                )  # , return_tensors='pt'
                if len(sentence.tokens) not in sentence_lenght_statistics:
                    sentence_lenght_statistics[len(sentence.tokens)] = 0
                sentence_lenght_statistics[len(sentence.tokens)] += 1

    print(f"sentence_lenght_statistics={sorted(sentence_lenght_statistics.items())}")
    most_common_sentence_lenght = max(
        sentence_lenght_statistics, key=sentence_lenght_statistics.get
    )
    # most_common_sentence_lenght = 12  # 11  # 19  #

    print(
        f"most_common_sentence_lenght={most_common_sentence_lenght} "
        f"(with {sentence_lenght_statistics[most_common_sentence_lenght]} sentences)"
    )

    all_sentences_with_most_common_lenght = []
    for acceptable_sentence in all_acceptable_sentences:
        if len(acceptable_sentence.sent.tokens) is most_common_sentence_lenght:
            all_sentences_with_most_common_lenght.append(acceptable_sentence)

    print_sentences_scores(
        all_sentences_with_most_common_lenght, model_type, model, tokenizer, device
    )


def print_sentences_scores(
    typed_sentences_of_same_lenght: List[TypedSentence],
    model_type,
    model,
    tokenizer,
    device,
):
    sentence_lenght = len(typed_sentences_of_same_lenght[0].sent.tokens)
    avg_cross_entropies_by_token_position: Dict[int, float] = {
        n: 0 for n in range(sentence_lenght)
    }
    for typed_sentence in typed_sentences_of_same_lenght:
        sentence = typed_sentence.sent
        # text_len = len(sentence.tokens)
        (
            lp_softmax,
            lp_logistic,
            score_per_masking,
            logistic_score_per_masking,
        ) = get_sentence_acceptability_score(
            model_type,
            model,
            tokenizer,
            sentence,
            device,
            at_once_mask_all_tokens_of_a_word=False,
        )

        # print the score for each token in the sentence (its prediction when it's masked)
        print(f"Printing per token scores for sentence --'{sentence.txt}'")
        print(
            f"sentence.tokens: {sentence.tokens} \n score_per_masking: {score_per_masking}"
        )
        if score_per_masking is not None:
            for token_pos, (token, score) in enumerate(
                zip(sentence.tokens, score_per_masking)
            ):
                print(f"{score:.3f} {token}")
                avg_cross_entropies_by_token_position[token_pos] += score / len(
                    typed_sentences_of_same_lenght
                )
        else:
            print(f"lp_softmax: {lp_softmax:.3f} ")
        # plt.plot(score_per_masking, marker="o", linestyle="dashed")
        # plt.show()
    # todo: separately for each island phenomenon, chose a sentence lenght, and plot two plots, one for acceptable and one
    # for unacceptable sentences. Compare differences

    # check that the cross-entropy calculation is correct, by comparing it with the paper results
    # problem: does not reproduce results that bert with PLL outperforms Gpt2

    # note on the thesis: need to control for sentence lenght and words frequencies ..when comparing a minimal pair
    # this is not always the case in out "minimal" pairs
    # try the different alpha coefficient (1 instead of 0.8). 1 should work better for bert, 0.6 for gpt, for the reasons
    # outlined in the paper
    # reproduce values for the same sentence exampled in the paper (ie S.Francisco) with the same models
    # try results with logistic fun instead of softmax, see the fluctuation of values in a sentences from token to token

    # todo: add to plot: model name, datasource descr and size
    # fixe x axis int numbers
    # plot tokens (words/subwords) as label of x axis

    # todo: run tests on whole Blimp dataset on island effects, compare with gpt2, bert and roberta results
    # run on the server

    # todo: analize (..quantitatively) the island effects with the peaks in score on the individual tokens of the sentencce
    #  see which parts of the sentences generate peaks, if it's ..rare words, or constructs, etc.
    # do it for our dataset, sprouse datasets english and italian, blimp dataset (cola dataset?)

    # use the ranking of sentences, Plot (with the spikes) the worst classified sentences,
    # those at the limit (diff btw acceptable and unacceptable close to zero).
    # subplots all in one colum, all with the same range of y values (so the spikes are evident)
    # x axis with the token names

    # todo: compare plts of variations sentences: eg chi si domanda se io/tu/Gianni ..
    # in this case, sentence lenght and token frequencies are controlled,
    # but are the plots significantly different? (different positions and number of the spikes)

    # todo: dynamic plots (on_update), to enter sentences, get them tokenized and scored and plotted https://towardsdatascience.com/intro-to-dynamic-visualization-with-python-animations-and-interactive-plots-f72a7fb69245

    plt.plot(
        avg_cross_entropies_by_token_position.values(),
        marker="o",
        linestyle="dashed",
        color="blue",
    )

    plt.show()
    # datapooints = zip(*sorted(avg_cross_entropies_by_token_position.items()))
    # plt.plot(datapooints, marker='o', linestyle='dashed')
    # plt.show()


def print_model_scores_numerical_properties():

    # testset_filenames = ["wh_island"]  # "wh_island"  # "wh_whether_island"
    # p = get_test_data_dir() / "blimp"
    # p = get_syntactic_tests_dir() / "sprouse/"
    # model_name = "bert-base-uncased"
    # model_type = ModelTypes.BERT
    # experimental_design=ExperimentalDesigns.MINIMAL_PAIRS

    config = "mro"  # "mro", "sprouse",  "blimp"

    if config == "blimp":

        testset_filenames = [
            "wh_island",
            "adjunct_island",
            "complex_NP_island",
        ]
        p = get_syntactic_tests_dir() / "blimp/from_blim_en/islands/"

        model_type = ModelTypes.ROBERTA  # ModelTypes.BERT
        model_name = "roberta-large"  # "bert-base-cased"

        experimental_design = ExperimentalDesigns.MINIMAL_PAIRS
        datasource = DataSources.BLIMP_EN
        max_examples = 9999

    elif config == "mro":

        testset_filenames = [
            "wh_whether_island",
            "wh_subject_islands",
            "wh_complex_np_islands",
            "wh_adjunct_islands",
        ]
        p = get_syntactic_tests_dir() / "mdd2/"  # "syntactic_tests_it/"

        model_name = "dbmdz/bert-base-italian-xxl-cased"
        model_type = ModelTypes.BERT
        # model_name = "idb-ita/gilberto-uncased-from-camembert"  #
        # model_type = ModelTypes.GILBERTO  # ModelTypes.BERT

        # todo:
        # inspect the loss from gpt2/geppetto ..

        experimental_design = ExperimentalDesigns.FACTORIAL
        datasource = DataSources.MADEDDU  # DataSources.BLIMP_EN,
        max_examples = 9999

    testset_dir_path = str(p)
    print_model_scores_numerical_properties_helper(
        testset_filenames,
        testset_dir_path,
        model_name,
        model_type,
        experimental_design,
        datasource,
        max_examples=max_examples,
    )


def online_test():
    # models: english roberta large and gpt2 (345M)
    sentence_txt = "San Francisco"
    typed_sentence = parse_typed_sentence(SentenceNames.SENTENCE_GOOD, sentence_txt)
    print(f"typed_sentence.sent.txt: {typed_sentence.sent.txt}")

    device = DEVICES.CUDA_0
    # online_test_helper_roberta(typed_sentence, device)

    model_type = ModelTypes.GPT
    model_name = "gpt2-medium"  # 355M params
    model, tokenizer = load_model(model_type, model_name, device)
    print(
        f"tokenizer.cls_token: {tokenizer.cls_token}, tokenizer.sep_token: {tokenizer.sep_token}, tokenizer.mask_token: {tokenizer.mask_token}"
    )
    typed_sentence.sent.tokens = tokenizer.tokenize(typed_sentence.sent.txt)
    print_sentences_scores([typed_sentence], model_type, model, tokenizer, device)

    # todo: int test, loading tokenizers and checking their special tokens for different models

    # todo: check logits for non masked words

    # sentence.tokens: ['San', 'ĠFrancisco']

    # model: roberta-large
    # PLL score_per_masking: [0.006103568, 0.999546]
    # 0.006 San
    # 1.000 ĠFrancisco
    # Paper: -0.006 + (-1.000) = -1.006 PLL_roberta(W)

    # model: gpt-medium
    # lp_softmax: -8.693
    # Paper: -7.749 + (-0.944) = -8.693 = log Pgpt2(W)


def online_test_helper_roberta(typed_sentence, device):

    model_type = ModelTypes.ROBERTA
    model_name = "roberta-large"
    model, tokenizer = load_model(model_type, model_name, device)
    print(
        f"tokenizer.cls_toke: {tokenizer.cls_token}, tokenizer.sep_token: {tokenizer.sep_token}, tokenizer.mask_token: {tokenizer.mask_token}"
    )
    typed_sentence.sent.tokens = tokenizer.tokenize(typed_sentence.sent.txt)
    print(f"typed_sentence.sent.tokens: {typed_sentence.sent.tokens}")
    print_sentences_scores([typed_sentence], model_type, model, tokenizer, device)


if __name__ == "__main__":
    print_model_scores_numerical_properties()
    # online_test()
