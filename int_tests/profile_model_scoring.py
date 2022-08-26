from typing import Dict
from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.linguistic_tests.compute_model_score import get_sentence_acceptability_score
from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import get_syntactic_tests_dir
from src.linguistic_tests.lm_utils import load_model
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import ScoringMeasures
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.testset import parse_testsets
from src.linguistic_tests.testset import parse_typed_sentence
from src.linguistic_tests.testset import TypedSentence


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
):

    examples_format = "json_lines"
    device = DEVICES.CPU

    max_examples = 9999
    # rescore_testsets_and_save_pickles(...)

    parsed_testsets = parse_testsets(
        testset_dir_path,
        testset_filenames,
        DataSources.BLIMP_EN,
        examples_format,
        experimental_design,
        model_name,
        scoring_measures=[ScoringMeasures.LP, ScoringMeasures.PenLP],
        max_examples=max_examples,
    )

    model, tokenizer = load_model(model_type, model_name, device)

    all_acceptable_sentences = []
    sentence_lenght_statistics: Dict[
        int, int
    ] = dict()  # sentence lenght, how many with that lenght
    for parsed_testset in parsed_testsets:
        for example_idx, example in enumerate(tqdm(parsed_testset.examples)):

            for _idx, typed_sentence in enumerate(example.sentences):

                if (
                    typed_sentence.stype
                    is not example.get_type_of_unacceptable_sentence()
                ):
                    all_acceptable_sentences.append(typed_sentence)

                    sentence = typed_sentence.sent
                    sentence.tokens = tokenizer.tokenize(
                        sentence.txt
                    )  # , return_tensors='pt'
                    if len(sentence.tokens) not in sentence_lenght_statistics:
                        sentence_lenght_statistics[len(sentence.tokens)] = 0
                    sentence_lenght_statistics[len(sentence.tokens)] += 1

    print(f"sentence_lenght_statistics={sorted(sentence_lenght_statistics.items())}")
    most_common_sentence_lenght = (
        19  # max(sentence_lenght_statistics, key=sentence_lenght_statistics.get)
    )
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
        lp_softmax, lp_logistic, score_per_masking = get_sentence_acceptability_score(
            model_type, model, tokenizer, sentence.tokens, device
        )

        # print the score for each token in the sentence (its prediction when it's masked)
        print(f"Printing per token scores for sentence --'{sentence.txt}'")
        print(
            f"sentence.tokens: {sentence.tokens} \n score_per_masking: {score_per_masking}"
        )
        for token_pos, (token, score) in enumerate(
            zip(sentence.tokens, score_per_masking)
        ):
            print(f"{score:.3f} {token}")
            avg_cross_entropies_by_token_position[token_pos] += score / len(
                typed_sentences_of_same_lenght
            )
        plt.plot(score_per_masking, marker="o", linestyle="dashed")
        plt.show()
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
    # p = get_syntactic_tests_dir() / "blimp/from_blim_en/islands/"
    # p = get_syntactic_tests_dir() / "sprouse/"
    # model_name = "bert-base-uncased"
    # model_type = ModelTypes.BERT
    # experimental_design=ExperimentalDesigns.MINIMAL_PAIRS

    testset_filenames = [
        "wh_whether_island",
        "wh_subject_islands",
        "wh_complex_np_islands",
        "wh_adjunct_islands",
    ]
    p = get_syntactic_tests_dir() / "syntactic_tests_it/"
    model_name = "dbmdz/bert-base-italian-xxl-cased"
    model_type = ModelTypes.BERT
    experimental_design = ExperimentalDesigns.FACTORIAL

    testset_dir_path = str(p)
    print_model_scores_numerical_properties_helper(
        testset_filenames,
        testset_dir_path,
        model_name,
        model_type,
        experimental_design,
    )


def online_test():
    # models: english roberta large and gpt2 (345M)
    sentence_txt = "San Francisco"
    typed_sentence = parse_typed_sentence(SentenceNames.SENTENCE_GOOD, sentence_txt)
    print(f"typed_sentence.sent.txt: {typed_sentence.sent.txt}")

    device = DEVICES.CPU

    model_type = ModelTypes.ROBERTA
    model_name = "roberta-large"
    model, tokenizer = load_model(model_type, model_name, device)
    print(
        f"tokenizer.cls_toke: {tokenizer.cls_token}, tokenizer.sep_token: {tokenizer.sep_token}, tokenizer.mask_token: {tokenizer.mask_token}"
    )
    typed_sentence.sent.tokens = tokenizer.tokenize(typed_sentence.sent.txt)
    print(f"typed_sentence.sent.tokens: {typed_sentence.sent.tokens}")
    print_sentences_scores([typed_sentence], model_type, model, tokenizer, device)

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
    # PLL score_per_masking: [0.54014975, 0.43931577]
    # 0.540 San
    # 0.439 ĠFrancisco
    # Paper: -0.006 + (-1.000) = -1.006 PLL_roberta(W)


if __name__ == "__main__":
    # print_model_scores_numerical_properties()
    online_test()
