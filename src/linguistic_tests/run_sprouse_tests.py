import os.path
import time
from typing import List

from linguistic_tests.compute_model_score import print_accuracy_scores
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.file_utils import parse_testsets
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_results_dir
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.testset import Example
from linguistic_tests.testset import load_testset_from_pickle
from linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
from linguistic_tests.testset import TestSet
from matplotlib import pyplot as plt
from tqdm import tqdm


model_names_it = {
    ModelTypes.GEPPETTO: "LorenzoDeMattei/GePpeTto",
    ModelTypes.BERT: "dbmdz/bert-base-italian-xxl-cased",
    ModelTypes.GILBERTO: "idb-ita/gilberto-uncased-from-camembert",
}  # ModelTypes.GPT # ModelTypes.ROBERTA  #

model_names_en = {
    ModelTypes.BERT: "bert-base-uncased",  # "bert-large-uncased"  #
    ModelTypes.GPT: "gpt2-large",
    ModelTypes.ROBERTA: "roberta-large",
}

sprouse_testsets_root_filenames = [  # 'rc_adjunct_island',
    # 'rc_complex_np', 'rc_subject_island', 'rc_wh_island', # fixme: rc_wh_island empty file
    "wh_adjunct_island",
    "wh_complex_np",
    "wh_subject_island",
    "wh_whether_island",
]
custom_it_island_testsets_root_filenames = [
    # "wh_adjunct_islands",
    # "wh_complex_np_islands",
    # "wh_whether_island",
    # "wh_subject_islands",
    "wh_whether_island",
    "wh_complex_np_islands",
    "wh_subject_islands",
    "wh_adjunct_islands",
]

# todo: parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file


def score_sprouse_testsets(
    model_type,
    model,
    tokenizer,
    device,
    parsed_testsets: List[TestSet],
) -> list[TestSet]:
    # todo: compare results (for each phenomena) on the 8 original Sprouse sentences, and the new 50 italian ones

    # todo: see activation levels in the model layers, try to identify several phenomena: clause segmentation,
    #  different constructs, long vs short dependencies, wh vs rc dependencies, islands vs non islands

    # todo: see if the pretrained models by Bostrom et al. 2020 perform better (on Sprouse and Blimp english test data )
    #  when they use more linguistically plausible subwords units for tokenization.

    # todo: add a max_examples variable to limit the tested examples to a fixed number, while still having more for some phenomena

    # testset_filepath = get_out_dir() + "blimp/from_blim_en/islands/complex_NP_island.jsonl"  # wh_island.jsonl' # adjunct_island.jsonl'

    scored_testsets = []
    for parsed_testset in parsed_testsets:

        scored_testset = score_sprouse_testset(
            model_type,
            model,
            tokenizer,
            device,
            parsed_testset,
        )
        scored_testsets.append(scored_testset)

    return scored_testsets


def plot_results(scored_testsets: list[TestSet], score_name):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))  # default figsize=(6.4, 4.8)

    window_title = f"{scored_testsets[0].dataset_source[:7]}_{scored_testsets[0].model_descr}_{score_name}"
    window_title = window_title.replace(" ", "_").replace("/", "_")

    fig.canvas.manager.set_window_title(window_title)
    axs_list = axs.reshape(-1)
    print(f"type axs_list: {type(axs_list)}, {len(axs_list)=}, {axs_list=}")

    preferred_axs_order = {"whether": 0, "complex": 1, "subject": 2, "adjunct": 3}
    for phenomenon_short_name, preferred_index in preferred_axs_order.items():
        if (
            phenomenon_short_name
            not in scored_testsets[preferred_index].linguistic_phenomenon
        ):
            for idx, testset in enumerate(scored_testsets):
                if phenomenon_short_name in testset.linguistic_phenomenon:
                    scored_testsets.remove(testset)
                    scored_testsets.insert(preferred_index, testset)
                    break

    for scored_testset, ax in zip(scored_testsets, axs_list):
        _plot_results_subplot(scored_testset, score_name, ax)

    fig.suptitle(
        f"Model: {scored_testsets[0].model_descr}, "
        f"\n Dataset: {scored_testset.dataset_source}"
    )

    filename = f"{window_title}.png"
    saving_dir = str(get_results_dir())
    filepath = os.path.join(saving_dir, filename)
    if os.path.exists(filepath):
        timestamp = time.strftime("%Y-%m-%d_h%Hm%Ms%S")
        filename = f"{window_title}-{timestamp}.png"
        filepath = os.path.join(saving_dir, filename)
    print_orange(f"Saving plot to file {filepath} ..")
    plt.savefig(filepath)  # , dpi=300
    # plt.show()


def _plot_results_subplot(scored_testset: TestSet, score_name, ax):
    # plt.subplot(2, 2, testset_idx)
    # plt.plot(x, y)

    # todo: normalize the scores centering them to 0 like the z scores in the paper
    DD_value = scored_testset.get_avg_DD(score_name)
    ax.legend(title=f"DD = {DD_value:.2f}")

    # todo? in the legend also plot p value across all the testset examples
    # in the legend also plot accuracy %
    score_averages = scored_testset.get_avg_scores(score_name)

    # nonisland line
    short_nonisland_average = [0, score_averages[SentenceNames.SHORT_NONISLAND]]
    long_nonisland_avg = [1, score_averages[SentenceNames.LONG_NONISLAND]]
    x_values = ["SHORT", "LONG"]
    y_values = [short_nonisland_average[1], long_nonisland_avg[1]]
    # ax.set_ylim([-32.5, -26.5])  # todo: set limits as min/max across all testsets
    ax.plot(x_values, y_values, label="non-island structure")

    # island line
    short_island_avg = [0, score_averages[SentenceNames.SHORT_ISLAND]]
    long_island_avg = [1, score_averages[SentenceNames.LONG_ISLAND]]
    x_values = [short_island_avg[0], long_island_avg[0]]
    y_values = [short_island_avg[1], long_island_avg[1]]
    ax.plot(x_values, y_values, linestyle="--", label="island structure")
    ax.set_title(scored_testset.linguistic_phenomenon)
    ax.set_ylabel(f"{score_name} values")


def score_sprouse_testset(
    model_type, model, tokenizer, device, testset: TestSet
) -> TestSet:
    # todo set scorebase param
    for example_idx, example in enumerate(tqdm(testset.examples)):
        score_example(
            device,
            example,
            model,
            model_type,
            tokenizer,
        )

        (
            example.DD_with_lp,
            example.DD_with_penlp,
            example.dd_with_ll,
            example.dd_with_pll,
        ) = get_dd_scores_wdataclasses(example)
        if example.DD_with_lp > 0:
            testset.accuracy_by_DD_lp += 1 / len(testset.examples)
        if example.DD_with_penlp > 0:
            testset.accuracy_by_DD_penlp += 1 / len(testset.examples)
        if example.DD_with_ll > 0:
            testset.accuracy_by_DD_ll += 1 / len(testset.examples)
        if example.DD_with_penll > 0:
            testset.accuracy_by_DD_penll += 1 / len(testset.examples)

        for _idx, typed_sentence in enumerate(example.sentences):
            stype = typed_sentence.stype
            sentence = typed_sentence.sent

            testset.lp_average_by_sentence_type[stype] += sentence.lp
            testset.penlp_average_by_sentence_type[stype] += sentence.pen_lp
            if model_type in BERT_LIKE_MODEL_TYPES:
                testset.ll_average_by_sentence_type[stype] += sentence.log_logistic
                testset.penll_average_by_sentence_type[
                    stype
                ] += sentence.pen_log_logistic

    for stype in testset.lp_average_by_sentence_type.keys():
        testset.lp_average_by_sentence_type[stype] /= len(testset.examples)
        testset.penlp_average_by_sentence_type[stype] /= len(testset.examples)
        if model_type in BERT_LIKE_MODEL_TYPES:
            testset.ll_average_by_sentence_type[stype] /= len(testset.examples)
            testset.penll_average_by_sentence_type[stype] /= len(testset.examples)

    # todo: lp scores should be normalized across the whole testset
    testset.avg_DD_lp = get_dd_score_parametric(
        a_short_nonisland_score=testset.lp_average_by_sentence_type[
            SentenceNames.SHORT_NONISLAND
        ],
        b_long_nonisland_score=testset.lp_average_by_sentence_type[
            SentenceNames.LONG_NONISLAND
        ],
        c_short_island_score=testset.lp_average_by_sentence_type[
            SentenceNames.SHORT_ISLAND
        ],
        d_long_island_score=testset.lp_average_by_sentence_type[
            SentenceNames.LONG_ISLAND
        ],
    )
    testset.avg_DD_penlp = get_dd_score_parametric(
        a_short_nonisland_score=testset.penlp_average_by_sentence_type[
            SentenceNames.SHORT_NONISLAND
        ],
        b_long_nonisland_score=testset.penlp_average_by_sentence_type[
            SentenceNames.LONG_NONISLAND
        ],
        c_short_island_score=testset.penlp_average_by_sentence_type[
            SentenceNames.SHORT_ISLAND
        ],
        d_long_island_score=testset.penlp_average_by_sentence_type[
            SentenceNames.LONG_ISLAND
        ],
    )
    if model_type in BERT_LIKE_MODEL_TYPES:
        testset.avg_DD_ll = get_dd_score_parametric(
            a_short_nonisland_score=testset.ll_average_by_sentence_type[
                SentenceNames.SHORT_NONISLAND
            ],
            b_long_nonisland_score=testset.ll_average_by_sentence_type[
                SentenceNames.LONG_NONISLAND
            ],
            c_short_island_score=testset.ll_average_by_sentence_type[
                SentenceNames.SHORT_ISLAND
            ],
            d_long_island_score=testset.ll_average_by_sentence_type[
                SentenceNames.LONG_ISLAND
            ],
        )
        testset.avg_DD_penll = get_dd_score_parametric(
            a_short_nonisland_score=testset.penll_average_by_sentence_type[
                SentenceNames.SHORT_NONISLAND
            ],
            b_long_nonisland_score=testset.penll_average_by_sentence_type[
                SentenceNames.LONG_NONISLAND
            ],
            c_short_island_score=testset.penll_average_by_sentence_type[
                SentenceNames.SHORT_ISLAND
            ],
            d_long_island_score=testset.penll_average_by_sentence_type[
                SentenceNames.LONG_ISLAND
            ],
        )

    for scoring_measure in testset.accuracy_per_score_type_per_sentence_type.keys():
        for (
            stype_acceptable_sentence
        ) in testset.accuracy_per_score_type_per_sentence_type[scoring_measure].keys():
            accurate_count = 0
            for example_idx, example in enumerate(tqdm(testset.examples)):
                if example.is_scored_accurately(
                    scoring_measure, stype_acceptable_sentence
                ):
                    accurate_count += 1
            accuracy = accurate_count / len(testset.examples)
            testset.accuracy_per_score_type_per_sentence_type[scoring_measure][
                stype_acceptable_sentence
            ] = accuracy

    return testset


def print_example(example_data, sentence_ordering):
    print(
        f"sentence ordering is {type(sentence_ordering)}"
        f"\nSHORT_NONISLAND: {example_data[sentence_ordering.SHORT_NONISLAND]}"
        f"\nLONG_NONISLAND : {example_data[sentence_ordering.LONG_NONISLAND]}"
        f"\nSHORT_ISLAND : {example_data[sentence_ordering.SHORT_ISLAND]}"
        f"\nLONG_ISLAND : {example_data[sentence_ordering.LONG_ISLAND]}"
    )


def get_dd_scores_wdataclasses(example):

    # todo, fixme: ddscore should be normalized across the example and across the testset
    #  (according to min and max token weights)
    #  store absolute values and normalized values

    example_dd_with_lp = get_example_dd_score(example, ScoringMeasures.LP)
    example_dd_with_penlp = get_example_dd_score(example, ScoringMeasures.PenLP)
    example_dd_with_ll = get_example_dd_score(example, ScoringMeasures.LL)
    example_dd_with_pll = get_example_dd_score(example, ScoringMeasures.PLL)

    return (
        example_dd_with_lp,
        example_dd_with_penlp,
        example_dd_with_ll,
        example_dd_with_pll,
    )


def get_example_dd_score(example: Example, score_name):
    for typed_sentence in example.sentences:
        stype = typed_sentence.stype
        sent = typed_sentence.sent
        if stype == SentenceNames.SHORT_NONISLAND:
            a_short_nonisland = sent
        elif stype == SentenceNames.LONG_NONISLAND:
            b_long_nonisland = sent
        elif stype == SentenceNames.SHORT_ISLAND:
            c_short_island = sent
        elif stype == SentenceNames.LONG_ISLAND:
            d_long_island = sent
        else:
            raise ValueError(f"Unexpected sentence type: {stype}")

    # todo, fixme: use normalized scores (normalized according to min max token weight across testset)
    return get_dd_score_parametric(
        a_short_nonisland.get_score(score_name),
        b_long_nonisland.get_score(score_name),
        c_short_island.get_score(score_name),
        d_long_island.get_score(score_name),
    )


def get_dd_score_parametric(
    a_short_nonisland_score,
    b_long_nonisland_score,
    c_short_island_score,
    d_long_island_score,
):
    example_lenght_effect = a_short_nonisland_score - b_long_nonisland_score
    example_structure_effect = a_short_nonisland_score - c_short_island_score
    example_total_effect = a_short_nonisland_score - d_long_island_score
    example_island_effect = example_total_effect - (
        example_lenght_effect + example_structure_effect
    )
    example_dd = example_structure_effect - (
        b_long_nonisland_score - d_long_island_score
    )

    example_dd *= -1
    assert_almost_equale(example_island_effect, example_dd)
    return example_dd


def get_dd_score(sentences_scores, sentences_ordering=SprouseSentencesOrder):
    a_short_nonisland_idx = sentences_ordering.SHORT_NONISLAND
    b_long_nonisland = sentences_ordering.LONG_NONISLAND
    c_short_island = sentences_ordering.SHORT_ISLAND
    d_long_island = sentences_ordering.LONG_ISLAND
    example_lenght_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[b_long_nonisland]
    )
    example_structure_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[c_short_island]
    )
    example_total_effect_with_lp = (
        sentences_scores[a_short_nonisland_idx] - sentences_scores[d_long_island]
    )
    example_island_effect_with_lp = example_total_effect_with_lp - (
        example_lenght_effect_with_lp + example_structure_effect_with_lp
    )
    example_dd_with_lp = example_structure_effect_with_lp - (
        sentences_scores[b_long_nonisland] - sentences_scores[d_long_island]
    )
    example_dd_with_lp *= -1
    assert_almost_equale(example_island_effect_with_lp, example_dd_with_lp)
    return example_dd_with_lp


def assert_almost_equale(val1, val2, precision=14):
    assert abs(val1 - val2) < 10 ** (-1 * precision), (
        f"val1:{val1}, val2: {val2}, " f"diff: {val1 - val2}"
    )


def plot_all_phenomena(phenomena_names, lp_avg_scores):
    for idx, phenomenon in enumerate(phenomena_names):
        plot_results(phenomenon, lp_avg_scores[idx], ScoringMeasures.LP.name)


def save_scored_testsets(
    scored_testsets: List[TestSet], model_name: str, broader_test_type: str
):
    for scored_testset in scored_testsets:
        scored_testset.model_descr = model_name
        filename = get_pickle_filename(
            scored_testset.linguistic_phenomenon,
            model_name,
            broader_test_type=broader_test_type,
        )

        scored_testset.save_to_pickle(filename)


def get_pickle_filename(
    linguistic_phenomenon,
    model_name,
    broader_test_type,
):
    # todo: filenames as pyplot filenames
    #  rename as get_pickle_filepath, ad results dir (same as pyplot images)
    filename = (
        f"{broader_test_type}_"
        f"{linguistic_phenomenon}_"
        f"{model_name.replace('/', '_')}.testset.pickle"
    )
    return filename


def load_pickles(phenomena, model_name, broader_test_type) -> list[TestSet]:
    # phenomena = [
    #     "custom-wh_whether_island",
    #     "custom-wh_complex_np_islands",
    #     "custom-wh_subject_islands",
    #     "custom-wh_adjunct_islands",
    # ]
    loaded_testsets = []
    for phenomenon in phenomena:
        filename = get_pickle_filename(phenomenon, model_name, broader_test_type)
        loaded_testset = load_testset_from_pickle(filename)
        loaded_testsets.append(loaded_testset)

    return loaded_testsets


def load_and_plot_pickle(
    phenomena,
    model_name,
    broader_test_type,
    model_type: ModelTypes,
    loaded_testsets=None,
):

    if loaded_testsets is None:
        loaded_testsets = load_pickles(phenomena, model_name, broader_test_type)

    plot_testsets(loaded_testsets, model_type)


def plot_testsets(scored_testsets: List[TestSet], model_type: ModelTypes):
    plot_results(scored_testsets, ScoringMeasures.LP.name)
    plot_results(scored_testsets, ScoringMeasures.PenLP.name)

    if model_type in BERT_LIKE_MODEL_TYPES:
        plot_results(scored_testsets, ScoringMeasures.LL.name)
        plot_results(scored_testsets, ScoringMeasures.PLL.name)


def print_sorted_sentences_to_check_spelling_errors2(
    score_descr, phenomena, model_name, broader_test_type, loaded_testsets=None
):

    if loaded_testsets is None:
        loaded_testsets = load_pickles(phenomena, model_name, broader_test_type)

    for testset in loaded_testsets:
        print(
            f"\nprinting for testset {testset.linguistic_phenomenon} calculated from {testset.model_descr}"
        )
        typed_sentences = testset.get_all_sentences_sorted_by_score(
            score_descr, reverse=False
        )
        print(f"{'idx':<5}" f"{score_descr:<10}" f"{'stype':<20}" f"{'txt':<85}")
        for idx, tsent in enumerate(typed_sentences):
            print(
                f"{idx : <5}"
                f"{tsent.sent.get_score(score_descr) : <10.2f}"
                f"{tsent.stype : <20}"
                f"{tsent.sent.txt : <85}"
            )


def print_sorted_sentences_to_check_spelling_errors(
    score_descr, phenomena, model_name, broader_test_type, loaded_testsets=None
):
    print("printing sorted_sentences_to_check_spelling_errors")

    if loaded_testsets is None:
        loaded_testsets = load_pickles(phenomena, model_name, broader_test_type)

    for testset in loaded_testsets:
        print(
            f"\nprinting for testset {testset.linguistic_phenomenon} calculated from {testset.model_descr}"
        )
        for stype in SPROUSE_SENTENCE_TYPES:
            print(f"printing for sentence type {stype}..")
            examples = testset.get_examples_sorted_by_sentence_type_and_score(
                stype, score_descr, reverse=False
            )
            print(f"{'idx':<5}" f"{score_descr:^10}" f"{'txt':<85}")
            for idx, example in enumerate(examples):
                print(
                    f"{idx : <5}"
                    f"{example[stype].get_score(score_descr) : <10.2f}"
                    f"{example[stype].txt : <85}"
                )


def print_examples_compare_diff(
    score_descr,
    sent_type1,
    sent_type2,
    phenomena,
    model_name,
    broader_test_type,
    testsets=None,
):
    if testsets is None:
        testsets = load_pickles(phenomena, model_name, broader_test_type)

    max_testsets = 4
    for testset in testsets[:max_testsets]:
        print(
            f"\nprinting testset for {testset.linguistic_phenomenon} from {testset.model_descr}"
        )
        print(f"comparing {sent_type1} and {sent_type2}")
        examples = testset.get_examples_sorted_by_score_diff(
            score_descr, sent_type1, sent_type2, reverse=False
        )
        max_prints = 50
        print(
            f"{'diff':<8}"
            f"{'s1 '+score_descr:^10}"
            f"{'s1 txt (' + sent_type1 + ')':<95}"
            f"{'s2 '+score_descr:^10}"
            f"{'s2 txt (' + sent_type2 + ')':<5}"
        )
        for example in examples[0:max_prints]:
            print(
                f"{example.get_score_diff(score_descr, sent_type1, sent_type2) : <8.2f}"
                f"{example[sent_type1].get_score(score_descr) : ^10.2f}"
                f"{example[sent_type1].txt : <95}"
                f"{example[sent_type2].get_score(score_descr) :^10.2f}"
                f"{example[sent_type2].txt : <5}"
            )


def print_testset_results(
    scored_testsets: List[TestSet],
    broader_test_type: str,
    model_type: ModelTypes,
    testsets_root_filenames: List[str],
):
    print("Printing accuracy scores..")
    for scored_testset in scored_testsets:
        # todo: also print results in table format or csv for excel export or word doc report
        print_accuracy_scores(scored_testset)
        print(
            f"Testset accuracy with DDs_with_lp: {scored_testset.accuracy_by_DD_lp:%}"
        )
        print(
            f"Testset accuracy with DDs_with_penlp: {scored_testset.accuracy_by_DD_penlp:%}"
        )
        lp_averages = scored_testset.lp_average_by_sentence_type
        penlp_averages = scored_testset.penlp_average_by_sentence_type
        print(f"{lp_averages=}")
        print(f"{penlp_averages=}")

        if model_type in BERT_LIKE_MODEL_TYPES:
            print(
                f"Testset accuracy with DDs_with_ll: {scored_testset.accuracy_by_DD_ll:%}"
            )
            print(
                f"Testset accuracy with DDs_with_penll: {scored_testset.accuracy_by_DD_penll:%}"
            )
            ll_averages = scored_testset.ll_average_by_sentence_type
            penll_averages = scored_testset.penll_average_by_sentence_type
            print(f"{ll_averages=}")
            print(f"{penll_averages=}")

    score_descr = ScoringMeasures.PenLP.name

    print_sorted_sentences_to_check_spelling_errors2(
        score_descr,
        testsets_root_filenames,
        model_names_it[model_type],
        broader_test_type,
        scored_testsets,
    )
    print_sorted_sentences_to_check_spelling_errors(
        score_descr,
        testsets_root_filenames,
        model_names_it[model_type],
        broader_test_type,
        scored_testsets,
    )

    print_examples_compare_diff(
        score_descr,
        SentenceNames.SHORT_ISLAND,
        SentenceNames.LONG_ISLAND,
        testsets_root_filenames,
        model_names_it[model_type],
        broader_test_type,
        testsets=scored_testsets,
    )
    print_examples_compare_diff(
        score_descr,
        SentenceNames.LONG_NONISLAND,
        SentenceNames.LONG_ISLAND,
        testsets_root_filenames,
        model_names_it[model_type],
        broader_test_type,
        testsets=scored_testsets,
    )
    print_examples_compare_diff(
        score_descr,
        SentenceNames.SHORT_NONISLAND,
        SentenceNames.SHORT_ISLAND,
        testsets_root_filenames,
        model_names_it[model_type],
        broader_test_type,
        testsets=scored_testsets,
    )


def rescore_testsets_and_save_pickles(
    model_type,
    broader_test_type,
    testset_dir_path,
    testsets_root_filenames,
    dataset_source,
):
    model_name = model_names_it[model_type]
    examples_format = "sprouse"  # "blimp", "json_lines", "sprouse"
    sent_types_descr = "sprouse"  # "blimp" or "sprouse"
    # sentence_ordering = SprouseSentencesOrder  # BlimpSentencesOrder
    print(f"Running testsets from dir {testset_dir_path}")
    scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
    if model_type in BERT_LIKE_MODEL_TYPES:
        scoring_measures += [ScoringMeasures.LL, ScoringMeasures.PLL]
    parsed_testsets = parse_testsets(
        # todo: add scorebase var in testset class
        testset_dir_path,
        testsets_root_filenames,
        dataset_source,
        examples_format,
        sent_types_descr,
        model_name,
        model_type,
        scoring_measures,
        max_examples=1000,
    )

    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)

    scored_testsets = score_sprouse_testsets(
        model_type,
        model,
        tokenizer,
        device,
        parsed_testsets,
    )
    save_scored_testsets(scored_testsets, model_name, broader_test_type)


def get_testset_params(tests_subdir):
    if tests_subdir == "syntactic_tests_it/":
        testsets_root_filenames = custom_it_island_testsets_root_filenames
        broader_test_type = "it_tests"
        dataset_source = "Madeddu (50 items per phenomenon)"
    elif tests_subdir == "sprouse/":
        testsets_root_filenames = sprouse_testsets_root_filenames
        broader_test_type = "sprouse"
        dataset_source = "Sprouse et al. 2016 (8 items per phenomenon)"

    return testsets_root_filenames, broader_test_type, dataset_source


def main():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_types",
        help=f"specify the models to run. { {i.name: i.value for i in ModelTypes} }",
        nargs="+",  # 0 or more values expected => creates a list
        type=int,
        choices=[i.value for i in model_names_it.keys()],
        default=[i.value for i in model_names_it.keys()],
    )
    args = arg_parser.parse_args()
    model_types_to_run = [
        ModelTypes(model_type_int) for model_type_int in args.model_types
    ]
    print(f"Will run tests with models: {model_types_to_run}")

    # todo: also add command line option for tests subdir path
    tests_subdir = "syntactic_tests_it/"  # "sprouse/"  #
    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

    testsets_root_filenames, broader_test_type, dataset_source = get_testset_params(
        tests_subdir
    )

    for model_type in model_types_to_run:
        print_orange(f"Starting test session for {model_type=}, and {dataset_source=}")
        # add score with logistic function (instead of softmax)

        # todo: check that accuracy values are scored and stored correctly
        #  (it seems they are scored twice and not shown when loading pickles)
        # save results to csv (for import in excel table)
        # autosave plots as *.png

        # create_test_jsonl_files_tests()

        rescore = True
        if rescore:
            rescore_testsets_and_save_pickles(
                model_type,
                broader_test_type,
                testset_dir_path,
                testsets_root_filenames,
                dataset_source,
            )

        loaded_testsets = load_pickles(
            testsets_root_filenames, model_names_it[model_type], broader_test_type
        )

        print_testset_results(
            loaded_testsets, broader_test_type, model_type, testsets_root_filenames
        )

        load_and_plot_pickle(
            testsets_root_filenames,
            model_names_it[model_type],
            broader_test_type,
            model_type,
            loaded_testsets=loaded_testsets,
        )
        print_orange(f"Finished test session for {model_type=}")

    # todo:
    # plot with 7+1x7 subplots of a testset (one subplot for each example)

    # todo:
    # normalize sentence/tokens scores for Bert models, to have scores comparables across a whole testset
    #
    # normalize results to a likert scale, for comparison with Sprouse et al 2016


if __name__ == "__main__":
    main()
