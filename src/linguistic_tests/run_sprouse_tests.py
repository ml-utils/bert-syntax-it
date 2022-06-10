import os.path

from linguistic_tests.compute_model_score import compute_example_scores_wdataclasses
from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.file_utils import get_file_root
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.testset import Example
from linguistic_tests.testset import load_testset_from_pickle
from linguistic_tests.testset import parse_testset
from linguistic_tests.testset import TestSet
from matplotlib import pyplot as plt
from tqdm import tqdm


# todo: parse the csv file
# 4 sentences for each examples (long vs short, island vs non island)
# turn into 3 examples: island long vs the other 3 sentences
# one file for each phenomena (2x4), ..8x3 examples in each file


def run_sprouse_tests(
    model_type,
    model,
    tokenizer,
    device,
    phenomena=None,
    testset_dir_path=None,
    examples_format="sprouse",
    sentence_ordering=SprouseSentencesOrder,
) -> list[TestSet]:
    # todo: compare results (for each phenomena) on the 8 original Sprouse sentences, and the new 50 italian ones

    # todo: see activation levels in the model layers, try to identify several phenomena: clause segmentation,
    #  different constructs, long vs short dependencies, wh vs rc dependencies, islands vs non islands

    # todo: see if the pretrained models by Bostrom et al. 2020 perform better (on Sprouse and Blimp english test data )
    #  when they use more linguistically plausible subwords units for tokenization.

    # todo: add a max_examples variable to limit the tested examples to a fixed number, while still having more for some phenomena

    # testset_filepath = get_out_dir() + "blimp/from_blim_en/islands/complex_NP_island.jsonl"  # wh_island.jsonl' # adjunct_island.jsonl'
    if phenomena is None:
        phenomena = [  # 'rc_adjunct_island',
            # 'rc_complex_np', 'rc_subject_island', 'rc_wh_island', # fixme: rc_wh_island empty file
            "wh_adjunct_island",
            "wh_complex_np",
            "wh_subject_island",
            "wh_whether_island",
        ]
    if testset_dir_path is None:
        testset_dir_path = str(get_syntactic_tests_dir() / "sprouse/")
    print(f"Running testsets from dir {testset_dir_path}")
    scored_testsets = []
    for phenomenon_name in phenomena:
        print(f"Running testset for {phenomenon_name}..")
        filename = phenomenon_name + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        scored_testset = run_sprouse_test(
            filepath,
            model_type,
            model,
            tokenizer,
            device,
            examples_format=examples_format,
            sentence_ordering=sentence_ordering,
        )
        scored_testsets.append(scored_testset)

    return scored_testsets


def plot_results(scored_testsets, score_name):
    fig, axs = plt.subplots(2, 2)
    axs_list = axs.reshape(-1)
    print(f"type axs_list: {type(axs_list)}, {len(axs_list)=}, {axs_list=}")
    for scored_testset, ax in zip(scored_testsets, axs_list):
        _plot_results_subplot(scored_testset, score_name, ax)

    plt.suptitle(f"Model: {scored_testsets[0].model_descr}")
    plt.show()


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
    x_values = [short_nonisland_average[0], long_nonisland_avg[0]]
    y_values = [short_nonisland_average[1], long_nonisland_avg[1]]
    ax.plot(x_values, y_values, label="non-island structure")

    # island line
    short_island_avg = [0, score_averages[SentenceNames.SHORT_ISLAND]]
    long_island_avg = [1, score_averages[SentenceNames.LONG_ISLAND]]
    x_values = [short_island_avg[0], long_island_avg[0]]
    y_values = [short_island_avg[1], long_island_avg[1]]
    ax.plot(x_values, y_values, linestyle="--", label="island structure")
    ax.set_title(scored_testset.linguistic_phenomenon)
    ax.set_ylabel(f"{score_name} values")


def run_sprouse_test(
    filepath,
    model_type,
    model,
    tokenizer,
    device,
    examples_format="sprouse",
    sentence_ordering=SprouseSentencesOrder,
):
    testset_dict = load_testset_data(filepath, examples_format=examples_format)
    examples_list = testset_dict["sentences"]
    phenomenon_name = get_file_root(filepath)
    parsed_testset = parse_testset(
        phenomenon_name, str(type(model)), examples_list, "sprouse"
    )

    # run_testset(model_type, model, tokenizer, device, testset)
    # lp_averages = run_sprouse_test_helper(
    #     model_type,
    #     model,
    #     tokenizer,
    #     device,
    #     testset,
    #     sentence_ordering=sentence_ordering,
    # )
    scored_testset = run_sprouse_test_helper_wdataclasses(
        model_type,
        model,
        tokenizer,
        device,
        parsed_testset,
    )
    lp_averages = scored_testset.lp_average_by_sentence_type
    print(f"{lp_averages=}")
    return scored_testset


def run_sprouse_test_helper_wdataclasses(
    model_type, model, tokenizer, device, testset: TestSet
):
    for example_idx, example in enumerate(tqdm(testset.examples)):
        compute_example_scores_wdataclasses(
            device,
            example,
            model,
            model_type,
            tokenizer,
        )

        example.DD_with_lp, example.DD_with_penlp = get_dd_scores_wdataclasses(example)
        if example.DD_with_lp > 0:
            testset.accuracy_by_DD_lp += 1 / len(testset.examples)
        if example.DD_with_penlp > 0:
            testset.accuracy_by_DD_penlp += 1 / len(testset.examples)

        for _idx, typed_sentence in enumerate(example.sentences):
            stype = typed_sentence.stype
            sentence = typed_sentence.sent

            testset.lp_average_by_sentence_type[stype] += sentence.lp
            testset.penlp_average_by_sentence_type[stype] += sentence.pen_lp

    for stype in testset.lp_average_by_sentence_type.keys():
        testset.lp_average_by_sentence_type[stype] /= len(testset.examples)
        testset.penlp_average_by_sentence_type[stype] /= len(testset.examples)

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

    print(f"Testset accuracy with DDs_with_lp: {testset.accuracy_by_DD_lp}")
    print(f"Testset accuracy with DDs_with_penlp: {testset.accuracy_by_DD_penlp}")

    return testset


def run_sprouse_test_helper(
    model_type,
    model,
    tokenizer,
    device,
    testset,
    examples_in_sprouse_format=True,
    sentence_ordering=SprouseSentencesOrder,
    max_examples=50,
    verbose=False,
):
    sent_ids = []
    sentences_per_example = 4

    if len(testset["sentences"]) > 50:
        testset["sentences"] = testset["sentences"][:max_examples]

    examples_count = len(testset["sentences"])
    lp_short_nonisland_average = 0
    lp_long_nonisland_avg = 0
    lp_short_island_avg = 0
    lp_long_island_avg = 0
    penlp_short_nonisland_average = 0
    DDs_with_lp = []
    DDs_with_pen_lp = []

    for example_idx, example_data in enumerate(tqdm(testset["sentences"])):

        (
            lps,
            pen_lps,
            pen_sentence_log_weights,
            sentence_log_weights,
            sentences,
        ) = get_example_scores(
            device,
            example_data,
            model,
            model_type,
            sent_ids,
            tokenizer,
            sentences_per_example,
            sprouse_format=examples_in_sprouse_format,
        )
        if verbose:
            print_example(sentences, sentence_ordering)

        DDs_with_lp.append(get_dd_score(lps, sentence_ordering))
        DDs_with_pen_lp.append(get_dd_score(pen_lps, sentence_ordering))
        lp_short_nonisland_average += lps[sentence_ordering.SHORT_NONISLAND]
        lp_long_nonisland_avg += lps[sentence_ordering.LONG_NONISLAND]
        lp_short_island_avg += lps[sentence_ordering.SHORT_ISLAND]
        lp_long_island_avg += lps[sentence_ordering.LONG_ISLAND]
        penlp_short_nonisland_average += pen_lps[0]

    lp_short_nonisland_average /= examples_count
    lp_long_nonisland_avg /= examples_count
    lp_short_island_avg /= examples_count
    lp_long_island_avg /= examples_count
    penlp_short_nonisland_average /= examples_count
    lp_averages = [
        lp_short_nonisland_average,
        lp_long_nonisland_avg,
        lp_short_island_avg,
        lp_long_island_avg,
    ]

    correc_count_DD_lp = len([x for x in DDs_with_lp if x > 0])
    accuracy_DD_lp = correc_count_DD_lp / len(DDs_with_lp)
    print(f"accuracy with DDs_with_lp: {accuracy_DD_lp}")
    correc_count_DD_penlp = len([x for x in DDs_with_pen_lp if x > 0])
    accuracy_DD_penlp = correc_count_DD_penlp / len(DDs_with_lp)
    print(f"accuracy with DDs_with_penlp: {accuracy_DD_penlp}")

    return lp_averages


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

    example_dd_with_lp = get_example_dd_score(example, "lp")
    example_dd_with_penlp = get_example_dd_score(example, "pen_lp")

    return example_dd_with_lp, example_dd_with_penlp


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
        plot_results(phenomenon, lp_avg_scores[idx], "lp")


def main():
    # create_test_jsonl_files_tests()

    model_type = (
        model_types.BERT  # model_types.GEPPETTO  #
    )  # model_types.GPT # model_types.ROBERTA  #
    model_name = "dbmdz/bert-base-italian-xxl-cased"  # "LorenzoDeMattei/GePpeTto"  #
    # "bert-base-uncased"  # "gpt2-large"  # "roberta-large" # "bert-large-uncased"  #
    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)

    run_custom_testsets = True
    # scored_testsets = []
    if run_custom_testsets:
        tests_dir = str(get_syntactic_tests_dir() / "sprouse/")  # "syntactic_tests_it/"
        phenomena = [
            # "wh_adjunct_islands",
            # "wh_complex_np_islands",
            # "wh_whether_island",
            # "wh_subject_islands",
            # "custom-wh_adjunct_islands",
            "custom-wh_complex_np_islands",
            "custom-wh_whether_island",
            "custom-wh_subject_islands",
        ]
        examples_format = "sprouse"  # "blimp"
        sentence_ordering = SprouseSentencesOrder  # BlimpSentencesOrder
        scored_testsets = run_sprouse_tests(
            model_type,
            model,
            tokenizer,
            device,
            phenomena=phenomena,
            testset_dir_path=tests_dir,
            examples_format=examples_format,
            sentence_ordering=sentence_ordering,
        )
    else:
        scored_testsets = run_sprouse_tests(model_type, model, tokenizer, device)

    for scored_testset in scored_testsets:
        scored_testset.model_descr = model_name
        scored_testset.save_to_picle(
            scored_testset.linguistic_phenomenon + ".testset.pickle"
        )


def load_pickles() -> list[TestSet]:
    phenomena = [
        "custom-wh_adjunct_islands",
        "custom-wh_complex_np_islands",
        "custom-wh_whether_island",
        "custom-wh_subject_islands",
    ]
    loaded_testsets = []
    for phenomenon in phenomena:
        loaded_testset = load_testset_from_pickle(phenomenon + ".testset.pickle")
        loaded_testsets.append(loaded_testset)

    return loaded_testsets


def load_and_plot_pickle():
    loaded_testsets = load_pickles()
    plot_testsets(loaded_testsets)


def plot_testsets(scored_testsets):
    plot_results(scored_testsets, "lp")
    plot_results(scored_testsets, "pen_lp")


def print_testset_detailed_analysis():

    loaded_testsets = load_pickles()

    max_testsets = 1
    for testset in loaded_testsets[:max_testsets]:
        print(
            f"printing testset for {testset.linguistic_phenomenon} from {testset.model_descr}"
        )
        examples = testset.get_examples_sorted_by_score_diff_1vs2("lp", reverse=False)
        max_prints = 10

        for example in examples[0:max_prints]:
            print(
                f"lp_diff: {example.get_score_diff_1vs2('lp'):.2f}, "
                f"s1 (SHORT_ISLAND, {example[SentenceNames.SHORT_ISLAND].get_score('lp'):.2f}): "
                f"{example[SentenceNames.SHORT_ISLAND].txt}, "
                f"s2 (LONG_ISLAND, {example[SentenceNames.LONG_ISLAND].get_score('lp'):.2f}): "
                f"{example[SentenceNames.LONG_ISLAND].txt}"
            )

    # todo:
    #  sort and print examples by diff in acceptability btw short island and long island sentences
    #  sort and print examples by DD value
    # ..
    # show the examples with ..
    # plot with 7+1x7 subplots of a testset (one subplot for each example)
    # ..
    # normalize sentence/tokens scrores


if __name__ == "__main__":
    # main()
    # load_and_plot_pickle()
    print_testset_detailed_analysis()
