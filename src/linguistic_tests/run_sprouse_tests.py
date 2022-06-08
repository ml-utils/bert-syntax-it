import os.path
from enum import IntEnum

from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from matplotlib import pyplot as plt
from tqdm import tqdm


class SprouseSentencesOrder(IntEnum):
    SHORT_NONISLAND = 0
    LONG_NONISLAND = 1
    SHORT_ISLAND = 2
    LONG_ISLAND = 3


class BlimpSentencesOrder(IntEnum):
    SHORT_ISLAND = 0
    LONG_ISLAND = 1
    LONG_NONISLAND = 2
    SHORT_NONISLAND = 3


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
    tests_dir=None,
    examples_format="sprouse",
    sentence_ordering=SprouseSentencesOrder,
):
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
    if tests_dir is None:
        tests_dir = str(get_syntactic_tests_dir() / "sprouse/")
    print(f"Running testsets from dir {tests_dir}")
    for phenomenon_name in phenomena:
        print(f"Running testset for {phenomenon_name}..")
        filename = phenomenon_name + ".jsonl"
        filepath = os.path.abspath(os.path.join(tests_dir, filename))
        score_averages = run_sprouse_test(
            filepath,
            model_type,
            model,
            tokenizer,
            device,
            examples_format=examples_format,
            sentence_ordering=sentence_ordering,
        )
        plot_results(phenomenon_name, score_averages, "lp")


def plot_results(phenomenon_name, score_averages, score_descr):
    # todo: plot values
    #     lp_averages = [lp_short_nonisland_average, lp_long_nonisland_avg,
    #                    lp_short_island_avg, lp_long_island_avg]

    # nonisland line
    short_nonisland_average = [0, score_averages[0]]
    long_nonisland_avg = [1, score_averages[1]]
    x_values = [short_nonisland_average[0], long_nonisland_avg[0]]
    y_values = [short_nonisland_average[1], long_nonisland_avg[1]]
    plt.plot(x_values, y_values)

    # island line
    short_island_avg = [0, score_averages[2]]
    long_island_avg = [1, score_averages[3]]
    x_values = [short_island_avg[0], long_island_avg[0]]
    y_values = [short_island_avg[1], long_island_avg[1]]
    plt.plot(x_values, y_values, linestyle="--")
    plt.title(phenomenon_name)
    plt.ylabel(f"{score_descr} values")
    plt.show()


def run_sprouse_test(
    filepath,
    model_type,
    model,
    tokenizer,
    device,
    examples_format="sprouse",
    sentence_ordering=SprouseSentencesOrder,
):
    testset = load_testset_data(filepath, examples_format=examples_format)

    # run_testset(model_type, model, tokenizer, device, testset)
    lp_averages = run_sprouse_test_helper(
        model_type,
        model,
        tokenizer,
        device,
        testset,
        sentence_ordering=sentence_ordering,
    )
    print(f"{lp_averages=}")
    return lp_averages


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


def main():
    # create_test_jsonl_files_tests()

    model_type = (
        model_types.GEPPETTO
    )  # model_types.BERT  #  model_types.GPT # model_types.ROBERTA  #
    model_name = "LorenzoDeMattei/GePpeTto"
    # "LorenzoDeMattei/GePpeTto"
    # "dbmdz/bert-base-italian-xxl-cased"
    # "bert-base-uncased"  # "gpt2-large"  # "roberta-large" # "bert-large-uncased"  #
    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)

    run_custom_testsets = True
    if run_custom_testsets:
        tests_dir = str(get_syntactic_tests_dir() / "syntactic_tests_it/")
        phenomena = [
            "wh_adjunct_islands",
            # "wh_complex_np_islands",
            # "wh_whether_island",
            # "wh_subject_islands",
        ]
        examples_format = "blimp"
        sentence_ordering = BlimpSentencesOrder
        run_sprouse_tests(
            model_type,
            model,
            tokenizer,
            device,
            phenomena=phenomena,
            tests_dir=tests_dir,
            examples_format=examples_format,
            sentence_ordering=sentence_ordering,
        )
    else:
        run_sprouse_tests(model_type, model, tokenizer, device)


if __name__ == "__main__":
    main()


def plot_all_phenomena(phenomena_names, lp_avg_scores):
    for idx, phenomenon in enumerate(phenomena_names):
        plot_results(phenomenon, lp_avg_scores[idx], "lp")
