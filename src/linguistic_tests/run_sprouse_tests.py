import os.path

from linguistic_tests.compute_model_score import get_example_scores
from linguistic_tests.compute_model_score import print_accuracy_scores
from linguistic_tests.compute_model_score import score_example
from linguistic_tests.file_utils import get_file_root
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from linguistic_tests.lm_utils import SprouseSentencesOrder
from linguistic_tests.testset import Example
from linguistic_tests.testset import load_testset_from_pickle
from linguistic_tests.testset import parse_testset
from linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
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
    phenomena_root_filenames,
    testset_dir_path,
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

    print(f"Running testsets from dir {testset_dir_path}")
    scored_testsets = []
    for phenomenon_name in phenomena_root_filenames:
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


def plot_results(scored_testsets: list[TestSet], score_name):
    fig, axs = plt.subplots(2, 2)
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
    scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
    parsed_testset = parse_testset(
        phenomenon_name, str(type(model)), examples_list, "sprouse", scoring_measures
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
    scored_testset = score_sprouse_testset(
        model_type,
        model,
        tokenizer,
        device,
        parsed_testset,
    )
    lp_averages = scored_testset.lp_average_by_sentence_type
    print(f"{lp_averages=}")
    return scored_testset


def score_sprouse_testset(
    model_type, model, tokenizer, device, testset: TestSet
) -> TestSet:
    for example_idx, example in enumerate(tqdm(testset.examples)):
        score_example(
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

    print(f"Testset accuracy with DDs_with_lp: {testset.accuracy_by_DD_lp:%}")
    print(f"Testset accuracy with DDs_with_penlp: {testset.accuracy_by_DD_penlp:%}")

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

    example_dd_with_lp = get_example_dd_score(example, ScoringMeasures.LP)
    example_dd_with_penlp = get_example_dd_score(example, ScoringMeasures.PenLP)

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
        plot_results(phenomenon, lp_avg_scores[idx], ScoringMeasures.LP.name)


def score_testsets(
    model_type: model_types,
    model_name: str,
    broader_test_type: str,
    testset_root_filenames: list[str] = None,
    testset_dir_path: str = None,
):
    # create_test_jsonl_files_tests()

    device = DEVICES.CPU
    model, tokenizer = load_model(model_type, model_name, device)

    examples_format = "sprouse"  # "blimp"
    sentence_ordering = SprouseSentencesOrder  # BlimpSentencesOrder
    scored_testsets = run_sprouse_tests(
        model_type,
        model,
        tokenizer,
        device,
        phenomena_root_filenames=testset_root_filenames,
        testset_dir_path=testset_dir_path,
        examples_format=examples_format,
        sentence_ordering=sentence_ordering,
    )

    for scored_testset in scored_testsets:
        scored_testset.model_descr = model_name
        filename = get_pickle_filename(
            scored_testset.linguistic_phenomenon,
            model_name,
            broader_test_type=broader_test_type,
        )
        scored_testset.save_to_picle(filename)


def get_pickle_filename(
    linguistic_phenomenon,
    model_name,
    broader_test_type,
):
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
    phenomena, model_name, broader_test_type, loaded_testsets=None
):

    if loaded_testsets is None:
        loaded_testsets = load_pickles(phenomena, model_name, broader_test_type)

    plot_testsets(loaded_testsets)


def plot_testsets(scored_testsets):
    plot_results(scored_testsets, ScoringMeasures.LP.name)
    plot_results(scored_testsets, ScoringMeasures.PenLP.name)


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


def main():

    # todo: enable command line arguments to choose which models to run
    # runs_args = []
    # run_args0 = {"model_type": }

    # add score with logistic function (instead of softmax)

    # todo: check that accuracy values are scored and stored correctly
    #  (it seems they are scored twice and not shown when loading pickles)
    # save results to csv (for import in excel table)
    # autosave plots as *.png

    # todo: make a list with all the models to test and run them
    # Bert (dbmdz/bert-base-italian-xxl-cased)
    # GePpeTto
    # GilBERTo (idb-ita/gilberto-uncased-from-camembert)

    model_type1 = model_types.GEPPETTO
    model_type2 = model_types.BERT  # model_types.GPT # model_types.ROBERTA  #
    model_type3 = model_types.GILBERTO
    model_types_to_run = [model_type1, model_type2, model_type3]

    model_name1 = "LorenzoDeMattei/GePpeTto"
    model_name2 = "dbmdz/bert-base-italian-xxl-cased"
    model_name3 = "idb-ita/gilberto-uncased-from-camembert"
    model_names = [model_name1, model_name2, model_name3]

    # "bert-base-uncased"  # "gpt2-large"  # "roberta-large" # "bert-large-uncased"  #
    tests_subdir = "sprouse/"  # "syntactic_tests_it/"  #
    testset_dir_path = str(get_syntactic_tests_dir() / tests_subdir)

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
    if tests_subdir == "syntactic_tests_it/":
        testsets_root_filenames = custom_it_island_testsets_root_filenames
        broader_test_type = "it_tests"
    elif tests_subdir == "sprouse/":
        testsets_root_filenames = sprouse_testsets_root_filenames
        broader_test_type = "sprouse"

    for model_type, model_name in zip(model_types_to_run, model_names):
        score_testsets(
            model_type,
            model_name,
            broader_test_type=broader_test_type,
            testset_root_filenames=testsets_root_filenames,
            testset_dir_path=testset_dir_path,
        )

        loaded_testsets = load_pickles(
            testsets_root_filenames, model_name, broader_test_type
        )

        print("Printing accuracy scores..")
        for scored_testset in loaded_testsets:
            print_accuracy_scores(scored_testset)

        # print results in table format for the doc report

        score_descr = ScoringMeasures.PenLP.name

        print_sorted_sentences_to_check_spelling_errors2(
            score_descr,
            testsets_root_filenames,
            model_name,
            broader_test_type,
            loaded_testsets,
        )
        print_sorted_sentences_to_check_spelling_errors(
            score_descr,
            testsets_root_filenames,
            model_name,
            broader_test_type,
            loaded_testsets,
        )

        print_examples_compare_diff(
            score_descr,
            SentenceNames.SHORT_ISLAND,
            SentenceNames.LONG_ISLAND,
            testsets_root_filenames,
            model_name,
            broader_test_type,
            testsets=loaded_testsets,
        )
        print_examples_compare_diff(
            score_descr,
            SentenceNames.LONG_NONISLAND,
            SentenceNames.LONG_ISLAND,
            testsets_root_filenames,
            model_name,
            broader_test_type,
            testsets=loaded_testsets,
        )
        print_examples_compare_diff(
            score_descr,
            SentenceNames.SHORT_NONISLAND,
            SentenceNames.SHORT_ISLAND,
            testsets_root_filenames,
            model_name,
            broader_test_type,
            testsets=loaded_testsets,
        )

        load_and_plot_pickle(
            testsets_root_filenames,
            model_name,
            broader_test_type,
            loaded_testsets=loaded_testsets,
        )

    # todo:
    #  sort and print examples by DD value
    # ..
    # show the examples with ..
    # plot with 7+1x7 subplots of a testset (one subplot for each example)
    # ..
    # normalize sentence/tokens scores for Bert models, to have scores comparables across a whole testset
    #
    # normalize results to a likert scale, for comparison with Sprouse et al 2016
    #


if __name__ == "__main__":
    main()
