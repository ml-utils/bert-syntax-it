import csv
import getopt
import json
import os
import sys
from collections import Counter

from linguistic_tests.bert_utils import analize_example
from linguistic_tests.bert_utils import analize_sentence
from linguistic_tests.bert_utils import check_unknown_words
from linguistic_tests.bert_utils import estimate_sentence_probability_from_text
from linguistic_tests.bert_utils import get_probs_for_words
from linguistic_tests.bert_utils import get_score_descr
from linguistic_tests.bert_utils import tokenize_sentence
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import sentence_score_bases
from linguistic_tests.utils import vocab_it
from torch.utils.hipify.hipify_python import bcolors
from tqdm import tqdm


def load_it():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run
    # "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ >
    # marvin_linzen_dataset.tsv"
    out = []
    for line in open("it_dataset.tsv"):
        case = line.strip().split("\t")
        cc[case[1]] += 1
        g, ug = case[-2], case[-1]
        g = g.split()
        ug = ug.split()
        assert len(g) == len(ug), (g, ug)
        diffs = [i for i, pair in enumerate(zip(g, ug)) if pair[0] != pair[1]]
        if len(diffs) != 1:
            # print(diffs)
            # print(g,ug)
            continue
        assert len(diffs) == 1, diffs
        gv = g[diffs[0]]  # good
        ugv = ug[diffs[0]]  # bad
        g[diffs[0]] = "***mask***"
        g.append(".")
        out.append((case[0], case[1], " ".join(g), gv, ugv))
    return out


def load_marvin():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run
    # "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ >
    # marvin_linzen_dataset.tsv"
    out = []
    for line in open("marvin_linzen_dataset.tsv"):
        case = line.strip().split("\t")
        cc[case[1]] += 1
        g, ug = case[-2], case[-1]
        g = g.split()
        ug = ug.split()
        assert len(g) == len(ug), (g, ug)
        diffs = [i for i, pair in enumerate(zip(g, ug)) if pair[0] != pair[1]]
        if len(diffs) != 1:
            # print(diffs)
            # print(g,ug)
            continue
        assert len(diffs) == 1, diffs
        gv = g[diffs[0]]  # good
        ugv = ug[diffs[0]]  # bad
        g[diffs[0]] = "***mask***"
        g.append(".")
        out.append((case[0], case[1], " ".join(g), gv, ugv))
    return out


def eval_it(bert, tokenizer):
    o = load_it()
    print(len(o), file=sys.stderr)
    import time

    # rc = defaultdict(Counter)
    # tc = Counter()
    start = time.time()
    print_orange(f"{len(o)} sentences to process..")
    for i, (case, tp, s, good_word, bad_word) in enumerate(o):
        ps = get_probs_for_words(bert, tokenizer, s, good_word, bad_word)
        if ps is None:
            ps = [0, 1]
        gp = ps[0]
        bp = ps[1]
        print(gp > bp, case, tp, good_word, bad_word, s)
        if i % 100 == 0:
            print(f"{bcolors.WARNING}{i}{bcolors.ENDC}")
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()


def eval_marvin(bert, tokenizer):
    o = load_marvin()
    print(len(o), file=sys.stderr)

    import time

    # rc = defaultdict(Counter)
    # tc = Counter()
    start = time.time()
    for i, (case, tp, s, g, b) in enumerate(o):
        ps = get_probs_for_words(bert, tokenizer, s, g, b)
        if ps is None:
            ps = [0, 1]
        gp = ps[0]
        bp = ps[1]
        print(gp > bp, case, tp, g, b, s)
        if i % 100 == 0:
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()


def eval_lgd(bert, tokenizer):
    for i, line in enumerate(open("lgd_dataset_with_is_are.tsv", encoding="utf8")):
        na, _, masked, good, bad = line.strip().split("\t")
        ps = get_probs_for_words(bert, tokenizer, masked, good, bad)
        if ps is None:
            continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp > bp), na, good, gp, bad, bp, masked.encode("utf8"), sep="\t")
        if i % 100 == 0:
            print(i, file=sys.stderr)
            sys.stdout.flush()


def read_gulordava():
    rows = csv.DictReader(open("generated.tab", encoding="utf8"), delimiter="\t")
    data = []
    for row in rows:
        row2 = next(rows)
        assert row["sent"] == row2["sent"]
        assert row["class"] == "correct"
        assert row2["class"] == "wrong"
        sent = row["sent"].lower().split()[:-1]  # dump the <eos> token.
        good_form = row["form"]
        bad_form = row2["form"]
        sent[int(row["len_prefix"])] = "***mask***"
        sent = " ".join(sent)
        data.append((sent, row["n_attr"], good_form, bad_form))
    return data


def eval_gulordava(bert, tokenizer):
    for i, (masked, natt, good, bad) in enumerate(read_gulordava()):
        if good in ["is", "are"]:
            print("skipping is/are")
            continue
        ps = get_probs_for_words(bert, tokenizer, masked, good, bad)
        if ps is None:
            continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp > bp), natt, good, gp, bad, bp, masked.encode("utf8"), sep="\t")
        if i % 100 == 0:
            print(i, file=sys.stderr)
            sys.stdout.flush()


# choose_eval()


def run_eval(eval_suite, bert, tokenizer):
    print(f"running eval, eval_suite: {eval_suite}")
    if "it" == eval_suite:
        eval_it(bert, tokenizer)
    elif "marvin" == eval_suite:
        eval_marvin(bert, tokenizer)
    elif "gul" == eval_suite:
        eval_gulordava(bert, tokenizer)
    else:
        eval_lgd(bert, tokenizer)


def arg_parse():
    print("parsing args..")
    # Python program to demonstrate
    # command line arguments

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # options = "be:"

    # Long options
    # long_options = ["bert_model", "eval_suite"]

    DEFAULT_MODEL = "bert-large-uncased"
    DEFAULT_EVAL_SUITE = "lgd"
    model_name = DEFAULT_MODEL
    eval_suite = DEFAULT_EVAL_SUITE

    try:
        # Parsing argument
        print(f"argumentList: {argumentList}")

        # checking each argument
        for arg_idx, currentArgument in enumerate(argumentList):
            print(f"persing currentArgument {currentArgument}")
            if currentArgument in ("-h", "--Help"):
                print("Displaying Help")

            elif currentArgument in ("-b", "--bert_model"):

                argValue = argumentList[arg_idx + 1]
                print(f"{currentArgument=}, {argValue=}")
                if argValue == "base":
                    model_name = "bert-base-uncased"
                else:
                    model_name = argValue
                    print(f"set model_name: {model_name}")

            elif currentArgument in ("-e", "--eval_suite"):
                argValue = argumentList[arg_idx + 1]
                print(f"{currentArgument=}, {argValue=}")
                eval_suite = argValue

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    print(f"model_name {model_name}, eval_suite {eval_suite}")
    return model_name, eval_suite


def custom_eval(sentence, bert, tokenizer):
    bert, tokenizer = load_model_and_tokenizer(model_types.BERT, "bert-base-uncased")

    compare_tokens, compare_target_idx = tokenize_sentence(
        tokenizer, "What is ***your*** name?"
    )

    bare_sentence_tokens = tokenizer.tokenize(sentence)

    # paper_logprobs = bert_get_logprobs(bare_sentence_tokens, None, bert,
    #                                   tokenizer, device=None)

    tokens_list = ["[CLS]"] + bare_sentence_tokens + ["[SEP]"]
    target_idx = 3
    masked_word = tokens_list[target_idx]
    tokens_list[target_idx] = "[MASK]"

    print(f"tokens: {tokens_list}, masked_word: {masked_word}")
    print(f"compare_tokens: {compare_tokens}")

    # input_ids = tokenizer.convert_tokens_to_ids(tokens_list)

    # try:
    #     masked_word_id = tokenizer.convert_tokens_to_ids([masked_word])
    # except KeyError:
    #     print(f"unable to convert {masked_word} to id")
    #     return None
    # tens = torch.LongTensor(input_ids).unsqueeze(0)

    # res_unsliced = bert(tens)
    # res = res_unsliced[0, target_idx]

    # res=torch.nn.functional.softmax(res,-1)

    # pred = bert("What is [MASK] name?")

    # Set the maximum sequence length.
    # MAX_LEN = 128
    # Pad our input tokens
    # input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt
    #                            in tokenized_texts],
    #                         maxlen=MAX_LEN, dtype="long", truncating="post",
    #                           padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in
    # the BERT vocabulary
    # input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
    #                          truncating="post", padding="post")


def print_sentence_pairs_probabilities(bert, tokenizer, sentence_data):
    sentence_good_no_extraction = sentence_data["sentence_good_no_extraction"]
    sentence_bad_extraction = sentence_data["sentence_bad_extraction"]
    sentence_good_extraction_resumption = sentence_data[
        "sentence_good_extraction_resumption"
    ]
    sentence_good_extraction_as_subject = sentence_data[
        "sentence_good_extraction_as_subject"
    ]
    print(f"sentence_good_no_extraction: {sentence_good_no_extraction}")
    print(f"sentence_bad_extraction: {sentence_bad_extraction}")
    print(f"{sentence_good_extraction_resumption=}")
    print(f"{sentence_good_extraction_as_subject=}")

    prob_sentence_good_no_extraction = estimate_sentence_probability_from_text(
        bert, tokenizer, sentence_good_no_extraction
    )
    prob_sentence_bad_extraction = estimate_sentence_probability_from_text(
        bert, tokenizer, sentence_bad_extraction
    )
    prob_sentence_good_extraction_resumption = estimate_sentence_probability_from_text(
        bert, tokenizer, sentence_good_extraction_resumption
    )
    prob_sentence_good_extraction_as_subject = estimate_sentence_probability_from_text(
        bert, tokenizer, sentence_good_extraction_as_subject
    )

    print(f"{prob_sentence_good_no_extraction=}")
    print(f"prob_sentence_bad_extraction: {prob_sentence_bad_extraction}")
    print(f"{prob_sentence_good_extraction_resumption}")
    print(f"{prob_sentence_good_extraction_as_subject=}")


def run_testset_bert(
    testsets_dir: str,
    filename: str,
    model,
    tokenizer,
    sentences_per_example,
    score_based_on=sentence_score_bases.SOFTMAX,
):
    filepath = os.path.join(testsets_dir, filename)
    print_orange(f"running test {filepath}")
    testset_data = load_testset_data(filepath)
    examples_count = len(testset_data["sentences"])
    print(f"examples_count: {examples_count}")

    # todo: add checks that there is enough variability / less repetitive
    #  examples (subjects proper names or pronouns,
    #  plural and singular, 1st, 2nd and 3rd person, ..

    # only_examples = [3, 6, 8, 10, 14, 15, 16, 18, 19, 21, 22, 23, 26, 29,
    # 31, 32, 33, 39, 43, 46, 47, 48, 49]
    # print(f'incorrect examples count: {len(only_examples)} out of 50
    # ({len(only_examples)/50})')
    error_count_base_sentence = 0
    error_count_second_sentence = 0
    error_count_either = 0
    no_errors_examples_indexes = []
    examples_by_base_sentence_acceptability_diff = {}
    examples_by_second_sentence_acceptability_diff = {}
    second_sentences_by_score = {}
    bad_sentences_by_score = {}
    base_sentences_by_score = {}
    for example_idx, sentence_data in enumerate(testset_data["sentences"]):
        # print(f"json_str, type: {type(sentence_data)}: {sentence_data}")
        # print_sentence_pairs_probabilities(bert, tokenizer, sentence_data)
        (
            base_sentence_less_acceptable,
            second_sentence_less_acceptable,
            acceptability_diff_base_sentence,
            acceptability_diff_second_sentence,
            score_base_sentence,
            score_bad_sentence,
            score_2nd_good_sentence,
            oov_counts,
        ) = analize_example(
            model,
            tokenizer,
            example_idx,
            sentence_data,
            sentences_per_example,
            score_based_on,
        )
        # return
        sentences = get_sentences_from_example(sentence_data, sentences_per_example)

        second_sentences_by_score[score_2nd_good_sentence] = sentences[2]
        bad_sentences_by_score[score_bad_sentence] = sentences[1]
        base_sentences_by_score[score_base_sentence] = sentences[0]
        examples_by_base_sentence_acceptability_diff[
            acceptability_diff_base_sentence
        ] = get_example_analysis_as_tuple(
            example_idx,
            score_base_sentence,
            score_bad_sentence,
            score_2nd_good_sentence,
            oov_counts,
            sentences[0],
            sentences[1],
        )
        examples_by_second_sentence_acceptability_diff[
            acceptability_diff_second_sentence
        ] = get_example_analysis_as_tuple(
            example_idx,
            score_base_sentence,
            score_bad_sentence,
            score_2nd_good_sentence,
            oov_counts,
            sentences[2],
            sentences[1],
        )

        if base_sentence_less_acceptable:
            error_count_base_sentence += 1
        if second_sentence_less_acceptable:
            error_count_second_sentence += 1

        if base_sentence_less_acceptable or second_sentence_less_acceptable:
            error_count_either += 1
        else:
            no_errors_examples_indexes.append(example_idx)

    print_red(
        f"error count and accuracy rates from {examples_count} "
        f"examples: "
        f"base sentence {error_count_base_sentence} "
        f"(acc: {get_perc(examples_count - error_count_base_sentence, examples_count)}), "
        f"second sentence: {error_count_second_sentence} "
        f"(acc: {get_perc(examples_count - error_count_second_sentence, examples_count)}), "
        f"either: {error_count_either} "
        f"(acc: {get_perc(examples_count - error_count_either, examples_count)}), "
        f"filename: {filename}"
    )

    print(
        f"error count out of {examples_count} examples: "
        f"base sentence {error_count_base_sentence} "
        f"({get_perc(error_count_base_sentence, examples_count)}), "
        f"second sentence: {error_count_second_sentence} "
        f"({get_perc(error_count_second_sentence, examples_count)}), "
        f"either: {error_count_either} "
        f"({get_perc(error_count_either, examples_count)}), {filename=}"
    )

    # print examples getting no errors:
    print_orange("Examples getting no errors:")
    for example_idx in no_errors_examples_indexes:
        no_error_example = testset_data["sentences"][example_idx]
        print(f"{get_sentences_from_example(no_error_example, 2)}")

    print_orange("examples sorted by sentence_acceptability diff, " "second sentence:")
    sorted_examples = dict(
        sorted(examples_by_second_sentence_acceptability_diff.items())
    ).items()
    for acceprability_diff, example_analysis in sorted_examples:
        print_example(
            example_analysis,
            acceprability_diff,
            score_based_on,
            compare_with_base_sentence=False,
        )

    print_orange("examples sorted by sentence_acceptability diff, " "base sentence:")
    for acceprability_diff, example_analysis in dict(
        sorted(examples_by_base_sentence_acceptability_diff.items())
    ).items():
        print_example(
            example_analysis,
            acceprability_diff,
            score_based_on,
            compare_with_base_sentence=True,
        )

    score_descr = get_score_descr(score_based_on)
    print_sentences_sorted_by_score(
        second_sentences_by_score, f"second sentences sorted by " f"{score_descr}:"
    )
    print_sentences_sorted_by_score(
        bad_sentences_by_score, f"bad sentences sorted by {score_descr}:"
    )
    print_sentences_sorted_by_score(
        base_sentences_by_score, f"base sentences sorted by {score_descr}:"
    )


def print_sentences_sorted_by_score(sentences_by_score, msg):
    print_orange(msg)
    for score, sentence in dict(sorted(sentences_by_score.items())).items():
        print(f"{score:.1f} {sentence}")


def get_example_analysis_as_tuple(
    example_idx,
    score_base_sentence,
    score_bad_sentence,
    score_2nd_good_sentence,
    oov_counts,
    sentence_good,
    sentence_bad,
):
    return (
        example_idx,
        score_base_sentence,
        score_bad_sentence,
        score_2nd_good_sentence,
        oov_counts,
        sentence_good,
        sentence_bad,
    )


def print_example(
    example_analysis,
    acceprability_diff,
    score_based_on,
    compare_with_base_sentence=True,
):
    example_idx = example_analysis[0]
    penLP_base_sentence = example_analysis[1]
    penLP_bad_sentence = example_analysis[2]
    penLP_2nd_good_sentence = example_analysis[3]
    oov_counts = example_analysis[4]
    sentence_good = example_analysis[5]
    sentence_bad = example_analysis[6]
    if compare_with_base_sentence:
        diff_descr = "accept_diff_w_base_sent"
    else:
        diff_descr = "accept_diff_w_2nd_sent"

    score_descr = get_score_descr(score_based_on)
    print(
        f"{diff_descr}: {rnd(acceprability_diff, 3)}, "
        f"({score_descr} values: {rnd(penLP_base_sentence, 1)}, "
        f"{rnd(penLP_bad_sentence, 1)}, {rnd(penLP_2nd_good_sentence, 1)}), "
        f"example (oov_counts: {oov_counts}): ({example_idx}, "
        f"'{sentence_good}', '{sentence_bad}'"
    )


def rnd(num, decimal_places):
    if num is not None:
        return round(num, decimal_places)
    else:
        return None


def get_perc(value, total):
    return f"{perc(value, total):.1f} %"


def basic_sentence_test(model, tokenizer):
    check_unknown_words(tokenizer)
    sentence_to_analizse = (
        "Di che cosa Marco si chiede se è stata riparata da ***Luca***?"
    )
    topk_tokens, topk_probs, topk_probs_nonsoftmax = analize_sentence(
        model, tokenizer, sentence_to_analizse
    )
    print(f"sentence: {sentence_to_analizse}")
    print(f" {topk_tokens=}, {topk_probs=}, {topk_probs_nonsoftmax=}")


def profile_slowdowns():
    import cProfile
    import pstats

    model_type = model_types.ROBERTA  # model_types.GPT  #
    model_name = "roberta-large"  # "roberta-base" #"gpt2-medium"
    # "gpt2-large"  # 'gpt2' #  "bert-large-uncased"
    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

    p = get_syntactic_tests_dir() / "blimp/from_blim_en/islands"
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

        sentence_good = example["sentence_good"]
        sentence_bad = example["sentence_bad"]
        examples.append(
            {
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "sentence_good_2nd": "",
            }
        )
    testset = {"sentences": examples}
    sentences_per_example = 2

    with cProfile.Profile() as pr:
        run_testset(
            model_type, model, tokenizer, DEVICES.CPU, testset, sentences_per_example
        )

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


def t_determiner_noun_agreement_1():
    from generation_projects.blimp import determiner_noun_agreement_1

    generator = determiner_noun_agreement_1.DetNGenerator()
    generator.generate_paradigm(
        rel_output_path="outputs/blimp/%s.jsonl" % generator.uid
    )


def print_profession_nouns():
    for noun in vocab_it.nouns_professions:
        print(noun + " ")
