import csv
import getopt
import json
import os.path
import sys
from collections import Counter

from linguistic_tests.bert_utils import analize_example
from linguistic_tests.bert_utils import analize_sentence
from linguistic_tests.bert_utils import check_unknown_words
from linguistic_tests.bert_utils import estimate_sentence_probability
from linguistic_tests.bert_utils import estimate_sentence_probability_from_text
from linguistic_tests.bert_utils import get_probs_for_words
from linguistic_tests.bert_utils import get_score_descr
from linguistic_tests.bert_utils import tokenize_sentence
from linguistic_tests.compute_model_score import perc
from linguistic_tests.compute_model_score import run_testset
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import model_types
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import red_txt
from linguistic_tests.lm_utils import sentence_score_bases
from torch.utils.hipify.hipify_python import bcolors
from tqdm import tqdm


def run_agreement_tests():
    return 0


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


def get_masked_word_probability(bert, tokenizer):
    return 0


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


def run_tests_goldberg():
    # todo: use sentence acceptability estimates (PenLP e PenNL), and see
    #  results on goldberg testset
    # also for blimp testset with tests non intended for bert, compare with
    # the results on gpt and other models
    return 0


def run_tests_blimp():
    # todo
    return 0


def run_tests_lau_et_al():
    # todo
    return 0


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


def interactive_mode():
    print("interactive mode")

    # load model than wait for input sentences
    model_name = str(get_models_dir() / "bert-base-italian-xxl-cased")
    # eval_suite = 'it'
    bert, tokenizer = load_model_and_tokenizer(
        model_types.BERT, model_name, do_lower_case=False
    )

    print("model loaded, waiting for sentences..")

    # given two sentences, print PenLPs, and diff btw PenLPs
    end_program = False
    while not end_program:
        good_sentence = input("Enter first sentence (good): ")
        if good_sentence == "exit":
            return
        bad_sentence = input("Enter 2nd sentence (bad): ")

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
            logits_normalized_bad_sentence,
            logits_normalized_base_sentence,
            logits_normalized_2nd_good_sentence,
            oov_counts,
        ) = analize_example(bert, tokenizer, -1, example, sentences_per_example)
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
            print_detailed_sentence_info(bert, tokenizer, good_sentence)
            print_detailed_sentence_info(bert, tokenizer, bad_sentence)


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


def print_detailed_sentence_info(bert, tokenizer, sentence_txt):
    print_red(f"printing details for sentence {sentence_txt}")
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    estimate_sentence_probability(bert, tokenizer, sentence_ids, verbose=True)


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
def run_blimp_en(model_type=None, model_name=None, testset_filenames=None):
    if model_type is None:
        model_type = model_types.ROBERTA  # model_types.GPT  #
        model_name = "roberta-large"  # "roberta-base" #"gpt2-medium"
        # "gpt2-large"  # 'gpt2'  #  "bert-large-uncased"
        # "bert-base-uncased"  #    'dbmdz/bert-base-italian-xxl-cased' #
    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

    if testset_filenames is None:
        testset_filenames = [
            "wh_island.jsonl",
            "adjunct_island.jsonl",
            "complex_NP_island.jsonl",
        ]

    p = get_syntactic_tests_dir() / "blimp/from_blim_en/islands"
    testset_dir_path = str(p)

    for testset_filename in testset_filenames:
        testset_filepath = os.path.join(testset_dir_path, testset_filename)
        # './outputs/blimp/from_blim_en/islands/adjunct_island.jsonl'

        print(f"loading testset file {testset_filepath}..")
        with open(testset_filepath, "r") as json_file:
            json_list = list(json_file)
        print("testset loaded.")

        examples = []
        for json_str in tqdm(json_list):
            example = json.loads(json_str)
            # print(f"result: {example}")
            # print(isinstance(example, dict))
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
        run_testset(
            model_type, model, tokenizer, DEVICES.CPU, testset, sentences_per_example
        )


def run_tests_it(model_type, testset_files=None):
    if model_type == model_types.GPT:
        model_name = "GroNLP/gpt2-small-italian"
    if model_type == model_types.GEPPETTO:
        model_name = "LorenzoDeMattei/GePpeTto"
    elif model_type == model_types.BERT:
        model_name = "bert-base-uncased"  # NB bert large uncased is about 1GB
        model_name = str(get_models_dir() / "bert-base-italian-uncased")
        model_name = str(get_models_dir() / "bert-base-italian-cased/")
        model_name = str(get_models_dir() / "bert-base-italian-xxl-cased")
        model_name = "dbmdz/bert-base-italian-cased"
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        # model_name = # str(get_models_dir() / "gilberto-uncased-from-camembert.tar.gz")
        # eval_suite = 'it'
    elif model_type == model_types.GILBERTO:
        model_name = "idb-ita/gilberto-uncased-from-camembert"

    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
    p = (
        get_syntactic_tests_dir() / "syntactic_tests_it"
    )  # "./outputs/syntactic_tests_it/"
    testsets_dir = str(p)
    if testset_files is None:
        testset_files = [  # 'variations_tests.jsonl'
            "wh_adjunct_islands.jsonl",
            "wh_complex_np_islands.jsonl",
            # 'wh_subject_islands.jsonl',
            # 'wh_whether_island.jsonl'
        ]
    sentences_per_example = 3
    for test_file in testset_files:
        filepath = os.path.join(testsets_dir, test_file)
        print_orange(f"running test {filepath}")
        testset_data = load_testset_data(filepath)

        if model_type in [model_types.BERT, model_types.GILBERTO, model_types.ROBERTA]:
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
        elif model_type in [model_types.GPT, model_types.GEPPETTO]:
            run_testset(
                model_type,
                model,
                tokenizer,
                DEVICES.CPU,
                testset_data,
                sentences_per_example,
            )


def run_tests_for_model_type(model_type):
    print("model_type: {model_type}")
    # model_name, eval_suite = arg_parse()

    # todo: run on the following testsets (minimal pairs):
    # (use same pretrained models.. or comparable ones to those in the papers)
    # blimp: ..
    # golderg: ..
    # Lau et al: https://github.com/ml-utils/
    # acceptability-prediction-in-context/tree/
    # 0a274d1d9f70f389ddc6b6d796bd8f815833056c/code

    run_tests_it(model_type)

    # if model_type == model_types.GPT:
    #    print('importing gpt_tests..')
    #     from gpt_tests import main as main2
    #    print('imported.')
    #    main2()

    # run_eval(eval_suite, bert, tokenizer)
    # prob1 = estimate_sentence_probability_from_text(bert, tokenizer,
    # 'What is your name?')
    # prob2 = estimate_sentence_probability_from_text(bert, tokenizer,
    # 'What is name your?')
    # print(f'prob1: {prob1}, prob2: {prob2}')
    # eval_it(bert, tokenizer)
    # custom_eval("What is your name?", bert, tokenizer)


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


def main():
    if len(sys.argv) > 1:
        interactive_mode()
    else:
        # run_blimp_en()
        # raise SystemExit
        # print('choosing model type ..')
        model_type = model_types.BERT
        run_tests_for_model_type(model_type)


if __name__ == "__main__":
    main()
    # profile_slowdowns()
