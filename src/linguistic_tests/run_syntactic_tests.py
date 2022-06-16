import os

from linguistic_tests.bert_utils import estimate_sentence_probability
from linguistic_tests.compute_model_score import score_dataclass_testset
from linguistic_tests.file_utils import get_file_root
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.testset import parse_testset


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
    # todo: compare results with other models
    return 0


def print_detailed_sentence_info(bert, tokenizer, sentence_txt):
    print_red(f"printing details for sentence {sentence_txt}")
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    estimate_sentence_probability(bert, tokenizer, sentence_ids, verbose=True)


def run_blimp_en(
    model_type,
    model_name,
    testset_filenames,
    testset_dir_path,
    max_examples=1000,
):
    parsed_testsets = []
    for testset_filename in testset_filenames:
        testset_filepath = os.path.join(testset_dir_path, testset_filename)
        print(f"Parsing testset {testset_filepath}")
        testset_dict = load_testset_data(testset_filepath, examples_format="json_lines")
        examples_list = testset_dict["sentences"]
        phenomenon_name = get_file_root(testset_filename)
        scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        parsed_testset = parse_testset(
            phenomenon_name,
            model_name,
            examples_list,
            "blimp",
            scoring_measures,
            max_examples=max_examples,
        )
        parsed_testsets.append(parsed_testset)

    model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
    for parsed_testset in parsed_testsets:
        print(
            f"Scoring testset {parsed_testset.linguistic_phenomenon}, on {model_type=} {model_name=}"
        )
        parsed_testset.examples = parsed_testset.examples[0:max_examples]

        scored_testset = score_dataclass_testset(
            model_type, model, tokenizer, DEVICES.CPU, parsed_testset
        )

        scored_testset.model_descr = model_name
        scored_testset.save_to_picle(
            # todo: filename: blimp/sprouse/.., datetime, phenomena, ..
            scored_testset.linguistic_phenomenon
            + ".testset.pickle"
        )

    return parsed_testsets


def run_tests_for_model_type(model_type):
    print(f"model_type: {model_type}")
    # model_name, eval_suite = arg_parse()

    # todo: run on the following testsets (minimal pairs):
    # (use same pretrained models.. or comparable ones to those in the papers)
    # blimp: ..
    # golderg: ..
    # Lau et al: https://github.com/ml-utils/
    # acceptability-prediction-in-context/tree/
    # 0a274d1d9f70f389ddc6b6d796bd8f815833056c/code

    # run_syntactic_tests_it_legacy_impl(model_type)

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


def main():
    # print_profession_nouns()
    # t_determiner_noun_agreement_1()
    pass


if __name__ == "__main__":
    main()
