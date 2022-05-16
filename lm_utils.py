import os.path
import json

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_pretrained_bert import BertForMaskedLM, tokenization


class model_types:
    BERT = 0
    GPT = 1


class sentence_score_bases:
    SOFTMAX = 0
    NORMALIZED_LOGITS = 1


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_pen_score(unnormalized_score, text_len):
    penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
    return unnormalized_score / penalty


def red_txt(txt: str):
    return f'{bcolors.RED}{txt}{bcolors.ENDC}'


def print_red(txt: str):
    print_in_color(txt, bcolors.RED)


def print_orange(txt: str):
    print_in_color(txt, bcolors.WARNING)


def print_in_color(txt, color: bcolors):
    print(f'{color}{txt}{bcolors.ENDC}')


def load_model_and_tokenizer(model_type, model_name, dict_name=None, do_lower_case=False):
    print(f'loading model_name: {model_name}..')

    if model_type == model_types.GPT:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif model_type == model_types.BERT:
        model = BertForMaskedLM.from_pretrained(model_name)
        print("Pretrained model loaded, getting the tokenizer..")

        if dict_name is None:
            vocab_filepath = model_name
        else:
            vocab_filepath = os.path.join(model_name, 'dict.txt')
        tokenizer = tokenization.BertTokenizer.from_pretrained(vocab_filepath, do_lower_case=do_lower_case)
    else:
        print(f'Supported models: Bert, Gpt.')
        raise SystemExit

    print("tokenizer ready.")

    model.eval()
    return model, tokenizer


def load_testset_data(file_path):
    with open(file_path, mode='r', encoding="utf-8") as json_file:
        #json_list = list(json_file)
        testset_data = json.load(json_file)

        #for i in data:
        #    print(i)

    return testset_data


def get_sentences_from_example(example : dict):
    by_sentence_variant_name = False

    if by_sentence_variant_name:
        sentence_names_wh_wheter_islands = ['sentence_good_no_extraction', 'sentence_bad_extraction',
                          'sentence_good_extraction_resumption', 'sentence_good_extraction_as_subject']
        sentence_names_wh_complex_np_islands = ['sentence_good_no_extraction', 'sentence_bad_extraction',
                                            'sentence_good_no_island', 'sentence_good_no_island_as_subject']
        sentence_names = sentence_names_wh_complex_np_islands
        sentences = []
        for sentence_name in sentence_names:
            sentences.append(example[sentence_name])
    else:
        sentences = list(example.values())[0:3]

    return sentences