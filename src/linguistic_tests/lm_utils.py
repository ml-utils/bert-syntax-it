import json
import os.path

import cython
import torch
from torch.utils.hipify.hipify_python import bcolors
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertForMaskedLM  # BertModel as BertForMaskedLM  #
from transformers import BertTokenizer
from transformers import CamembertForMaskedLM
from transformers import CamembertTokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer


class sent_idx:
    GOOD_1: int = 0
    # GOOD_1 = cython.declare(cython.int, 0)
    # GOOD_SENTENCE_1_IDX : int = 0
    BAD: int = 1
    GOOD_2: int = 2


class special_tokens:
    UNK: str = "[UNK]"


@cython.cclass
class model_types:
    BERT = 0
    GPT = 1
    ROBERTA = 2
    GILBERTO = 3
    GEPPETTO = 4


class sentence_score_bases:
    SOFTMAX = 0
    NORMALIZED_LOGITS = 1


class DEVICES:
    CPU = "cpu"
    CUDA = "cuda:X"


def get_pen_score(unnormalized_score, text_lenght):
    return unnormalized_score / get_penalty_term(text_lenght)


def get_penalty_term(text_lenght, alpha=0.8):
    return (5 + text_lenght) ** alpha / (5 + 1) ** alpha


def color_txt(txt: str, color: str):
    return f"{color}{txt}{bcolors.ENDC}"


def print_in_color(txt, color: str):
    print(color_txt(txt, color))


def red_txt(txt: str):
    return color_txt(txt, bcolors.FAIL)


def print_red(txt: str):
    print(red_txt(txt))


def print_orange(txt: str):
    print_in_color(txt, bcolors.WARNING)


def load_pretrained(
    model_type,
    model_name,
    device=DEVICES.CPU,
    dict_name=None,
    do_lower_case=False,
    force_automodel=False,
    local_files_only=False,
):
    print(f"loading model_name: {model_name}..")

    # Load pre-trained model and tokenizer
    if force_automodel:
        print(f"loading model {model_name}..")
        model = AutoModel.from_pretrained(model_name)
        print(f"model loaded of type {type(model)}. Loading tokenizer {model_name}..")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"tokenizer loaded of type {type(tokenizer)}.")
    elif model_type == model_types.GPT:
        print(f"loading model {model_name}..")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"model loaded. Loading tokenizer {model_name}..")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("tokenizer loaded.")
    elif model_type == model_types.BERT:
        print(f"loading model {model_name}..")
        model = BertForMaskedLM.from_pretrained(
            model_name
        )  # BertForMaskedLM.from_pretrained(model_name)

        if dict_name is None:
            tokenizer_path = model_name
        else:
            tokenizer_path = os.path.join(model_name, dict_name)

        print(f"model loaded. Loading tokenizer {tokenizer_path}..")

        do_lower_case = True if "uncased" in model_name else False
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=do_lower_case
        )
        print("tokenizer loaded.")

    elif model_type == model_types.GEPPETTO:
        print(f"loading model {model_name}..")
        model = GPT2LMHeadModel.from_pretrained(model_name)  # GPT2Model
        print(f"model loaded. Loading tokenizer {model_name}..")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("tokenizer loaded.")

    elif model_type in [model_types.ROBERTA]:
        print(f"loading model {model_name}..")
        model = RobertaForMaskedLM.from_pretrained(model_name)
        print(f"model loaded. Loading tokenizer {model_name}..")
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        print("tokenizer loaded.")
    elif model_type == model_types.GILBERTO:
        print(f"loading model {model_name}..")
        model = CamembertForMaskedLM.from_pretrained(model_name)
        print(f"model loaded. Loading tokenizer {model_name}..")
        tokenizer = CamembertTokenizer.from_pretrained(model_name, do_lower_case=True)
        print("tokenizer loaded.")
    else:
        raise ValueError(
            f"ValueError, unsupported model_name: {model_name}. "
            f"Supported models: Bert, Gpt."
        )

    # put model to device (GPU/CPU)
    device = torch.device(device)
    model.to(device)

    # eval mode; no dropout
    model.eval()

    return model, tokenizer


def load_model_and_tokenizer(
    model_type, model_name, dict_name=None, do_lower_case=False
):
    return load_pretrained(
        model_type, model_name, dict_name=dict_name, do_lower_case=do_lower_case
    )


def load_model(model_type, model_name, device):

    return load_pretrained(model_type, model_name, device=device)


def load_testset_data(file_path):
    with open(file_path, mode="r", encoding="utf-8") as json_file:
        # json_list = list(json_file)
        testset_data = json.load(json_file)

        # for i in data:
        #    print(i)

    return testset_data


def get_sentences_from_example(
    example: dict, sentences_per_example=2, sprouse_format=False
):
    by_sentence_variant_name = False

    if by_sentence_variant_name:
        # sentence_names_wh_wheter_islands = ['sentence_good_no_extraction',
        #                                     'sentence_bad_extraction',
        #                                     'sentence_good_extraction_resumption',
        #                                     'sentence_good_extraction_as_subject']
        sentence_names_wh_complex_np_islands = [
            "sentence_good_no_extraction",
            "sentence_bad_extraction",
            "sentence_good_no_island",
            "sentence_good_no_island_as_subject",
        ]
        sentence_names = sentence_names_wh_complex_np_islands
        sentences = []
        for sentence_name in sentence_names:
            sentences.append(example[sentence_name])
    else:
        sentences = list(example.values())[0:sentences_per_example]  #

    return sentences
