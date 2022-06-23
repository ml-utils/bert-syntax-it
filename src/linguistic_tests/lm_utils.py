import json
import os.path
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import Union

import sentencepiece as spm
import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.hipify.hipify_python import bcolors
from tqdm import tqdm
from transformers import AlbertTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertForMaskedLM  # BertModel as BertForMaskedLM  #
from transformers import BertTokenizer
from transformers import CamembertForMaskedLM
from transformers import CamembertTokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaModel
from transformers import RobertaTokenizer


class StrEnum(str, Enum):
    pass


class SentenceNames(StrEnum):
    SHORT_NONISLAND = "short_nonisland"
    LONG_NONISLAND = "long_nonisland"
    SHORT_ISLAND = "short_island"
    LONG_ISLAND = "long_island"
    SENTENCE_BAD = "sentence_bad"
    SENTENCE_GOOD = "sentence_good"
    SENTENCE_GOOD_2ND = "sentence_good_2nd"

    def __repr__(self):
        return self.name


class ScoringMeasures(StrEnum):
    LP = "LogProbability-softmax"
    PenLP = "PenaltyLogProbability-softmax"
    LL = "LogProbability-logistic"
    PLL = "PenaltyLogProbability-logistic"

    def __eq__(self, b):

        return (
            self is b
            or self.name == b
            or self.name == str(b)
            or (hasattr(b, "value") and b.value == self.name)
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)


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


class sent_idx:
    GOOD_1: int = 0
    # GOOD_1 = cython.declare(cython.int, 0)
    # GOOD_SENTENCE_1_IDX : int = 0
    BAD: int = 1
    GOOD_2: int = 2


class special_tokens:
    UNK: str = "[UNK]"


class ModelTypes(IntEnum):
    BERT = 0
    GPT = 1
    ROBERTA = 2
    GILBERTO = 3
    GEPPETTO = 4


class sentence_score_bases:
    SOFTMAX = 0
    LOGISTIC_FUN = 2


class DEVICES:
    CPU = "cpu"
    CUDA = "cuda:X"


BERT_LIKE_MODEL_TYPES = [ModelTypes.BERT, ModelTypes.ROBERTA, ModelTypes.GILBERTO]


class CustomModelWrapper:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch/")
        self.model = RobertaModel.from_pretrained(model_dir)


class CustomTokenizerWrapper:
    def __init__(self, model_dir: str = None, custom_tokens=None):

        # load tokenizer:
        if model_dir is None:
            model_dir = str(get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch/")
        _tokenizer_filename = "tokenizer.model"
        _tokenizer_filepath = os.path.join(model_dir, _tokenizer_filename)
        print(f"{_tokenizer_filepath=}")
        self.sp_tokenizer: SentencePieceProcessor = spm.SentencePieceProcessor(
            model_file=_tokenizer_filepath
        )

        self._ids_to_tokens = {
            id: self.sp_tokenizer.id_to_piece(id)
            for id in range(self.sp_tokenizer.get_piece_size())
        }

        self._tokens_to_ids = {
            self.sp_tokenizer.id_to_piece(id): id
            for id in range(self.sp_tokenizer.get_piece_size())
        }

        # in https://huggingface.co/transformers/v4.5.1/_modules/transformers/models/deberta_v2/tokenization_deberta_v2.html
        #         # self.vocab['[PAD]'] = 0
        #         # self.vocab['[CLS]'] = 1
        #         # self.vocab['[SEP]'] = 2
        #         # self.vocab['[UNK]'] = 3
        # in the fairseq Dictionary order:
        #         #         bos="<s>", bos/cls
        #         #         pad="<pad>",
        #         #         eos="</s>", eos/sep
        #         #         unk="<unk>",
        if custom_tokens is None:
            self._custom_tokens = {}
            # self.custom_tokens = {
            #     "[PAD]": 20000,
            #     "[CLS]": 20001,
            #     "[SEP]": 20002,
            #     "[UNK]": 20003,
            #     "[MASK]": 20004,
            # }
        else:
            self._custom_tokens = custom_tokens

    def _change_custom_tokens(self, custom_tokens):
        self._custom_tokens = custom_tokens
        print(f"the new custom tokens mapping is {self._custom_tokens=}")

    @property
    def bos_token(self):
        return self.id_to_piece(self.sp_tokenizer.bos_id())

    @property
    def eos_token(self):
        return self.id_to_piece(self.sp_tokenizer.eos_id())

    @property
    def unk_token(self):
        return self.id_to_piece(self.sp_tokenizer.unk_id())

    @property
    def pad_token(self):
        return self.id_to_piece(self.sp_tokenizer.pad_id())

    @property
    def all_special_tokens(self):
        # https://github.com/NVIDIA/NeMo/issues/2404
        # "No pad_token,eos_token,sep_token,cls_token in SentencePieceTokenizer"
        #
        # https://huggingface.co/transformers/v3.5.1/_modules/transformers/tokenization_albert.html
        #         bos_token="[CLS]",
        #         eos_token="[SEP]",
        #         unk_token="<unk>",
        #         sep_token="[SEP]",
        #         pad_token="<pad>",
        #         cls_token="[CLS]",
        #         mask_token="[MASK]",
        return [
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.pad_token,
        ]

    @property
    def vocab_size(self):
        return self.sp_tokenizer.get_piece_size()

    @property
    def ids_to_tokens(self):
        return self._ids_to_tokens

    # todo/check, fix: this is not actually compatible with gpt2 output of get_vocab (idx value probably has not same meaning)
    def get_vocab(self):
        return self._tokens_to_ids

    def get_piece_size(self):
        return self.sp_tokenizer.get_piece_size()

    def tokenize(self, text: str) -> list[str]:
        return self.sp_tokenizer.encode_as_pieces(text)

    def encode_as_pieces(self, text: str) -> list[str]:
        return self.sp_tokenizer.encode_as_pieces(text)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:

        ids = [self.piece_to_id(token) for token in tokens]
        return ids

    def convert_ids_to_tokens(self, ids: list[int]):

        tokens = [self.id_to_piece(id) for id in ids]
        return tokens

    def id_to_piece(self, id: int) -> Union[str, None]:  # typing.Optional[str]
        # id <=> piece conversion
        if id == -1:
            return None
        return self.sp_tokenizer.id_to_piece(id)

    def piece_to_id(self, token: str) -> int:
        # id <=> piece conversion
        if token is None:
            return None

        if token in self._custom_tokens.keys():
            return self._custom_tokens[token]
        else:
            return self.sp_tokenizer.piece_to_id(token)

    def encode_as_ids(self, text: str):
        return self.sp_tokenizer.encode_as_ids(text)

    def decode_pieces(self, tokens: list[str]):
        # decode: id => text
        return self.sp_tokenizer.decode_pieces(tokens)

    def decode_ids(self, ids: list[int]):
        # decode: id => text
        return self.sp_tokenizer.decode_ids(ids)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str = None, *model_args, **kwargs
    ):
        return CustomTokenizerWrapper(model_dir=pretrained_model_name_or_path)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_models_dir() -> Path:
    return get_project_root() / "models"


def get_syntactic_tests_dir() -> Path:
    return get_project_root() / "outputs"


def get_results_dir() -> Path:
    return get_project_root() / "results"


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
    model_name: str,
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
    elif model_type == ModelTypes.GPT:
        print(f"loading model {model_name}..")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"model loaded. Loading tokenizer {model_name}..")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("tokenizer loaded.")
    elif model_type == ModelTypes.BERT:
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

    elif model_type == ModelTypes.GEPPETTO:
        print(f"loading model {model_name}..")
        model = GPT2LMHeadModel.from_pretrained(model_name)  # GPT2Model
        print(f"model loaded. Loading tokenizer {model_name}..")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("tokenizer loaded.")

    elif model_type in [ModelTypes.ROBERTA]:
        print(f"loading model {model_name}..")
        model = RobertaForMaskedLM.from_pretrained(model_name)  # RobertaForMaskedLM
        # todo: try using RobertaModel.from_pretrained with a different output format
        print(f"model loaded. Loading tokenizer {model_name}..")

        if "bostromkaj" in model_name:
            tokenizer_model_path = os.path.join(model_name, "tokenizer.model")
            print(f"{tokenizer_model_path=}")
            tokenizer = AlbertTokenizer.from_pretrained(tokenizer_model_path)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        print("tokenizer loaded.")
    elif model_type == ModelTypes.GILBERTO:
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


def load_testset_data(file_path, examples_format="blimp"):
    if examples_format == "blimp":
        with open(file_path, mode="r", encoding="utf-8") as json_file:
            # json_list = list(json_file)
            testset_data = json.load(json_file)

            # for i in data:
            #    print(i)
        return testset_data
    elif examples_format in ["sprouse", "json_lines"]:
        print(f"loading testset file {file_path}..")
        with open(file_path, mode="r", encoding="utf-8") as json_file:
            json_list = list(json_file)
        print("testset loaded.")

        examples = []
        for json_str in tqdm(json_list):
            example = json.loads(json_str)
            # print(f"result: {example}")
            # print(isinstance(example, dict))
            # parsed_example = read_sentences_item(example)
            # sentence_good = example['sentence_good']
            # sentence_bad = example['sentence_bad']
            examples.append(
                example
            )  # {'sentence_good': sentence_good, 'sentence_bad': sentence_bad, 'sentence_good_2nd': ""})
        testset = {"sentences": examples}
        return testset
    else:
        raise ValueError(f"unrecognized testset file format arg: {examples_format}")


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


def assert_almost_equal(val1, val2, precision=13):
    # todo: convert this to a warning
    assert abs(val1 - val2) < 10 ** (-1 * precision), (
        f"val1:{val1}, val2: {val2}, " f"diff: {val1 - val2}"
    )
