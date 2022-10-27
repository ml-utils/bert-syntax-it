import json
import logging
import os.path
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
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


LIKERT_SCALE_POINTS = 7


class StrEnum(str, Enum):
    pass


class Conditions(StrEnum):
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

    def __str__(self):
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

    def __str__(self):
        return f"ModelType({self.name})"


class sentence_score_bases(IntEnum):
    SOFTMAX = 0
    LOGISTIC_FUN = 2


class DEVICES(StrEnum):
    CPU = "cpu"
    CUDA = "cuda:X"
    CUDA_0 = "cuda:0"


BERT_LIKE_MODEL_TYPES = [ModelTypes.BERT, ModelTypes.ROBERTA, ModelTypes.GILBERTO]
GPT_LIKE_MODEL_TYPES = [ModelTypes.GPT, ModelTypes.GEPPETTO]


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
        print(f"_tokenizer_filepath={_tokenizer_filepath}")
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
        print(
            f"the new custom tokens mapping is self._custom_tokens: {self._custom_tokens}"
        )

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

    def tokenize(self, text: str) -> List[str]:
        return self.sp_tokenizer.encode_as_pieces(text)

    def encode_as_pieces(self, text: str) -> List[str]:
        return self.sp_tokenizer.encode_as_pieces(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:

        ids = [self.piece_to_id(token) for token in tokens]
        return ids

    def convert_ids_to_tokens(self, ids: List[int]):

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

    def decode_pieces(self, tokens: List[str]):
        # decode: id => text
        return self.sp_tokenizer.decode_pieces(tokens)

    def decode_ids(self, ids: List[int]):
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


def get_penalty_term(text_lenght, alpha=0.8):  # alpha=1.0, 0.8, 0.6
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
    device: DEVICES,
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
            print(f"tokenizer_model_path={tokenizer_model_path}")
            tokenizer = AlbertTokenizer.from_pretrained(tokenizer_model_path)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        print("tokenizer loaded.")
    elif model_type == ModelTypes.GILBERTO:
        print(f"loading model {model_name}..")
        tokenizer = CamembertTokenizer.from_pretrained(model_name, do_lower_case=True)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = CamembertForMaskedLM.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
        print(f"model loaded. Loading tokenizer {model_name}..")

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
    model_type, model_name, device, dict_name=None, do_lower_case=False
):
    return load_pretrained(
        model_type,
        model_name,
        device=device,
        dict_name=dict_name,
        do_lower_case=do_lower_case,
    )


def load_model(model_type, model_name, device: DEVICES):

    return load_pretrained(model_type, model_name, device=device)


def load_testset_data(file_path, examples_format="blimp") -> List[dict]:
    if examples_format == "blimp":
        with open(file_path, mode="r", encoding="utf-8") as json_file:
            # json_list = list(json_file)
            testset_data = json.load(json_file)

            # for i in data:
            #    print(i)

    elif examples_format in ["sprouse", "json_lines"]:
        print(f"loading testset file {file_path}..")
        with open(file_path, mode="r", encoding="utf-8-sig") as json_file:
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
        testset_data = {"sentences": examples}

    else:
        raise ValueError(f"unrecognized testset file format arg: {examples_format}")

    # integrity checks:
    testset_examples: List[dict] = testset_data["sentences"]
    assert (
        Conditions.SENTENCE_GOOD in testset_examples[0].keys()
        or Conditions.SHORT_NONISLAND in testset_examples[0].keys()
    )

    return testset_data


def get_sentences_from_example(
    example: dict, sentences_per_example=2, sprouse_format=False
) -> List:
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
    almost_equal = abs(val1 - val2) < 10 ** (-1 * precision)
    msg = (
        f"These values are not almost equal: val1:{val1}, val2: {val2}, "
        f"diff: {val1 - val2}, precision={precision}"
    )
    # assert almost_equal, msg
    if not almost_equal:
        print_orange(msg)
        logging.warning(msg)


MODEL_TYPES_AND_NAMES_EN: Dict[str, ModelTypes] = {
    "gpt2": ModelTypes.GPT,
    "gpt2-medium": ModelTypes.GPT,
    "gpt2-large": ModelTypes.GPT,
    "bert-base-uncased": ModelTypes.BERT,
    "bert-base-cased": ModelTypes.BERT,
    "bert-large-uncased": ModelTypes.BERT,
    "bert-large-cased": ModelTypes.BERT,
    "roberta-base": ModelTypes.ROBERTA,
    "roberta-large": ModelTypes.ROBERTA,
}

MODEL_TYPES_AND_NAMES_IT: Dict[str, ModelTypes] = {
    "LorenzoDeMattei/GePpeTto": ModelTypes.GEPPETTO,
    "dbmdz/bert-base-italian-xxl-cased": ModelTypes.BERT,
    "dbmdz/bert-base-italian-cased": ModelTypes.BERT,
    "idb-ita/gilberto-uncased-from-camembert": ModelTypes.GILBERTO,
}

MODEL_NAMES_IT: Dict[ModelTypes, str] = {
    ModelTypes.GEPPETTO: "LorenzoDeMattei/GePpeTto",
    ModelTypes.BERT: "dbmdz/bert-base-italian-xxl-cased",
    ModelTypes.GILBERTO: "idb-ita/gilberto-uncased-from-camembert",
}  # ModelTypes.GPT # ModelTypes.ROBERTA  #
MODEL_NAMES_EN: Dict[ModelTypes, str] = {
    ModelTypes.BERT: "bert-base-uncased",  # "bert-large-uncased"  #
    ModelTypes.GPT: "gpt2-large",
    ModelTypes.ROBERTA: "roberta-large",
}

BLIMP_TESTSETS_ROOT_FILENAMES = [
    "adjunct_island",
    "complex_NP_island",
    # "coordinate_structure_constraint_complex_left_branch",
    # "coordinate_structure_constraint_object_extraction",
    # "left_branch_island_echo_question",
    # "left_branch_island_simple_question",
    # "sentential_subject_island",
    "wh_island",  # .jsonl
]


def _get_test_session_descr(
    dataset_source: str,
    dependency_type_prefix: str,
    model_descr: str,
    score_name: str = "",
):
    session_descr = (
        f"{dataset_source[:7]}_{dependency_type_prefix[:2]}_{model_descr}_{score_name}"
    )
    session_descr = session_descr.replace(" ", "_").replace("/", "_")
    return session_descr


SPROUSE_TESTSETS_ROOT_FILENAMES_WH = [
    "wh_adjunct_island",
    "wh_complex_np",
    "wh_subject_island",
    "wh_whether_island",
]
SPROUSE_TESTSETS_ROOT_FILENAMES_RC = [
    "rc_adjunct_island",
    "rc_complex_np",
    "rc_subject_island",
    "rc_wh_island",  # fixme: rc_wh_island empty file
]
CUSTOM_IT_ISLAND_TESTSETS_ROOT_FILENAMES = [
    # "wh_adjunct_islands",
    # "wh_complex_np_islands",
    # "wh_whether_island",
    # "wh_subject_islands",
    "wh_whether_island",
    "wh_complex_np_islands",
    "wh_subject_islands",
    "wh_adjunct_islands",
]


class ExperimentalDesigns(IntEnum):
    MINIMAL_PAIRS = 0
    FACTORIAL = 1
    MINIMAL_PAIRS_VARIATIONS = 2


class DataSources(StrEnum):
    BLIMP_EN = "Blimp Warstadt et al. 2020"  # "Blimp paper"
    SPROUSE = "Sprouse et al. 2016"  # "sprouse", "Sprouse et al. 2016"
    MADEDDU = "Madeddu"
    VARIATIONS = "variations"

    def __repr__(self):
        return self.value

    def __eq__(self, b):

        return (
            self is b
            or self.name == b
            or self.name.lower() == str(b).lower()
            or (hasattr(b, "name") and b.name == self.name)
            or (hasattr(b, "value") and b.value.lower() == self.name.lower())
        )

    def __hash__(self):
        return hash(self.name)


def get_testset_params(
    tests_subdir,
) -> Tuple[List[str], str, DataSources, ExperimentalDesigns]:
    if tests_subdir == "syntactic_tests_it/":  # "mdd2/"
        testsets_root_filenames = CUSTOM_IT_ISLAND_TESTSETS_ROOT_FILENAMES
        broader_test_type = "it_tests"
        dataset_source = DataSources.MADEDDU
        experimental_design = ExperimentalDesigns.FACTORIAL
    elif tests_subdir == "sprouse/":
        testsets_root_filenames = SPROUSE_TESTSETS_ROOT_FILENAMES_WH  # SPROUSE_TESTSETS_ROOT_FILENAMES_RC + SPROUSE_TESTSETS_ROOT_FILENAMES_WH
        broader_test_type = "sprouse"
        dataset_source = DataSources.SPROUSE
        experimental_design = ExperimentalDesigns.FACTORIAL
    elif tests_subdir == "blimp/from_blim_en/islands/":
        testsets_root_filenames = BLIMP_TESTSETS_ROOT_FILENAMES
        broader_test_type = "blimp"
        dataset_source = DataSources.BLIMP_EN
        experimental_design = ExperimentalDesigns.MINIMAL_PAIRS
    elif tests_subdir == "variations/":
        testsets_root_filenames = [
            # "wh_complex_np_islands",
            # "wh_complex_np_islands_2", # all items with the "ha intuito che" construct
            # "wh_complex_np_islands_3",
            # "wh_complex_np_islands_4",  # like n.2, plus merged Sprouse and Madeddu testsuites and added more items with variations
            # "wh_complex_np_islands_5",  # like n.4, but replaced "percepire"/"avere la percezione che" with "avvertire"/"avere il sentore che"
            # "wh_complex_np_islands_6",  # like n.5, but replaced with "avere il sentore che / ..avvertito che"
            # "wh_complex_np_islands_7",  # like n.6, but replaced with "avere la premonizione che / ..premonito che"
            # "wh_complex_np_islands_8",  # like n.7, but replaced main clause with "sapeva che" / "sapeva del fatto che"
            # "eri a conoscenza che" / "conoscevi il fatto che" (NB: "light nouns" effect that avoids the island effect?)
            "wh_complex_np_islands_9",  # like n.7, but replaced main clause with  "sapevi che" / "conoscevi la novità che"
            # like n.4, but replaced "intuire"/"avere l'intuizione" with "percepire"/"avere la percezione che"
            "wh_adjunct_islands_2",  # all in present perfect as the complex np
            "wh_whether_island",
            # "wh_whether_island2",
            # "wh_whether_island3",
            "wh_subject_islands",  # same as the standard test suite
        ]
        broader_test_type = "variations"
        dataset_source = DataSources.VARIATIONS
        experimental_design = ExperimentalDesigns.FACTORIAL
    else:
        raise ValueError(f"Invalid tests_subdir specified: {tests_subdir}")

    return (
        testsets_root_filenames,
        broader_test_type,
        dataset_source,
        experimental_design,
    )


def get_num_of_available_cuda_gpus():
    import torch

    if torch.cuda.is_available():
        print_red("Cuda is available")
        return torch.cuda.device_count()
    else:
        print_red("Cuda is NOT available")
        return 0


def discretize(
    x,
    groups,
    labels=None,
    retbins: bool = False,
    use_quantiles=False,
):
    # print(
    #     f"discretize: {len(x)}, groups={groups}, labels={labels}, retbins={retbins}, use_quantiles={use_quantiles}"
    # )
    if use_quantiles:
        return pd.qcut(x, q=groups, labels=labels, retbins=retbins)
    else:
        # use bins
        return pd.cut(x, bins=groups, labels=labels, retbins=retbins)
