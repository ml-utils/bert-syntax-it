import builtins
import json
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from linguistic_tests.lm_utils import DEVICES
from linguistic_tests.lm_utils import get_models_dir
from linguistic_tests.lm_utils import get_pen_score
from linguistic_tests.lm_utils import get_penalty_term
from linguistic_tests.lm_utils import get_project_root
from linguistic_tests.lm_utils import get_sentences_from_example
from linguistic_tests.lm_utils import get_syntactic_tests_dir
from linguistic_tests.lm_utils import load_model
from linguistic_tests.lm_utils import load_model_and_tokenizer
from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import print_red
from linguistic_tests.lm_utils import red_txt
from linguistic_tests.testset import parse_example
from linguistic_tests.testset import SPROUSE_SENTENCE_TYPES
from torch.utils.hipify.hipify_python import bcolors
from transformers import BertForMaskedLM as BRT_M
from transformers import BertTokenizer as BRT_T
from transformers import CamembertForMaskedLM as CM_M
from transformers import CamembertTokenizer as CM_T
from transformers import GPT2LMHeadModel as GPT_M
from transformers import GPT2Tokenizer as GPT_T
from transformers import RobertaForMaskedLM as RB_M
from transformers import RobertaTokenizer as RB_T


class TestLMUtils(TestCase):
    def test_get_pen_score(self):
        unnormalized_score = 0.5
        text_len = 1
        assert get_pen_score(unnormalized_score, text_len) > get_pen_score(
            unnormalized_score, text_len + 1
        )

    def test_get_penalty_term(self):

        assert get_penalty_term(text_lenght=1, alpha=1) == 1
        assert get_penalty_term(text_lenght=1, alpha=2) == 1

        assert get_penalty_term(text_lenght=0, alpha=1) == 5 / 6
        assert get_penalty_term(text_lenght=2, alpha=1) == 7 / 6

        assert get_penalty_term(text_lenght=0) < 1
        assert get_penalty_term(text_lenght=1) == 1
        assert get_penalty_term(text_lenght=2) > 1
        assert get_penalty_term(text_lenght=3) < get_penalty_term(text_lenght=4)

    def test_get_project_root(self):
        p = get_project_root()
        assert p.exists()
        assert p.is_dir()
        # todo, fixme: these dir tests should be moved to integration tests
        # for subdir_name in ['src', 'unit_tests', 'models', 'outputs', 'int_tests']:
        #    assert len(list(p.glob(subdir_name))) == 1, f'from {p.name} no {subdir_name}, subdirs: {str(list(p.glob("""*""")))}'

    def test_get_models_dir(self):
        p = get_models_dir()
        assert p.name == "models"
        # assert p.exists()
        # assert p.is_dir()

    def test_get_syntactic_tests_dir(self):
        p = get_syntactic_tests_dir()
        assert p.name == "outputs"

    def test_get_sentences_from_example(self):
        example_blimp_it_str = """  {"sentence_good_no_extraction": "Chi conosceva la novità che ti avrebbero affidato l’incarico?",
  "sentence_bad_extraction": "Cosa conoscevi la novità che ti avrebbero affidato?",
  "sentence_good_no_island": "Cosa sapevi che ti avrebbero affidato?",
  "field": "syntax",
  "linguistics_term": "island_effects",
  "UID": "wh_island",
  "island_subcategory": "complex_np_island",
  "pairID": "11"}"""
        example_blimp_it_dict = json.loads(example_blimp_it_str)
        sentences = get_sentences_from_example(
            example_blimp_it_dict, sentences_per_example=3
        )
        assert isinstance(sentences, list)
        assert len(sentences) == 3

        example_blimp_en_str = """{"sentence_good": "What had Rebecca forgotten she tours?",
        "sentence_bad": "What had Rebecca forgotten who tours?", "one_prefix_prefix":
        "What had Rebecca forgotten", "one_prefix_word_good": "she", "one_prefix_word_bad":
        "who", "field": "syntax", "linguistics_term": "island_effects", "UID": "wh_island",
        "simple_LM_method": true, "one_prefix_method": true, "two_prefix_method": false,
        "lexically_identical": false, "pairID": "67"}"""
        example_blimp_en_dict = json.loads(example_blimp_en_str)
        sentences = get_sentences_from_example(example_blimp_en_dict)
        assert isinstance(sentences, list)
        assert len(sentences) == 2

    def test_get_sentences_from_example_2(self):

        example_data = get_basic_example_data_dict()

        sentences1 = get_sentences_from_example(
            example_data, sentences_per_example=4, sprouse_format=None
        )
        expected_sent_types = SPROUSE_SENTENCE_TYPES
        parsed_example = parse_example(example_data, expected_sent_types)

        # zip loop and assert that the sentences are in the same order
        # nb for get_sentences_from_example() implementation, it actually depends
        # on the order in which the ditctionary items are declared
        for idx, tsent in enumerate(parsed_example.sentences):
            assert tsent.sent.txt == sentences1[idx]

    @patch.object(
        CM_T, CM_T.from_pretrained.__name__, return_value=Mock(spec=CM_T)
    )  # "from_pretrained"
    @patch.object(CM_M, CM_M.from_pretrained.__name__, return_value=Mock(spec=CM_M))
    @patch.object(RB_T, RB_T.from_pretrained.__name__, return_value=Mock(spec=RB_T))
    @patch.object(RB_M, RB_M.from_pretrained.__name__, return_value=Mock(spec=RB_M))
    @patch.object(GPT_M, GPT_M.from_pretrained.__name__, return_value=Mock(spec=GPT_M))
    @patch.object(GPT_T, GPT_T.from_pretrained.__name__, return_value=Mock(spec=GPT_T))
    @patch.object(BRT_M, BRT_M.from_pretrained.__name__, return_value=Mock(spec=BRT_M))
    @patch.object(BRT_T, BRT_T.from_pretrained.__name__, return_value=Mock(spec=BRT_T))
    def test_load_model(self, mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8):

        for mock in [mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8]:
            assert mock in [
                CM_T.from_pretrained,
                CM_M.from_pretrained,
                RB_T.from_pretrained,
                RB_M.from_pretrained,
                GPT_M.from_pretrained,
                GPT_T.from_pretrained,
                BRT_M.from_pretrained,
                BRT_T.from_pretrained,
            ]

        self.__test_load_model_helper(
            ModelTypes.BERT, "bert-base-uncased", BRT_M, BRT_T
        )
        self.__test_load_model_helper(ModelTypes.GPT, "gpt2", GPT_M, GPT_T)
        self.__test_load_model_helper(
            ModelTypes.GEPPETTO, "LorenzoDeMattei/GePpeTto", GPT_M, GPT_T
        )
        self.__test_load_model_helper(ModelTypes.ROBERTA, "roberta-base", RB_M, RB_T)
        self.__test_load_model_helper(
            ModelTypes.GILBERTO, "idb-ita/gilberto-uncased-from-camembert", CM_M, CM_T
        )

    @staticmethod
    def __test_load_model_helper(
        model_type, model_name, expected_model_class, expected_tokenizer_class
    ):
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)
        assert isinstance(model, expected_model_class)
        assert isinstance(tokenizer, expected_tokenizer_class)

    @patch.object(CM_T, CM_T.from_pretrained.__name__, return_value=Mock(spec=CM_T))
    @patch.object(CM_M, CM_M.from_pretrained.__name__, return_value=Mock(spec=CM_M))
    @patch.object(RB_T, RB_T.from_pretrained.__name__, return_value=Mock(spec=RB_T))
    @patch.object(RB_M, RB_M.from_pretrained.__name__, return_value=Mock(spec=RB_M))
    @patch.object(GPT_M, GPT_M.from_pretrained.__name__, return_value=Mock(spec=GPT_M))
    @patch.object(GPT_T, GPT_T.from_pretrained.__name__, return_value=Mock(spec=GPT_T))
    @patch.object(BRT_M, BRT_M.from_pretrained.__name__, return_value=Mock(spec=BRT_M))
    @patch.object(BRT_T, BRT_T.from_pretrained.__name__, return_value=Mock(spec=BRT_T))
    def test_load_model_and_tokenizer(
        self, mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8
    ):

        for mock in [mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8]:
            assert mock in [
                CM_T.from_pretrained,
                CM_M.from_pretrained,
                RB_T.from_pretrained,
                RB_M.from_pretrained,
                GPT_M.from_pretrained,
                GPT_T.from_pretrained,
                BRT_M.from_pretrained,
                BRT_T.from_pretrained,
            ]

        bert_name = "bert-base-uncased"
        bert, b_tokenizer = load_model_and_tokenizer(ModelTypes.BERT, bert_name)
        assert isinstance(bert, BRT_M)
        assert isinstance(b_tokenizer, BRT_T)

        gpt2_name = "gpt2"
        gpt2, g_tokenizer = load_model_and_tokenizer(ModelTypes.GPT, gpt2_name)
        assert isinstance(gpt2, GPT_M)
        assert isinstance(g_tokenizer, GPT_T)

        gpt2_it_name = "LorenzoDeMattei/GePpeTto"
        gpt2_it, g_it_tokenizer = load_model_and_tokenizer(ModelTypes.GPT, gpt2_it_name)
        assert isinstance(gpt2_it, GPT_M)
        assert isinstance(g_it_tokenizer, GPT_T)

        rob_name = "roberta-base"
        roberta, rob_tokenizer = load_model_and_tokenizer(ModelTypes.ROBERTA, rob_name)
        assert isinstance(roberta, RB_M)
        assert isinstance(rob_tokenizer, RB_T)

        gil_name = "idb-ita/gilberto-uncased-from-camembert"
        roberta, rob_tokenizer = load_model_and_tokenizer(ModelTypes.GILBERTO, gil_name)
        assert isinstance(roberta, CM_M)
        assert isinstance(rob_tokenizer, CM_T)

        unknown_model_type = 10
        with self.assertRaises(ValueError):
            load_model_and_tokenizer(unknown_model_type, "")

    @pytest.mark.skip("todo")
    def test_load_testset_data(self):
        load_testset_data()
        raise NotImplementedError

    def test_red_txt(self):
        txt = "Lorem"
        result = red_txt(txt)
        assert isinstance(result, str)
        assert len(result) > len(txt)

    @patch.object(builtins, builtins.print.__name__)  # @patch("builtins.print")
    def test_print_in_color(self, mock_print: Mock):
        assert builtins.print is mock_print

        txt = "Lorem"
        print_orange(txt)
        mock_print.assert_called_with(bcolors.WARNING + txt + bcolors.ENDC)
        print_red(txt)
        mock_print.assert_called_with(bcolors.FAIL + txt + bcolors.ENDC)


def get_basic_example_data_str():
    return """{"short_nonisland": "Chi dice che lo studio rilascerà la nuova versione?",
     "long_nonisland": "Che cosa dici che lo studio rilascerà?",
     "short_island": "Chi aumenterà le vendite se rilascerà la nuova versione?",
     "long_island": "Che cosa lo studio aumenterà le vendite se rilascerà?"}"""


def get_basic_example_data_dict():
    return json.loads(get_basic_example_data_str())
