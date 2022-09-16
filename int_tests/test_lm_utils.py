from unittest import TestCase

import pytest
import torch

from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import load_model
from src.linguistic_tests.lm_utils import ModelTypes


class TestLMUtils(TestCase):
    @pytest.mark.enable_socket
    def test_lowercasing_gilbert(self):
        model_name = "idb-ita/gilberto-uncased-from-camembert"  #
        model_type = ModelTypes.GILBERTO
        # tokenizer: CamembertTokenizer
        model, tokenizer = load_model(model_type, model_name, DEVICES.CPU)

        # check that lowecasing is done
        txt = "Io sono italiano e mi chiamo GilBERTo!"
        txt = txt.lower()
        input_ids = torch.tensor(tokenizer.encode(txt))
        input_ids_batch = input_ids.unsqueeze(0)
        print(f"input_ids: {input_ids}")
        print(f"input_ids_batch: {input_ids_batch}")
        print(f"input_ids redecoded: {tokenizer.convert_ids_to_tokens(input_ids)}")
        expected_input_ids_batch = torch.tensor(
            [[5, 755, 181, 1413, 25, 155, 12513, 14397, 16247, 31976, 6]]
        )

        token_list = tokenizer.convert_ids_to_tokens(tokenizer.encode(txt))
        expected_token_list = [
            "<s>",
            "▁io",
            "▁sono",
            "▁italiano",
            "▁e",
            "▁mi",
            "▁chiamo",
            "▁gil",
            "berto",
            "!",
            "</s>",
        ]

        self.assertListEqual(token_list, expected_token_list)
        assert torch.equal(input_ids_batch, expected_input_ids_batch)
