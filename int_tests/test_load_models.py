import os.path
from unittest import TestCase

from transformers import AlbertTokenizer

# import pytest


class TestLoadModels(TestCase):
    def test_load_with_AlbertTokenizer(self):
        model_dir = "../models/bostromkaj/bpe_20k_ep20_pytorch/"
        dict_name = "dict.txt"
        dict_path = os.path.join(model_dir, dict_name)
        print("loading with AlbertTokenizer..")
        tokenizer = AlbertTokenizer.from_pretrained(dict_path)
        print(f"Tokenizer loaded: {type(tokenizer)}.")

        txt = "The pen is on the table"
        tokens = tokenizer.tokenize(txt)
        print(f"Sentence tokens: {tokens}")
