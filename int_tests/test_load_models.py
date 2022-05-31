import os.path
from itertools import islice
from unittest import TestCase

import torch
from transformers import AlbertTokenizer
from transformers import BertTokenizer

# import pytest


class TestLoadModels(TestCase):
    model_dir = "../models/bostromkaj/bpe_20k_ep20_pytorch/"
    dict_name = "dict.txt"
    dict_path = os.path.join(model_dir, dict_name)

    def load_with_BertTokenizer(self):
        # fixme: loaded tokenizer doen not work properly (all tokens are UNK)
        #  it's using sentencepiece module, with split tokens like:
        #  'This is a test' = ['▁This', '▁is', '▁a', '▁', 't', 'est']
        #  'Hello world' = '▁He', 'll', 'o', '▁world'

        # from the paper:
        # "We pre-train four transformer masked language models
        # using the architecture and training objective of
        # ROBERTA-BASE (Liu et al., 2019) using the reference
        # fairseq implementation (Ott et al., 2019)
        # .. "We subsequently fine-tune each of the pretrained
        # English models .. We base our fine-tuning implementations on
        # those of the transformers toolkit (Wolf et al.,
        # 2019)."

        print("loading with BertTokenizer..")
        tokenizer = BertTokenizer.from_pretrained(TestLoadModels.dict_path)
        print(f"Tokenizer loaded: {type(tokenizer)}.")

        txt = "The pen is on the table"
        tokens = tokenizer.tokenize(txt)
        print(f"Sentence tokens: {tokens}")

        print(f"{txt} to {tokens=}")
        print(
            f"Special tokens: {tokenizer.bos_token=}, {tokenizer.unk_token=}, "
            f"{tokenizer.cls_token=}, {tokenizer.eos_token=}, {tokenizer.mask_token=}, "
            f"{tokenizer.sep_token=}, {tokenizer.pad_token=}, "
            f"\n{tokenizer.all_special_tokens=}, "
            f"\nvocab size: {tokenizer.vocab_size}"
        )
        vocab = tokenizer.get_vocab()
        for vocab_item in vocab.items():
            print(
                f"{vocab_item=}, {type(vocab_item)=}, {type(vocab_item[0])=}, {type(vocab_item[1])=}"
            )

        print(f"vocab: {dict(islice(vocab.items(), 0, 20))}")

    def test_load_with_AlbertTokenizer(self):
        print("loading with AlbertTokenizer..")
        tokenizer = AlbertTokenizer.from_pretrained(TestLoadModels.dict_path)
        print(f"Tokenizer loaded: {type(tokenizer)}.")

        txt = "The pen is on the table"
        tokens = tokenizer.tokenize(txt)
        print(f"Sentence tokens: {tokens}")

    def test_load_with_Torch(self):
        print("loading with Torch..")
        tokenizer = torch.hub.load(
            TestLoadModels.model_dir, "tokenizer", source="local", pretrained=True
        )
        print(f"Tokenizer loaded: {type(tokenizer)}.")
