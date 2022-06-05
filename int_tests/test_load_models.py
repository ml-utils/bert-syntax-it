import os.path
from itertools import islice
from unittest import TestCase

import pytest
import torch
from linguistic_tests.lm_utils import get_models_dir
from pytest_socket import SocketBlockedError
from transformers import AlbertTokenizer
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import CamembertTokenizer
from transformers import RobertaTokenizer

# import pytest


class TestLoadModels(TestCase):
    model_dir = str(
        get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch"
    )  # "../models/bostromkaj/bpe_20k_ep20_pytorch/"
    dict_name = "dict.txt"
    dict_path = os.path.join(model_dir, dict_name)
    torch_model_filename = "pytorch_model.bin"
    torch_model_path = model_dir + "/" + torch_model_filename

    def test_load_remotely(self):
        # todo: re enable remote calls for integration tests
        with pytest.raises(SocketBlockedError):
            _ = BertForMaskedLM.from_pretrained("bert-base")

    @pytest.mark.skip(
        "fails if run from pytest, passes from jb (pycharm) pytest runner"
    )
    def test_load_with_AutoTokenizer(self):
        with pytest.raises(ValueError) as val_err:
            tokenizer = AutoTokenizer.from_pretrained(TestLoadModels.model_dir)
            self.__test_tokenizer_helper(tokenizer)
        print(f"{val_err}=")
        self.assertInErrorMsg("Unrecognized model", val_err)
        self.assertInErrorMsg(
            "Should have a `model_type` key in its config.json, or contain one of the following strings in its name",
            val_err,
        )
        self.assertInErrorMsg("bert, openai-gpt, gpt2, transfo-xl, xlnet", val_err)

    def test_load_with_CamembertTokenizer(self):
        with pytest.raises(RuntimeError) as run_err:
            tokenizer = CamembertTokenizer.from_pretrained(TestLoadModels.dict_path)
            self.__test_tokenizer_helper(tokenizer)
        print(f"{run_err}=")
        self.assertInErrorMsg("Internal", run_err)
        self.assertInErrorMsg("sentencepiece_processor.cc", run_err)

    @pytest.mark.skip("todo")
    def test_load_with_RobertaTokenizer(self):

        # Load the model in fairseq
        from fairseq.models.roberta import RobertaModel

        roberta = RobertaModel.from_pretrained(
            TestLoadModels.model_dir
        )  # , checkpoint_file='pytorch_model.bin'
        roberta.eval()  # disable dropout (or leave in train mode to finetune)
        tokens = roberta.encode("Hello world!")
        print(f"tokens: {tokens}")
        assert tokens.tolist() == [0, 31414, 232, 328, 2]
        print(roberta.decode(tokens))  # 'Hello world!'

        tokenizer = RobertaTokenizer.from_pretrained(
            r"E:/dev/code/bert-syntax-it/models/bostromkaj/bpe_20k_ep20_pytorch/"
        )
        # TestLoadModels.model_dir + "/", do_lower_case=True)
        # tokenizer = RobertaTokenizer.from_pretrained(
        self.__test_tokenizer_helper(tokenizer)

    def test_load_with_Torch(self):
        import torch

        # model = Net()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        checkpoint = torch.load(TestLoadModels.torch_model_path)
        print(f"{type(checkpoint)}, len: {len(checkpoint)}")

        # gives "roberta.embeddings"

        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        #
        # model.eval()

    def test_load_with_BertTokenizer(self):
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

        # see also:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_pytorch_checkpoint_to_original_tf.py

        print("loading with BertTokenizer..")
        tokenizer = BertTokenizer.from_pretrained(TestLoadModels.dict_path)
        self.__test_tokenizer_helper(tokenizer)

    def test_load_with_AlbertTokenizer(self):
        print("loading with AlbertTokenizer..")

        with pytest.raises(RuntimeError) as run_err:
            tokenizer = AlbertTokenizer.from_pretrained(TestLoadModels.dict_path)
            self.__test_tokenizer_helper(tokenizer)
        print(f"{run_err=}")
        self.assertInErrorMsg("Internal", run_err)
        self.assertInErrorMsg("sentencepiece_processor.cc", run_err)

    def assertInErrorMsg(self, expected_str, error):
        if error.type in [FileNotFoundError]:
            msg = error.value.filename
        elif error.type in [RuntimeError, ValueError]:
            msg = error.value.args[0]
        else:
            msg = str(error)
        self.assertIn(expected_str, msg)

    def test_load_with_TorchHub(self):
        print("loading with Torch..")

        # see https://pytorch.org/hub/huggingface_pytorch-transformers/
        # config = torch.hub.load("local", 'config', TestLoadModels.model_dir)

        with pytest.raises(FileNotFoundError) as f_err:
            tokenizer = torch.hub.load(
                TestLoadModels.model_dir,
                "tokenizer",
                source="local",
                pretrained=True
                # "local", "tokenizer", TestLoadModels.model_dir,  # config=config
            )
            self.__test_tokenizer_helper(tokenizer)
        print(f"{f_err=}")
        self.assertInErrorMsg("hubconf.py", f_err)

    @staticmethod
    def __test_tokenizer_helper(tokenizer):
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
        max_prints = 20
        for token, idx in vocab.items():
            if idx > max_prints:
                break
            print(f"{token=}, {idx=}")

        print(f"vocab: {dict(islice(vocab.items(), 0, 20))}")
