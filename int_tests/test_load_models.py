import os.path
from itertools import islice
from unittest import TestCase

import pytest
import torch
from _pytest._code.code import ExceptionInfo
from fairseq.models.roberta import RobertaModel as RobertaModel_FS
from linguistic_tests.lm_utils import CustomTokenizerWrapper
from linguistic_tests.lm_utils import get_models_dir
from pytest_socket import SocketBlockedError
from transformers import AlbertTokenizer
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import CamembertTokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaModel
from transformers import RobertaTokenizer


class TestLoadModels(TestCase):
    model_dir = str(
        get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch"
    )  # "../models/bostromkaj/bpe_20k_ep20_pytorch/"
    dict_name = "dict.txt"
    dict_path = os.path.join(model_dir, dict_name)
    torch_model_filename = "pytorch_model.bin"
    torch_model_path = model_dir + "/" + torch_model_filename

    def test_load_with_sentencepiece_unigram(self):
        self._test_load_with_sentencepiece_helper("bostromkaj/uni_20k_ep20_pytorch/")

    def test_load_with_sentencepiece_bpe(self):
        self._test_load_with_sentencepiece_helper("bostromkaj/bpe_20k_ep20_pytorch/")

    def _test_load_with_sentencepiece_helper(self, model_subdir: str):
        # filename = "tokenizer.model"
        # filepath = str(get_models_dir() / (model_subdir + filename))
        sp = CustomTokenizerWrapper(
            str(get_models_dir() / model_subdir)
        )  # spm.SentencePieceProcessor(model_file=filepath)

        print("encode: text => id")
        print(sp.encode_as_pieces("This is a test"))
        print(sp.encode_as_ids("This is a test"))

        print("decode: id => text")
        print(sp.decode_pieces(["▁This", "▁is", "▁a", "▁t", "est"]))
        print(sp.decode_ids([201, 39, 5, 379, 586]))

        vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
        print(f"vocab first {100} {vocabs[0:100]}")
        tokens = sp.encode_as_pieces("Hello world!")
        print(f"tokens: {tokens}")
        # assert tokens.tolist() == [0, 31414, 232, 328, 2]
        # print(roberta.decode(tokens))  # 'Hello world!'

        self.__test_tokenizer_helper(sp)

    def test_load_remotely(self):
        # remote calls are blocked, enabled with annotations  only for specific tests
        with pytest.raises(SocketBlockedError):
            _ = BertForMaskedLM.from_pretrained("bert-base")

    @pytest.mark.skip("using an edited config.json (not the default)")
    def test_load_with_AutoTokenizer_with_default_config_json(self):
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

    def test_load_with_AutoTokenizer_with_edited_config_json(self):
        with pytest.raises(OSError) as os_err:
            tokenizer = AutoTokenizer.from_pretrained(TestLoadModels.model_dir)
            self.__test_tokenizer_helper(tokenizer)
        self.assertInErrorMsg("Can't load tokenizer for ", os_err)
        self.assertInErrorMsg(
            " is the correct path to a directory containing all relevant files for a RobertaTokenizerFast tokenizer.",
            os_err,
        )

    def test_load_with_CamembertTokenizer(self):
        with pytest.raises(RuntimeError) as run_err:
            tokenizer = CamembertTokenizer.from_pretrained(TestLoadModels.dict_path)
            self.__test_tokenizer_helper(tokenizer)
        print(f"{run_err}=")
        self.assertInErrorMsg("Internal", run_err)
        self.assertInErrorMsg("sentencepiece_processor.cc", run_err)

    def test_load_as_huggingfaces_RobertaModel(self):
        roberta = RobertaModel.from_pretrained(TestLoadModels.model_dir)
        print(type(roberta))

        roberta2 = RobertaForMaskedLM.from_pretrained(TestLoadModels.model_dir)
        print(type(roberta2))

        # todo: asserts on the model
        # outputs = model(tokens_tensor, token_type_ids=segment_tensor)
        # check output type format, how to access the loss and token scores

    def test_load_as_huggingfaces_RobertaTokenizer(self):
        model_dir2 = str(
            get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch"  # + "/"
        )  # + "\\"
        with pytest.raises(OSError) as os_err:
            tokenizer = RobertaTokenizer.from_pretrained(model_dir2)
            self.__test_tokenizer_helper(tokenizer)
        self.assertInErrorMsg("Can't load tokenizer for ", os_err)
        self.assertInErrorMsg("make sure ", os_err)
        self.assertInErrorMsg(
            "is the correct path to a directory containing all relevant files for a RobertaTokenizer tokenizer.",
            os_err,
        )

    def test_load_as_custom_transformers_tokenizer(self):
        from tokenizers import Tokenizer

        filename = "tokenizer.model"  # nb: default and custom tokenizers saved
        # with hugginface transformers have a tokenizer.json instead (not a tokenizer.model file)
        # see https://huggingface.co/roberta-base/tree/main
        # and https://discuss.huggingface.co/t/creating-a-custom-tokenizer-for-roberta/2809
        filepath = get_models_dir() / ("bostromkaj/bpe_20k_ep20_pytorch/" + filename)
        filepath = str(filepath)
        print(f"{filepath=}")
        with pytest.raises(Exception) as stream_exception:
            tokenizer = Tokenizer.from_file(
                filepath  # expected: A path to a local JSON file representing a previously serialized
            )
            print(type(tokenizer))
        self.assertInErrorMsg("stream did not contain valid UTF-8", stream_exception)

    def test_load_with_fairseq_RobertaModel(self):

        with pytest.raises(OSError) as os_err:
            roberta = RobertaModel_FS.from_pretrained(
                TestLoadModels.model_dir,
            )
            # skipped because of raised error above:
            roberta.eval()  # disable dropout (or leave in train mode to finetune)
            tokens = roberta.encode("Hello world!")
            print(f"tokens: {tokens}")
            assert tokens.tolist() == [0, 31414, 232, 328, 2]
            print(roberta.decode(tokens))  # 'Hello world!'
        print(f"{os_err=}")
        self.assertInErrorMsg("Model file not found", os_err)
        self.assertInErrorMsg("model.pt", os_err)

        with pytest.raises(KeyError) as key_err:
            roberta = RobertaModel_FS.from_pretrained(
                TestLoadModels.model_dir,
                checkpoint_file="pytorch_model.bin",  # by default looks for a "model.pt", so a pythorch model file, note a .bin file (like roberta base from huggingfaces)
            )
            print(type(roberta))
        self.assertInErrorMsg("best_loss", key_err)

    def test_load_checkpoint_with_Torch(self):
        import torch

        # model = Net()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        from collections import OrderedDict

        checkpoint = torch.load(TestLoadModels.torch_model_path)

        assert isinstance(checkpoint, OrderedDict)
        assert len(checkpoint) == 201

        vocab_size = 20005
        hidden_size = 768
        intermediate_size = 3072
        max_position_embeddings = 514
        # num_attention_heads = 12
        # num_hidden_layers = 12
        num_labels = 2

        expected_keys_and_tensor_shapes = {
            "roberta.embeddings.word_embeddings.weight": (vocab_size, hidden_size),
            "roberta.embeddings.position_embeddings.weight": (
                max_position_embeddings,
                hidden_size,
            ),
            "roberta.embeddings.token_type_embeddings.weight": (1, hidden_size),
            "roberta.embeddings.LayerNorm.weight": (hidden_size,),
            "roberta.embeddings.LayerNorm.bias": (hidden_size,),
            "roberta.encoder.layer.11.attention.self.query.weight": (
                hidden_size,
                hidden_size,
            ),
            "roberta.encoder.layer.11.attention.self.query.bias": (hidden_size,),
            "roberta.encoder.layer.11.attention.self.key.weight": (
                hidden_size,
                hidden_size,
            ),
            "roberta.encoder.layer.11.attention.self.key.bias": (hidden_size,),
            "roberta.encoder.layer.11.attention.self.value.weight": (
                hidden_size,
                hidden_size,
            ),
            "roberta.encoder.layer.11.attention.self.value.bias": (hidden_size,),
            "roberta.encoder.layer.11.attention.output.dense.weight": (
                hidden_size,
                hidden_size,
            ),
            "roberta.encoder.layer.11.attention.output.dense.bias": (hidden_size,),
            "roberta.encoder.layer.11.attention.output.LayerNorm.weight": (
                hidden_size,
            ),
            "roberta.encoder.layer.11.attention.output.LayerNorm.bias": (hidden_size,),
            "roberta.encoder.layer.11.intermediate.dense.weight": (
                intermediate_size,
                hidden_size,
            ),
            "roberta.encoder.layer.11.intermediate.dense.bias": (intermediate_size,),
            "roberta.encoder.layer.11.output.dense.weight": (
                hidden_size,
                intermediate_size,
            ),
            "roberta.encoder.layer.11.output.dense.bias": (hidden_size,),
            "roberta.encoder.layer.11.output.LayerNorm.weight": (hidden_size,),
            "roberta.encoder.layer.11.output.LayerNorm.bias": (hidden_size,),
            "roberta.pooler.dense.weight": (hidden_size, hidden_size),
            "roberta.pooler.dense.bias": (hidden_size,),
            "qa_outputs.weight": (num_labels, hidden_size),
            "qa_outputs.bias": (num_labels,),
        }
        checkpoint_actual_keys = list(checkpoint.keys())
        for expected_key, expected_shape in expected_keys_and_tensor_shapes.items():
            assert expected_key in checkpoint_actual_keys
            assert isinstance(checkpoint[expected_key], torch.Tensor)
            assert checkpoint[expected_key].shape == expected_shape

        assert isinstance(checkpoint._metadata, OrderedDict)
        assert len(checkpoint._metadata) == 218
        expected_metadata_keys = [
            "",
            "roberta",
            "roberta.embeddings",
        ]
        metadata_actual_keys = list(checkpoint._metadata.keys())
        for key in expected_metadata_keys:
            assert key in metadata_actual_keys
            assert isinstance(checkpoint._metadata[key], dict)
            assert checkpoint._metadata[key] == {"version": 1}
            assert len(checkpoint._metadata[key]) == 1
            assert checkpoint._metadata[key]["version"] == 1

        # print(f"{type(checkpoint)}, len: {len(checkpoint)}")

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

    def assertInErrorMsg(self, expected_str, error: ExceptionInfo):
        if error.type in [FileNotFoundError]:
            msg = f"{str(error.value)} {error.value.args[1]}  {error.value.strerror}  {error.value.filename}"
        elif error.type in [RuntimeError, ValueError, OSError]:
            msg = error.value.args[0]
        else:  # KeyError?
            msg = str(error)
        self.assertIn(expected_str, msg)

    def test_load_with_TorchHub(self):
        print("loading with Torch..")

        # see https://pytorch.org/hub/huggingface_pytorch-transformers/
        # config = torch.hub.load("local", 'config', TestLoadModels.model_dir)  # repo_owner, repo_name = repo_info.split('/') ValueError: not enough values to unpack (expected 2, got 1)

        with pytest.raises(FileNotFoundError) as f_err:
            tokenizer = torch.hub.load(
                TestLoadModels.model_dir,
                "tokenizer",
                source="local",
                pretrained=True,
                # config=config
                # "local", "tokenizer", TestLoadModels.model_dir,  # config=config
            )
            self.__test_tokenizer_helper(tokenizer)
        print(f"{f_err=}")
        self.assertInErrorMsg("hubconf.py", f_err)

        with pytest.raises(FileNotFoundError) as f_err2:
            roberta = torch.hub.load(
                TestLoadModels.model_dir,  # repo_or_dir # 'pytorch/fairseq',
                "roberta",  # model arg: "..", # 'roberta.large' # model (string):
                # the name of a callable (entrypoint) defined in the
                # repo/dir's ``hubconf.py``.
                source="local",
                pretrained=True,
            )
            roberta.eval()
        self.assertInErrorMsg("No such file or directory", f_err2)
        self.assertInErrorMsg("hubconf.py", f_err2)
        return

        # https://github.com/facebookresearch/fairseq/tree/main/examples/roberta
        # Apply Byte-Pair Encoding (BPE) to input text:
        tokens = roberta.encode("Hello world!")
        print(f"tokens: {tokens}")
        assert tokens.tolist() == [0, 31414, 232, 328, 2]
        print(roberta.decode(tokens))  # 'Hello world!'

        # Extract features from RoBERTa:
        # Extract the last layer's features
        last_layer_features = roberta.extract_features(tokens)
        assert last_layer_features.size() == torch.Size([1, 5, 1024])
        # Extract all layer's features (layer 0 is the embedding layer)
        all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
        assert len(all_layers) == 25
        assert torch.all(all_layers[-1] == last_layer_features)

    @staticmethod
    def __test_tokenizer_helper(tokenizer):
        print(f"Tokenizer loaded: {type(tokenizer)}.")

        txt = "The pen is on the table"
        tokens = tokenizer.tokenize(txt)
        print(f"Sentence tokens: {tokens}")

        print(f"{txt} to {tokens=}")
        special_tokens_properties1 = [
            "bos_token",
            "unk_token",
            "eos_token",
            "pad_token",
        ]
        print(
            f"Special tokens: {tokenizer.all_special_tokens=}, {tokenizer.vocab_size=}"
        )
        for special_token_property in special_tokens_properties1:
            token = getattr(tokenizer, special_token_property)
            print(
                f"{special_token_property}: "
                f"{token=}"
                f", with id: {tokenizer.convert_tokens_to_ids([token])}"
            )

        special_tokens_properties2 = ["cls_token", "mask_token", "sep_token"]
        for special_token_property in special_tokens_properties2:
            if hasattr(tokenizer, special_token_property):
                token = getattr(tokenizer, special_token_property)
                print(
                    f"special_token_property: "
                    f"{token}"
                    f", with id: {tokenizer.convert_tokens_to_ids([token])}"
                )
            else:
                print(f"this tokenizer has no {special_token_property} property")

        vocab = tokenizer.get_vocab()
        max_prints = 20
        for token, idx in vocab.items():
            if idx > max_prints:
                break
            print(f"{token=}, {idx=}")

        print(f"vocab: {dict(islice(vocab.items(), 0, 20))}")
