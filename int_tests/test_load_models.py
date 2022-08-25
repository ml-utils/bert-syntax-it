import os.path
from itertools import islice
from unittest import TestCase

import pytest
import torch
from _pytest._code.code import ExceptionInfo
from pytest_socket import SocketBlockedError
from transformers import AlbertTokenizer
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import CamembertTokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers.convert_slow_tokenizer import SentencePieceExtractor

from int_tests.integration_tests_utils import is_internet_on
from src.linguistic_tests.bert_utils import convert_ids_to_tokens
from src.linguistic_tests.bert_utils import get_bert_output_single_masking
from src.linguistic_tests.lm_utils import CustomTokenizerWrapper
from src.linguistic_tests.lm_utils import get_models_dir


def _is_fairseq_installed_helper() -> bool:
    is_installed = False

    name = "fairseq"
    import sys
    from importlib.util import find_spec
    from importlib.util import module_from_spec

    if name in sys.modules:
        is_installed = True
        print(f"{name!r} is installed, already in sys.modules")
    else:
        spec = find_spec(name)
        if spec is not None:
            # performing the actual import ...
            _ = module_from_spec(spec)  # module =
            # sys.modules[name] = module
            # spec.loader.exec_module(module)
            is_installed = True
            print(f"{name!r} is installed.")
        else:
            print(f"can't find the {name!r} module")

    return is_installed


class TestLoadModels(TestCase):

    is_fairseq_installed = _is_fairseq_installed_helper()

    model_bpe_subdirs = "bostromkaj/bpe_20k_ep20_pytorch"
    model_uni_subdirs = "bostromkaj/uni_20k_ep20_pytorch"
    model_bpe_edited_subdirs = "bostromkaj/bpe_20k_ep20_pytorch_edited"
    model_uni_edited_subdirs = "bostromkaj/uni_20k_ep20_pytorch_edited"
    model_dir_uni = str(get_models_dir() / model_uni_subdirs)
    model_dir_bpe = str(get_models_dir() / model_bpe_subdirs)
    model_dir_bpe_edited = str(get_models_dir() / model_bpe_edited_subdirs)
    model_dir_uni_edited = str(get_models_dir() / model_uni_edited_subdirs)
    dict_name = "dict.txt"
    sentencepiece_tokenizer_filename = "tokenizer.model"
    torch_model_filename = "pytorch_model.bin"
    vocab_filename = "vocab.json"
    dict_path_uni = os.path.join(model_dir_uni, dict_name)
    tok_bpe_path = os.path.join(model_dir_bpe, sentencepiece_tokenizer_filename)
    tok_uni_path = os.path.join(model_dir_uni, sentencepiece_tokenizer_filename)
    torch_model_path_bpe = os.path.join(model_dir_bpe, torch_model_filename)
    torch_model_path_uni = os.path.join(model_dir_uni, torch_model_filename)
    model_dir_bpe_edited_vocabfile = os.path.join(model_dir_bpe_edited, vocab_filename)

    def _test_sentencepiece_robertatokenizer_helper(
        self,
        roberta_tokenizer: RobertaTokenizer,
        model,
        mask_token="<mask>",
        override_mask_id=None,
    ):
        masked_sentence = "The pen is on the " + mask_token + "."
        masked_index_in_sentence = len(roberta_tokenizer.tokenize("The pen is on the"))
        tokenized_as = roberta_tokenizer.tokenize(masked_sentence)
        input_ids = roberta_tokenizer(masked_sentence)["input_ids"]

        if override_mask_id is not None and isinstance(override_mask_id, int):
            print(f"overriding mask_token_id with {override_mask_id}")
            input_ids = [
                override_mask_id if x == roberta_tokenizer.mask_token_id else x
                for x in input_ids
            ]

        print(
            f"\nmasked_sentence={masked_sentence}"
            f"\ninput_ids={input_ids}"
            f"\ntokenized_as={tokenized_as}"
        )
        (
            logits,
            res_softmax,
            res_logistic,
            # res_normalized,
            # logits_shifted_above_zero,
        ) = get_bert_output_single_masking(model, input_ids, masked_index_in_sentence)
        k = 5
        topk_probs, topk_ids = torch.topk(res_softmax, k)
        topk_ids = list(topk_ids)
        topk_tokens = convert_ids_to_tokens(
            roberta_tokenizer, topk_ids
        )  # nb: topk_ids might be a list of tensors, not int
        print(f"topk_tokens, topk_ids = {topk_tokens}, {topk_ids}")
        # decode the topk ids to tokens, and see if "table" is contained
        # (or encode "table" first and see if the id is in the topk)

        exprected_prediction_token = roberta_tokenizer.tokenize("table")[0]
        exprected_prediction_token_id = roberta_tokenizer.convert_tokens_to_ids(
            [exprected_prediction_token]
        )

        print(f"{type(exprected_prediction_token_id)}, {type(topk_ids)}")

        topk_prediction_includes_expected = exprected_prediction_token_id in topk_ids

        return topk_prediction_includes_expected, topk_ids, topk_tokens

    def _test_sentencepiece_custom_tokens_helper(
        self,
        custom_tokenizer,
        model,
        cls_token="[CLS]",
        mask_token="[MASK]",
        sep_token="[SEP]",
    ):
        # encode the sentence: "The coffee is on the ***mask*** ."
        # tokens = ["[CLS]", "The", "coffee", "is", "on", "the", "[MASK]", ".", "[SEP]"]

        # can't use this: tokens_tot = custom_tokenizer.encode_as_pieces("[CLS] The coffee is on the [MASK] . [SEP]")
        tokens_l = custom_tokenizer.tokenize("The pen is on the")
        # tokens_r = custom_tokenizer.tokenize(".")  # removed because it adds also a spece token before
        tokens_tot = (
            [cls_token] + tokens_l + [mask_token] + [sep_token]
        )  # + tokens_r + ["[SEP]"]
        # bos_id = custom_tokenizer.piece_to_id("[CLS]")
        # mask_id = custom_tokenizer.piece_to_id("[MASK]")
        # eos_id = custom_tokenizer.piece_to_id("[SEP]")
        # sentence_ids = [bos_id] + tokens_l + [mask_id] + tokens_r + [eos_id]
        exprected_prediction_token = custom_tokenizer.encode_as_pieces("table")[0]
        exprected_prediction_token_id = custom_tokenizer.piece_to_id(
            exprected_prediction_token
        )
        masked_index_in_sentence = len(["[CLS]"] + tokens_l)
        sentence_ids = custom_tokenizer.convert_tokens_to_ids(tokens_tot)
        print(f"tokenized as: {tokens_tot} \n " f"with ids: {sentence_ids}")

        # pass the sentence ids to the model and predict topk
        (
            logits,
            res_softmax,
            res_logistic,
            res_normalized,
            logits_shifted_above_zero,
        ) = get_bert_output_single_masking(
            model, sentence_ids, masked_index_in_sentence
        )
        k = 5
        topk_probs, topk_ids = torch.topk(res_softmax, k)
        topk_tokens = convert_ids_to_tokens(
            custom_tokenizer, topk_ids
        )  # nb: topk_ids might be a list of tensors, not int

        # decode the topk ids to tokens, and see if "table" is contained
        # (or encode "table" first and see if the id is in the topk)
        topk_prediction_includes_expected = exprected_prediction_token_id in topk_ids

        return topk_prediction_includes_expected, topk_ids, topk_tokens

    def test_load_with_sententepiece_tokenizers_huggingface(self):
        # SentencePieceExtractor
        pass

    def test_sentencepiece_special_tokens(self):
        model_subdir = "bostromkaj/bpe_20k_ep20_pytorch/"
        tokenizer = CustomTokenizerWrapper(str(get_models_dir() / model_subdir))

        # todo: see processor.SetEncodeExtraOptions("bos:eos");   // add <s> and </s>.
        # https://github.com/google/sentencepiece/blob/master/doc/api.md
        # tokenizer.sp_tokenizer.SetEncodeExtraOptions("bos:eos")

        special_tokens = [
            "<mask>",
            "[MASK]",
            "<sep>",
            "[SEP]",
            "<bos>",
            "<eos>",
            "[CLS]",
            "<cls>",
            "<pad>",
            "<s>",
            "</s>",
        ]
        sentence_with_special_tokens = (
            "<mask> [MASK] <sep> [SEP] <bos> <eos> [CLS] <cls> <pad> <s> </s>"
        )
        special_pieces = tokenizer.encode_as_pieces(sentence_with_special_tokens)
        special_ids = tokenizer.convert_tokens_to_ids(special_tokens)
        print(f"{special_tokens}")
        print(f"{sentence_with_special_tokens}")
        print(f"{special_pieces}")
        print(f"{special_ids}")

        # processor.IsControl(10);     // returns true if the given id is a control token. e.g., <s>, </s>
        print(f"{tokenizer.sp_tokenizer.IsControl(10)}")
        print(f"{tokenizer.sp_tokenizer.IsControl(0)}")

        ids_to_check = [0, tokenizer.vocab_size - 1, tokenizer.vocab_size, 20004]
        for id in ids_to_check:
            # with pytest.raises(IndexError)as idx_err:
            try:
                print(f"id {id} corresponds to token: {tokenizer.id_to_piece(id)}")
            except IndexError:  # as idx_err
                print(
                    f"id {id} is out of range in the vocabulary size of {tokenizer.vocab_size}"
                )

        # todo: check if in the vocabulary there are tokens, and print them, with "<"
        vocab = tokenizer.get_vocab()
        print("printing tokens containing '<' or '['")
        for token, id in vocab.items():
            if "<" in token or "[" in token:
                print(f"token {token} has id {id}")

        # print(f"{tokenizer.sp_tokenizer.sep_token}")

        first_tokens_count = 10
        print(
            f"first {first_tokens_count} tokens are: "
            f"{[(i, tokenizer.ids_to_tokens[i]) for i in range(first_tokens_count)]}"
        )

        # from https://huggingface.co/docs/transformers/model_doc/roberta
        # https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/models/roberta/tokenization_roberta.py#L103
        #         bos_token="<s>",
        #         eos_token="</s>",
        #         sep_token="</s>",
        #         cls_token="<s>",
        #         unk_token="<unk>",
        #         pad_token="<pad>",
        #         mask_token="<mask>",

        # from https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/data/dictionary.py
        #         bos="<s>",
        #         pad="<pad>",
        #         eos="</s>",
        #         unk="<unk>",
        #
        #         self.bos_index = self.add_symbol(bos)
        #         self.pad_index = self.add_symbol(pad)
        #         self.eos_index = self.add_symbol(eos)
        #         self.unk_index = self.add_symbol(unk)
        #
        # https://github.com/facebookresearch/fairseq/issues/1309

        # see https://github.com/google/sentencepiece
        # by default the unk token has id 0, as in the Bostrom models.
        # the bos and eos tokens by default have ids 1 and 2, but they can be disabled,
        # while the unk token cannot.
        # so they must have been disabled when training the Bostrom tokenizers.
        # https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md

        print("bos=", tokenizer.sp_tokenizer.bos_id())
        print("eos=", tokenizer.sp_tokenizer.eos_id())
        print("unk=", tokenizer.sp_tokenizer.unk_id())
        print("pad=", tokenizer.sp_tokenizer.pad_id())  # disabled by default
        # gives:
        # bos= -1
        # eos= -1
        # unk= 0
        # pad= -1

        # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
        # Training sentencepiece model from the word list with frequency
        # We can train the sentencepiece model from the pair of <word, frequency>.
        # First, you make a TSV file where the first column is the word and the
        # second column is the frequency. Then, feed this TSV file with
        # --input_format=tsv flag. Note that when feeding TSV as training data,
        # we implicitly assume that --split_by_whtespace=true.
        #
        # import sentencepiece as spm
        #
        # spm.SentencePieceTrainer.train('--input=word_freq_list.tsv --input_format=tsv --model_prefix=m --vocab_size=2000')
        # sp = spm.SentencePieceProcessor()
        # sp.load('m.model')
        #
        # print(sp.encode_as_pieces('this is a test.'))

        # https://github.com/google/sentencepiece/blob/master/python/README.md
        print(tokenizer.sp_tokenizer.encode("This is a test"))
        print(
            tokenizer.sp_tokenizer.encode(
                ["This is a test", "Hello world"], out_type=int
            )
        )
        print(tokenizer.sp_tokenizer.encode("This is a test", out_type=str))
        print(
            tokenizer.sp_tokenizer.encode(
                ["This is a test", "Hello world"], out_type=str
            )
        )

        return_str = "\r"
        print(f"{tokenizer.sp_tokenizer.piece_to_id(return_str)}")
        print(f"{tokenizer.sp_tokenizer.piece_to_id('▁')}")

        # https://github.com/facebookresearch/fairseq/issues/459

    def test_fill_mask(self):
        model_subdir = "bostromkaj/bpe_20k_ep20_pytorch/"
        tokenizer = CustomTokenizerWrapper(str(get_models_dir() / model_subdir))

        topk = self.fill_mask("My name is <mask>.", tokenizer)
        print(f"{topk}")

    def fill_mask(
        self, masked_input: str, tokenizer: CustomTokenizerWrapper, topk: int = 5
    ):
        masked_token = "<mask>"
        assert (
            masked_token in masked_input and masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token
        )

        text_spans = masked_input.split(masked_token)
        print(f"{text_spans}")
        # text_spans_bpe = (' {0} '.format(masked_token)).join(
        #     [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
        # ).strip()
        print(f"{tokenizer.encode_as_pieces(text_spans[0].rstrip())}")
        text_spans_bpe0 = (
            (" {} ".format(masked_token))
            .join(
                [
                    tokenizer.encode_as_pieces(text_span.rstrip())
                    for text_span in text_spans
                ][0]
            )
            .strip()
        )
        print(f"{text_spans_bpe0}")
        # text_spans_bpe = (' {0} '.format(masked_token)).join(
        #     [tokenizer.encode_as_pieces(text_span.rstrip()) for text_span in text_spans]
        # ).strip()

        return
        tokens = self.task.source_dictionary.encode_line(
            "<s> " + text_spans_bpe0,
            append_eos=True,
        )

        _ = (tokens == self.task.mask_idx).nonzero()  # masked_index =
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # with utils.eval(self.model):
        #     features, extra = self.model(
        #         tokens.long().to(device=self.device),
        #         features_only=False,
        #         return_all_hiddens=False,
        #     )
        # logits = features[0, masked_index, :].squeeze()
        # prob = logits.softmax(dim=0)
        # values, index = prob.topk(k=topk, dim=0)
        # topk_predicted_token_bpe = self.task.source_dictionary.string(index)
        #
        topk_filled_outputs = []
        # for index, predicted_token_bpe in enumerate(
        #     topk_predicted_token_bpe.split(" ")
        # ):
        #     predicted_token = self.bpe.decode(predicted_token_bpe)
        #     if " {}".format(masked_token) in masked_input:
        #         topk_filled_outputs.append(
        #             (
        #                 masked_input.replace(
        #                     " {}".format(masked_token), predicted_token
        #                 ),
        #                 values[index].item(),
        #                 predicted_token,
        #             )
        #         )
        #     else:
        #         topk_filled_outputs.append(
        #             (
        #                 masked_input.replace(masked_token, predicted_token),
        #                 values[index].item(),
        #                 predicted_token,
        #             )
        #         )
        return topk_filled_outputs

    @pytest.mark.skipif(not is_internet_on(), reason="No internet connection available")
    def test_load_remotely(self):
        # remote calls are blocked, enabled with annotations  only for specific tests
        not_cached_model = "hfl/chinese-macbert-base"  # "bert-base-uncased"
        with pytest.raises(SocketBlockedError):
            _ = BertForMaskedLM.from_pretrained(not_cached_model)

    @pytest.mark.skip("cannot reproduce the error")
    @pytest.mark.enable_socket
    def test_load_with_AutoTokenizer_with_default_config_json(self):
        with pytest.raises(ValueError) as val_err:
            # TestLoadModels.model_dir_uni: "OSError"
            # TestLoadModels.torch_model_path_uni gives "Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection
            # is on.'"
            tokenizer = AutoTokenizer.from_pretrained(TestLoadModels.model_dir_uni)
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
            tokenizer = CamembertTokenizer.from_pretrained(TestLoadModels.dict_path_uni)
            self.__test_tokenizer_helper(tokenizer)
        print(f"{run_err}=")
        self.assertInErrorMsg("Internal", run_err)
        self.assertInErrorMsg("sentencepiece_processor.cc", run_err)

    def test_load_as_huggingfaces_RobertaModel(self):
        roberta = RobertaModel.from_pretrained(TestLoadModels.model_dir_uni)
        print(type(roberta))

        roberta2 = RobertaForMaskedLM.from_pretrained(TestLoadModels.model_dir_uni)
        print(type(roberta2))

        # check: no need to specify model type in the config.json?

        # todo: asserts on the model
        # outputs = model(tokens_tensor, token_type_ids=segment_tensor)
        # check output type format, how to access the loss and token scores

    def test_load_with_AutoTokenizer_from_edited_files(self):
        with pytest.raises(OSError) as os_err1:
            tokenizer0 = AutoTokenizer.from_pretrained(
                TestLoadModels.model_dir_uni_edited
            )
            self.__test_tokenizer_helper(tokenizer0)
        self.assertInErrorMsg("Can't load tokenizer for ", os_err1)
        self.assertInErrorMsg(
            "is the correct path to a directory containing all relevant files for a "
            "BertTokenizerFast tokenizer.",
            os_err1,
        )

        # with pytest.raises(TypeError) as type_err:
        with pytest.raises(OSError) as _:
            tokenizer1 = AutoTokenizer.from_pretrained(
                TestLoadModels.model_dir_bpe_edited
            )
            self.__test_tokenizer_helper(tokenizer1)
        # self.assertInErrorMsg("Can't convert", type_err)
        # self.assertInErrorMsg("(list) to Union[Merges, Filename]", type_err)

        with pytest.raises(OSError) as os_err2:
            tokenizer2 = AutoTokenizer.from_pretrained(TestLoadModels.model_dir_bpe)
            self.__test_tokenizer_helper(tokenizer2)
        self.assertInErrorMsg("Can't load tokenizer for ", os_err2)
        self.assertInErrorMsg(
            " is the correct path to a directory containing all relevant files for a RobertaTokenizerFast tokenizer.",
            os_err2,
        )

    def test_load_as_huggingfaces_RobertaTokenizer(self):
        model_dir2 = str(
            get_models_dir() / "bostromkaj/bpe_20k_ep20_pytorch"  # + "/"
        )  # + "\\"
        with pytest.raises(OSError) as os_err:
            tokenizer3 = RobertaTokenizer.from_pretrained(model_dir2)
            self.__test_tokenizer_helper(tokenizer3)
        self.assertInErrorMsg("Can't load tokenizer for ", os_err)
        self.assertInErrorMsg("make sure ", os_err)
        self.assertInErrorMsg(
            "is the correct path to a directory containing all relevant files for a RobertaTokenizer tokenizer.",
            os_err,
        )

        # todo: try using an earlier verison of transformers (model was saved in 2019)
        # the tokenizer gets recognized (when loaded from AlbertTokenizer, which gives a warning)
        # as a Bert tokenizer. Try the 2019 BertTokenizer version ?
        with pytest.raises(ValueError) as val_err:
            _ = RobertaTokenizer.from_pretrained(TestLoadModels.tok_uni_path)
        self.assertInErrorMsg(
            "Calling RobertaTokenizer.from_pretrained() with the path to a single file or url is "
            "not supported for this tokenizer. Use a model identifier or the path to a directory instead.",
            val_err,
        )

    @pytest.mark.skip("done with command line script")
    def test_use_SentencePieceExtractor(self):
        _ = SentencePieceExtractor(
            TestLoadModels.tok_uni_path
        )  # TestLoadModels.tok_bpe_path

        # generate files (vocab, merge) with the extractor
        # try to load with ..BertTokenizerFast and RobertaTokenizer(Fast or not)
        # from pretrained
        # check if special tokens have been added (vocab count, ids above 19999,
        # any special token recognized ("[MASK]", etc)

        # see test_load_with_AutoTokenizer_with_edited_config_json

    @pytest.mark.skip("sentencepiece import not working properly")
    def test_load_as_custom_transformers_tokenizer(self):

        special_tokens = [
            "<mask>",
            "[MASK]",
            "<sep>",
            "[SEP]",
            "<bos>",
            "<eos>",
            "[CLS]",
            "<cls>",
            "<pad>",
            "<s>",
            "</s>",
            "[PAD]",
            "<unk>",
            " [UNK]",
        ]
        sentence_with_special_tokens = "<mask> [MASK] <sep> [SEP] <bos> <eos> [CLS] <cls> <pad> [PAD] <s> </s> <unk> [UNK]"

        # https://pypi.org/project/tokenizers/
        # Initialize a tokenizer

        # expects filename ..
        from tokenizers.implementations.sentencepiece_unigram import (
            Tokenizer,
        )
        from tokenizers.implementations import (
            SentencePieceUnigramTokenizer,
        )

        # import tokenizers
        # import sentencepiece
        # from unittest.mock import MagicMock
        # # from tokenizers.implementations.
        # # from sentencepiece.sentencepiece_model_pb2 import
        # with patch.object(
        #         tokenizers.implementations.sentencepiece_unigram,
        #         sentencepiece.sentencepiece_model_pb2.__name__,
        #         # return_value=MagicMock(),
        #         side_effect=lambda : ,
        # ) as mock_sentencepiece_model_pb2:
        # # assert sentencepiece.sentencepiece_model_pb2 is mock_sentencepiece_model_pb2
        # mock_sentencepiece_model_pb2.ModelProto.return_value =

        tokenizer_uni2 = SentencePieceUnigramTokenizer.from_spm(
            TestLoadModels.tok_uni_path
        )

        print("\n")
        for special_token in special_tokens:
            special_token_id = tokenizer_uni2.token_to_id(special_token)
            print(f"special token {special_token} id: {special_token_id}")
        encoded = tokenizer_uni2.encode(sentence_with_special_tokens)
        print(encoded.ids)
        print(encoded.tokens)
        # tokenizer_uni2.save("tokenizer_uni2.json", pretty=True)

        print(f"{tokenizer_uni2.get_vocab_size()}")  # 20000
        special_tokens_to_add = ["[MASK]", "[SEP]", "[CLS]"]
        tokenizer_uni2.add_special_tokens(special_tokens_to_add)
        for added_special_token in special_tokens_to_add:
            added_special_token_id = tokenizer_uni2.token_to_id(added_special_token)
            print(
                f"added special token {added_special_token} id: {added_special_token_id}"
            )

        # vocab = "./path/to/vocab.json"
        # merges = "./path/to/merges.txt"
        # tokenizer_uni1 = SentencePieceUnigramTokenizer(vocab, merges)
        # encoded = tokenizer_uni1.encode(sentence_with_special_tokens)
        # print(encoded.ids)
        # print(encoded.tokens)
        # tokenizer_uni1.save("tokenizer_uni1.json", pretty=True)

        filename = "tokenizer.model"  # nb: default and custom tokenizers saved
        # with hugginface transformers have a tokenizer.json instead (not a tokenizer.model file)
        # see https://huggingface.co/roberta-base/tree/main
        # and https://discuss.huggingface.co/t/creating-a-custom-tokenizer-for-roberta/2809
        filepath = get_models_dir() / ("bostromkaj/bpe_20k_ep20_pytorch/" + filename)
        filepath = str(filepath)
        print(f"{filepath}")
        with pytest.raises(Exception) as stream_exception:
            tokenizer = Tokenizer.from_file(
                filepath  # expected: A path to a local JSON file representing a previously serialized
            )
            print(type(tokenizer))
        self.assertInErrorMsg("stream did not contain valid UTF-8", stream_exception)

    @pytest.mark.skipif(not is_fairseq_installed, reason="Fairseq is not installed")
    def test_load_with_fairseq_RobertaModel(self):

        if not self.is_fairseq_installed:
            return

        from fairseq.models.roberta import RobertaModel as RobertaModel_FS

        with pytest.raises(OSError) as os_err:
            roberta = RobertaModel_FS.from_pretrained(
                TestLoadModels.model_dir_uni,
            )
            # skipped because of raised error above:
            roberta.eval()  # disable dropout (or leave in train mode to finetune)
            tokens = roberta.encode("Hello world!")
            print(f"tokens: {tokens}")
            assert tokens.tolist() == [0, 31414, 232, 328, 2]
            print(roberta.decode(tokens))  # 'Hello world!'
        print(f"{os_err}")
        self.assertInErrorMsg("Model file not found", os_err)
        self.assertInErrorMsg("model.pt", os_err)

        with pytest.raises(KeyError):  # as key_err
            roberta = RobertaModel_FS.from_pretrained(
                TestLoadModels.model_dir_uni,
                checkpoint_file="pytorch_model.bin",  # by default looks for a "model.pt", so a pythorch model file, note a .bin file (like roberta base from huggingfaces)
            )
            print(type(roberta))
        # fixme: this files in py36 (but not in py39): AssertionError: 'best_loss' not found in "<ExceptionInfo KeyError('args',) tblen=5>"
        # self.assertInErrorMsg("best_loss", key_err)

        # from fairseq.models.roberta import RobertaModel
        # roberta = RobertaModel.from_pretrained('checkpoints',
        #                                        'checkpoint_best.pt',
        #                                        'path/to/data')
        # assert isinstance(roberta.model, torch.nn.Module)

    def test_load_checkpoint_with_Torch(self):
        import torch

        # model = Net()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        from collections import OrderedDict

        checkpoint = torch.load(TestLoadModels.torch_model_path_uni)

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

    def test_load_with_BertTokenizer_dict_file(self):
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
        tokenizer = BertTokenizer.from_pretrained(TestLoadModels.dict_path_uni)
        self.__test_tokenizer_helper(tokenizer)

    def test_load_with_BertTokenizer_model_dir(self):
        pass
        # OSError: Can't load tokenizer for
        # is the correct path to a directory containing all relevant files for a BertTokenizer tokenizer
        # tokenizer = BertTokenizer.from_pretrained(TestLoadModels.model_dir_bpe_edited)
        # tokenizer = BertTokenizer.from_pretrained(TestLoadModels.model_dir_uni_edited)

        # bert_tokenizer = BertTokenizer.from_pretrained(
        #     TestLoadModels.model_dir_uni_edited)

        # OSError: Can't load tokenizer
        # is the correct path to a directory containing all relevant files for a BertTokenizerFast tokenizer.
        # tokenizer = BertTokenizerFast.from_pretrained(TestLoadModels.model_dir_uni_edited)
        # tokenizer = BertTokenizerFast.from_pretrained(TestLoadModels.model_dir_bpe_edited)

        # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 28: invalid start byte
        # tokenizer = BertTokenizer.from_pretrained(TestLoadModels.tok_bpe_path)

        # TypeError: __init__() got multiple values for argument 'self'
        # transformers\configuration_utils.py:675: TypeError
        # tokenizer = BertTokenizer.from_pretrained(TestLoadModels.model_dir_bpe_edited_vocabfile)

        # ValueError: Calling BertTokenizerFast.from_pretrained() with the path to a single file or url is not supported for this tokenizer.
        # Use a model identifier or the path to a directory instead.
        # tokenizer = BertTokenizerFast.from_pretrained(TestLoadModels.model_dir_bpe_edited_vocabfile)

        # OSError: Can't load tokenizer for is the correct path to a directory containing all relevant files for a BertTokenizerFast tokenizer
        # tokenizer = BertTokenizerFast.from_pretrained(TestLoadModels.model_dir_bpe_edited)

        # tokenizer = RobertaTokenizerFast.from_pretrained(TestLoadModels.model_dir_bpe_edited)

        # warning: The tokenizer class you load from this checkpoint is 'BertTokenizer'.
        # typeerror ..
        # tokenizer = RobertaTokenizerFast.from_pretrained(TestLoadModels.model_dir_bpe_edited)

    def test_load_with_BertTokenizer_model_file(self):
        print("loading with BertTokenizer..")

        print(f"{TestLoadModels.tok_uni_path}")
        with pytest.raises(UnicodeDecodeError) as utf8_err:
            tokenizer = BertTokenizer.from_pretrained(TestLoadModels.tok_uni_path)
            print(f"{type(tokenizer)}")
            self.__test_tokenizer_helper(tokenizer)
        self.assertInErrorMsg("utf-8", utf8_err)
        self.assertInErrorMsg("invalid start byte", utf8_err)
        # self.assertInErrorMsg("codec can't decode byte 0xa3 in position 27", utf8_err)

    def test_load_with_AlbertTokenizer(self):
        print("loading with AlbertTokenizer..")

        with pytest.raises(RuntimeError) as run_err:
            tokenizer = AlbertTokenizer.from_pretrained(TestLoadModels.dict_path_uni)
            self.__test_tokenizer_helper(tokenizer)
        print(f"{run_err}")
        self.assertInErrorMsg("Internal", run_err)
        self.assertInErrorMsg("sentencepiece_processor.cc", run_err)

        print(f"{TestLoadModels.tok_uni_path}")
        tokenizer = AlbertTokenizer.from_pretrained(TestLoadModels.tok_uni_path)
        print(f"{type(tokenizer)}")

        self.__test_tokenizer_helper(tokenizer)

        # todo: test tokens indexes correspondence with dict.txt
        #  try to decode/encode tokens based on dict.txt indexes?
        # lines in uni dict.txt:
        #  2393 <unk>
        #  1991 ▁bridge
        #  2285 ▁Bridge
        #  2410 ▁letter
        # 16042 northwest
        # 20000 筤
        print(
            f"{convert_ids_to_tokens(tokenizer, [1,2,3,4,5,2360,2374,2375,2376,2393,16042,19999])}"
        )
        print(f"{convert_ids_to_tokens(tokenizer, [15,16,17,18,19,20])}")
        print(f"{convert_ids_to_tokens(tokenizer, [21,22,23,24,25])}")
        print(f"{convert_ids_to_tokens(tokenizer, [5,10,15,20,25,30,35])}")

        # todo:
        # test predicting masked sentences, see topk and check that the most
        # likely prediction is in the topk

    def assertInErrorMsg(self, expected_str, error: ExceptionInfo):
        if error.type in [FileNotFoundError]:
            msg = f"{str(error.value)} {error.value.args[1]}  {error.value.strerror}  {error.value.filename}"
        elif error.type in [RuntimeError, ValueError, OSError]:
            msg = error.value.args[0]
        elif error.type in [UnicodeDecodeError]:
            print(f"{error.value.args[0]}")
            msg = str(error)
        else:  # KeyError?
            msg = str(error)
        self.assertIn(expected_str, msg)

    def test_load_with_TorchHub(self):
        print("loading with Torch..")

        # see https://pytorch.org/hub/huggingface_pytorch-transformers/
        # config = torch.hub.load("local", 'config', TestLoadModels.model_dir_uni)  # repo_owner, repo_name = repo_info.split('/') ValueError: not enough values to unpack (expected 2, got 1)

        with pytest.raises(FileNotFoundError) as f_err:
            tokenizer = torch.hub.load(
                TestLoadModels.model_dir_uni,
                "tokenizer",
                source="local",
                pretrained=True,
                # config=config
                # "local", "tokenizer", TestLoadModels.model_dir_uni,  # config=config
            )
            self.__test_tokenizer_helper(tokenizer)
        print(f"{f_err}")
        self.assertInErrorMsg("hubconf.py", f_err)

        with pytest.raises(FileNotFoundError) as f_err2:
            roberta = torch.hub.load(
                TestLoadModels.model_dir_uni,  # repo_or_dir # 'pytorch/fairseq',
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

        print(f"{txt} to {tokens}")
        special_tokens_properties1 = [
            "bos_token",
            "unk_token",
            "eos_token",
            "pad_token",
        ]
        print(f"Special tokens: {tokenizer.all_special_tokens}, {tokenizer.vocab_size}")
        for special_token_property in special_tokens_properties1:
            token = getattr(tokenizer, special_token_property)
            print(
                f"{special_token_property}: "
                f"{token}"
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
            print(f"{token}, {idx}")

        print(f"vocab: {dict(islice(vocab.items(), 0, 20))}")

        #  <class 'transformers.models.albert.tokenization_albert.AlbertTokenizer'>.

        if isinstance(tokenizer, AlbertTokenizer) and hasattr(
            tokenizer, "convert_ids_to_tokens"
        ):
            print(f"{convert_ids_to_tokens(tokenizer, [20003])}")

        # IndexError: piece id is out of range.
        # print(f"{tokenizer.convert_ids_to_tokens([20004])}")
