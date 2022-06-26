from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer

from int_tests.test_load_models import TestLoadModels
from src.linguistic_tests.lm_utils import CustomTokenizerWrapper
from src.linguistic_tests.lm_utils import get_models_dir
from src.linguistic_tests.lm_utils import print_orange


class LegacyTests:
    def test_load_with_sentencepiece_unigram(self):
        self._test_load_with_sentencepiece_helper("bostromkaj/uni_20k_ep20_pytorch/")

    def test_load_with_sentencepiece_bpe(self):
        self._test_load_with_sentencepiece_helper("bostromkaj/bpe_20k_ep20_pytorch/")

    def test_sentencepiece_custom_tokens(self):
        model_subdir = "bostromkaj/uni_20k_ep20_pytorch/"
        model_dir_path = str(get_models_dir() / model_subdir)
        custom_tokenizer = CustomTokenizerWrapper(  # custom_tokenizer
            model_dir=model_dir_path, custom_tokens={}
        )
        # roberta = RobertaModel.from_pretrained(model_dir_path)
        roberta2 = RobertaForMaskedLM.from_pretrained(model_dir_path)
        print(
            # f"{type(roberta)}, "
            f"{type(roberta2)}, "
        )

        # generate all 5x4x3 combinations of custom tokens ids
        custom_tokens_deberta = {
            "[PAD]": 20000,
            "[CLS]": 20001,
            "[SEP]": 20002,
            "[UNK]": 20003,
            "[MASK]": 20004,
        }
        custom_tokens_fairseq = {  # custom_tokens_fairseq
            "[CLS]": 20000,  # bos
            "[PAD]": 20001,
            "[SEP]": 20002,  # eos
            # "[UNK]": 20003,
            "[MASK]": 20003,
        }
        custom_tokens_albert_tokenizer = {  # custom_tokens_actual
            "[CLS]": 20000,  # bos, <s>
            "[SEP]": 20001,  # eos, </s>
            "<pad>": 20002,
            "[MASK]": 20003,
        }

        # tokenizer.all_special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
        # tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)=[20000, 20001, 0, 20002, 20003]
        custom_tokens_roberta_tokenizer = {
            "<s>": 20000,
            "</s>": 20001,
            "<pad>": 20002,
            "<mask>": 20003,
        }

        custom_tokens_configs = [
            custom_tokens_deberta,
            custom_tokens_fairseq,
            custom_tokens_albert_tokenizer,
            custom_tokens_roberta_tokenizer,
        ]

        # generate all 5x4x3 combinations assignig ids to ["[CLS]", "[SEP]", "[MASK]"]
        possible_ids = [20000, 20001, 20002, 20003, 20004]
        custom_tokens_configs = []
        for id_for_bos in possible_ids:
            remaining_ids = possible_ids.copy()
            remaining_ids.remove(id_for_bos)
            for id_for_eos in remaining_ids:
                remaining_ids2 = remaining_ids.copy()
                remaining_ids2.remove(id_for_eos)
                for id_for_mask in remaining_ids2:
                    custom_tokens_config = {
                        "[CLS]": id_for_bos,  # [CLS], bos, <s>
                        "[SEP]": id_for_eos,  # [SEP], eos, </s>
                        "[MASK]": id_for_mask,
                    }
                    custom_tokens_configs.append(custom_tokens_config)
        assert len(custom_tokens_configs) == 60

        # todo: check/print which configuration's predictions generate the sentences with lowest loss

        # custom_tokens_configs = [custom_tokens_roberta_tokenizer]

        working_config_count = 0
        for custom_tokens_config in custom_tokens_configs:
            # load custom tokenizer only once, add method to change the special tokens ids
            custom_tokenizer._change_custom_tokens(custom_tokens_config)

            (
                topk_prediction_includes_expected,
                topk_ids,
                topk_tokens,
            ) = self._test_sentencepiece_custom_tokens_helper(
                custom_tokenizer, roberta2
            )

            verbose = False
            if verbose:
                if topk_prediction_includes_expected:
                    print(
                        f"This custom_tokens configuration predicts correctly: {custom_tokens_config}"
                    )
                    print(f"{topk_ids}")
                    print(f"{topk_tokens}")
                    working_config_count += 1
                else:
                    print(
                        f"This custom_tokens configuration does NOT predict correctly: {custom_tokens_config}"
                    )
                    print(f"{topk_ids}")
                    print_orange(f"{topk_tokens}")

        print(f"{working_config_count}")
        # assert working_config_count > 0

        _ = "The coffee is on the [MASK]"  # sentence =
        _ = 1  # or 2?  # mask_idx =
        # todo: get topk
        # check which results contain "table"

        # print: with custom_tokens1: .., topk are ..

        # training a Roberta model with fairseq and sentencepiece tokenizer:
        # https://github.com/musixmatchresearch/umberto/issues/2
        # # Encode Data with SentencePiece Tokenizer
        # spm_encode \
        #     --model=spm.bpe.model \ [ model that is from output of sp training ]
        #     --extra_options=bos:eos \ [ saying that you want begin of sequence and end of sequence encoded ]
        #     --output_format=piece \ [ here you are telling that encoded data will be as tokens of spm ]
        #     < file.raw \ [ raw data in input]
        #     > file.bpe [ encoded data in output ]
        #
        # https://github.com/google/sentencepiece
        # % spm_encode --extra_options=bos:eos (add <s> and </s>)

    def test_load_with_RobertaTokenizer(self):

        # Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
        roberta_tokenizer = RobertaTokenizer.from_pretrained(
            TestLoadModels.model_dir_uni_edited
        )
        roberta2 = RobertaForMaskedLM.from_pretrained(
            TestLoadModels.model_dir_uni_edited
        )

        print(
            f"\n{roberta_tokenizer.vocab_size}, "
            f"\n{roberta_tokenizer.sep_token}, "
            f"\n{roberta_tokenizer.mask_token}, "
            f"\n{roberta_tokenizer.eos_token}, "
            f"\n{roberta_tokenizer.cls_token}, "
            f"\n{roberta_tokenizer.bos_token}, "
            f"\n{roberta_tokenizer.all_special_tokens}, "
            f"\n{roberta_tokenizer.convert_tokens_to_ids(roberta_tokenizer.all_special_tokens)}"
        )
        # tokenizer.vocab_size=20000,
        # tokenizer.sep_token='</s>',
        # tokenizer.mask_token='<mask>',
        # tokenizer.eos_token='</s>',
        # tokenizer.cls_token='<s>',
        # tokenizer.bos_token='<s>',
        # tokenizer.all_special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>'],

        self.__test_tokenizer_helper(roberta_tokenizer)

        (
            topk_prediction_includes_expected,
            topk_ids,
            topk_tokens,
        ) = self._test_sentencepiece_robertatokenizer_helper(
            roberta_tokenizer, roberta2
        )
        print(f"{topk_prediction_includes_expected}")
        print(f"{topk_ids}")
        print_orange(f"{topk_tokens}")

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

        from test_load_models import TestLoadModels

        TestLoadModels.__test_tokenizer_helper(sp)
