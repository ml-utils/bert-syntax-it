import pytest
from transformers import RobertaForMaskedLM
from transformers import RobertaForQuestionAnswering

from int_tests.test_load_models import TestLoadModels
from src.linguistic_tests.tokenizer import RobertaSentencePieceTokenizer

# from src.linguistic_tests.bert_utils import convert_ids_to_tokens
# from src.linguistic_tests.bert_utils import get_bert_output_single_masking
# from src.linguistic_tests.lm_utils import CustomTokenizerWrapper
# from src.linguistic_tests.lm_utils import get_models_dir
# import torch
# from _pytest._code.code import ExceptionInfo
# from pytest_socket import SocketBlockedError
# from transformers import AlbertTokenizer
# from transformers import AutoTokenizer
# from transformers import BertForMaskedLM
# from transformers import BertTokenizer
# from transformers import CamembertTokenizer
# from transformers import RobertaModel
# from transformers import RobertaTokenizer
# from transformers.convert_slow_tokenizer import SentencePieceExtractor
#
# from int_tests.integration_tests_utils import is_internet_on
# import os.path
# from itertools import islice
# from unittest import TestCase


class TestRobertaSentencePiece(TestLoadModels):
    @pytest.mark.skip(
        "todo, fix error when calling qa_model(..).  IndexError: index out of range in self. Maybe need older python and transformers version?"
    )
    def test_for_question_answering(self):

        models_dirs = [
            self.model_dir_uni_edited,
            self.model_dir_bpe_edited,
        ]

        for model_dir in models_dirs:
            print(f"model dir: {model_dir}")
            # self.model_dir_bpe
            tokenizer = RobertaSentencePieceTokenizer._from_pretrained(model_dir)

            # special_tokens_dict = {"mask_token": "<mask>"}
            # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

            model = RobertaForQuestionAnswering.from_pretrained(  # RobertaForMaskedLM.from_pretrained(
                model_dir
            )

            self._question_answering_helper(model, tokenizer)

    def test_for_word_prediction(self):
        # problem: there is no <mask> token
        # vocab goes from 0 to 20003 (20004 tokens). But model has 20005.
        # maybe index for 20004 is <mask>
        # try adding it manually to the tokenizer and see the model output

        # when adding it manually, the mask token gets assigned idx 3, taking the place left by the duplicated <unk>
        # test if with mask_token idx 3 model prediction are correct, or with mask_token idx 20004

        # model dir: ./uni_20k_ep20_pytorch
        # tokenizer.SPECIAL_TOKENS_ATTRIBUTES=['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'additional_special_tokens']
        # We have added 0 tokens
        # the mask_token is <mask>
        # the mask_token has id 3
        # the mask_token has id [3]
        # tokenizer.all_special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>']
        # tokenizer.additional_special_tokens_ids=[]
        # tokenizer.all_special_tokens_extended=['<s>', '</s>', '<unk>', '<pad>', '<mask>']
        # tokenizer.additional_special_tokens_ids=[]
        # all_special_tokens ids=[0, 2, 2396, 1, 3]
        # ids and tokens: [(0, '<s>'), (1, '<pad>'), (2, '</s>'), (3, '<unk>'), (4, '▁the'), (5, ','), (2394, 'kh'), (2395, '▁recently'), (2396, '<unk>'), (2397, '▁Ch'), (2398, '▁stars'), (20003, '筤')]
        # Dictionary symbols with ['<', '>']
        # <s> with id [0]
        # <pad> with id [1]
        # </s> with id [2]
        # <unk> with id [2396]
        # <unk> with id [2396]
        # model dir: ./bpe_20k_ep20_pytorch
        # tokenizer.SPECIAL_TOKENS_ATTRIBUTES=['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'additional_special_tokens']
        # We have added 0 tokens
        # the mask_token is <mask>
        # the mask_token has id 3
        # the mask_token has id [3]
        # tokenizer.all_special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>']
        # tokenizer.additional_special_tokens_ids=[]
        # tokenizer.all_special_tokens_extended=['<s>', '</s>', '<unk>', '<pad>', '<mask>']
        # tokenizer.additional_special_tokens_ids=[]
        # all_special_tokens ids=[0, 2, 2548, 1, 3]
        # ids and tokens: [(0, '<s>'), (1, '<pad>'), (2, '</s>'), (3, '<unk>'), (4, '▁the'), (5, ','), (2546, '▁recently'), (2547, '▁Sand'), (2548, '<unk>'), (2549, 'ope'), (2550, 'ises'), (20003, '筤')]
        # Dictionary symbols with ['<', '>']
        # <s> with id [0]
        # <pad> with id [1]
        # </s> with id [2]
        # <unk> with id [2548]
        # <unk> with id [2548]

        # (0, '<s>'), (1, '<pad>'), (2, '</s>'),
        # (3, '<unk>') but this value is overwritten by the duplicate
        # (2396, '<unk>') for uni_20k_ep20_pytorch
        # (2548, '<unk>') for bpe_20k_ep20_pytorch
        # Dictionary symbols with ['<', '>']
        # <s> with id [0]
        # <pad> with id [1]
        # </s> with id [2]
        # <unk> with id [2548]
        # <unk> with id [2548]

        models_dirs = [
            self.model_dir_uni_edited,
            self.model_dir_bpe_edited,
        ]

        for model_dir in models_dirs:
            print(f"model dir: {model_dir}")
            # self.model_dir_bpe
            tokenizer = RobertaSentencePieceTokenizer._from_pretrained(model_dir)

            tokenizer.verbose = True
            print(
                f"tokenizer.SPECIAL_TOKENS_ATTRIBUTES={tokenizer.SPECIAL_TOKENS_ATTRIBUTES}"
            )

            special_tokens_dict = {"mask_token": "<mask>"}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print("We have added", num_added_toks, "tokens")
            print(f"the mask_token is {tokenizer.mask_token}")
            print(f"the mask_token has id {tokenizer.mask_token_id}")
            print(
                f"the mask_token has id {tokenizer.convert_tokens_to_ids([tokenizer.mask_token])}"
            )

            # special_tokens_dict2 = {"additional_special_tokens": ["<mask>"]}

            # tokenizer.add_tokens()
            # tokenizer.mask_token_id

            print(f"tokenizer.all_special_tokens={tokenizer.all_special_tokens}")
            print(
                f"tokenizer.additional_special_tokens_ids={tokenizer.additional_special_tokens_ids}"
            )
            print(
                f"tokenizer.all_special_tokens_extended={tokenizer.all_special_tokens_extended}"
            )
            print(
                f"tokenizer.additional_special_tokens_ids={tokenizer.additional_special_tokens_ids}"
            )
            print(
                f"all_special_tokens ids={tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens_extended)}"
            )
            # print(f"tokenizer.get_special_tokens_mask={tokenizer.get_special_tokens_mask()}")

            vocab_size = len(tokenizer.dictionary)
            unk_id = tokenizer.dictionary.index("<unk>")
            ids = [
                0,
                1,
                2,
                3,
                4,
                5,
                unk_id - 2,
                unk_id - 1,
                unk_id,
                unk_id + 1,
                unk_id + 2,
                vocab_size - 1,
            ]
            tokens_from_ids = tokenizer.convert_ids_to_tokens(ids)
            print(f"ids and tokens: {list(zip(ids, tokens_from_ids))}")
            # tokenizer.batch_encode_plus()
            # todo print vocabulary showing any missing ids that might correspond to the mask token
            chars = ["<", ">"]
            print(f"Dictionary symbols with {chars}")
            for symbol in tokenizer.dictionary.symbols:
                if chars[0] in symbol and chars[1] in symbol:
                    print(
                        f"{symbol} with id {tokenizer.convert_tokens_to_ids([symbol])}"
                    )

            topk = self.fill_mask2(
                "My name is <mask>.", tokenizer, mask_token=tokenizer.mask_token
            )
            print(f"topk={topk}")

            model = RobertaForMaskedLM.from_pretrained(model_dir)
            print(f"model type={type(model)}")

            self._test_sentencepiece_robertatokenizer_helper(
                tokenizer,
                model,
                mask_token=tokenizer.mask_token,
                override_mask_id=vocab_size,
            )

    def _question_answering_helper(self, model, tokenizer):
        from transformers import pipeline

        # example
        # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        #         start_positions = torch.tensor([1])
        #         end_positions = torch.tensor([3])
        #         outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        #         loss, start_scores, end_scores = outputs[:2]

        qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "Where do I live?"
        context = "My name is Merve and I live in İstanbul."
        model_output = qa_model(question=question, context=context)
        print(
            f"question={question} \ncontext={context} \nanswer={model_output['answer']} \nmodel_output={model_output}"
        )
        # {'answer': 'İstanbul', 'end': 39, 'score': 0.953, 'start': 31}

        # from datasets import load_dataset
        # squad = load_dataset("squad")
        # squad["train"][0]

        # squad_train_0 = {'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
        #  'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        #  'id': '5733be284776f41900661182',
        #  'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
        #  'title': 'University_of_Notre_Dame'
        # }

    def fill_mask2(
        self, masked_input: str, tokenizer, topk: int = 5, mask_token="<mask>"
    ):
        masked_token = mask_token
        assert (
            masked_token in masked_input and masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token
        )

        text_spans = masked_input.split(masked_token)
        print(f"text_spans={text_spans}")
        # text_spans_bpe = (' {0} '.format(masked_token)).join(
        #     [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
        # ).strip()

        print(
            f"tokenizer._tokenize(text_spans[0].rstrip())={tokenizer._tokenize(text_spans[0].rstrip())}"
        )
        text_spans_bpe0 = (
            (" {} ".format(masked_token))
            .join(
                [tokenizer._tokenize(text_span.rstrip()) for text_span in text_spans][0]
            )
            .strip()
        )
        print(f"text_spans_bpe0={text_spans_bpe0}")
        # text_spans_bpe = (' {0} '.format(masked_token)).join(
        #     [tokenizer.encode_as_pieces(text_span.rstrip()) for text_span in text_spans]
        # ).strip()

        return
        # tokens = self.task.source_dictionary.encode_line(
        #             "<s> " + text_spans_bpe0,
        #             append_eos=True,
        #         )


def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
