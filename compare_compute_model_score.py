import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # pytorch_transformers

from transformers import BertTokenizer
from transformers import BertForMaskedLM  # BertModel as BertForMaskedLM

from scipy.special import softmax
import numpy as np

from lm_utils import model_types, get_sentences_from_example


class DEVICES:
    CPU = 'cpu'
    CUDA = 'cuda:X'


def load_model(model_type, model_name, device):
    # Load pre-trained model and tokenizer
    if model_type == model_types.GPT:
        print(f'loading model {model_name}..')
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f'model loaded. Loading tokenizer {model_name}..')
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print(f'tokenizer loaded.')
    elif model_type == model_types.BERT:
        model = BertForMaskedLM.from_pretrained(model_name)  # BertForMaskedLM.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                  do_lower_case=(True if "uncased" in model_name else False))
    else:
        return

    # put model to device (GPU/CPU)
    device = torch.device(device)
    model.to(device)

    # eval mode; no dropout
    model.eval()
    return model, tokenizer


def run_testset(model_type, model, tokenizer, device, testset):

    sent_ids = []

    from lm_utils import GOOD_SENTENCE_2_IDX, GOOD_SENTENCE_1_IDX, SENTENCE_BAD_IDX
    correct_lps_1st_sentence = 0
    correct_pen_lps_1st_sentence = 0
    correct_lps_2nd_sentence = 0
    correct_pen_lps_2nd_sentence = 0
    for example_idx, example_data in enumerate(tqdm(testset['sentences'])):
        sentences = get_sentences_from_example(example_data)
        lps = []
        mean_lps = []
        pen_lps = []
        for sent_id, sentence in enumerate(sentences):
            sentence_tokens = tokenizer.tokenize(sentence)
            text_len = len(sentence_tokens)
            lp = get_sentence_score_JHLau(model_type, model, tokenizer, sentence_tokens, device)

            # acceptability measures by sentence idx
            penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
            lps.append(lp)
            # mean_lps.append(lp / text_len)
            pen_lps.append(lp / penalty)
            sent_ids.append(sent_id)
        if lps[GOOD_SENTENCE_1_IDX] > lps[SENTENCE_BAD_IDX]:
            correct_lps_1st_sentence += 1
        if pen_lps[GOOD_SENTENCE_1_IDX] > pen_lps[SENTENCE_BAD_IDX]:
            correct_pen_lps_1st_sentence += 1
        if len(sentences) > 2:
            if lps[GOOD_SENTENCE_2_IDX] > lps[SENTENCE_BAD_IDX]:
                correct_lps_2nd_sentence += 1
            if pen_lps[GOOD_SENTENCE_2_IDX] > pen_lps[SENTENCE_BAD_IDX]:
                correct_pen_lps_2nd_sentence += 1

    examples_count = len(testset['sentences'])
    print(f'test results report:')
    print(f'acc. correct_lps_1st_sentence: {perc(correct_lps_1st_sentence, examples_count):.1f} %')
    print(f'acc. correct_pen_lps_1st_sentence: {perc(correct_pen_lps_1st_sentence, examples_count):.1f} %')
    print(f'acc. correct_lps_2nd_sentence: {perc(correct_lps_2nd_sentence, examples_count):.1f} %')
    print(f'acc. correct_pen_lps_2nd_sentence: {perc(correct_pen_lps_2nd_sentence, examples_count):.1f} %')


def perc(value, total):
    return 100 * (value / total)

# nb, for bert it uses softmax
def get_sentence_score_JHLau(model_type: model_types, model, tokenizer, sentence_tokens, device):

    if model_type == model_types.GPT:

        # not use context variant:
        #prepend the sentence with <|endoftext|> token, so that the loss is computed correctly
        tensor_input = torch.tensor([[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(sentence_tokens)], device=device)
        labels = torch.tensor([[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(sentence_tokens)], device=device)
        labels[:,:1] = -1
        loss = model(tensor_input, labels=tensor_input)

        return float(loss[0]) * -1.0 * len(sentence_tokens)

    elif model_type == model_types.BERT:

        batched_indexed_tokens = []
        batched_segment_ids = []

        # not use_context variant:
        tokenize_combined = ["[CLS]"] + sentence_tokens + ["[SEP]"]

        for i in range(len(sentence_tokens)):
            # Mask a token that we will try to predict back with `BertForMaskedLM`
            masked_index = i + 1 + 0  # not use_context variant
            tokenize_masked = tokenize_combined.copy()
            tokenize_masked[masked_index] = '[MASK]'
            # unidir bert
            # for j in range(masked_index, len(tokenize_combined)-1):
            #    tokenize_masked[j] = '[MASK]'

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segment_ids = [0] * len(tokenize_masked)

            batched_indexed_tokens.append(indexed_tokens)
            batched_segment_ids.append(segment_ids)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)
        segment_tensor = torch.tensor(batched_segment_ids, device=device)

        # Predict all tokens
        with torch.no_grad():
            # print(f'type(model): {type(model)}')
            outputs = model(tokens_tensor, token_type_ids=segment_tensor)
            predictions = outputs[0]

        # go through each word and sum their logprobs
        lp = 0.0
        for i in range(len(sentence_tokens)):
            masked_index = i + 1 + 0  # not use_context variant
            predicted_score = predictions[i, masked_index]
            predicted_prob = softmax(predicted_score.cpu().numpy())
            lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])

        return lp

