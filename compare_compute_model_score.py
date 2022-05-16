import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # pytorch_transformers

from transformers import BertTokenizer, BertForMaskedLM
from transformers import BertForMaskedLM

from scipy.special import softmax
import numpy as np

from lm_utils import model_types, get_sentences_from_example


def run_testset(model_type, model_name, device, testset):
    # system scores
    lps = []
    mean_lps = []
    pen_lps = []
    div_lps = []
    sub_lps = []
    slors = []
    pen_slors = []
    sent_ids = []

    # Load pre-trained model and tokenizer
    if model_type == model_types.GPT:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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

    for example_idx, example_data in enumerate(testset['sentences']):
        sentences = get_sentences_from_example(example_data)
        for sent_id, sentence in enumerate(sentences):
            sentence_tokens = tokenizer.tokenize(sentence)
            text_len = len(sentence_tokens)
            lp = get_sentence_score_JHLau(model_type, model, tokenizer, sentence_tokens, device)

            # acceptability measures by sentence idx
            penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
            lps.append(lp)
            mean_lps.append(lp / text_len)
            pen_lps.append(lp / penalty)
            sent_ids.append(sent_id)


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

