import torch
from transformers import AutoConfig
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from lm_utils import get_pen_score


class ml_libraries:
    PYTORCH = "pt"
    TENSORFLOW = "tf"


def get_model():
    model_name = "gpt2"  # 'bert-base-uncased'
    model = torch.hub.load("huggingface/pytorch-transformers", "model", model_name)
    # Download model and configuration from S3 and cache.

    assert model.config.output_attentions
    # Loading from a TF checkpoint file instead of a PyTorch model (slower)
    config = AutoConfig.from_pretrained("./tf_model/gpt_tf_model_config.json")

    checkpoint_path = "./tf_model/gpt_tf_checkpoint.ckpt.index"
    model = torch.hub.load(
        "huggingface/transformers",
        "modelForCausalLM",
        checkpoint_path,
        from_tf=True,
        config=config,
    )


def get_model2():
    # https://huggingface.co/gpt2
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained("GroNLP/gpt2-small-italian")  # 'gpt2'
    model = GPT2Model.from_pretrained("GroNLP/gpt2-small-italian")  # 'gpt2'
    return model, tokenizer


def get_model3():
    # https://huggingface.co/GroNLP/gpt2-small-italian
    # from transformers import pipeline
    # pipe = pipeline("text-generation", model="GroNLP/gpt2-small-italian")
    from transformers import AutoTokenizer, AutoModel  # , TFAutoModel

    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    model = AutoModel.from_pretrained("GroNLP/gpt2-small-italian")  # PyTorch
    # model = TFAutoModel.from_pretrained("GroNLP/gpt2-small-italian")  # Tensorflow
    return model, tokenizer


def get_model4():
    model = GPT2LMHeadModel.from_pretrained("GroNLP/gpt2-small-italian")
    tokenizer = GPT2Tokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    return model, tokenizer


def get_gpt_sentence_score(model, tokenizer: GPT2Tokenizer, tokenized_input, device):
    # prepend the sentence with <|endoftext|> token, so that the loss is computed correctly
    # bos_token (str, optional, defaults to <|endoftext|>) — The beginning of sequence token
    # bos_token = "<|endoftext|>"
    # bos_id = tokenizer.convert_tokens_to_ids([tokenizer.bos_token_id])  # 50256
    print(
        f"tokenizer.bos_token: {tokenizer.bos_token}, tokenizer.bos_token: {tokenizer.unk_token}"
    )
    tensor_input = torch.tensor(
        [[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenized_input)],
        device=device,
    )

    labels = torch.tensor(
        [[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenized_input)],
        device=device,
    )
    labels[:, :1] = -1

    # IndexError: index out of range in self
    # 50256 for <|endoftext|> token
    loss = model(tensor_input, labels=tensor_input)
    score = float(loss[0]) * -1.0 * len(tokenized_input)
    return score


def main():
    print("getting gpt model..")
    model, tokenizer = get_model4()
    print("model loaded")
    sentence_good_base = (
        "Gianni ha detto che il libro di linguistia ha duecento pagine."
    )
    sentence_bad = "Di che cosa Gianni ha detto che il libro ha duecento pagine?"
    score_sentence_good_base = __get_gpt_sentence_score(
        model, tokenizer, sentence_good_base
    )
    score_sentence_bad = __get_gpt_sentence_score(model, tokenizer, sentence_bad)
    print(
        f"score_sentence_good_base: {score_sentence_good_base}, score_sentence_bad: {score_sentence_bad}"
    )


def __get_gpt_sentence_score(model, tokenizer, text, verbose=False):
    # https://huggingface.co/gpt2
    encoded_inputs = tokenizer(text, return_tensors=ml_libraries.PYTORCH)
    #
    # # output = model(**encoded_inputs)
    # print(f'type(output) {type(output)}, len(output) {len(output)}')
    # #type(output) <class 'transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions'>
    # print(f'type(output[0]) {type(output[0])} len(output[0]) {len(output[0])}, output[0].size(): {output[0].size()}')
    # # type(output[0]) <class 'torch.Tensor'> len(output[0]) 1, torch.Size([1, 13, 768])
    # print(f'type(output[1]) {type(output[1])} len(output[1]) {len(output[1])}')
    # # type(output[1]) <class 'tuple'> len(output[1]) 12
    # print(f'type(output[1][0]): {type(output[1][0])}, len(output[1][0]): {len(output[1][0])}')
    # # type(output[1][0]): <class 'tuple'>, len(output[1][0]): 2
    # print(f'type(output[1][0][0]): {type(output[1][0][0])}, len(output[1][0][0]): {len(output[1][0][0])}, '
    #       f'output[1][0][0].size(): {output[1][0][0].size()}')
    # # type(output[1][0][0]): <class 'torch.Tensor'>, len(output[1][0][0]): 1
    # # output[1][0][0].size(): torch.Size([1, 12, 13, 64])
    # print(f'type(output[1][0][1]): {type(output[1][0][1])}, len(output[1][0][1]): {len(output[1][0][1])}, '
    #       f'output[1][0][1].size(): {output[1][0][1].size()}')
    # # torch.Size([1, 12, 13, 64])
    # # print(output)

    output = model(**encoded_inputs, labels=encoded_inputs["input_ids"])
    # type(output) <class 'transformers.modeling_outputs.CausalLMOutputWithCrossAttentions'>, len(output) 3
    loss = output.loss
    # logits = output.logits

    device = None
    tokenize_input = tokenizer.tokenize(text)
    sentence_score = get_gpt_sentence_score(model, tokenizer, tokenize_input, device)
    print(f"text lenght: {len(tokenize_input)}")

    sentence_score = get_pen_score(sentence_score, len(tokenize_input))

    if verbose:
        print(
            f"type(output.loss) {type(output.loss)}, output.loss.size(): {output.loss.size()}"
        )
        print(
            f"output[0].size() {output[0].size()}, output[1].size() {output[1].size()}, len(output[2]) {len(output[2])}"
        )
        # output[0].size() torch.Size([]), output[1].size() torch.Size([1, 13, 30001]), len(output[2]) 12
        print(f"loss: {loss}")
        print(f"type(output.logits) {type(output.logits)}")
        # type(output.logits) <class 'torch.Tensor'>
        print(f"size: {output.logits.size()}")
        # size: torch.Size([1, 13, 30001])
        print(sentence_score)

    return sentence_score
