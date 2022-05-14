import torch
from transformers import AutoConfig


class ml_libraries:
    PYTORCH = 'pt'
    TENSORFLOW = 'tf'


def get_model():
    model_name = 'gpt2'  # 'bert-base-uncased'
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name )
    # Download model and configuration from S3 and cache.

    assert model.config.output_attentions == True
    # Loading from a TF checkpoint file instead of a PyTorch model (slower)
    config = AutoConfig.from_pretrained('./tf_model/gpt_tf_model_config.json')

    checkpoint_path = './tf_model/gpt_tf_checkpoint.ckpt.index'
    model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', checkpoint_path,
                           from_tf=True, config=config)


def get_model2():
    # https://huggingface.co/gpt2
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    return model, tokenizer


def get_model3():
    # https://huggingface.co/GroNLP/gpt2-small-italian
    from transformers import pipeline
    pipe = pipeline("text-generation", model="GroNLP/gpt2-small-italian")
    from transformers import AutoTokenizer, AutoModel, TFAutoModel
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    model = AutoModel.from_pretrained("GroNLP/gpt2-small-italian")  # PyTorch
    # model = TFAutoModel.from_pretrained("GroNLP/gpt2-small-italian")  # Tensorflow
    return model, tokenizer


def main():
    print(f'getting gpt model..')
    model, tokenizer = get_model3()
    print(f'model loaded')
    text = "Di che cosa Gianni ha detto che il libro ha duecento pagine?"
    # https://huggingface.co/gpt2
    encoded_input = tokenizer(text, return_tensors=ml_libraries.PYTORCH)
    output = model(**encoded_input)
    print(f'type(output) {type(output)}, len(output) {len(output)}')
    #type(output) <class 'transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions'>
    print(f'type(output[0]) {type(output[0])} len(output[0]) {len(output[0])}, output[0].size(): {output[0].size()}')
    # type(output[0]) <class 'torch.Tensor'> len(output[0]) 1, torch.Size([1, 13, 768])
    print(f'type(output[1]) {type(output[1])} len(output[1]) {len(output[1])}')
    # type(output[1]) <class 'tuple'> len(output[1]) 12
    print(f'type(output[1][0]): {type(output[1][0])}, len(output[1][0]): {len(output[1][0])}')
    # type(output[1][0]): <class 'tuple'>, len(output[1][0]): 2
    print(f'type(output[1][0][0]): {type(output[1][0][0])}, len(output[1][0][0]): {len(output[1][0][0])}, '
          f'output[1][0][0].size(): {output[1][0][0].size()}')
    # type(output[1][0][0]): <class 'torch.Tensor'>, len(output[1][0][0]): 1
    # output[1][0][0].size(): torch.Size([1, 12, 13, 64])
    print(f'type(output[1][0][1]): {type(output[1][0][1])}, len(output[1][0][1]): {len(output[1][0][1])}, '
          f'output[1][0][1].size(): {output[1][0][1].size()}')
    # torch.Size([1, 12, 13, 64])
    # print(output)


