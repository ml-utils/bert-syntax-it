# from tokenizer import *
from transformers import AutoTokenizer, AutoModel
# from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
# from fairseq.modules import TransformerSentenceEncoderLayer


def get_filberto_model():
    tokenizer_gil = AutoTokenizer.from_pretrained("idb-ita/gilberto-uncased-from-camembert", do_lower_case=True)
    model = AutoModel.from_pretrained("idb-ita/gilberto-uncased-from-camembert")
    # Import GilBERTo with pytorch\fairseq Library
    #gilberto_model = FairseqRobertaModel.from_pretrained('path/to/checkpoints_folder', bpe='sentencepiece')
    # Mask Predictions
    #gilberto_model.fill_mask('Buongiorno mi <mask> Gilberto!', topk=3)  # Fill mask token with GilBERTo

    # type(model: <class 'transformers.models.camembert.modeling_camembert.CamembertModel'>,
    # tokenizer_gil: <class 'transformers.models.camembert.tokenization_camembert_fast.CamembertTokenizerFast'>

    return model, tokenizer_gil


def main():
    get_filberto_model()


if __name__ == "__main__":
    main()
