class model_types:
    BERT = 0
    GPT = 1


def get_pen_score(unnormalized_score, text_len):
    penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
    return unnormalized_score / penalty