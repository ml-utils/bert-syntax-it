class model_types:
    BERT = 0
    GPT = 1


class sentence_score_bases:
    SOFTMAX = 0
    NORMALIZED_LOGITS = 1


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_pen_score(unnormalized_score, text_len):
    penalty = ((5 + text_len) ** 0.8 / (5 + 1) ** 0.8)
    return unnormalized_score / penalty


def red_txt(txt: str):
    return f'{bcolors.RED}{txt}{bcolors.ENDC}'


def print_red(txt: str):
    print_in_color(txt, bcolors.RED)


def print_orange(txt: str):
    print_in_color(txt, bcolors.WARNING)


def print_in_color(txt, color: bcolors):
    print(f'{color}{txt}{bcolors.ENDC}')
