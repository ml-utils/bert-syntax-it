from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from enum import Enum


class StrEnum(str, Enum):
    pass


class SentenceNames(StrEnum):
    SHORT_NONISLAND = "short_nonisland"
    LONG_NONISLAND = "long_nonisland"
    SHORT_ISLAND = "short_island"
    LONG_ISLAND = "long_island"
    SENTENCE_BAD = "sentence_bad"
    SENTENCE_GOOD = "sentence_good"
    SENTENCE_GOOD_2ND = "sentence_good_2nd"


@dataclass
class Sentence:
    txt: str

    lp: float = -200
    pen_lp: float = -200
    sentence_log_weight: float = -200
    pen_sentence_log_weight: float = -200
    # lps: list[float] = field(default_factory=list)
    # pen_lps: list[float] = field(default_factory=list)
    token_weights: list[float] = field(default_factory=list)

    # todo? add sentence ids
    tokens: list[str] = field(default_factory=list)

    def get_score(self, score_name):
        if score_name == "lp":
            return self.lp
        elif score_name == "pen_lp":
            return self.pen_lp
        else:
            raise ValueError(f"Unexcpected score name: {score_name}")


@dataclass
class TypedSentence:
    stype: SentenceNames
    sent: Sentence


@dataclass
class Example:
    sentences: list[TypedSentence]

    min_token_weight: float = 200
    max_token_weight: float = -200
    # token_weights_by_sentence = [] # ..todo

    DD_with_lp: float = -200
    DD_with_penlp: float = -200


@dataclass
class TestSet:
    linguistic_phenomenon: str
    model_descr: str
    examples: list[Example]

    sent_types: InitVar[list[SentenceNames]]
    lp_average_by_sentence_type: dict[SentenceNames, float] = field(
        default_factory=dict
    )
    penlp_average_by_sentence_type: dict[SentenceNames, float] = field(
        default_factory=dict
    )
    avg_DD_lp: float = -200
    avg_DD_penlp: float = -200
    accuracy_by_DD_lp: float = 0
    accuracy_by_DD_penlp: float = 0

    min_token_weight: float = 200
    max_token_weight: float = -200

    # todo: add model descriptor, indicating from which model the scrores where calculated
    # todo: normalize score_averages ..

    def __post_init__(self, sent_types):
        for stype in sent_types:
            self.lp_average_by_sentence_type[stype] = 0
            self.penlp_average_by_sentence_type[stype] = 0

    def get_avg_scores(self, score_name):
        if score_name == "lp":
            return self.lp_average_by_sentence_type
        elif score_name == "pen_lp":
            return self.penlp_average_by_sentence_type
        else:
            raise ValueError(f"Unexcpected score name: {score_name}")

    def get_avg_DD(self, score_name):
        if score_name == "lp":
            return self.avg_DD_lp
        elif score_name == "pen_lp":
            return self.avg_DD_penlp
        else:
            raise ValueError(f"Unexcpected score name: {score_name}")


def parse_testset(
    linguistic_phenomenon,
    model_descr,
    examples_list: list,
    sent_types_descr: str,
    max_examples=50,
):
    print(f"len examples: {len(examples_list)}, max: {max_examples}")

    if sent_types_descr == "sprouse":
        sent_types = [
            SentenceNames.SHORT_NONISLAND,
            SentenceNames.LONG_NONISLAND,
            SentenceNames.SHORT_ISLAND,
            SentenceNames.LONG_ISLAND,
        ]
    elif sent_types_descr == "blimp":
        sent_types = [
            SentenceNames.SENTENCE_GOOD,
            SentenceNames.SENTENCE_BAD,
        ]
    else:
        raise ValueError(f"unrecognized sentence types format: {sent_types_descr}")

    if len(examples_list) > max_examples:
        # print(f"slicing the number of examples to {max_examples}")
        examples_list = examples_list[:max_examples]

    parsed_examples = []
    for example in examples_list:
        parsed_example = parse_example(example, sent_types)
        parsed_examples.append(parsed_example)

    return TestSet(linguistic_phenomenon, model_descr, parsed_examples, sent_types)


def parse_example(example: dict, sent_types: list):
    typed_senteces = []
    # print(f"example: {example}")

    for sent_type in sent_types:
        typed_sentece = parse_typed_sentence(sent_type, example[sent_type])
        typed_senteces.append(typed_sentece)

    return Example(typed_senteces)


def parse_typed_sentence(stype: SentenceNames, txt: str):
    sent = Sentence(txt)
    return TypedSentence(stype, sent)
