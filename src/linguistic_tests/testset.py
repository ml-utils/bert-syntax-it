from dataclasses import dataclass
from dataclasses import field
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
    sentence_log_weights: list[float] = field(default_factory=list)
    pen_sentence_log_weights: list[float] = field(default_factory=list)
    # lps: list[float] = field(default_factory=list)
    # pen_lps: list[float] = field(default_factory=list)
    token_weights: list[float] = field(default_factory=list)


@dataclass
class TypedSentence:
    type: SentenceNames
    sent: Sentence


@dataclass
class Example:
    sentences: list[TypedSentence]

    min_token_weight: float = 200
    max_token_weight: float = -200
    # token_weights_by_sentence = [] # ..todo


@dataclass
class TestSet:
    examples: list[Example]
    min_token_weight: float = 200
    max_token_weight: float = -200


def parse_testset(examples_list: list, sent_types_descr: str):
    print(f"len examples: {len(examples_list)}")

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

    parsed_examples = []
    for example in examples_list:
        parsed_example = parse_example(example, sent_types)
        parsed_examples.append(parsed_example)

    return TestSet(parsed_examples)


def parse_example(example: dict, sent_types: list):
    typed_senteces = []
    print(f"example: {example}")

    for sent_type in sent_types:
        typed_sentece = parse_typed_sentence(sent_type, example[sent_type])
        typed_senteces.append(typed_sentece)

    return Example(typed_senteces)


def parse_typed_sentence(stype: SentenceNames, txt: str):
    sent = Sentence(txt)
    return TypedSentence(stype, sent)
