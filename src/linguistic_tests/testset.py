import os
import pickle
import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import KeysView
from typing import List

import numpy as np
from linguistic_tests.lm_utils import assert_almost_equale
from linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from linguistic_tests.lm_utils import get_results_dir
from linguistic_tests.lm_utils import ModelTypes
from linguistic_tests.lm_utils import print_orange
from linguistic_tests.lm_utils import ScoringMeasures
from linguistic_tests.lm_utils import SentenceNames
from scipy.stats import zmap

SPROUSE_SENTENCE_TYPES = [
    SentenceNames.SHORT_NONISLAND,
    SentenceNames.LONG_NONISLAND,
    SentenceNames.SHORT_ISLAND,
    SentenceNames.LONG_ISLAND,
]

BLIMP_SENTENCE_TYPES = [
    SentenceNames.SENTENCE_GOOD,
    SentenceNames.SENTENCE_BAD,
]

ERROR_LP: float = -200.0


@dataclass
class Sentence:
    txt: str

    lp_softmax: float = ERROR_LP
    pen_lp_softmax: float = ERROR_LP
    lp_logistic: float = ERROR_LP
    pen_lp_logistic: float = ERROR_LP

    # todo? add sentence ids
    tokens: list[str] = field(default_factory=list)

    def get_score(self, scoring_measure: ScoringMeasures):
        if scoring_measure == ScoringMeasures.LP:
            score = self.lp_softmax
        elif scoring_measure == ScoringMeasures.PenLP:
            score = self.pen_lp_softmax
        elif scoring_measure == ScoringMeasures.LL:
            score = self.lp_logistic
        elif scoring_measure == ScoringMeasures.PLL:
            score = self.pen_lp_logistic
        else:
            raise ValueError(f"Unexpected scoring_measure: {scoring_measure}")

        if score == ERROR_LP:
            raise ValueError(
                f"Score {scoring_measure} "
                f"for this sentence not set {score},"
                f" '{self.txt}' "
            )
        return score


@dataclass
class TypedSentence:
    stype: SentenceNames
    sent: Sentence


@dataclass
class Example:
    # todo: convert to dict[]
    sentences: list[TypedSentence]

    DD_with_lp: float = ERROR_LP
    DD_with_penlp: float = ERROR_LP
    DD_with_ll: float = ERROR_LP
    DD_with_penll: float = ERROR_LP

    def __getitem__(self, key: SentenceNames) -> Sentence:
        for typed_sentence in self.sentences:
            if typed_sentence.stype == key:
                return typed_sentence.sent
        raise ValueError(f"Invalid key, it's not a SentenceName: {key}")

    def get_score(self, scoring_measure: ScoringMeasures, sent_type: SentenceNames):
        self[sent_type].get_score(scoring_measure)

    def get_structure_effect(self, score_descr) -> float:
        raise NotImplementedError

    def get_score_diff(
        self, score_descr: ScoringMeasures, stype1: SentenceNames, stype2: SentenceNames
    ) -> float:
        return self[stype1].get_score(score_descr) - self[stype2].get_score(score_descr)

    def get_type_of_unacceptable_sentence(self) -> SentenceNames:

        unacceptable_sentences_types = [
            SentenceNames.LONG_ISLAND,
            SentenceNames.SENTENCE_BAD,
        ]
        for unacceptable_sentences_type in unacceptable_sentences_types:
            for typed_sentence in self.sentences:
                if typed_sentence.stype == unacceptable_sentences_type:
                    return unacceptable_sentences_type
        raise AttributeError(
            f"No unacceptable sentence type found in this example: {self.sentences}"
        )

    def is_scored_accurately(
        self, score_descr: ScoringMeasures, stype: SentenceNames
    ) -> bool:
        score_diff = self.get_score_diff(
            score_descr, stype, self.get_type_of_unacceptable_sentence()
        )
        return score_diff > 0


@dataclass
class TestSet:
    linguistic_phenomenon: str
    model_descr: str
    dataset_source: str
    examples: list[Example]

    sent_types: InitVar[list[SentenceNames]]
    scoring_measures: InitVar[list[ScoringMeasures]]
    model_type: InitVar[ModelTypes]

    lp_average_by_sentence_type: dict[SentenceNames, float] = field(
        default_factory=dict
    )
    penlp_average_by_sentence_type: dict[SentenceNames, float] = field(
        default_factory=dict
    )
    ll_average_by_sentence_type: dict[SentenceNames, float] = field(
        default_factory=dict
    )
    penll_average_by_sentence_type: dict[SentenceNames, float] = field(
        default_factory=dict
    )

    # z scores for a testset, to be used in the sprouse-like plots, so just for the DD caclulations
    avg_zscores_by_measure_and_by_stype: dict[
        ScoringMeasures, dict[SentenceNames, float]
    ] = field(default_factory=dict)
    std_error_of_zscores_by_measure_and_by_stype: dict[
        ScoringMeasures, dict[SentenceNames, float]
    ] = field(default_factory=dict)

    avg_DD_lp: float = ERROR_LP
    avg_DD_penlp: float = ERROR_LP
    avg_DD_ll: float = ERROR_LP
    avg_DD_penll: float = ERROR_LP

    accuracy_by_DD_lp: float = 0
    accuracy_by_DD_penlp: float = 0
    accuracy_by_DD_ll: float = 0
    accuracy_by_DD_penll: float = 0

    accuracy_per_score_type_per_sentence_type: dict[
        ScoringMeasures, dict[SentenceNames, float]
    ] = field(default_factory=dict)

    min_token_weight: float = -1 * ERROR_LP
    max_token_weight: float = ERROR_LP

    def __post_init__(self, sent_types, scoring_measures, model_type: ModelTypes):
        for stype in sent_types:
            self.lp_average_by_sentence_type[stype] = 0
            self.penlp_average_by_sentence_type[stype] = 0

            if (
                model_type in BERT_LIKE_MODEL_TYPES
            ):  # todo: use isinstance(var, (classinfo1, classinfo2, classinfo3))
                self.ll_average_by_sentence_type[stype] = 0
                self.penll_average_by_sentence_type[stype] = 0

        for scoring_measure in scoring_measures:

            self.accuracy_per_score_type_per_sentence_type[scoring_measure] = dict()
            self.avg_zscores_by_measure_and_by_stype[scoring_measure] = dict()
            self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure] = dict()

            for stype in sent_types:

                self.avg_zscores_by_measure_and_by_stype[scoring_measure][stype] = 0
                self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                    stype
                ] = 0

                # an example scores accurately a sentence type if it gives it an higher acceptability score
                # than the ungrammatical/unacceptable sentence
                # so for the ungrammatical/unacceptable sentence there is no point in storing an accuracy value
                # (it would be like comparing it with iself)
                if (
                    stype is not SentenceNames.SENTENCE_BAD
                    and stype is not SentenceNames.LONG_ISLAND
                ):
                    self.accuracy_per_score_type_per_sentence_type[scoring_measure][
                        stype
                    ] = 0

    def get_sentence_types(self) -> KeysView[SentenceNames]:
        return self.lp_average_by_sentence_type.keys()

    def get_scoring_measures(self) -> KeysView[ScoringMeasures]:
        return self.accuracy_per_score_type_per_sentence_type.keys()

    def get_acceptable_sentence_types(self) -> KeysView[SentenceNames]:
        some_scoring_measure = next(iter(self.get_scoring_measures()))
        return self.accuracy_per_score_type_per_sentence_type[
            some_scoring_measure
        ].keys()

    def set_avg_zscores_by_measure_and_by_stype(
        self, scoring_measure: ScoringMeasures, merged_scores
    ):

        for stype in self.get_sentence_types():

            # convert all the scores to zscores
            all_scores_this_measure_and_stype: List[float] = []
            for example in self.examples:
                score = example[stype].get_score(scoring_measure)
                all_scores_this_measure_and_stype.append(score)
            all_zscores_this_measure_and_stype: List[float] = zmap(
                all_scores_this_measure_and_stype, merged_scores
            )

            # then calculate the averages
            self.avg_zscores_by_measure_and_by_stype[scoring_measure][
                stype
            ] = np.average(all_zscores_this_measure_and_stype)

            # also calculate the standard errors for the error bars:
            # std_error = np.std(data, ddof=1) / np.sqrt(len(data))
            self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                stype
            ] = get_std_error(all_zscores_this_measure_and_stype)

    def get_avg_zscores_by_measure_and_by_stype(
        self, scoring_measure: ScoringMeasures, stype: SentenceNames
    ):
        return self.avg_zscores_by_measure_and_by_stype[scoring_measure][stype]

    def get_std_error_of_zscores_by_measure_and_by_stype(
        self, scoring_measure: ScoringMeasures, stype: SentenceNames
    ):
        return self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][stype]

    def get_std_errors_of_zscores_by_measure_and_sentence_structure(
        self, scoring_measure: ScoringMeasures, sentence_structure: SentenceNames
    ):
        if sentence_structure in [
            SentenceNames.SHORT_NONISLAND,
            SentenceNames.LONG_NONISLAND,
        ]:
            return [
                self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                    SentenceNames.SHORT_NONISLAND
                ],
                self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                    SentenceNames.LONG_NONISLAND
                ],
            ]
        if sentence_structure in [
            SentenceNames.SHORT_ISLAND,
            SentenceNames.LONG_ISLAND,
        ]:
            return [
                self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                    SentenceNames.SHORT_ISLAND
                ],
                self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                    SentenceNames.LONG_ISLAND
                ],
            ]
        raise ValueError(
            f"Unrecognized sentence type as sentenc structure type: {sentence_structure}"
        )

    def get_avg_scores(self, scoring_measure: ScoringMeasures):
        if scoring_measure == ScoringMeasures.LP:
            return self.lp_average_by_sentence_type
        elif scoring_measure == ScoringMeasures.PenLP:
            return self.penlp_average_by_sentence_type
        if scoring_measure == ScoringMeasures.LL:
            return self.ll_average_by_sentence_type
        elif scoring_measure == ScoringMeasures.PLL:
            return self.penll_average_by_sentence_type
        else:
            raise ValueError(f"Unexpected scoring_measure: {scoring_measure}")

    def get_avg_DD_zscores(self, scoring_measure: ScoringMeasures):
        return get_dd_score_parametric(
            a_short_nonisland_score=self.avg_zscores_by_measure_and_by_stype[
                scoring_measure
            ][SentenceNames.SHORT_NONISLAND],
            b_long_nonisland_score=self.avg_zscores_by_measure_and_by_stype[
                scoring_measure
            ][SentenceNames.LONG_NONISLAND],
            c_short_island_score=self.avg_zscores_by_measure_and_by_stype[
                scoring_measure
            ][SentenceNames.SHORT_ISLAND],
            d_long_island_score=self.avg_zscores_by_measure_and_by_stype[
                scoring_measure
            ][SentenceNames.LONG_ISLAND],
        )

    def get_pvalues_of_avg_DD_zscores(self, scoring_measure: ScoringMeasures):
        raise NotImplementedError

    def get_avg_DD(self, scoring_measure: ScoringMeasures):
        if scoring_measure == ScoringMeasures.LP:
            return self.avg_DD_lp
        elif scoring_measure == ScoringMeasures.PenLP:
            return self.avg_DD_penlp
        elif scoring_measure == ScoringMeasures.LL:
            return self.avg_DD_ll
        elif scoring_measure == ScoringMeasures.PLL:
            return self.avg_DD_penll
        else:
            raise ValueError(f"Unexpected scoring_measure: {scoring_measure}")

    def get_examples_sorted_by_score_diff(
        self,
        score_descr,
        sent_type1: SentenceNames,
        sent_type2: SentenceNames,
        reverse=True,
    ) -> list[Example]:
        return sorted(
            self.examples,
            key=lambda x: x.get_score_diff(score_descr, sent_type1, sent_type2),
            reverse=reverse,
        )

    def get_all_sentences_sorted_by_score(
        self, score_descr: ScoringMeasures, reverse=True
    ) -> list[TypedSentence]:
        all_sentences = []
        for example in self.examples:
            for typed_sent in example.sentences:
                all_sentences.append(typed_sent)
        return sorted(
            all_sentences,
            key=lambda x: x.sent.get_score(score_descr),
            reverse=reverse,
        )

    def get_examples_sorted_by_sentence_type_and_score(
        self, stype: SentenceNames, score_descr: ScoringMeasures, reverse=True
    ) -> list[Example]:
        return sorted(
            self.examples,
            key=lambda x: x[stype].get_score(score_descr),
            reverse=reverse,
        )

    def save_to_pickle(self, filename):

        for example in self.examples:
            for typed_sent in example.sentences:
                assert ERROR_LP != typed_sent.sent.get_score(ScoringMeasures.LP)
                assert ERROR_LP != typed_sent.sent.get_score(ScoringMeasures.PenLP)

        saving_dir = str(get_results_dir())
        filepath = os.path.join(saving_dir, filename)
        if os.path.exists(filepath):
            timestamp = time.strftime("%Y-%m-%d_h%Hm%Ms%S")
            filename = f"{filename}-{timestamp}.testset.pickle"
            filepath = os.path.join(saving_dir, filename)
        print_orange(f"Saving testset to {filepath}")
        with open(filepath, "wb") as file:
            pickle.dump(self, file)


def load_testset_from_pickle(filename) -> TestSet:
    saving_dir = str(get_results_dir())

    # todo, fixme: should actually look for all the files in that dir that
    #  start with {filename}, and pick the most recent one.
    #  or the save method, when the file already exists, could rename the
    #  existing one and move it to the prev_pickles subdir, before saving
    #  the new one.
    filepath = os.path.join(saving_dir, filename)
    print(f"Loading testset from {filepath}..")
    with open(filepath, "rb") as file:
        testset = pickle.load(file)

    for example in testset.examples:

        assert ERROR_LP != example.DD_with_lp
        assert ERROR_LP != example.DD_with_penlp

        for typed_sent in example.sentences:
            assert ERROR_LP != typed_sent.sent.get_score(ScoringMeasures.LP)
            assert ERROR_LP != typed_sent.sent.get_score(ScoringMeasures.PenLP)

    return testset


def parse_testset(
    linguistic_phenomenon,
    model_type: ModelTypes,
    model_descr,
    dataset_source: str,
    examples_list: list,
    sent_types_descr: str,
    scoring_measures: list[ScoringMeasures],
    max_examples=50,
) -> TestSet:
    print(f"len examples: {len(examples_list)}, max: {max_examples}")

    if sent_types_descr == "sprouse":
        sent_types = SPROUSE_SENTENCE_TYPES
    elif sent_types_descr == "blimp":
        sent_types = BLIMP_SENTENCE_TYPES
    else:
        raise ValueError(f"unrecognized sentence types format: {sent_types_descr}")

    if len(examples_list) > max_examples:
        # print(f"slicing the number of examples to {max_examples}")
        examples_list = examples_list[:max_examples]

    parsed_examples = []
    for example in examples_list:
        parsed_example = parse_example(example, sent_types)
        parsed_examples.append(parsed_example)

    return TestSet(
        linguistic_phenomenon,
        model_descr,
        dataset_source,
        parsed_examples,
        sent_types,
        scoring_measures,
        model_type,
    )


def get_merged_score_across_testsets(
    scoring_measure: ScoringMeasures, testsets: List[TestSet]
):
    merged_scores = []
    for testset in testsets:
        for example in testset.examples:
            for typed_sentence in example.sentences:
                merged_scores.append(typed_sentence.sent.get_score(scoring_measure))

    return merged_scores


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


def get_dd_score_parametric(
    a_short_nonisland_score,
    b_long_nonisland_score,
    c_short_island_score,
    d_long_island_score,
):
    example_lenght_effect = a_short_nonisland_score - b_long_nonisland_score
    example_structure_effect = a_short_nonisland_score - c_short_island_score
    example_total_effect = a_short_nonisland_score - d_long_island_score
    example_island_effect = example_total_effect - (
        example_lenght_effect + example_structure_effect
    )
    example_dd = example_structure_effect - (
        b_long_nonisland_score - d_long_island_score
    )

    example_dd *= -1
    assert_almost_equale(example_island_effect, example_dd)
    return example_dd


def get_std_error(arr):
    return np.std(arr, ddof=1) / np.sqrt(len(arr))
