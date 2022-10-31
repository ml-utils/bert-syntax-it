import logging
import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.stats import zmap

from src.linguistic_tests.file_utils import get_file_root
from src.linguistic_tests.file_utils import get_pickle_filename
from src.linguistic_tests.file_utils import load_object_from_pickle
from src.linguistic_tests.file_utils import save_obj_to_pickle
from src.linguistic_tests.lm_utils import assert_almost_equal
from src.linguistic_tests.lm_utils import BERT_LIKE_MODEL_TYPES
from src.linguistic_tests.lm_utils import Conditions
from src.linguistic_tests.lm_utils import DataSources
from src.linguistic_tests.lm_utils import discretize
from src.linguistic_tests.lm_utils import ExperimentalDesigns
from src.linguistic_tests.lm_utils import load_testset_data
from src.linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_EN
from src.linguistic_tests.lm_utils import MODEL_TYPES_AND_NAMES_IT
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import ScoringMeasures


SPROUSE_SENTENCE_TYPES: List[Conditions] = [
    Conditions.SHORT_NONISLAND,
    Conditions.LONG_NONISLAND,
    Conditions.SHORT_ISLAND,
    Conditions.LONG_ISLAND,
]

BLIMP_SENTENCE_TYPES: List[Conditions] = [
    Conditions.SENTENCE_GOOD,
    Conditions.SENTENCE_BAD,
]

MINIMAL_PAIRS_VARIATIONS_SENTENCE_TYPES: List[Conditions] = [
    Conditions.SHORT_NONISLAND,
    Conditions.LONG_NONISLAND,
    Conditions.SHORT_ISLAND,
    Conditions.LONG_ISLAND,
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
    tokens: List[str] = field(default_factory=list)
    # these are the individual words (one OOV word might correspond to n tokens)
    # for each pretoken (str), there is a Tuple[int, int] indicating the sentence idxes of first and last token
    pretokens: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)

    def __str__(self):
        return (
            f"Sent(lp_s={self.lp_softmax}, plp_s={self.pen_lp_softmax}, "
            f"lp_l={self.lp_logistic}, plp_l={self.pen_lp_logistic})"
        )

    def get_score(self, scoring_measure: ScoringMeasures) -> float:
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

        if score is None or score == ERROR_LP:
            raise ValueError(
                f"Score {scoring_measure} "
                f"for this sentence not set {score},"
                f" '{self.txt}' "
            )
        return score


@dataclass
class TypedSentence:
    stype: Conditions
    sent: Sentence

    def __str__(self):
        return f"TSent({self.stype}: {self.sent})"


@dataclass
class TestItem:
    # todo: convert to Dict[]
    sentences: List[TypedSentence]

    DD_with_lp: float = ERROR_LP
    DD_with_penlp: float = ERROR_LP
    DD_with_ll: float = ERROR_LP
    DD_with_penll: float = ERROR_LP

    def __str__(self, scoring_measure: ScoringMeasures = None):
        if scoring_measure is None:
            return f"Example({self.sentences})"
        else:
            # Sent(lp_s={self.lp_softmax}, plp_s={self.pen_lp_softmax}, " \
            #                f"lp_l={self.lp_logistic}, plp_l={self.pen_lp_logistic})
            descr = ""
            for stype in self.get_sentence_types():
                descr += f"{stype}={self.get_score(scoring_measure, stype)}, "
            return f"Example({scoring_measure.name}: {descr})"

    def __getitem__(self, key: Conditions) -> Sentence:
        for typed_sentence in self.sentences:
            if typed_sentence.stype == key:
                return typed_sentence.sent
        raise ValueError(f"Invalid key, it's not a SentenceName: {key}")

    def get_score(
        self, scoring_measure: ScoringMeasures, sent_type: Conditions
    ) -> float:
        return self[sent_type].get_score(scoring_measure)

    def get_lenght_effect(self, scoring_measure: ScoringMeasures) -> float:
        short_nonisland = self[Conditions.SHORT_NONISLAND]
        long_nonisland = self[Conditions.LONG_NONISLAND]
        return short_nonisland.get_score(scoring_measure) - long_nonisland.get_score(
            scoring_measure
        )

    def get_structure_effect(self, scoring_measure: ScoringMeasures) -> float:
        short_nonisland = self[Conditions.SHORT_NONISLAND]
        short_island = self[Conditions.SHORT_ISLAND]
        return short_nonisland.get_score(scoring_measure) - short_island.get_score(
            scoring_measure
        )

    def get_total_effect(self, scoring_measure: ScoringMeasures) -> float:
        short_nonisland = self[Conditions.SHORT_NONISLAND]
        long_island = self[Conditions.LONG_ISLAND]
        return short_nonisland.get_score(scoring_measure) - long_island.get_score(
            scoring_measure
        )

    def get_dd_score(self, scoring_measure: ScoringMeasures) -> float:

        for typed_sentence in self.sentences:
            stype = typed_sentence.stype
            sent = typed_sentence.sent
            if stype == Conditions.SHORT_NONISLAND:
                a_short_nonisland = sent
            elif stype == Conditions.LONG_NONISLAND:
                b_long_nonisland = sent
            elif stype == Conditions.SHORT_ISLAND:
                c_short_island = sent
            elif stype == Conditions.LONG_ISLAND:
                d_long_island = sent
            else:
                raise ValueError(f"Unexpected sentence type: {stype}")

        return get_dd_score_parametric(
            a_short_nonisland.get_score(scoring_measure),
            b_long_nonisland.get_score(scoring_measure),
            c_short_island.get_score(scoring_measure),
            d_long_island.get_score(scoring_measure),
        )

    def get_dd_scores(
        self, model_type: ModelTypes
    ) -> Tuple[float, float, Optional[float], Optional[float]]:

        example_dd_with_lp = self.get_dd_score(ScoringMeasures.LP)
        example_dd_with_penlp = self.get_dd_score(ScoringMeasures.PenLP)

        example_dd_with_ll, example_dd_with_pll = None, None
        if model_type in BERT_LIKE_MODEL_TYPES:
            example_dd_with_ll = self.get_dd_score(ScoringMeasures.LL)
            example_dd_with_pll = self.get_dd_score(ScoringMeasures.PLL)

        return (
            example_dd_with_lp,
            example_dd_with_penlp,
            example_dd_with_ll,
            example_dd_with_pll,
        )

    def get_score_diff(
        self, score_descr: ScoringMeasures, stype1: Conditions, stype2: Conditions
    ) -> float:
        return self[stype1].get_score(score_descr) - self[stype2].get_score(score_descr)

    def get_sentence_types(self) -> List[Conditions]:
        return [typed_sentence.stype for typed_sentence in self.sentences]

    def get_types_of_acceptable_sentences(self) -> List[Conditions]:
        types_of_acceptable_sentences = []
        for typed_sentence in self.sentences:
            if typed_sentence.stype is not self.get_type_of_unacceptable_sentence():
                types_of_acceptable_sentences.append(typed_sentence.stype)
        return types_of_acceptable_sentences

    def get_type_of_unacceptable_sentence(self) -> Conditions:

        unacceptable_sentences_types = [
            Conditions.LONG_ISLAND,
            Conditions.SENTENCE_BAD,
        ]
        for unacceptable_sentences_type in unacceptable_sentences_types:
            for typed_sentence in self.sentences:
                if typed_sentence.stype == unacceptable_sentences_type:
                    return unacceptable_sentences_type
        raise AttributeError(
            f"No unacceptable sentence type found in this example: {self.sentences}"
        )

    def is_scored_accurately_for(
        self, score_descr: ScoringMeasures, stype: Conditions
    ) -> bool:
        score_diff = self.get_score_diff(
            score_descr, stype, self.get_type_of_unacceptable_sentence()
        )
        return score_diff > 0


@dataclass
class TestSuite:
    linguistic_phenomenon: str
    model_descr: str
    dataset_source: DataSources
    experimental_design: ExperimentalDesigns
    examples: List[TestItem]

    scoring_measures: InitVar[List[ScoringMeasures]]

    phenomenon_properties: Dict[str, str] = field(default_factory=dict)
    # todo from the jsonl header, fields "conditions" and "factorial_properties_and_levels",
    #  can be used to check the integrity of the json testsuite

    lp_average_by_sentence_type: Dict[Conditions, float] = field(default_factory=dict)
    penlp_average_by_sentence_type: Dict[Conditions, float] = field(
        default_factory=dict
    )
    ll_average_by_sentence_type: Dict[Conditions, float] = field(default_factory=dict)
    penll_average_by_sentence_type: Dict[Conditions, float] = field(
        default_factory=dict
    )

    # z scores for a testset, to be used in the sprouse-like plots, so just for the DD caclulations
    avg_zscores_by_measure_and_by_stype: Dict[
        ScoringMeasures, Dict[Conditions, float]
    ] = field(default_factory=dict)
    avg_zscores_of_likerts_by_measure_and_by_stype: Dict[
        ScoringMeasures, Dict[Conditions, float]
    ] = field(default_factory=dict)
    std_error_of_zscores_by_measure_and_by_stype: Dict[
        ScoringMeasures, Dict[Conditions, float]
    ] = field(default_factory=dict)
    std_error_of_zscores_of_likerts_by_measure_and_by_stype: Dict[
        ScoringMeasures, Dict[Conditions, float]
    ] = field(default_factory=dict)

    avg_DD_lp: float = ERROR_LP
    avg_DD_penlp: float = ERROR_LP
    avg_DD_ll: float = ERROR_LP
    avg_DD_penll: float = ERROR_LP

    accuracy_by_DD_lp: float = 0
    accuracy_by_DD_penlp: float = 0
    accuracy_by_DD_ll: float = 0
    accuracy_by_DD_penll: float = 0

    accuracy_per_score_type_per_sentence_type: Dict[
        ScoringMeasures, Dict[Conditions, float]
    ] = field(default_factory=dict)

    # todo: check that no longer used and remove
    min_token_weight: float = -1 * ERROR_LP
    max_token_weight: float = ERROR_LP

    def __post_init__(self, scoring_measures):

        model_type = self.get_model_type()
        sent_types = self.examples[0].get_sentence_types()

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
            self.avg_zscores_of_likerts_by_measure_and_by_stype[
                scoring_measure
            ] = dict()
            self.std_error_of_zscores_of_likerts_by_measure_and_by_stype[
                scoring_measure
            ] = dict()

            for stype in sent_types:

                self.avg_zscores_by_measure_and_by_stype[scoring_measure][stype] = 0
                self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                    stype
                ] = 0
                self.avg_zscores_of_likerts_by_measure_and_by_stype[scoring_measure][
                    stype
                ] = 0
                self.std_error_of_zscores_of_likerts_by_measure_and_by_stype[
                    scoring_measure
                ][stype] = 0

                # an example scores accurately a sentence type if it gives it an higher acceptability score
                # than the ungrammatical/unacceptable sentence
                # so for the ungrammatical/unacceptable sentence there is no point in storing an accuracy value
                # (it would be like comparing it with iself)
                if (
                    stype is not Conditions.SENTENCE_BAD
                    and stype is not Conditions.LONG_ISLAND
                ):
                    self.accuracy_per_score_type_per_sentence_type[scoring_measure][
                        stype
                    ] = 0

    def get_model_type(self):
        return get_model_type_from_model_name(self.model_descr)

    def get_expected_scoring_measures(self):
        expected_scoring_measures = [ScoringMeasures.LP, ScoringMeasures.PenLP]
        if self.get_model_type() in BERT_LIKE_MODEL_TYPES:
            expected_scoring_measures += [
                ScoringMeasures.LL,
                ScoringMeasures.PLL,
            ]
        return expected_scoring_measures

    def get_expected_sentence_types(self):
        if self.dataset_source in [
            DataSources.SPROUSE,
            DataSources.MADEDDU,
            DataSources.VARIATIONS,
            DataSources.MULTIPLE,
        ]:
            return [
                Conditions.SHORT_NONISLAND,
                Conditions.LONG_NONISLAND,
                Conditions.SHORT_ISLAND,
                Conditions.LONG_ISLAND,
            ]
        elif self.dataset_source == DataSources.BLIMP_EN:
            return [Conditions.SENTENCE_GOOD, Conditions.SENTENCE_BAD]
        else:
            raise ValueError(f"Unexpected dataset_source: {self.dataset_source}")

    def get_item_count_per_phenomenon(self) -> int:
        return len(self.examples)

    def get_sentence_types(self) -> KeysView[Conditions]:
        return self.lp_average_by_sentence_type.keys()

    def get_scoring_measures(self) -> KeysView[ScoringMeasures]:
        return self.accuracy_per_score_type_per_sentence_type.keys()

    def get_acceptable_sentence_types(self) -> KeysView[Conditions]:
        some_scoring_measure = next(iter(self.get_scoring_measures()))
        return self.accuracy_per_score_type_per_sentence_type[
            some_scoring_measure
        ].keys()

    def set_avg_zscores_by_measure_and_by_stype(
        self,
        scoring_measure: ScoringMeasures,
        merged_scores_for_this_measure,
        merged_likert_scores_for_this_measure,
        likert_bins_for_this_measure,
        likert_labels,
    ):

        for stype in self.get_sentence_types():

            # convert all the scores to zscores
            all_scores_this_measure_and_stype: List[float] = []
            for example in self.examples:
                score = example[stype].get_score(scoring_measure)
                all_scores_this_measure_and_stype.append(score)
            all_scores_to_likert_for_this_measure_and_stype = discretize(
                all_scores_this_measure_and_stype,
                groups=likert_bins_for_this_measure,
                labels=likert_labels,
            )
            all_scores_to_likert_for_this_measure_and_stype = np.asarray(
                all_scores_to_likert_for_this_measure_and_stype
            )

            all_zscores_this_measure_and_stype: List[float] = zmap(
                all_scores_this_measure_and_stype, merged_scores_for_this_measure
            )
            all_zscores_from_likert_for_this_measure_and_stype: List[float] = zmap(
                all_scores_to_likert_for_this_measure_and_stype,
                merged_likert_scores_for_this_measure,
            )

            # then calculate the averages
            self.avg_zscores_by_measure_and_by_stype[scoring_measure][
                stype
            ] = np.average(all_zscores_this_measure_and_stype)
            self.avg_zscores_of_likerts_by_measure_and_by_stype[scoring_measure][
                stype
            ] = np.average(all_zscores_from_likert_for_this_measure_and_stype)

            # also calculate the standard errors for the error bars:
            # std_error = np.std(data, ddof=1) / np.sqrt(len(data))
            self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                stype
            ] = get_std_error(all_zscores_this_measure_and_stype)
            self.std_error_of_zscores_of_likerts_by_measure_and_by_stype[
                scoring_measure
            ][stype] = get_std_error(all_zscores_this_measure_and_stype)

    def get_avg_zscores_by_measure_and_by_stype(
        self,
        scoring_measure: ScoringMeasures,
        stype: Conditions,
        likert=False,
    ):
        if not likert:
            return self.avg_zscores_by_measure_and_by_stype[scoring_measure][stype]
        else:
            return self.avg_zscores_of_likerts_by_measure_and_by_stype[scoring_measure][
                stype
            ]

    def get_std_error_of_zscores_by_measure_and_by_stype(
        self,
        scoring_measure: ScoringMeasures,
        stype: Conditions,
        likert=False,
    ):
        if not likert:
            return self.std_error_of_zscores_by_measure_and_by_stype[scoring_measure][
                stype
            ]
        else:
            return self.std_error_of_zscores_of_likerts_by_measure_and_by_stype[
                scoring_measure
            ][stype]

    def get_std_errors_of_zscores_by_measure_and_sentence_structure(
        self,
        scoring_measure: ScoringMeasures,
        sentence_structure: Conditions,
        likert=False,
    ):
        if not likert:
            std_error_of_zscores = self.std_error_of_zscores_by_measure_and_by_stype
        else:
            std_error_of_zscores = (
                self.std_error_of_zscores_of_likerts_by_measure_and_by_stype
            )

        if sentence_structure in [
            Conditions.SHORT_NONISLAND,
            Conditions.LONG_NONISLAND,
        ]:
            return [
                std_error_of_zscores[scoring_measure][Conditions.SHORT_NONISLAND],
                std_error_of_zscores[scoring_measure][Conditions.LONG_NONISLAND],
            ]
        if sentence_structure in [
            Conditions.SHORT_ISLAND,
            Conditions.LONG_ISLAND,
        ]:
            return [
                std_error_of_zscores[scoring_measure][Conditions.SHORT_ISLAND],
                std_error_of_zscores[scoring_measure][Conditions.LONG_ISLAND],
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

    def get_avg_DD_zscores(self, scoring_measure: ScoringMeasures, likert=False):

        if not likert:
            avg_zscores = self.avg_zscores_by_measure_and_by_stype
        else:
            avg_zscores = self.avg_zscores_of_likerts_by_measure_and_by_stype

        return get_dd_score_parametric(
            a_short_nonisland_score=avg_zscores[scoring_measure][
                Conditions.SHORT_NONISLAND
            ],
            b_long_nonisland_score=avg_zscores[scoring_measure][
                Conditions.LONG_NONISLAND
            ],
            c_short_island_score=avg_zscores[scoring_measure][Conditions.SHORT_ISLAND],
            d_long_island_score=avg_zscores[scoring_measure][Conditions.LONG_ISLAND],
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

    def get_examples_sorted_by_DD_score(
        self,
        scoring_measure: ScoringMeasures,
        reverse=False,
    ) -> List[TestItem]:

        return sorted(
            self.examples,
            key=lambda x: x.get_dd_score(scoring_measure),
            reverse=reverse,
        )

    def get_examples_sorted_by_score_diff(
        self,
        score_descr,
        sent_type1: Conditions,
        sent_type2: Conditions,
        reverse=True,
    ) -> List[TestItem]:
        return sorted(
            self.examples,
            key=lambda x: x.get_score_diff(score_descr, sent_type1, sent_type2),
            reverse=reverse,
        )

    def get_all_sentences_sorted_by_score(
        self, score_descr: ScoringMeasures, reverse=True
    ) -> List[TypedSentence]:
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
        self, stype: Conditions, score_descr: ScoringMeasures, reverse=True
    ) -> List[TestItem]:
        return sorted(
            self.examples,
            key=lambda x: x[stype].get_score(score_descr),
            reverse=reverse,
        )

    def save_to_pickle(self, filename):

        self.assert_is_well_formed()
        save_obj_to_pickle(self, filename)

    def assert_is_well_formed(self, expected_experimental_design=None):
        testset = self
        if expected_experimental_design is not None:
            logging.debug(
                f"expected_experimental_design={expected_experimental_design}, testset.experimental_design={testset.experimental_design}"
            )
            assert expected_experimental_design == testset.experimental_design

        factorial = testset.experimental_design == ExperimentalDesigns.FACTORIAL
        assert len(testset.linguistic_phenomenon) > 0
        assert len(testset.model_descr) > 0
        assert type(testset.dataset_source) is not str

        for scoring_measure in testset.get_scoring_measures():
            assert scoring_measure in testset.get_expected_scoring_measures()

        if factorial:
            assert ERROR_LP != testset.avg_DD_lp
            assert ERROR_LP != testset.avg_DD_penlp
            assert ERROR_LP != testset.accuracy_by_DD_lp
            assert ERROR_LP != testset.accuracy_by_DD_penlp

            if testset.get_model_type() in BERT_LIKE_MODEL_TYPES:
                assert ERROR_LP != testset.avg_DD_ll
                assert ERROR_LP != testset.avg_DD_penll
                assert ERROR_LP != testset.accuracy_by_DD_ll
                assert ERROR_LP != testset.accuracy_by_DD_penll

            for stype in testset.get_sentence_types():
                assert 0 != testset.lp_average_by_sentence_type[stype] != ERROR_LP
                assert 0 != testset.penlp_average_by_sentence_type[stype] != ERROR_LP
                if testset.get_model_type() in BERT_LIKE_MODEL_TYPES:
                    assert 0 != testset.ll_average_by_sentence_type[stype] != ERROR_LP
                    assert (
                        0 != testset.penll_average_by_sentence_type[stype] != ERROR_LP
                    )
                for scoring_measure in testset.get_scoring_measures():
                    assert_non_default_value(
                        testset.avg_zscores_by_measure_and_by_stype[scoring_measure][
                            stype
                        ]
                    )

                    # todo: check why this fails a test  # LP, SHORT_ISLAND avg = 0
                    assert_non_default_value(
                        testset.avg_zscores_of_likerts_by_measure_and_by_stype[
                            scoring_measure
                        ][stype],
                        warning=True,
                        details=f"{scoring_measure}, {stype}, avg_zscores_of_likerts_by_measure_and_by_stype",
                    )

                    assert_non_default_value(
                        testset.std_error_of_zscores_by_measure_and_by_stype[
                            scoring_measure
                        ][stype]
                    )
                    assert_non_default_value(
                        testset.std_error_of_zscores_of_likerts_by_measure_and_by_stype[
                            scoring_measure
                        ][stype]
                    )

        for acceptable_stype in testset.get_acceptable_sentence_types():
            for scoring_measure in testset.get_scoring_measures():
                assert_btw_0_1(
                    testset.accuracy_per_score_type_per_sentence_type[scoring_measure][
                        acceptable_stype
                    ]
                )

        assert len(testset.examples) > 0
        for example in testset.examples:
            assert len(example.sentences) > 0

            for typed_sent in example.sentences:

                assert typed_sent.stype in testset.get_expected_sentence_types()
                assert len(typed_sent.sent.txt) > 0
                assert len(typed_sent.sent.tokens) > 0

                for scoring_measure in testset.get_scoring_measures():
                    assert ERROR_LP != typed_sent.sent.get_score(scoring_measure)
                    assert typed_sent.sent.get_score(scoring_measure) is not None
                    # print(f"score: {typed_sent.stype} {scoring_measure.name} = {typed_sent.sent.get_score(scoring_measure)}")

            if factorial:
                assert ERROR_LP != example.DD_with_lp
                assert ERROR_LP != example.DD_with_penlp
                if testset.get_model_type() in BERT_LIKE_MODEL_TYPES:
                    assert ERROR_LP != example.DD_with_ll
                    assert ERROR_LP != example.DD_with_penll
                else:
                    assert_is_default(example.DD_with_ll)
                    assert_is_default(example.DD_with_penll)
            else:
                assert_is_default(example.DD_with_lp)
                assert_is_default(example.DD_with_penlp)
                assert_is_default(example.DD_with_ll)
                assert_is_default(example.DD_with_penll)


def assert_is_default(value):
    assert value is None or value == ERROR_LP


def assert_non_default_value(value: float, warning=False, details=""):
    assert value != ERROR_LP, f"{value}"

    if not warning:
        assert value != 0, f"{value}"
    else:
        if value == 0:
            logging.warning(f"{value}, {details}")


def assert_btw_0_1(value: float):
    assert 0 <= value <= 1, f"{value}"


def get_model_type_from_model_name(model_name: str) -> ModelTypes:
    model_name_to_model_type = {**MODEL_TYPES_AND_NAMES_EN, **MODEL_TYPES_AND_NAMES_IT}
    return model_name_to_model_type[model_name]


def load_testset_from_pickle(
    filename, expected_experimental_design: ExperimentalDesigns
) -> TestSuite:
    testset = load_object_from_pickle(filename)
    testset.assert_is_well_formed(expected_experimental_design)

    return testset


def parse_testset(
    linguistic_phenomenon,
    model_descr: str,
    dataset_source: DataSources,
    examples_list: list,
    experimental_design: ExperimentalDesigns,
    scoring_measures: List[ScoringMeasures],
    max_examples,
    phenomenon_properties: Dict[str, str] = dict(),
) -> TestSuite:
    print(f"len examples: {len(examples_list)}, max: {max_examples}")
    do_lower_case = True if "uncased" in model_descr else False
    if experimental_design == ExperimentalDesigns.FACTORIAL:
        sent_types = SPROUSE_SENTENCE_TYPES
    elif experimental_design == ExperimentalDesigns.MINIMAL_PAIRS:
        sent_types = BLIMP_SENTENCE_TYPES
    elif experimental_design == ExperimentalDesigns.MINIMAL_PAIRS_VARIATIONS:
        sent_types = MINIMAL_PAIRS_VARIATIONS_SENTENCE_TYPES
    else:
        raise ValueError(f"unrecognized experimental_design: {experimental_design}")

    if len(examples_list) > max_examples:
        # print(f"slicing the number of examples to {max_examples}")
        examples_list = examples_list[:max_examples]

    parsed_examples = []
    for example in examples_list:
        parsed_example = parse_example(example, sent_types, do_lower_case=do_lower_case)
        parsed_examples.append(parsed_example)

    return TestSuite(
        linguistic_phenomenon,
        model_descr,
        dataset_source,
        experimental_design,
        parsed_examples,
        scoring_measures,
        phenomenon_properties,
    )


def get_merged_score_across_testsets(
    scoring_measure: ScoringMeasures, testsets: List[TestSuite]
):
    merged_scores = []
    for testset in testsets:
        for example in testset.examples:
            for typed_sentence in example.sentences:
                merged_scores.append(typed_sentence.sent.get_score(scoring_measure))

    return merged_scores


def parse_example(
    example: Dict[Conditions, str], sent_types: list, do_lower_case=False
):
    typed_senteces = []
    # print(f"example: {example}")

    for sent_type in sent_types:
        sentence_txt: str = example[sent_type]
        if do_lower_case:
            sentence_txt = sentence_txt.lower()
        typed_sentece = parse_typed_sentence(sent_type, sentence_txt)
        typed_senteces.append(typed_sentece)

    return TestItem(typed_senteces)


def parse_typed_sentence(stype: Conditions, txt: str) -> TypedSentence:
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
    assert_almost_equal(example_island_effect, example_dd)
    return example_dd


def get_std_error(arr):
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


def load_testsets_from_pickles(
    dataset_source,
    phenomena,
    model_name,
    expected_experimental_design: ExperimentalDesigns,
) -> List[TestSuite]:

    loaded_testsets = []
    for phenomenon in phenomena:
        filename = get_pickle_filename(dataset_source, phenomenon, model_name)
        loaded_testset = load_testset_from_pickle(
            filename, expected_experimental_design
        )
        loaded_testsets.append(loaded_testset)

    return loaded_testsets


def parse_testsuites_dir(
    testset_dir_path: str,
    model_name: str,
    scoring_measures: List[ScoringMeasures],
    dataset_source: DataSources = DataSources.MULTIPLE,
) -> List[TestSuite]:
    # todo: list oll *.jsonl files in dir
    from os import listdir
    from os.path import isfile, join

    jsonl_filenames = [
        f
        for f in listdir(testset_dir_path)
        if (isfile(join(testset_dir_path, f)) and f.endswith(".jsonl"))
    ]

    # open each file, read all lines, get the header json line
    # parse all the other lines according to the header
    parsed_testsets: List[TestSuite] = []
    for testsuite_filename in jsonl_filenames:
        testsuite_filepath = os.path.join(testset_dir_path, testsuite_filename)

        print(f"Parsing testsuite {testsuite_filepath}")
        testsuite_list = load_testset_data(
            testsuite_filepath, examples_format="jsonl_w_header"
        )  # es.: "json_lines"

        testsuite_header = testsuite_list[0]
        assert "is_header" in testsuite_header
        is_header = testsuite_header["is_header"]
        is_header = is_header.lower() in ["true"]

        assert "phenomenon_properties" in testsuite_header
        phenomenon_properties = testsuite_header["phenomenon_properties"]
        assert type(phenomenon_properties) is dict

        phenomenon_name = phenomenon_properties["phenomenon_short_name"]
        items_list = testsuite_list[1:]
        parsed_testset = parse_testset(
            phenomenon_name,
            model_name,
            dataset_source,
            items_list,
            ExperimentalDesigns.FACTORIAL,
            scoring_measures,
            max_examples=999,
            phenomenon_properties=phenomenon_properties,
        )

        parsed_testsets.append(parsed_testset)

    return parsed_testsets


def parse_testsets(
    testset_dir_path: str,
    testset_filenames: List[str],
    dataset_source: DataSources,
    examples_format: str,
    experimental_design: ExperimentalDesigns,
    model_name: str,
    scoring_measures: List[ScoringMeasures],
    max_examples: int,
) -> List[TestSuite]:

    # todo: add scorebase var in testset class

    parsed_testsets = []
    for testset_filename in testset_filenames:
        testset_filepath = os.path.join(testset_dir_path, testset_filename + ".jsonl")
        print(f"Parsing testset {testset_filepath}")
        testset_dict = load_testset_data(
            testset_filepath, examples_format=examples_format
        )  # es.: "json_lines"
        examples_list = testset_dict["sentences"]
        phenomenon_name = get_file_root(testset_filename)

        parsed_testset = parse_testset(
            phenomenon_name,
            model_name,
            dataset_source,
            examples_list,
            experimental_design,
            scoring_measures,
            max_examples=max_examples,
        )

        parsed_testsets.append(parsed_testset)

    return parsed_testsets


def save_scored_testsets(
    scored_testsets: List[TestSuite], model_name: str, dataset_source: str
):
    for scored_testset in scored_testsets:
        scored_testset.model_descr = model_name
        filename = get_pickle_filename(
            dataset_source,
            scored_testset.linguistic_phenomenon,
            model_name,
        )

        scored_testset.save_to_pickle(filename)


def save_scored_testsets_to_single_pickle(
    scored_testsets: Dict[Tuple[str, ModelTypes], List[TestSuite]], filename: str
):
    for testsuite_list in scored_testsets.values():
        for testsuite in testsuite_list:
            testsuite.assert_is_well_formed()
    save_obj_to_pickle(scored_testsets, filename, suffix="pickle", add_timestamp=True)
