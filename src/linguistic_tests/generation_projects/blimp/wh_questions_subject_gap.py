from src.linguistic_tests.utils import data_generator
from src.linguistic_tests.utils.conjugate import *
from src.linguistic_tests.utils.constituent_building import *
from src.linguistic_tests.utils.randomize import choice
from src.linguistic_tests.utils.vocab_sets import *
from src.linguistic_tests.utils.vocab_table import get_matched_by
from src.linguistic_tests.utils.vocab_table import get_matches_of


class FillerGapGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="syntax",
            linguistics="filler_gap_dependency",
            uid="wh_questions_subject_gap",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=False,
            lexically_identical=False,
        )
        self.wh_np_verbs = get_all("wh_np_verb", "1")

    def sample(self):
        # John noticed the man that saw everyone.
        # N1   V1          N2  that V2  N3
        # John noticed who the man saw everyone.
        # N1   V1      wh      N2  V2  N3

        V1 = choice(self.wh_np_verbs)
        N1 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_1", all_nouns)))
        N2 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_2", all_common_nouns)))
        V2 = choice(get_matched_by(N2, "arg_1", all_transitive_verbs))
        N3 = N_to_DP_mutate(choice(get_matches_of(V2, "arg_2", all_nouns)))
        V1 = conjugate(V1, N1)
        V2 = conjugate(V2, N2)
        wh = choice(get_matched_by(N3, "arg_1", all_wh_words))

        data = {
            "sentence_good": "%s %s %s that %s %s."
            % (N1[0], V1[0], N2[0], V2[0], N3[0]),
            "sentence_bad": "%s %s %s %s %s %s."
            % (N1[0], V1[0], wh[0], N2[0], V2[0], N3[0]),
        }
        return data, data["sentence_good"]


generator = FillerGapGenerator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)