from linguistic_tests.utils import data_generator
from linguistic_tests.utils.conjugate import *
from linguistic_tests.utils.constituent_building import *
from linguistic_tests.utils.randomize import choice
from linguistic_tests.utils.vocab_sets import *
from linguistic_tests.utils.vocab_table import get_matched_by
from linguistic_tests.utils.vocab_table import get_matches_of


class FillerGapGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="syntax",
            linguistics="filler_gap_dependency",
            uid="wh_vs_that_with_gap",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=False,
            lexically_identical=False,
        )
        self.embedding_verbs = get_all("responsive", "1")
        self.robustly_transitive_verbs = get_all("strict_trans", "1")

    def sample(self):
        # I  know what the lion devoured.
        # N1 V1   wh       N2   V2
        # I  know that the lion   devoured.
        # N1 V1   THAT     N2   V2

        V1 = choice(self.embedding_verbs)
        N1 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_1", all_nouns)))
        V2 = choice(self.robustly_transitive_verbs)
        N2 = N_to_DP_mutate(choice(get_matches_of(V2, "arg_1", all_common_nouns)))
        N3 = N_to_DP_mutate(choice(get_matches_of(V2, "arg_2", all_nouns)))
        V1 = conjugate(V1, N1)
        V2 = conjugate(V2, N2)
        wh = choice(get_matched_by(N3, "arg_1", all_wh_words))

        data = {
            "sentence_good": "{} {} {} {} {}.".format(
                N1[0], V1[0], wh[0], N2[0], V2[0]
            ),
            "sentence_bad": "{} {} that {} {}.".format(N1[0], V1[0], N2[0], V2[0]),
        }
        return data, data["sentence_good"]


generator = FillerGapGenerator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
