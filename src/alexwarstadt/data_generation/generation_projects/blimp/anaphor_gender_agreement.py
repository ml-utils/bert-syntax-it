from functools import reduce

from linguistic_tests.utils import data_generator
from linguistic_tests.utils.conjugate import *
from linguistic_tests.utils.constituent_building import *
from linguistic_tests.utils.randomize import choice
from linguistic_tests.utils.vocab_sets import *
from linguistic_tests.utils.vocab_table import get_matched_by
from linguistic_tests.utils.vocab_table import get_matches_of


class AnaphorGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="morphology",
            linguistics="anaphor_agreement",
            uid="anaphor_gender_agreement",
            simple_lm_method=True,
            one_prefix_method=True,
            two_prefix_method=False,
            lexically_identical=False,
        )
        self.all_safe_nouns = np.setdiff1d(
            all_singular_nouns, all_singular_neuter_animate_nouns
        )
        self.all_singular_reflexives = reduce(
            np.union1d,
            (
                get_all("expression", "himself"),
                get_all("expression", "herself"),
                get_all("expression", "itself"),
            ),
        )

    def sample(self):
        # John knows himself
        # N1   V1    refl_match
        # John knows itself
        # N1   V1    refl_mismatch

        V1 = choice(all_refl_preds)
        N1 = N_to_DP_mutate(
            choice(
                get_matches_of(
                    V1, "arg_1", get_matches_of(V1, "arg_2", self.all_safe_nouns)
                )
            )
        )
        refl_match = choice(get_matched_by(N1, "arg_1", all_reflexives))
        refl_mismatch = choice(np.setdiff1d(self.all_singular_reflexives, [refl_match]))

        V1 = conjugate(V1, N1)

        data = {
            "sentence_good": "{} {} {}.".format(N1[0], V1[0], refl_match[0]),
            "sentence_bad": "{} {} {}.".format(N1[0], V1[0], refl_mismatch[0]),
            "one_prefix_prefix": "{} {}".format(N1[0], V1[0]),
            "one_prefix_word_good": refl_match[0],
            "one_prefix_word_bad": refl_mismatch[0],
        }
        return data, data["sentence_good"]


binding_generator = AnaphorGenerator()
binding_generator.generate_paradigm(
    rel_output_path="outputs/blimp/%s.jsonl" % binding_generator.uid
)
