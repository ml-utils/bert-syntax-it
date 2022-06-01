from linguistic_tests.utils import data_generator
from linguistic_tests.utils.conjugate import *
from linguistic_tests.utils.constituent_building import *
from linguistic_tests.utils.randomize import choice
from linguistic_tests.utils.vocab_sets import *
from linguistic_tests.utils.vocab_table import get_matches_of


class LeftBranchGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="syntax",
            linguistics="island_effects",
            uid="left_branch_island_simple_question",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=False,
            lexically_identical=True,
        )
        self.all_D_wh = get_all("category_2", "D_wh")
        self.which_what = np.append(
            get_all_conjunctive([("expression", "which")], self.all_D_wh),
            get_all_conjunctive([("expression", "what")], self.all_D_wh),
        )

    def sample(self):
        # Whose dog are  you petting?
        # wh    N2  V_do N1  V1
        # Whose are  you petting dog?
        # wh    V_do N1  V1      N2

        V1 = choice(all_non_finite_transitive_verbs)
        N1 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_1", all_nouns)))
        V_do = return_aux(V1, N1, allow_negated=False)
        N2 = choice(get_matches_of(V1, "arg_2", all_common_nouns))
        if N2["animate"] == "1":
            wh = choice(self.which_what)
        else:
            wh = choice(self.all_D_wh)

        data = {
            "sentence_good": "{} {} {} {} {}?".format(
                wh[0], N2[0], V_do[0], N1[0], V1[0]
            ),
            "sentence_bad": "{} {} {} {} {}?".format(
                wh[0], V_do[0], N1[0], V1[0], N2[0]
            ),
        }
        return data, data["sentence_good"]


generator = LeftBranchGenerator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
