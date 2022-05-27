from src.linguistic_tests.utils import data_generator
from src.linguistic_tests.utils.conjugate import *
from src.linguistic_tests.utils.constituent_building import *
from src.linguistic_tests.utils.randomize import choice
from src.linguistic_tests.utils.vocab_sets import *
from src.linguistic_tests.utils.vocab_table import get_matched_by
from src.linguistic_tests.utils.vocab_table import get_matches_of


class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="morphology",
            linguistics="irregular_forms",
            uid="irregular_past_participle_adjectives",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=True,
            lexically_identical=False,
        )
        self.all_trans_en_verbs = get_all("special_en_form", "1", all_transitive_verbs)

    def sample(self):
        # The eaten pie was delicious
        # THE V_en  N1  cop adj
        # The ate    pie was delicious
        # THE V_past N1 cop adj

        V_base = choice(self.all_trans_en_verbs)
        while " " in V_base[0]:
            V_base = choice(self.all_trans_en_verbs)
        Verbs = get_all("root", V_base["root"])
        V_past = get_all("past", "1", Verbs)
        V_en = get_all("en", "1", Verbs)
        N1 = choice(get_matches_of(V_base, "arg_2", all_common_nouns))
        cop = return_copula(N1)
        adj = choice(get_matched_by(N1, "arg_1", all_adjectives))

        data = {
            "sentence_good": "The {} {} {} {}.".format(
                V_en[0][0], N1[0], cop[0], adj[0]
            ),
            "sentence_bad": "The {} {} {} {}.".format(
                V_past[0][0], N1[0], cop[0], adj[0]
            ),
            "two_prefix_prefix_good": "The %s" % (V_en[0][0]),
            "two_prefix_prefix_bad": "The %s" % (V_past[0][0]),
            "two_prefix_word": N1[0],
        }
        return data, data["sentence_good"]


generator = AgreementGenerator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
