from src.linguistic_tests.utils import data_generator
from src.linguistic_tests.utils.conjugate import *
from src.linguistic_tests.utils.constituent_building import *
from src.linguistic_tests.utils.randomize import choice
from src.linguistic_tests.utils.vocab_table import get_matched_by
from src.linguistic_tests.utils.vocab_table import get_matches_of


class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="syntax",
            linguistics="argument_structure",
            uid="transitive",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=True,
            lexically_identical=False,
        )

        self.strict_intransitive = get_all("strict_intrans", "1")

    def sample(self):
        # The bear has attacked the girl.
        # Subj     Aux V_trans  obj
        # The bear has smiled    the girl.
        # Subj     Aux V_intrans obj

        V_trans = choice(all_transitive_verbs)
        Subj = N_to_DP_mutate(choice(get_matches_of(V_trans, "arg_1", all_nominals)))
        Aux = return_aux(V_trans, Subj)
        Obj = N_to_DP_mutate(choice(get_matches_of(V_trans, "arg_2", all_nominals)))
        V_intrans = choice(
            get_matched_by(
                Subj, "arg_1", get_matches_of(Aux, "arg_2", self.strict_intransitive)
            )
        )

        data = {
            "sentence_good": "{} {} {} {}.".format(Subj[0], Aux[0], V_trans[0], Obj[0]),
            "sentence_bad": "{} {} {} {}.".format(
                Subj[0], Aux[0], V_intrans[0], Obj[0]
            ),
            "two_prefix_prefix_good": "{} {} {}".format(Subj[0], Aux[0], V_trans[0]),
            "two_prefix_prefix_bad": "{} {} {}".format(Subj[0], Aux[0], V_intrans[0]),
            "two_prefix_word": Obj[0].strip().split(" ")[0],
        }
        return data, data["sentence_good"]


generator = Generator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
