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
            uid="passive_2",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=False,
            lexically_identical=False,
        )

        self.en_verbs = get_all("en", "1")
        self.intransitive = get_all("passive", "0", self.en_verbs)
        self.transitive = get_all("passive", "1", self.en_verbs)

    def sample(self):
        # The girl was attacked.
        # NP1      be  V_trans
        # The girl was smiled.
        # NP1      be  V_intrans

        V_intrans = choice(self.intransitive)
        NP1 = N_to_DP_mutate(choice(get_matches_of(V_intrans, "arg_1", all_nominals)))
        V_trans = choice(get_matched_by(NP1, "arg_2", self.transitive))
        be = return_copula(NP1)

        data = {
            "sentence_good": "{} {} {}.".format(NP1[0], be[0], V_trans[0]),
            "sentence_bad": "{} {} {}.".format(NP1[0], be[0], V_intrans[0]),
        }
        return data, data["sentence_good"]


generator = Generator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
