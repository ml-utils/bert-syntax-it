import inflect
from linguistic_tests.utils import data_generator
from linguistic_tests.utils.conjugate import *
from linguistic_tests.utils.constituent_building import *
from linguistic_tests.utils.randomize import choice
from linguistic_tests.utils.vocab_sets import *
from linguistic_tests.utils.vocab_table import get_matched_by
from linguistic_tests.utils.vocab_table import get_matches_of


class SuperlativeGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="semantics",
            linguistics="quantifiers",
            uid="superlative_quantifiers_1",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=False,
            lexically_identical=False,
        )
        self.quantifiers = [("more than", "at least"), ("fewer than", "at most")]
        self.safe_nouns = np.setdiff1d(all_plural_nouns, all_proper_names)

    def sample(self):
        # No professor graded more than three papers
        # NO N1        V      Q1        Num   N2
        # No professor graded at least four papers
        # NO N1        V      Q2       Num  N2

        V = choice(all_non_plural_transitive_verbs)
        N1 = choice(get_matches_of(V, "arg_1", all_singular_count_nouns))
        V = conjugate(V, N1, False)
        N2 = choice(get_matches_of(V, "arg_2", self.safe_nouns))
        quantifiers = random.choice(self.quantifiers)
        Q1 = quantifiers[0]
        Q2 = quantifiers[1]
        number_inflector = inflect.engine()
        Num = number_inflector.number_to_words(random.randint(2, 10))
        data = {
            "sentence_good": "No {} {} {} {} {}.".format(N1[0], V[0], Q1, Num, N2[0]),
            "sentence_bad": "No {} {} {} {} {}.".format(N1[0], V[0], Q2, Num, N2[0]),
        }
        return data, data["sentence_good"]


generator = SuperlativeGenerator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
