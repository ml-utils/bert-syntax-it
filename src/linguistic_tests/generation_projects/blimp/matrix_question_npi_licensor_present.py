from linguistic_tests.utils import data_generator
from linguistic_tests.utils.conjugate import *
from linguistic_tests.utils.constituent_building import *
from linguistic_tests.utils.randomize import choice
from linguistic_tests.utils.vocab_table import get_matched_by
from linguistic_tests.utils.vocab_table import get_matches_of


class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="semantics",
            linguistics="npi_licensing",
            uid="matrix_question_npi_licensor_present",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=True,
            lexically_identical=True,
        )
        self.safe_verbs = np.setdiff1d(all_non_finite_verbs, all_ing_verbs)

    def sample(self):
        # Have the teachers ever watched the waitresses?
        # Aux  Subj         EVER VP
        # The teachers have ever watched the waitresses.
        # Subj         Aux  EVER VP

        V = choice(self.safe_verbs)
        args = verb_args_from_verb(V, allow_negated=False)
        VP = V_to_VP_mutate(V, aux=False, args=args)

        data = {
            "sentence_good": "%s %s ever %s?"
            % (args["aux"][0], args["subj"][0], VP[0]),
            "sentence_bad": "{} {} ever {}.".format(
                args["subj"][0], args["aux"][0], VP[0]
            ),
            "two_prefix_prefix_good": "{} {}".format(args["aux"][0], args["subj"][0]),
            "two_prefix_prefix_bad": "{} {}".format(args["subj"][0], args["aux"][0]),
            "two_prefix_word": "ever",
        }
        return data, data["sentence_good"]


generator = Generator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
