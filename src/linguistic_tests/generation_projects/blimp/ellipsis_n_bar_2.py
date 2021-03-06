from linguistic_tests.utils import data_generator
from linguistic_tests.utils.conjugate import *
from linguistic_tests.utils.constituent_building import *
from linguistic_tests.utils.randomize import choice
from linguistic_tests.utils.vocab_sets import *
from linguistic_tests.utils.vocab_table import get_matched_by
from linguistic_tests.utils.vocab_table import get_matches_of


class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="syntax",
            linguistics="ellipsis",
            uid="ellipsis_n_bar_2",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=False,
            lexically_identical=False,
        )
        self.safe_verbs = all_transitive_verbs
        self.plural_dets = [
            ("two", "three"),
            ("three", "two"),
            ("four", "five"),
            ("five", "four"),
            ("several", "a few"),
            ("a few", "several"),
            ("many", "few"),
            ("few", "many"),
            ("a few", "a lot"),
            ("a lot of", "a few"),
            ("no", "some"),
            ("three", "more"),
            ("three", "fewer"),
            ("three", "at least as many"),
            ("three", "almost as many"),
        ]

        self.singular_dets = [
            ("one", "another"),
            ("each", "one"),
            ("this", "another"),
            ("that", "another"),
            ("this", "one"),
            ("that", "one"),
        ]

        # self.singular_dets = [("one", "two"),
        #                         ("one", "five"),
        #                         ("one", "four"),
        #                         ("one", "a few"),
        #                         ("one", "several"),
        #                         ("one", "few"),
        #                         ("one", "many"),
        #                         ("one", "a lot"),
        #                         ("one", "some"),
        #                         ("one", "more"),
        #                         ("one", "at least as many")
        #                     ]

        self.safe_objs_pl = np.setdiff1d(all_nominals, all_proper_names)
        self.safe_objs_sg = np.setdiff1d(all_nominals, all_proper_names)

    def sample(self):
        # John  has  had two cups and Jane  has  had three green cups.
        # Subj1 Aux1 V   D1  Obj  AND Subj2 Aux2 V   D2    Adj2  Obj
        # John  has  had two red  cups and Jane  has  had three green.
        # Subj1 Aux1 V   D1  Adj1 Obj  AND Subj2 Aux2 V   D2    Adj2

        V = choice(self.safe_verbs)
        Subj1 = choice(get_matches_of(V, "arg_1", all_nominals))
        Subj2 = choice(get_matches_of(V, "arg_1", all_nominals), avoid=Subj1)
        Subj1 = N_to_DP_mutate(Subj1)
        Subj2 = N_to_DP_mutate(Subj2)
        Aux1 = return_aux(V, subj=Subj1)
        if is_match_disj(Subj2, Aux1["arg_1"]):
            Aux2 = Aux1
        else:
            Aux2 = return_aux(V, subj=Subj2)
        Obj = choice(get_matches_of(V, "arg_2", get_all("mass", "0", self.safe_objs)))
        Adj1 = choice(get_matched_by(Obj, "arg_1", all_adjectives))
        Adj2 = choice(get_matched_by(Obj, "arg_1", all_adjectives), avoid=Adj1)
        if Obj["pl"] == "0":
            (D1, D2) = random.choice(self.singular_dets)
        else:
            (D1, D2) = random.choice(self.plural_dets)

        data = {
            "sentence_good": "%s %s %s %s %s and %s %s %s %s %s %s."
            % (
                Subj1[0],
                Aux1[0],
                V[0],
                D1,
                Obj[0],
                Subj2[0],
                Aux2[0],
                V[0],
                D2,
                Adj2[0],
                Obj[0],
            ),
            "sentence_bad": "%s %s %s %s %s %s and %s %s %s %s %s."
            % (
                Subj1[0],
                Aux1[0],
                V[0],
                D1,
                Adj1[0],
                Obj[0],
                Subj2[0],
                Aux2[0],
                V[0],
                D2,
                Adj2[0],
            ),
        }
        return data, data["sentence_good"]


generator = AgreementGenerator()
generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
