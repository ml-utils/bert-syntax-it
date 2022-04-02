from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.vocab_sets import *
import numpy
from os.path import exists

class DetNGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="determiner_noun_agreement",
                         uid="determiner_noun_agreement_1",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=True)
        self.init_field('all_null_plural_nouns', lambda: get_all("sgequalspl", "1"))
        self.init_field('all_missingPluralSing_nouns', lambda: get_all_conjunctive([("pluralform", ""), ("singularform", "")]))
        self.init_field('all_irregular_nouns', lambda: get_all("irrpl", "1"))
        self.init_field('all_unusable_nouns', lambda:np.union1d(self.all_null_plural_nouns, np.union1d(self.all_missingPluralSing_nouns, self.all_irregular_nouns)))
        self.init_field('all_pluralizable_nouns', lambda:lambda:np.setdiff1d(all_common_nouns, self.all_unusable_nouns))


    def init_field(self, field_name, get_fun):
        field_cache_file = field_name + '.npy'

        # todo, check if file already exists, or object is not null
        #x = getattr(self, source)
        #
        if exists(field_cache_file):
            setattr(self, field_name, numpy.load(field_cache_file, allow_pickle=True))
            # self.all_null_plural_nouns = numpy.load(field_cache_file, allow_pickle=True)
        else:
            setattr(self, field_name, get_fun())
            numpy.save(field_cache_file, getattr(self, field_name))

        print(f'{field_name} len: {len(getattr(self, field_name))}')

    def sample(self):
        # John cleaned this table.
        # N1   V1      Dem  N2_match

        # John cleaned this tables.
        # N1   V1      Dem  N2_mismatch

        V1 = choice(all_transitive_verbs)
        N1 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_1", all_nouns)))
        N2_match = choice(get_matches_of(V1, "arg_2", self.all_pluralizable_nouns))
        Dem = choice(get_matched_by(N2_match, "arg_1", all_demonstratives))
        if N2_match['pl'] == "1":
            N2_mismatch = N2_match['singularform']
        else:
            N2_mismatch = N2_match['pluralform']
        V1 = conjugate(V1, N1)

        data = {
            "sentence_good": "%s %s %s %s." % (N1[0], V1[0], Dem[0], N2_match[0]),
            "sentence_bad": "%s %s %s %s." % (N1[0], V1[0], Dem[0], N2_mismatch),
            "one_prefix_prefix": "%s %s %s" % (N1[0], V1[0], Dem[0]),
            "one_prefix_word_good": N2_match[0],
            "one_prefix_word_bad": N2_mismatch
        }
        return data, data["sentence_good"]

if __name__ == "__main__":
    generator = DetNGenerator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)
