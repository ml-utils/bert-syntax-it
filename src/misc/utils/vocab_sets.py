# from functools import reduce
import numpy as np

from .vocab_table import get_all
from .vocab_table import get_all_conjunctive

# from utils.randomize import *


# NOUNS
all_nouns = get_all_conjunctive([("category", "N"), ("frequent", "1")])
print(f"all_nouns len: {len(all_nouns)}")

all_singular_nouns = None  # get_all("sg", "1", all_nouns)
all_singular_count_nouns = None  # get_all("mass", "0", all_singular_nouns)
all_animate_nouns = None  # get_all("animate", "1", all_nouns)
all_inanimate_nouns = None  # get_all("animate", "0", all_nouns)
all_documents = None  # get_all_conjunctive([("category", "N"), ("document", "1")])
all_gendered_nouns = None  # np.union1d(get_all("gender", "m"), get_all("gender", "f"))
all_singular_neuter_animate_nouns = None  # get_all_conjunctive([("category", "N"), ("sg", "1"), ("animate", "1"), ("gender", "n")])
all_plural_nouns = (
    None  # get_all_conjunctive([("category", "N"), ("frequent", "1"), ("pl", "1")])
)
all_plural_animate_nouns = None  # np.intersect1d(all_animate_nouns, all_plural_nouns)


all_common_nouns = get_all_conjunctive([("category", "N"), ("properNoun", "0")])
print(f"all_common_nouns len: {len(all_common_nouns)}")
all_relational_nouns = None  # get_all("category", "N/NP")
all_nominals = None  # get_all_conjunctive([("noun", "1"), ("frequent", "1")])
all_relational_poss_nouns = None  # get_all("category", "N\\NP[poss]")
all_proper_names = None  # get_all("properNoun", "1")


# VERBS
all_verbs = get_all("verb", "1")
print(f"all_verbs len: {len(all_verbs)}")
all_transitive_verbs = get_all("category", "(S\\NP)/NP")
print(f"all_transitive_verbs len: {len(all_transitive_verbs)}")

all_intransitive_verbs = None  # get_all("category", "S\\NP")
all_non_recursive_verbs = (
    None  # np.union1d(all_transitive_verbs, all_intransitive_verbs)
)
all_finite_verbs = None  # get_all("finite", "1", all_verbs)
all_non_finite_verbs = None  # get_all("finite", "0", all_verbs)
all_ing_verbs = None  # get_all("ing", "1", all_verbs)
all_en_verbs = None  # get_all("en", "1", all_verbs)
all_bare_verbs = None  # get_all("bare", "1", all_verbs)
all_anim_anim_verbs = None  # get_matched_by(choice(all_animate_nouns), "arg_1", get_matched_by(choice(all_animate_nouns), "arg_2", all_transitive_verbs))
all_doc_doc_verbs = None  # get_matched_by(choice(all_documents), "arg_1", get_matched_by(choice(all_documents), "arg_2", all_transitive_verbs))
all_refl_nonverbal_predicates = None  # np.extract([x["arg_1"] == x["arg_2"] for x in get_all("category_2", "Pred")], get_all("category_2", "Pred"))
all_refl_preds = None  # reduce(np.union1d, (all_anim_anim_verbs, all_doc_doc_verbs))
all_non_plural_transitive_verbs = None  # np.extract(["sg=0" not in x["arg_1"] and "pl=1" not in x["arg_1"] for x in all_transitive_verbs], all_transitive_verbs)
all_strictly_plural_verbs = (
    None  # get_all_conjunctive([("pres", "1"), ("3sg", "0")], all_verbs)
)
all_strictly_singular_verbs = (
    None  # get_all_conjunctive([("pres", "1"), ("3sg", "1")], all_verbs)
)
all_strictly_plural_transitive_verbs = (
    None  # np.intersect1d(all_strictly_plural_verbs, all_transitive_verbs)
)
all_strictly_singular_transitive_verbs = (
    None  # np.intersect1d(all_strictly_singular_verbs, all_transitive_verbs)
)
all_possibly_plural_verbs = None  # np.setdiff1d(all_verbs, all_strictly_singular_verbs)
all_possibly_singular_verbs = None  # np.setdiff1d(all_verbs, all_strictly_plural_verbs)
all_non_finite_transitive_verbs = (
    None  # np.intersect1d(all_non_finite_verbs, all_transitive_verbs)
)
all_non_finite_intransitive_verbs = (
    None  # get_all("finite", "0", all_intransitive_verbs)
)

all_modals_auxs = get_all("category", "(S\\NP)/(S[bare]\\NP)")
print(f"all_modals_auxs len: {len(all_modals_auxs)}")

all_modals = None  # get_all("category_2", "modal")
all_auxs = None  # get_all("category_2", "aux")
all_negated_modals_auxs = None  # get_all("negated", "1", all_modals_auxs)
all_non_negated_modals_auxs = None  # get_all("negated", "0", all_modals_auxs)
all_negated_modals = None  # get_all("negated", "1", all_modals)
all_non_negated_modals = None  # get_all("negated", "0", all_modals)
all_negated_auxs = None  # get_all("negated", "1", all_auxs)
all_non_negated_auxs = None  # get_all("negated", "0", all_auxs)

all_copulas = None  # get_all("category_2", "copula")
all_finite_copulas = None  # np.setdiff1d(all_copulas, get_all("bare", "1"))
all_rogatives = None  # get_all("category", "(S\\NP)/Q")

all_agreeing_aux = None  # np.setdiff1d(all_auxs, get_all("arg_1", "sg=1;sg=0"))
all_non_negative_agreeing_aux = None  # get_all("negated", "0", all_agreeing_aux)
all_negative_agreeing_aux = None  # get_all("negated", "1", all_agreeing_aux)
all_auxiliaries_no_null = None  # np.setdiff1d(all_auxs, get_all("expression", ""))
all_non_negative_copulas = None  # get_all("negated", "0", all_finite_copulas)
all_negative_copulas = None  # get_all("negated", "1", all_finite_copulas)


# OTHER

all_determiners = get_all("category", "(S/(S\\NP))/N")
print(f"all_determiners len: {len(all_determiners)}")

all_frequent_determiners = None  # get_all("frequent", "1", all_determiners)
all_very_common_dets = None  # np.append(get_all("expression", "the"), np.append(get_all("expression", "a"), get_all("expression", "an")))
all_relativizers = None  # get_all("category_2", "rel")
all_reflexives = None  # get_all("category_2", "refl")
all_ACCpronouns = None  # get_all("category_2", "proACC")
all_NOMpronouns = None  # get_all("category_2", "proNOM")
all_embedding_verbs = None  # get_all("category_2", "V_embedding")
all_wh_words = None  # get_all("category", "NP_wh")


all_demonstratives = np.append(
    get_all("expression", "this"),
    np.append(
        get_all_conjunctive([("category_2", "D"), ("expression", "that")]),
        np.append(
            get_all("expression", "these"),
            get_all("expression", "those"),
        ),
    ),
)
print(f"all_demonstratives len: {len(all_demonstratives)}")

all_adjectives = (
    None  # np.append(get_all("category_2", "adjective"), get_all("category_2", "Adj"))
)
all_frequent = get_all("frequent", "1")
print(f"all_frequent len: {len(all_frequent)}")
