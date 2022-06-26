def check_oov_words():
    from generation_projects.blimp import determiner_noun_agreement_1

    generator = determiner_noun_agreement_1.DetNGenerator()
    generator.generate_paradigm(
        rel_output_path="outputs/blimp/%s.jsonl" % generator.uid
    )


def t_determiner_noun_agreement_1():
    from generation_projects.blimp import determiner_noun_agreement_1

    generator = determiner_noun_agreement_1.DetNGenerator()
    generator.generate_paradigm(
        rel_output_path="outputs/blimp/%s.jsonl" % generator.uid
    )


def print_profession_nouns():
    for noun in vocab_it.nouns_professions:
        print(noun + " ")
