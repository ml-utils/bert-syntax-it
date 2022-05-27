from vocabulary import vocab_it


def t_determiner_noun_agreement_1():
    from generation_projects.blimp import determiner_noun_agreement_1

    generator = determiner_noun_agreement_1.DetNGenerator()
    generator.generate_paradigm(
        rel_output_path="outputs/blimp/%s.jsonl" % generator.uid
    )


def print_profession_nouns():
    for noun in vocab_it.nouns_professions:
        print(noun + " ")


def main():
    # print_profession_nouns()
    t_determiner_noun_agreement_1()


if __name__ == "__main__":
    main()
