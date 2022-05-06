import vocab_it


def t_determiner_noun_agreement_1():
    from generation_projects.blimp import determiner_noun_agreement_1
    generator = determiner_noun_agreement_1.DetNGenerator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


def main():
    for noun in vocab_it.nouns_professions:
        print(noun + ' ')


if __name__ == "__main__":
    # main()
    t_determiner_noun_agreement_1()
