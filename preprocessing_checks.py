from vocabulary import vocab_it


def check_oov_words():
    from generation_projects.blimp import determiner_noun_agreement_1

    generator = determiner_noun_agreement_1.DetNGenerator()
    generator.generate_paradigm(
        rel_output_path="outputs/blimp/%s.jsonl" % generator.uid
    )


# def test_sentences():
#     testset_data = load_testset_data("./outputs/syntactic_tests_it/wh_island.jsonl")
#
#     for example_data in testset_data["sentences"]:
#         sentence_good_no_extraction = example_data["sentence_good_no_extraction"]
#         sentence_bad_extraction = example_data["sentence_bad_extraction"]
#         sentence_good_extraction_resumption = example_data[
#             "sentence_good_extraction_resumption"
#         ]
#         sentence_good_extraction_as_subject = example_data[
#             "sentence_good_extraction_as_subject"
#         ]


def test_sentence(sentence: str):
    # todo:
    # checks grammatical mistakes / words mispellings?
    # check frequency of words, detect too rare words
    # check that lenght of ungrammatical sentences is equal or less than grammatical ones
    # check that frequency of words in ungrammatical sentences  is greater or equal than those in grammatical ones
    # ..
    # check when ungrammatical sentence is estimated as more likely than grammatical one, es:
    # sentence_bad_extraction: Che cosa Marco si chiede se Luca ha riparato?
    # sentence_good_extraction_as_subject: Di che cosa Marco si chiede se Ã¨ stata riparata da Luca?
    # ..
    # check how words are converted to ids, handling of oov words
    # detailed summary of a sentence: lenght, prob of indivisual masking, topk words for each masking
    # ..
    # check for UNK, and if they are words that should be recognized instead
    # (try also to mask the UNK, to see what it's predicted
    # ..
    # check words that are split. Are those just oov words? ie "riparata" should be a common word (in vocabulary),
    # but it's split.
    # ..
    # as an approximation, check that in the bad sentences there are less or equal number of split words
    # (because it seems we don't have access to word frequencies)
    return


def print_profession_nouns():
    for noun in vocab_it.nouns_professions:
        print(noun + " ")


def main():
    # t_determiner_noun_agreement_1()
    print("main")


if __name__ == "__main__":
    main()
