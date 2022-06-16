from unittest import TestCase

import pytest
from linguistic_tests.notebook import interactive_mode
from linguistic_tests.notebook import main
from linguistic_tests.run_legacy_tests import arg_parse
from linguistic_tests.run_legacy_tests import basic_sentence_test
from linguistic_tests.run_legacy_tests import custom_eval
from linguistic_tests.run_legacy_tests import eval_gulordava
from linguistic_tests.run_legacy_tests import eval_it
from linguistic_tests.run_legacy_tests import eval_lgd
from linguistic_tests.run_legacy_tests import eval_marvin
from linguistic_tests.run_legacy_tests import get_example_analysis_as_tuple
from linguistic_tests.run_legacy_tests import get_perc
from linguistic_tests.run_legacy_tests import load_it
from linguistic_tests.run_legacy_tests import load_marvin
from linguistic_tests.run_legacy_tests import print_example
from linguistic_tests.run_legacy_tests import print_sentence_pairs_probabilities
from linguistic_tests.run_legacy_tests import print_sentences_sorted_by_score
from linguistic_tests.run_legacy_tests import read_gulordava
from linguistic_tests.run_legacy_tests import rnd
from linguistic_tests.run_legacy_tests import run_eval
from linguistic_tests.run_legacy_tests import run_testset_bert
from linguistic_tests.run_syntactic_tests import get_masked_word_probability
from linguistic_tests.run_syntactic_tests import print_detailed_sentence_info
from linguistic_tests.run_syntactic_tests import run_agreement_tests
from linguistic_tests.run_syntactic_tests import run_tests_blimp
from linguistic_tests.run_syntactic_tests import run_tests_goldberg
from linguistic_tests.run_syntactic_tests import run_tests_lau_et_al


class TestNotebook(TestCase):
    @pytest.mark.skip("todo")
    def test_arg_parse(self):
        arg_parse()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_basic_sentence_test(self):
        basic_sentence_test()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_custom_eval(self):
        custom_eval()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_eval_gulordava(self):
        eval_gulordava()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_eval_it(self):
        eval_it()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_eval_lgd(self):
        eval_lgd()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_eval_marvin(self):
        eval_marvin()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_example_analysis_as_tuple(self):
        get_example_analysis_as_tuple()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_masked_word_probability(self):
        get_masked_word_probability()
        raise NotImplementedError

    def test_get_perc(self):
        self.assertEqual("10.0 %", get_perc(10, 100))
        self.assertEqual("0.1 %", get_perc(1, 1000))

    @pytest.mark.skip("todo")
    def test_interactive_mode(self):
        interactive_mode()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_load_it(self):
        load_it()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_load_marvin(self):
        load_marvin()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_main(self):
        main()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_detailed_sentence_info(self):
        print_detailed_sentence_info()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_example(self):
        print_example()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_sentence_pairs_probabilities(self):
        print_sentence_pairs_probabilities()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_print_sentences_sorted_by_score(self):
        print_sentences_sorted_by_score()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_read_gulordava(self):
        read_gulordava()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_rnd(self):
        rnd()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_agreement_tests(self):
        run_agreement_tests()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_eval(self):
        run_eval()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_blimp(self):
        run_tests_blimp()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_goldberg(self):
        run_tests_goldberg()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_tests_lau_et_al(self):
        run_tests_lau_et_al()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_run_testset_bert(self):
        run_testset_bert()
        raise NotImplementedError
