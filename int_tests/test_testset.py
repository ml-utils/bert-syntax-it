import os
from unittest import TestCase

from linguistic_tests.lm_utils import load_testset_data
from linguistic_tests.testset import parse_testset

from int_tests.int_tests_utils import get_test_data_dir


class TestTestset(TestCase):
    def test_parse_testset_sprouse(self):

        p = get_test_data_dir() / "sprouse"
        testset_dir_path = str(p)
        filename = "mini_wh_adjunct_island" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="sprouse")
        examples_list = testset["sentences"]
        parsed_testset = parse_testset(examples_list, "sprouse")

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 4
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    def test_parse_testset_blimp(self):
        p = get_test_data_dir() / "blimp"
        testset_dir_path = str(p)
        filename = "mini_wh_island" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="json_lines")
        examples_list = testset["sentences"]
        parsed_testset = parse_testset(examples_list, "blimp")

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 2
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

    def test_parse_testset_custom_it(self):
        p = get_test_data_dir() / "custom_it"
        testset_dir_path = str(p)
        filename = "mini_wh_adjunct_islands" + ".jsonl"
        filepath = os.path.abspath(os.path.join(testset_dir_path, filename))
        testset = load_testset_data(filepath, examples_format="blimp")
        examples_list = testset["sentences"]
        parsed_testset = parse_testset(examples_list, "sprouse")

        assert len(parsed_testset.examples) == 2
        for example in parsed_testset.examples:
            assert len(example.sentences) == 4
            for typed_sentence in example.sentences:
                assert len(typed_sentence.sent.txt) > 0

        # todo: also test for the custom it file
