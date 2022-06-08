from unittest import TestCase

import pytest
from linguistic_tests.file_utils import create_test_jsonl_files_tests
from linguistic_tests.file_utils import get_sentence_from_row
from linguistic_tests.file_utils import read_sentences_item
from linguistic_tests.file_utils import write_sentence_item
from linguistic_tests.file_utils import write_sentence_pair


class TestFileUtils(TestCase):
    @pytest.mark.skip("todo")
    def test_create_test_jsonl_files_tests(self):
        create_test_jsonl_files_tests()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_from_row(self):
        get_sentence_from_row()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_read_sentences_item(self):
        read_sentences_item()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_write_sentence_item(self):
        write_sentence_item()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_write_sentence_pair(self):
        write_sentence_pair()
        raise NotImplementedError
