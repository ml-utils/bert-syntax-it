from unittest import TestCase

import pytest
from linguistic_tests.file_utils import __get_sentence_from_row
from linguistic_tests.file_utils import __write_sentence_item
from linguistic_tests.file_utils import convert_sprouse_csv_to_jsonl


class TestFileUtils(TestCase):
    @pytest.mark.skip("todo")
    def test_create_test_jsonl_files_tests(self):
        convert_sprouse_csv_to_jsonl()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_sentence_from_row(self):
        __get_sentence_from_row()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_write_sentence_item(self):
        __write_sentence_item()
        raise NotImplementedError
