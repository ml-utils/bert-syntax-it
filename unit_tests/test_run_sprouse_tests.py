from unittest import TestCase

import pytest
from linguistic_tests.run_sprouse_tests import create_test_jsonl_files_tests


class TestRunSprouseTests(TestCase):
    @pytest.mark.skip("todo")
    def test_create_test_jsonl_files_tests(self):
        create_test_jsonl_files_tests()
        raise NotImplementedError
