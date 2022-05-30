from unittest import TestCase

import pytest
from linguistic_tests import gpt_tests
from linguistic_tests.gpt_tests import get_gpt_sentence_score
from linguistic_tests.gpt_tests import get_model
from linguistic_tests.gpt_tests import get_model2
from linguistic_tests.gpt_tests import get_model3
from linguistic_tests.gpt_tests import get_model4
from linguistic_tests.gpt_tests import main


class TestGptTests(TestCase):
    @pytest.mark.skip("todo")
    def test__check_unk_and_num_tokens(self):
        gpt_tests.__get_gpt_sentence_score()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_gpt_sentence_score(self):
        get_gpt_sentence_score()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_model(self):
        get_model()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_model2(self):
        get_model2()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_model3(self):
        get_model3()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_get_model4(self):
        get_model4()
        raise NotImplementedError

    @pytest.mark.skip("todo")
    def test_main(self):
        main()
        raise NotImplementedError
