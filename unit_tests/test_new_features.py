from unittest import TestCase

import pytest
from pytest_socket import SocketBlockedError
from transformers import BertForMaskedLM

from int_tests.integration_tests_utils import is_internet_on


class TestNewFeatures(TestCase):
    @pytest.mark.skip("todo")
    def test_get_dd_score(self):
        raise NotImplementedError

    @pytest.mark.skipif(not is_internet_on(), reason="No internet connection available")
    def test_no_remote_Calls(self):
        not_cached_model = "hfl/chinese-macbert-base"  # "bert-base-uncased"
        with pytest.raises(SocketBlockedError):
            _ = BertForMaskedLM.from_pretrained(not_cached_model)
