from unittest import TestCase

import pytest
from pytest_socket import SocketBlockedError
from transformers import BertForMaskedLM


class TestNewFeatures(TestCase):
    @pytest.mark.skip("todo")
    def test_get_dd_score(self):
        raise NotImplementedError

    def test_no_remote_Calls(self):
        with pytest.raises(SocketBlockedError):
            _ = BertForMaskedLM.from_pretrained("bert-base-uncased")
