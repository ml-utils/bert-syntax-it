from unittest import TestCase

import pytest

from src.linguistic_tests.compute_model_score import get_sentence_acceptability_score
from src.linguistic_tests.lm_utils import DEVICES
from src.linguistic_tests.lm_utils import load_model
from src.linguistic_tests.lm_utils import ModelTypes
from src.linguistic_tests.lm_utils import SentenceNames
from src.linguistic_tests.testset import parse_typed_sentence


class TestModelScoring(TestCase):
    @pytest.mark.enable_socket
    def test_run_test_design_sprouse_bert(self):
        sentence_txt = "San Francisco"
        typed_sentence = parse_typed_sentence(SentenceNames.SENTENCE_GOOD, sentence_txt)
        print(f"typed_sentence.sent.txt: {typed_sentence.sent.txt}")

        device = DEVICES.CPU
        # online_test_helper_roberta(typed_sentence, device)

        model_type = ModelTypes.GPT
        model_name = "gpt2-medium"  # 355M params
        model, tokenizer = load_model(model_type, model_name, device)
        print(
            f"tokenizer.cls_token: {tokenizer.cls_token}, tokenizer.sep_token: {tokenizer.sep_token}, tokenizer.mask_token: {tokenizer.mask_token}"
        )
        typed_sentence.sent.tokens = tokenizer.tokenize(typed_sentence.sent.txt)
        (
            lp_softmax,
            lp_logistic,
            score_per_masking,
            logistic_score_per_masking,
        ) = get_sentence_acceptability_score(
            model_type, model, tokenizer, typed_sentence.sent.tokens, device
        )
        expected_sentence_log_Pgpt2 = -8.693
        decimals_precision = 3
        self.assertAlmostEqual(
            expected_sentence_log_Pgpt2, lp_softmax, decimals_precision
        )

        # sentence.tokens: ['San', 'ĠFrancisco']
        model_type = ModelTypes.ROBERTA
        model_name = "roberta-large"  # 355M params
        model, tokenizer = load_model(model_type, model_name, device)
        print(
            f"tokenizer.cls_token: {tokenizer.cls_token}, tokenizer.sep_token: {tokenizer.sep_token}, tokenizer.mask_token: {tokenizer.mask_token}"
        )
        typed_sentence.sent.tokens = tokenizer.tokenize(typed_sentence.sent.txt)
        (
            lp_softmax,
            lp_logistic,
            score_per_masking,
            logistic_score_per_masking,
        ) = get_sentence_acceptability_score(
            model_type, model, tokenizer, typed_sentence.sent.tokens, device
        )
        expected_sentence_PLL_roberta = -1.006
        expected_tokens_probs = [0.006, 1.000]
        decimals_precision = 3
        self.assertAlmostEqual(
            expected_sentence_PLL_roberta, lp_softmax, decimals_precision
        )
        for expected, actual in zip(expected_tokens_probs, score_per_masking):
            self.assertAlmostEqual(expected, actual, decimals_precision)
        # model: roberta-large
        # PLL score_per_masking: [0.006103568, 0.999546]
        # 0.006 San
        # 1.000 ĠFrancisco
        # Paper: -0.006 + (-1.000) = -1.006 PLL_roberta(W)

        # model: gpt-medium
        # lp_softmax: -8.693
        # Paper: -7.749 + (-0.944) = -8.693 = log Pgpt2(W)
