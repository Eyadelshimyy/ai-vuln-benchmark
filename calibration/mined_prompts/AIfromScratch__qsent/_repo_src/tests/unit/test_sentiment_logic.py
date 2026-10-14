"""
Unit tests for FinBERTModel and the safe() utility.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from qsf.common.utils import safe, company_search_name
from qsf.nlp.sentiment import FinBERTModel, SENTIMENT_MAP


# ---------------------------------------------------------------------------
# safe() - NaN/inf sanitizer
# ---------------------------------------------------------------------------

class TestSafe:
    def test_normal_float(self):
        assert safe(0.12345) == 0.1235

    def test_rounding(self):
        assert safe(0.123456789) == 0.1235

    def test_custom_decimals(self):
        assert safe(1.23456, 2) == 1.23

    def test_nan_returns_none(self):
        import math
        assert safe(float("nan")) is None

    def test_inf_returns_none(self):
        assert safe(float("inf")) is None

    def test_negative_inf_returns_none(self):
        assert safe(float("-inf")) is None

    def test_none_returns_none(self):
        assert safe(None) is None

    def test_zero(self):
        assert safe(0.0) == 0.0

    def test_negative(self):
        assert safe(-0.75) == -0.75


# ---------------------------------------------------------------------------
# company_search_name() - legal name → brand name
# ---------------------------------------------------------------------------

class TestCompanySearchName:
    def test_strips_inc_after_comma(self):
        assert company_search_name("FormFactor, Inc.") == "FormFactor"

    def test_strips_inc_after_comma_tesla(self):
        assert company_search_name("Tesla, Inc.") == "Tesla"

    def test_strips_corporation_suffix(self):
        assert company_search_name("NVIDIA Corporation") == "NVIDIA"

    def test_strips_inc_without_comma(self):
        assert company_search_name("Alphabet Inc.") == "Alphabet"

    def test_strips_ltd(self):
        assert company_search_name("Some Company Ltd.") == "Some Company"

    def test_strips_llc(self):
        assert company_search_name("Some Company LLC") == "Some Company"

    def test_no_suffix_returned_as_is(self):
        assert company_search_name("Apple") == "Apple"

    def test_empty_string_returns_empty(self):
        assert company_search_name("") == ""


# ---------------------------------------------------------------------------
# FinBERTModel.score() - HuggingFace FinBERT scoring
# ---------------------------------------------------------------------------

class TestFinBERTModel:
    def _make_hf_response(self, label, score):
        """Simulate HuggingFace response for one text: list of label dicts."""
        others = [l for l in SENTIMENT_MAP if l != label]
        remaining = (1 - score) / 2
        return [
            {"label": label, "score": score},
            {"label": others[0], "score": remaining},
            {"label": others[1], "score": remaining},
        ]

    @patch("qsf.nlp.sentiment.requests.post")
    def test_positive_sentiment(self, mock_post):
        mock_post.return_value.json.return_value = [
            self._make_hf_response("positive", 0.9)
        ]
        scores = FinBERTModel().score(["Great earnings report"])
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.9)

    @patch("qsf.nlp.sentiment.requests.post")
    def test_negative_sentiment(self, mock_post):
        mock_post.return_value.json.return_value = [
            self._make_hf_response("negative", 0.8)
        ]
        scores = FinBERTModel().score(["Company files for bankruptcy"])
        assert scores[0] == pytest.approx(-0.8)

    @patch("qsf.nlp.sentiment.requests.post")
    def test_neutral_sentiment(self, mock_post):
        mock_post.return_value.json.return_value = [
            self._make_hf_response("neutral", 0.7)
        ]
        scores = FinBERTModel().score(["Company releases quarterly report"])
        assert scores[0] == pytest.approx(0.0)

    @patch("qsf.nlp.sentiment.requests.post")
    def test_one_call_per_text(self, mock_post):
        """Each text must trigger exactly one API call and produce one score.

        This guards against batching bugs where the HF API only scores the
        first item in a batch, silently dropping the rest.
        """
        mock_post.return_value.json.return_value = [self._make_hf_response("positive", 0.9)]
        scores = FinBERTModel().score(["text"] * 5)
        assert mock_post.call_count == 6  # 1 health check + 5 item requests
        assert len(scores) == 5

    @patch("qsf.nlp.sentiment.requests.post")
    def test_failed_item_does_not_stop_others(self, mock_post):
        """A failed item returns None at its index — other items are unaffected."""
        good = MagicMock()
        good.json.return_value = [self._make_hf_response("positive", 0.9)]

        bad = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        bad.raise_for_status.side_effect = requests.HTTPError(response=mock_response)

        mock_post.side_effect = [good, good, bad, good]  # health check + 3 item requests
        scores = FinBERTModel().score(["text1", "text2", "text3"])
        assert mock_post.call_count == 4  # 1 health check + 3 item requests
        # Always same length as input — failed item is None, not omitted
        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.9)
        assert scores[1] is None  # failed item
        assert scores[2] == pytest.approx(0.9)

    @patch("qsf.nlp.sentiment.requests.post")
    def test_unexpected_response_format_is_skipped(self, mock_post):
        """A non-list API response (e.g. error dict) returns None at that index."""
        mock_post.return_value.json.return_value = {"error": "Model is loading"}
        scores = FinBERTModel().score(["text"])
        assert len(scores) == 1
        assert scores[0] is None

    @patch("qsf.nlp.sentiment.requests.post")
    def test_retries_on_token_limit_error(self, mock_post):
        """HTTP 400 'size of tensor' triggers a trim-and-retry; final score is returned."""
        token_error = MagicMock()
        token_error.status_code = 400
        token_error.text = '{"error":"The size of tensor a (531) must match the size of tensor b (512) at non-singleton dimension 1"}'
        token_error.raise_for_status = MagicMock()

        success = MagicMock()
        success.status_code = 200
        success.text = ""
        success.raise_for_status = MagicMock()
        success.json.return_value = [self._make_hf_response("positive", 0.9)]

        # health check, then item 1 fails once before succeeding on retry
        mock_post.side_effect = [success, token_error, success]
        scores = FinBERTModel().score(["a" * 600])
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.9)
        assert mock_post.call_count == 3  # health check + 1 failed attempt + 1 retry

    @patch("qsf.nlp.sentiment.requests.post")
    def test_empty_input(self, mock_post):
        scores = FinBERTModel().score([])
        assert scores == []
        mock_post.assert_not_called()
