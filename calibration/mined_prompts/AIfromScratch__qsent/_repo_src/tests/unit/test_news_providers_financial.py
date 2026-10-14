"""
Unit tests for the financial news providers (Massive, AlphaVantage, Marketaux).
"""
from unittest.mock import MagicMock, patch

import pytest

from qsf.ingestion.news_alphavantage import AlphaVantageNewsProvider
from qsf.ingestion.news_marketaux import MarketauxNewsProvider
from qsf.ingestion.news_massive import MassiveNewsProvider


# ---------------------------------------------------------------------------
# MassiveNewsProvider
# ---------------------------------------------------------------------------

class TestMassiveNewsProvider:
    @patch("qsf.ingestion.news_massive.requests.get")
    def test_returns_correctly_shaped_items(self, mock_get, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {
            "results": [
                {"title": "IonQ beats earnings", "description": "Strong Q4 results.", "published_utc": "2026-02-10T10:00:00Z"},
                {"title": "Quantum stocks rise", "description": None, "published_utc": "2026-02-11T14:00:00Z"},
            ]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        items = MassiveNewsProvider().get_articles("IONQ")
        assert len(items) == 2
        assert items[0]["text"] == "IonQ beats earnings. Strong Q4 results."
        assert items[0]["date"] == "2026-02-10"
        assert items[0]["source"] == "news"

    @patch("qsf.ingestion.news_massive.requests.get")
    def test_skips_articles_with_no_title(self, mock_get, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {
            "results": [
                {"title": None, "description": "Some description.", "published_utc": "2026-02-10T10:00:00Z"},
                {"title": "Valid article", "description": None, "published_utc": "2026-02-10T10:00:00Z"},
            ]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        items = MassiveNewsProvider().get_articles("IONQ")
        assert len(items) == 1
        assert items[0]["text"] == "Valid article"

    def test_raises_on_missing_key(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MASSIVE_API_KEY not set"):
            MassiveNewsProvider().get_articles("IONQ")

    @patch("qsf.ingestion.news_massive.requests.get")
    def test_passes_ticker_param(self, mock_get, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {"results": []}
        mock_get.return_value.raise_for_status = MagicMock()
        MassiveNewsProvider().get_articles("FORM")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert params["ticker"] == "FORM"


# ---------------------------------------------------------------------------
# AlphaVantageNewsProvider
# ---------------------------------------------------------------------------

class TestAlphaVantageNewsProvider:
    @patch("qsf.ingestion.news_alphavantage.requests.get")
    def test_returns_correctly_shaped_items(self, mock_get, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {
            "feed": [
                {"title": "IonQ beats earnings", "summary": "Strong Q4 results.", "time_published": "20260210T143022"},
                {"title": "Quantum stocks rise", "summary": None, "time_published": "20260211T090000"},
            ]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        items = AlphaVantageNewsProvider().get_articles("IONQ")
        assert len(items) == 2
        assert items[0]["text"] == "IonQ beats earnings. Strong Q4 results."
        assert items[0]["date"] == "2026-02-10"
        assert items[0]["source"] == "news"

    @patch("qsf.ingestion.news_alphavantage.requests.get")
    def test_date_parsing_uses_parse_date_not_slice(self, mock_get, monkeypatch):
        """time_published[:10] would give '20260210T1' — _parse_date() must be used."""
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {
            "feed": [
                {"title": "Test article", "summary": "", "time_published": "20260210T143022"},
            ]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        items = AlphaVantageNewsProvider().get_articles("IONQ")
        assert items[0]["date"] == "2026-02-10"

    @patch("qsf.ingestion.news_alphavantage.requests.get")
    def test_raises_on_quota_response(self, mock_get, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {
            "Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."
        }
        mock_get.return_value.raise_for_status = MagicMock()
        with pytest.raises(ValueError):
            AlphaVantageNewsProvider().get_articles("IONQ")

    def test_raises_on_missing_key(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY not set"):
            AlphaVantageNewsProvider().get_articles("IONQ")


# ---------------------------------------------------------------------------
# MarketauxNewsProvider
# ---------------------------------------------------------------------------

class TestMarketauxNewsProvider:
    @patch("qsf.ingestion.news_marketaux.requests.get")
    def test_returns_correctly_shaped_items(self, mock_get, monkeypatch):
        monkeypatch.setenv("MARKETAUX_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {
            "data": [
                {"title": "IonQ beats earnings", "description": "Strong Q4 results.", "published_at": "2026-02-10T10:00:00+00:00"},
                {"title": "Quantum stocks rise", "description": None, "published_at": "2026-02-11T14:00:00+00:00"},
            ]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        items = MarketauxNewsProvider().get_articles("IONQ")
        assert len(items) == 2
        assert items[0]["text"] == "IonQ beats earnings. Strong Q4 results."
        assert items[0]["date"] == "2026-02-10"
        assert items[0]["source"] == "news"

    @patch("qsf.ingestion.news_marketaux.requests.get")
    def test_sends_filter_entities_true(self, mock_get, monkeypatch):
        monkeypatch.setenv("MARKETAUX_API_KEY", "test-key")
        mock_get.return_value.json.return_value = {"data": []}
        mock_get.return_value.raise_for_status = MagicMock()
        MarketauxNewsProvider().get_articles("FORM")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert params["filter_entities"] == "true"

    def test_raises_on_missing_key(self, monkeypatch):
        monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MARKETAUX_API_KEY not set"):
            MarketauxNewsProvider().get_articles("IONQ")
