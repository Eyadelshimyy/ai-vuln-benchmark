"""
Integration tests for the LangGraph pipeline.
Providers are injected via build_pipeline() — no module-level patching.
Tests verify that nodes are wired correctly and state flows through the full
graph as expected.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from qsf.agents.workflow import build_pipeline

MOCK_HIST = pd.DataFrame(
    {"Close": [10.0, 10.5, 11.0], "Volume": [1000000, 1200000, 1100000]},
    index=pd.to_datetime(["2026-02-10", "2026-02-11", "2026-02-12"]),
)
MOCK_NEWS = [
    {"text": "IonQ beats earnings", "date": "2026-02-10", "source": "news"},
    {"text": "Quantum stocks surge", "date": "2026-02-11", "source": "news"},
]
MOCK_REDDIT = [
    {"text": "IONQ looking bullish", "date": "2026-02-10", "source": "social"},
]
MOCK_SCORES = [0.8, 0.6, 0.7]


def make_pipeline(history=None, articles=None, posts=None, scores=None):
    market = MagicMock()
    market.get_history.return_value = MOCK_HIST if history is None else history
    market.get_company_name.return_value = "IonQ, Inc."

    news = MagicMock()
    news.get_articles.return_value = MOCK_NEWS if articles is None else articles

    social = MagicMock()
    social.get_posts.return_value = MOCK_REDDIT if posts is None else posts

    model = MagicMock()
    model.score.return_value = MOCK_SCORES if scores is None else scores

    return build_pipeline(market, news, social, model), news, social


def test_happy_path_produces_result():
    pipeline, _, _ = make_pipeline()
    state = pipeline.invoke({"ticker": "IONQ"})

    assert state.get("error") is None
    assert state.get("result") is not None
    assert state["result"]["ticker"] == "IONQ"
    assert len(state["result"]["daily_data"]) == 3
    assert state["result"]["sentiment_score"] is not None


def test_invalid_ticker_sets_error():
    pipeline, _, _ = make_pipeline(history=pd.DataFrame())
    state = pipeline.invoke({"ticker": "FAKE"})

    assert "error" in state
    assert state.get("result") is None


def test_no_text_data_sets_error():
    pipeline, _, _ = make_pipeline(articles=[], posts=[], scores=[])
    state = pipeline.invoke({"ticker": "IONQ"})

    assert "error" in state
    assert state.get("result") is None


def test_ticker_is_available_throughout_pipeline():
    """Verify ticker is preserved in state from start to finish."""
    pipeline, mock_news, mock_social = make_pipeline()
    state = pipeline.invoke({"ticker": "IONQ"})

    assert state["ticker"] == "IONQ"
    mock_news.get_articles.assert_called_once_with("IONQ", "IonQ")
    mock_social.get_posts.assert_called_once_with("IONQ", "IonQ")
