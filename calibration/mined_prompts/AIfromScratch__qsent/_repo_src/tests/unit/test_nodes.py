"""
Unit tests for LangGraph node logic via build_pipeline() with mock providers.
Providers are injected directly — no patching of module-level symbols.
"""
import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import pytest

from qsf.agents.workflow import build_pipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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


def make_providers(
    history=None,
    articles=None,
    posts=None,
    scores=None,
):
    """Return four mock providers with default happy-path return values."""
    market = MagicMock()
    market.get_history.return_value = MOCK_HIST if history is None else history
    market.get_company_name.return_value = "IonQ, Inc."

    news = MagicMock()
    news.get_articles.return_value = MOCK_NEWS if articles is None else articles

    social = MagicMock()
    social.get_posts.return_value = MOCK_REDDIT if posts is None else posts

    model = MagicMock()
    model.score.return_value = MOCK_SCORES if scores is None else scores

    return market, news, social, model


# ---------------------------------------------------------------------------
# fetch_market_data
# ---------------------------------------------------------------------------

class TestFetchMarketData:
    def test_returns_stock_df_on_success(self):
        market, news, social, model = make_providers(
            history=pd.DataFrame(
                {"Close": [10.0], "Volume": [1000000]},
                index=pd.to_datetime(["2026-02-10"]),
            ),
            articles=[MOCK_NEWS[0]],
            posts=[MOCK_REDDIT[0]],
            scores=[0.8],
        )
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert state.get("error") is None
        assert "stock_df" in state
        assert "Close" in state["stock_df"].columns
        assert "ror" in state["stock_df"].columns

    def test_returns_error_on_empty_history(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "FAKE"})
        assert "error" in state
        assert "FAKE" in state["error"]

    def test_index_is_date_objects(self):
        market, news, social, model = make_providers(
            history=pd.DataFrame(
                {"Close": [10.0], "Volume": [1000000]},
                index=pd.to_datetime(["2026-02-10"]),
            ),
            articles=[MOCK_NEWS[0]],
            posts=[MOCK_REDDIT[0]],
            scores=[0.8],
        )
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert isinstance(state["stock_df"].index[0], dt.date)

    def test_downstream_providers_not_called_on_error(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        pipeline.invoke({"ticker": "FAKE"})
        news.get_articles.assert_not_called()
        social.get_posts.assert_not_called()
        model.score.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_news
# ---------------------------------------------------------------------------

class TestFetchNews:
    def test_calls_news_provider_with_ticker(self):
        market, news, social, model = make_providers()
        pipeline = build_pipeline(market, news, social, model)
        pipeline.invoke({"ticker": "IONQ"})
        news.get_articles.assert_called_once_with("IONQ", "IonQ")

    def test_skips_on_upstream_error(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        pipeline.invoke({"ticker": "FAKE"})
        news.get_articles.assert_not_called()

    def test_returns_empty_list_when_no_articles(self):
        market, news, social, model = make_providers(articles=[], posts=[], scores=[])
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert state.get("news_items") == []


# ---------------------------------------------------------------------------
# fetch_reddit
# ---------------------------------------------------------------------------

class TestFetchReddit:
    def test_calls_social_provider_with_ticker(self):
        market, news, social, model = make_providers()
        pipeline = build_pipeline(market, news, social, model)
        pipeline.invoke({"ticker": "IONQ"})
        social.get_posts.assert_called_once_with("IONQ", "IonQ")

    def test_skips_on_upstream_error(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        pipeline.invoke({"ticker": "FAKE"})
        social.get_posts.assert_not_called()

    def test_returns_empty_list_when_no_posts(self):
        market, news, social, model = make_providers(articles=[], posts=[], scores=[])
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert state.get("reddit_items") == []


# ---------------------------------------------------------------------------
# score_sentiment
# ---------------------------------------------------------------------------

class TestScoreSentimentNode:
    def test_scores_combined_items(self):
        market, news, social, model = make_providers()
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert "scored_items" in state
        assert len(state["scored_items"]) == 3
        assert state["scored_items"][0]["weighted_sentiment"] == 0.8

    def test_preserves_original_item_fields(self):
        market, news, social, model = make_providers()
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        first = state["scored_items"][0]
        assert first["text"] == "IonQ beats earnings"
        assert first["source"] == "news"
        assert first["date"] == "2026-02-10"

    def test_returns_error_when_no_items(self):
        market, news, social, model = make_providers(articles=[], posts=[], scores=[])
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert "error" in state
        assert "IONQ" in state["error"]

    def test_model_not_called_on_upstream_error(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        pipeline.invoke({"ticker": "FAKE"})
        model.score.assert_not_called()


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _run(self):
        market, news, social, model = make_providers()
        pipeline = build_pipeline(market, news, social, model)
        return pipeline.invoke({"ticker": "IONQ"})

    def test_returns_result_with_expected_keys(self):
        state = self._run()
        assert "result" in state
        r = state["result"]
        assert r["ticker"] == "IONQ"
        assert "sentiment_score" in r
        assert "breakdown" in r
        assert "daily_data" in r
        assert "last_updated" in r

    def test_daily_data_length_matches_stock_df(self):
        state = self._run()
        assert len(state["result"]["daily_data"]) == len(MOCK_HIST)

    def test_daily_data_contains_expected_fields(self):
        state = self._run()
        day = state["result"]["daily_data"][0]
        assert "date" in day
        assert "close" in day
        assert "volume" in day
        assert "ror" in day
        assert "news_sentiment" in day
        assert "social_sentiment" in day

    def test_trend_is_valid_value(self):
        state = self._run()
        assert state["result"]["breakdown"]["trend"] in ("increasing", "decreasing", "stable")

    def test_skips_on_upstream_error(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "FAKE"})
        assert state.get("result") is None

    def test_overall_sentiment_is_average_of_news_and_social(self):
        state = self._run()
        r = state["result"]
        expected = round(
            (r["breakdown"]["news_sentiment"] + r["breakdown"]["social_sentiment"]) / 2, 4
        )
        assert r["sentiment_score"] == expected


# ---------------------------------------------------------------------------
# source_status
#
# NOTE: written pre-implementation — these tests are expected to FAIL until
# code-builder adds the `source_status` field to PipelineState and wires
# per-node try/except + status bookkeeping in workflow.py.
# ---------------------------------------------------------------------------

class TestSourceStatus:
    def test_given_all_providers_succeed_when_pipeline_runs_then_all_sources_marked_ok(self):
        market, news, social, model = make_providers()
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})
        assert state["source_status"] == {
            "market": "ok",
            "news": "ok",
            "reddit": "ok",
            "sentiment": "ok",
        }

    def test_given_market_provider_raises_when_pipeline_runs_then_error_set_and_downstream_sources_skipped(self):
        market, news, social, model = make_providers()
        market.get_history.side_effect = Exception("connection timeout")
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})

        assert state.get("error")

        status = state["source_status"]
        assert status["market"] != "ok"
        assert isinstance(status["market"], str) and status["market"]
        assert status["news"] == "skipped"
        assert status["reddit"] == "skipped"
        assert status["sentiment"] == "skipped"

    def test_given_news_provider_raises_when_pipeline_runs_then_error_set_and_downstream_sources_skipped(self):
        market, news, social, model = make_providers()
        news.get_articles.side_effect = Exception("NewsAPI timeout")
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})

        assert state.get("error")

        status = state["source_status"]
        assert status["market"] == "ok"
        assert status["news"] != "ok"
        assert isinstance(status["news"], str) and status["news"]
        assert status["reddit"] == "skipped"
        assert status["sentiment"] == "skipped"

    def test_given_reddit_provider_raises_when_pipeline_runs_then_error_set_and_sentiment_skipped(self):
        market, news, social, model = make_providers()
        social.get_posts.side_effect = Exception("received 401 HTTP response")
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})

        assert state.get("error")

        status = state["source_status"]
        assert status["market"] == "ok"
        assert status["news"] == "ok"
        assert status["reddit"] != "ok"
        assert isinstance(status["reddit"], str)
        assert "401" in status["reddit"]
        assert status["sentiment"] == "skipped"

    def test_given_sentiment_model_raises_when_pipeline_runs_then_error_set_and_sentiment_status_describes_failure(self):
        market, news, social, model = make_providers()
        model.score.side_effect = Exception("HuggingFace API down")
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})

        assert state.get("error")

        status = state["source_status"]
        assert status["market"] == "ok"
        assert status["news"] == "ok"
        assert status["reddit"] == "ok"
        assert status["sentiment"] != "ok"
        assert isinstance(status["sentiment"], str) and status["sentiment"]

    def test_given_empty_market_history_when_pipeline_runs_then_market_status_matches_error_and_downstream_skipped(self):
        market, news, social, model = make_providers(history=pd.DataFrame())
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "FAKE"})

        status = state["source_status"]
        assert status["market"] == state["error"]
        assert status["market"] not in ("ok", "skipped")
        assert status["news"] == "skipped"
        assert status["reddit"] == "skipped"
        assert status["sentiment"] == "skipped"

    def test_given_no_news_or_social_items_when_pipeline_runs_then_sentiment_status_matches_error(self):
        market, news, social, model = make_providers(articles=[], posts=[], scores=[])
        pipeline = build_pipeline(market, news, social, model)
        state = pipeline.invoke({"ticker": "IONQ"})

        status = state["source_status"]
        assert status["market"] == "ok"
        assert status["news"] == "ok"
        assert status["reddit"] == "ok"
        assert status["sentiment"] == state["error"]
        assert status["sentiment"] not in ("ok", "skipped")
