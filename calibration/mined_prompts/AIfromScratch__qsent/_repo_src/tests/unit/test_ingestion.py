"""
Unit tests for ingestion providers (YFinanceMarketData, NewsAPIProvider, RedditProvider).
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qsf.ingestion.market import YFinanceMarketData
from qsf.ingestion.news import NewsAPIProvider, news_from_date, _article_text
from qsf.ingestion.social import RedditProvider, _post_text


# ---------------------------------------------------------------------------
# YFinanceMarketData.get_history()
# ---------------------------------------------------------------------------

class TestYFinanceMarketData:
    @patch("qsf.ingestion.market.yf.Ticker")
    def test_returns_dataframe_from_yfinance(self, mock_ticker):
        expected = pd.DataFrame(
            {"Close": [10.0], "Volume": [1000000]},
            index=pd.to_datetime(["2026-02-10"]),
        )
        mock_ticker.return_value.history.return_value = expected
        result = YFinanceMarketData().get_history("IONQ", "30d")
        assert result.equals(expected)

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_passes_ticker_to_yfinance(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        YFinanceMarketData().get_history("IONQ", "30d")
        mock_ticker.assert_called_once_with("IONQ")

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_passes_period_to_history(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        YFinanceMarketData().get_history("IONQ", "90d")
        mock_ticker.return_value.history.assert_called_once_with(period="90d")

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_returns_empty_dataframe_for_unknown_ticker(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        result = YFinanceMarketData().get_history("FAKE", "30d")
        assert result.empty


# ---------------------------------------------------------------------------
# YFinanceMarketData.get_company_name()
# ---------------------------------------------------------------------------

class TestYFinanceGetCompanyName:
    @patch("qsf.ingestion.market.yf.Ticker")
    def test_returns_long_name(self, mock_ticker):
        mock_ticker.return_value.info = {"longName": "FormFactor, Inc.", "shortName": "FormFactor Inc."}
        result = YFinanceMarketData().get_company_name("FORM")
        assert result == "FormFactor, Inc."

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_falls_back_to_short_name(self, mock_ticker):
        mock_ticker.return_value.info = {"longName": None, "shortName": "FormFactor Inc."}
        result = YFinanceMarketData().get_company_name("FORM")
        assert result == "FormFactor Inc."

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_returns_empty_string_when_no_name(self, mock_ticker):
        mock_ticker.return_value.info = {}
        result = YFinanceMarketData().get_company_name("FAKE")
        assert result == ""


# ---------------------------------------------------------------------------
# news_from_date()
# ---------------------------------------------------------------------------

class TestNewsFromDate:
    def test_28_day_window_stays_within_api_limit(self):
        # NewsAPI free plan has a 30-day exclusive rolling window.
        # We request 28 days to stay safely within that limit.
        # 29 and 30 days were tried but both failed in practice — the API server
        # clock differs from local time, so the boundary can shift by a day
        # depending on when the request is made. 28 days gives enough buffer.
        now = datetime(2026, 3, 1, 14, 30, 0)
        result = news_from_date(now, days=28)
        assert result == "2026-02-01T14:30:00"


# ---------------------------------------------------------------------------
# _article_text()
# ---------------------------------------------------------------------------

class TestArticleText:
    def test_combines_title_and_description(self):
        article = {"title": "IonQ beats earnings", "description": "Strong Q4 results."}
        assert _article_text(article) == "IonQ beats earnings. Strong Q4 results."

    def test_falls_back_to_title_when_description_is_none(self):
        article = {"title": "IonQ beats earnings", "description": None}
        assert _article_text(article) == "IonQ beats earnings"

    def test_falls_back_to_title_when_description_is_missing(self):
        article = {"title": "IonQ beats earnings"}
        assert _article_text(article) == "IonQ beats earnings"

    def test_falls_back_to_title_when_description_is_empty_string(self):
        article = {"title": "IonQ beats earnings", "description": ""}
        assert _article_text(article) == "IonQ beats earnings"


# ---------------------------------------------------------------------------
# NewsAPIProvider.get_articles()
# ---------------------------------------------------------------------------

class TestNewsAPIProvider:
    @patch("qsf.ingestion.news.NewsApiClient")
    def test_returns_correctly_shaped_items(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {
            "articles": [
                {"title": "IonQ beats earnings", "description": "Strong Q4 results.", "publishedAt": "2026-02-10T10:00:00Z"},
                {"title": "Quantum stocks rise", "description": None, "publishedAt": "2026-02-11T14:00:00Z"},
            ]
        }
        items = NewsAPIProvider().get_articles("IONQ")
        assert len(items) == 2
        assert items[0]["text"] == "IonQ beats earnings. Strong Q4 results."
        assert items[0]["date"] == "2026-02-10"
        assert items[0]["source"] == "news"

    @patch("qsf.ingestion.news.NewsApiClient")
    def test_skips_articles_with_no_title(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {
            "articles": [
                {"title": None, "description": "Some description.", "publishedAt": "2026-02-10T10:00:00Z"},
                {"title": "Valid article", "description": None, "publishedAt": "2026-02-10T10:00:00Z"},
            ]
        }
        items = NewsAPIProvider().get_articles("IONQ")
        assert len(items) == 1
        assert items[0]["text"] == "Valid article"

    @patch("qsf.ingestion.news.NewsApiClient")
    def test_empty_response(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {"articles": []}
        items = NewsAPIProvider().get_articles("IONQ")
        assert items == []

    @patch("qsf.ingestion.news.NewsApiClient")
    def test_searches_with_ticker_when_no_company_name(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {"articles": []}
        NewsAPIProvider().get_articles("FORM")
        call_kwargs = mock_client_class.return_value.get_everything.call_args.kwargs
        assert call_kwargs["q"] == "FORM"

    @patch("qsf.ingestion.news.NewsApiClient")
    def test_searches_with_quoted_company_name(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {"articles": []}
        NewsAPIProvider().get_articles("FORM", company_name="FormFactor")
        call_kwargs = mock_client_class.return_value.get_everything.call_args.kwargs
        assert call_kwargs["q"] == '"FormFactor"'


# ---------------------------------------------------------------------------
# _post_text()
# ---------------------------------------------------------------------------

class TestPostText:
    def _make_post(self, title, selftext="", comments=None):
        post = MagicMock()
        post.title = title
        post.selftext = selftext
        post.comments = MagicMock()
        post.comments.replace_more = MagicMock()
        mock_comments = []
        for body, score in (comments or []):
            c = MagicMock()
            c.body = body
            c.score = score
            mock_comments.append(c)
        post.comments.__iter__ = MagicMock(return_value=iter(mock_comments))
        return post

    def test_title_only_when_no_selftext_or_comments(self):
        post = self._make_post("IONQ hits new high")
        assert _post_text(post) == "IONQ hits new high"

    def test_combines_title_and_selftext(self):
        post = self._make_post("IONQ hits new high", selftext="Strong quarterly results.")
        assert _post_text(post) == "IONQ hits new high. Strong quarterly results."

    def test_includes_top_comments_by_upvote(self):
        post = self._make_post(
            "IONQ earnings",
            comments=[("Terrible results", 10), ("Great outlook", 50), ("Meh", 2)],
        )
        result = _post_text(post)
        # Highest upvoted comment should appear before lower ones
        assert "Great outlook" in result
        assert result.index("Great outlook") < result.index("Terrible results")

    def test_truncates_to_max_chars(self):
        long_comment = "x" * 600
        post = self._make_post(
            "IONQ earnings",
            comments=[(long_comment, 100), (long_comment, 50), (long_comment, 10)],
        )
        result = _post_text(post)
        from qsf.ingestion.social import REDDIT_MAX_CHARS
        assert len(result) <= REDDIT_MAX_CHARS

    def test_title_always_preserved(self):
        long_comment = "x" * 600
        post = self._make_post(
            "IONQ earnings",
            comments=[(long_comment, 100), (long_comment, 50), (long_comment, 10)],
        )
        result = _post_text(post)
        assert result.startswith("IONQ earnings")

    def test_ignores_empty_selftext(self):
        post = self._make_post("IONQ earnings", selftext="   ")
        assert _post_text(post) == "IONQ earnings"


# ---------------------------------------------------------------------------
# RedditProvider.get_posts()
# ---------------------------------------------------------------------------

class TestRedditProvider:
    def _make_post(self, title, days_ago):
        post = MagicMock()
        post.title = title
        post.selftext = ""
        post.created_utc = (datetime.now() - timedelta(days=days_ago)).timestamp()
        post.comments = MagicMock()
        post.comments.replace_more = MagicMock()
        post.comments.__iter__ = MagicMock(return_value=iter([]))
        return post

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_returns_recent_posts(self, mock_reddit_class):
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [
            self._make_post("IONQ hits new high", days_ago=5),
            self._make_post("Quantum computing outlook", days_ago=10),
        ]
        items = RedditProvider().get_posts("IONQ")
        assert len(items) == 12  # 2 posts × 6 subreddits
        assert items[0]["source"] == "social"
        assert items[0]["text"] == "IONQ hits new high"

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_filters_out_old_posts(self, mock_reddit_class):
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [
            self._make_post("Old post", days_ago=31),
        ]
        items = RedditProvider().get_posts("IONQ")
        assert items == []

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_searches_all_three_subreddits(self, mock_reddit_class):
        mock_subreddit = mock_reddit_class.return_value.subreddit.return_value
        mock_subreddit.search.return_value = [
            self._make_post("IONQ post", days_ago=3),
        ]
        RedditProvider().get_posts("IONQ")
        assert mock_reddit_class.return_value.subreddit.call_count == 6

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_calls_replace_more_on_each_post(self, mock_reddit_class):
        post = self._make_post("IONQ post", days_ago=3)
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [post]
        RedditProvider().get_posts("IONQ")
        # replace_more called once per post per subreddit (6 subreddits)
        assert post.comments.replace_more.call_count == 6

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_searches_with_cashtag_when_no_company_name(self, mock_reddit_class):
        mock_subreddit = mock_reddit_class.return_value.subreddit.return_value
        mock_subreddit.search.return_value = []
        RedditProvider().get_posts("FORM")
        mock_subreddit.search.assert_called_with("$FORM", sort="new", time_filter="month", limit=50)

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_searches_with_company_name_and_cashtag(self, mock_reddit_class):
        mock_subreddit = mock_reddit_class.return_value.subreddit.return_value
        mock_subreddit.search.return_value = []
        RedditProvider().get_posts("FORM", company_name="FormFactor")
        mock_subreddit.search.assert_called_with('"FormFactor" OR "$FORM"', sort="new", time_filter="month", limit=50)
