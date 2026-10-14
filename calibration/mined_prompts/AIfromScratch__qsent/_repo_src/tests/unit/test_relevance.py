"""
Unit tests for the RelevanceClassifier.
"""
from unittest.mock import MagicMock, patch

import pytest

from qsf.nlp.relevance import MAX_ARTICLES, RelevanceClassifier


def _make_openai_response(content: str):
    """Build a minimal mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_articles(n: int) -> list[dict]:
    return [{"text": f"Article {i}", "date": "2026-02-10", "source": "news"} for i in range(n)]


class TestRelevanceClassifier:
    @patch("qsf.nlp.relevance.OpenAI")
    def test_yes_returns_true(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "IonQ Q4 results", "date": "2026-02-10", "source": "news"}])
        assert result == [True]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_no_returns_false(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("no")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "Unrelated article", "date": "2026-02-10", "source": "news"}])
        assert result == [False]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_case_insensitive(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("Yes")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "IonQ article", "date": "2026-02-10", "source": "news"}])
        assert result == [True]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_failed_api_call_returns_false_no_raise(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "Some article", "date": "2026-02-10", "source": "news"}])
        assert result == [False]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_returns_same_length_as_input(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")
        articles = _make_articles(5)
        result = RelevanceClassifier().classify("IONQ", "IonQ", articles)
        assert len(result) == 5

    @patch("qsf.nlp.relevance.OpenAI")
    def test_caps_at_max_articles(self, mock_openai_cls, monkeypatch):
        """Only MAX_ARTICLES API calls should be made; positions beyond cap default to False."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")

        articles = _make_articles(MAX_ARTICLES + 5)
        result = RelevanceClassifier().classify("IONQ", "IonQ", articles)

        # Only MAX_ARTICLES API calls should have been made
        assert mock_client.chat.completions.create.call_count == MAX_ARTICLES
        # Result length matches input length
        assert len(result) == MAX_ARTICLES + 5
        # Positions beyond cap are False
        assert all(r is False for r in result[MAX_ARTICLES:])

    @patch("qsf.nlp.relevance.OpenAI")
    def test_empty_input_returns_empty_list(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        result = RelevanceClassifier().classify("IONQ", "IonQ", [])
        assert result == []
        mock_client.chat.completions.create.assert_not_called()

    # ---- classify_stream() tests ----

    @patch("qsf.nlp.relevance.OpenAI")
    def test_stream_yields_tuples_with_correct_index(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")
        articles = _make_articles(3)
        results = list(RelevanceClassifier().classify_stream("IONQ", "IonQ", articles))
        assert len(results) == 3
        for i, (index, article, relevant) in enumerate(results):
            assert index == i
            assert article == articles[i]
            assert relevant is True

    @patch("qsf.nlp.relevance.OpenAI")
    def test_stream_yes_true_no_false(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_openai_response(r) for r in ["yes", "no", "yes"]
        ]
        articles = _make_articles(3)
        results = list(RelevanceClassifier().classify_stream("IONQ", "IonQ", articles))
        assert [r for _, _, r in results] == [True, False, True]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_stream_api_error_yields_false_no_raise(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API down")
        articles = _make_articles(2)
        results = list(RelevanceClassifier().classify_stream("IONQ", "IonQ", articles))
        assert len(results) == 2
        assert all(r is False for _, _, r in results)

    @patch("qsf.nlp.relevance.OpenAI")
    def test_stream_caps_at_max_articles(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")
        articles = _make_articles(MAX_ARTICLES + 5)
        results = list(RelevanceClassifier().classify_stream("IONQ", "IonQ", articles))
        assert len(results) == MAX_ARTICLES + 5
        assert mock_client.chat.completions.create.call_count == MAX_ARTICLES
        assert all(r is False for _, _, r in results[MAX_ARTICLES:])
        for expected_i, (index, _, _) in enumerate(results[MAX_ARTICLES:], start=MAX_ARTICLES):
            assert index == expected_i

    @patch("qsf.nlp.relevance.OpenAI")
    def test_stream_empty_input_yields_nothing(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        results = list(RelevanceClassifier().classify_stream("IONQ", "IonQ", []))
        assert results == []
        mock_client.chat.completions.create.assert_not_called()
