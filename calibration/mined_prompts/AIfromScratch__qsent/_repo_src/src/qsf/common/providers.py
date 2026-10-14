"""
Protocol abstractions for external data providers and ML models.

These protocols define the interface boundary. Swapping a provider means
writing one new class that satisfies the protocol — nothing else changes.
"""
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_history(self, ticker: str, period: str) -> pd.DataFrame: ...
    def get_company_name(self, ticker: str) -> str: ...


@runtime_checkable
class NewsProvider(Protocol):
    def get_articles(self, ticker: str, company_name: str = "", days: int = 28) -> list[dict]: ...
    # each dict: {"text": str, "date": "YYYY-MM-DD", "source": "news"}


@runtime_checkable
class SocialProvider(Protocol):
    def get_posts(self, ticker: str, company_name: str = "", days: int = 30) -> list[dict]: ...
    # each dict: {"text": str, "date": "YYYY-MM-DD", "source": "social"}


@runtime_checkable
class SentimentModel(Protocol):
    def score(self, texts: list[str]) -> list[float | None]: ...
    # Always same length as input. None at an index means that item failed to score.
