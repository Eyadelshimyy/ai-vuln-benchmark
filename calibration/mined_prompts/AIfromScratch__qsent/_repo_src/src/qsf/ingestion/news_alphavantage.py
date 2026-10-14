"""
Alpha Vantage news sentiment provider.
Supports ticker-based filtering via the NEWS_SENTIMENT function.
"""
import os
from datetime import datetime, timedelta

import requests


class AlphaVantageNewsProvider:
    def get_articles(self, ticker: str, company_name: str = "", days: int = 28) -> list[dict]:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set")

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%dT%H%M")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 50,
            "time_from": from_date,
            "apikey": api_key,
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        data = response.json()

        if "feed" not in data:
            raise ValueError(f"Alpha Vantage quota exceeded or invalid response: {data}")

        return [
            {
                "text": _article_text(a),
                "date": _parse_date(a["time_published"]),
                "source": "news",
            }
            for a in data["feed"]
            if a.get("title")
        ]


def _parse_date(raw: str) -> str:
    """Parse Alpha Vantage date format '20260210T143022' -> '2026-02-10'."""
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def _article_text(article: dict) -> str:
    title = article.get("title", "")
    summary = article.get("summary") or ""
    if summary:
        return f"{title}. {summary}"
    return title
