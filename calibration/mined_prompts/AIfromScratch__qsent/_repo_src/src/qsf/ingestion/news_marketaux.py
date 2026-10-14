"""
Marketaux news provider.
Supports ticker-based filtering via filter_entities parameter.
"""
import os
from datetime import datetime, timedelta

import requests


class MarketauxNewsProvider:
    def get_articles(self, ticker: str, company_name: str = "", days: int = 28) -> list[dict]:
        api_key = os.getenv("MARKETAUX_API_KEY")
        if not api_key:
            raise ValueError("MARKETAUX_API_KEY not set")

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        params = {
            "symbols": ticker,
            "filter_entities": "true",
            "limit": 50,
            "published_after": from_date,
            "api_token": api_key,
        }
        response = requests.get("https://api.marketaux.com/v1/news/all", params=params)
        response.raise_for_status()
        data = response.json()

        return [
            {
                "text": _article_text(a),
                "date": a["published_at"][:10],
                "source": "news",
            }
            for a in data.get("data", [])
            if a.get("title")
        ]


def _article_text(article: dict) -> str:
    title = article.get("title", "")
    description = article.get("description") or ""
    if description:
        return f"{title}. {description}"
    return title
