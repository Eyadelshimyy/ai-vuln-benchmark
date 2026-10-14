"""
Massive.com (formerly Polygon.io) news provider.
Supports ticker-based filtering via the reference/news endpoint.
"""
import os
from datetime import datetime, timedelta

import requests


class MassiveNewsProvider:
    def get_articles(self, ticker: str, company_name: str = "", days: int = 28) -> list[dict]:
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            raise ValueError("MASSIVE_API_KEY not set")

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        params = {
            "ticker": ticker,
            "limit": 50,
            "published_utc.gte": from_date,
            "apiKey": api_key,
        }
        response = requests.get("https://api.massive.com/v2/reference/news", params=params)
        response.raise_for_status()
        data = response.json()

        return [
            {
                "text": _article_text(a),
                "date": a["published_utc"][:10],
                "source": "news",
            }
            for a in data.get("results", [])
            if a.get("title")
        ]


def _article_text(article: dict) -> str:
    title = article.get("title", "")
    description = article.get("description") or ""
    if description:
        return f"{title}. {description}"
    return title
