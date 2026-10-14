"""
NewsAPI news provider.
"""
import os
from datetime import datetime, timedelta

from newsapi import NewsApiClient

from qsf.common.logging import get_current_trace


def news_from_date(now: datetime, days: int = 28) -> str:
    """Return the earliest datetime string to pass to NewsAPI for `days` of history.

    Defaults to 28 days to stay safely within the NewsAPI free plan's 30-day
    exclusive rolling window.
    """
    return (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


class NewsAPIProvider:
    def get_articles(self, ticker: str, company_name: str = "", days: int = 28) -> list[dict]:
        client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
        from_date = news_from_date(datetime.now(), days)
        query = f'"{company_name}"' if company_name else ticker

        trace = get_current_trace()
        span = trace.span(name="newsapi.get_articles", input={"query": query, "days": days}) if trace else None
        response = client.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=100,
        )
        articles = [
            {
                "text": _article_text(a),
                "date": a["publishedAt"][:10],
                "source": "news",
            }
            for a in response.get("articles", [])
            if a.get("title")
        ]
        if span: span.end(output={"count": len(articles)})
        return articles


def _article_text(article: dict) -> str:
    """Combine headline and description for richer FinBERT input.

    NewsAPI free plan does not provide full article body, so we concatenate
    the title and description to give the model more context than a headline alone.
    """
    title = article.get("title", "")
    description = article.get("description") or ""
    if description:
        return f"{title}. {description}"
    return title
