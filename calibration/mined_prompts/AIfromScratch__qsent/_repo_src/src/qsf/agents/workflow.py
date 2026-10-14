"""
LangGraph pipeline for sentiment analysis.

Graph:
    fetch_market_data → fetch_news → fetch_reddit → score_sentiment → aggregate
"""
import operator
from datetime import datetime
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from qsf.common.logging import get_current_trace, get_logger
from qsf.common.providers import MarketDataProvider, NewsProvider, SentimentModel, SocialProvider
from qsf.common.utils import safe, company_search_name
from qsf.ingestion import YFinanceMarketData, NewsAPIProvider, RedditProvider
from qsf.nlp import FinBERTModel

logger = get_logger(__name__)


class PipelineState(TypedDict, total=False):
    ticker: str
    trace_id: str
    company_name: str
    stock_df: Any
    news_items: list
    reddit_items: list
    scored_items: list
    result: Optional[dict]
    error: Optional[str]
    source_status: Annotated[dict[str, str], operator.or_]


def build_pipeline(
    market: MarketDataProvider,
    news: NewsProvider,
    social: SocialProvider,
    model: SentimentModel,
):
    def _fetch_market_data(state: dict) -> dict:
        ticker = state["ticker"]
        trace = get_current_trace()
        span = trace.span(name="fetch_market_data", input={"ticker": ticker}) if trace else None
        try:
            hist = market.get_history(ticker, "30d")
        except Exception as e:
            msg = f"Failed to fetch market data: {e}"
            if span: span.end(output={"error": str(e)}, level="ERROR")
            return {"error": msg, "source_status": {"market": msg}}
        if hist.empty:
            msg = f"No price data found for '{ticker}'"
            if span: span.end(output={"error": "no price data"}, level="ERROR")
            return {"error": msg, "source_status": {"market": msg}}
        stock_df = hist[["Close", "Volume"]].copy()
        stock_df.index = stock_df.index.date
        stock_df["ror"] = stock_df["Close"].pct_change() * 100
        days = len(stock_df)
        logger.info("[%s] fetch_market_data: %d trading days returned", ticker, days)
        if days < 15:
            logger.warning(
                "[%s] fetch_market_data: only %d trading days returned, expected ~21",
                ticker, days,
            )
        try:
            raw_name = market.get_company_name(ticker)
        except Exception as e:
            msg = f"Failed to fetch market data: {e}"
            if span: span.end(output={"error": str(e)}, level="ERROR")
            return {"error": msg, "source_status": {"market": msg}}
        company_name = company_search_name(raw_name)
        logger.info("[%s] fetch_market_data: company name resolved to '%s'", ticker, company_name)
        if span: span.end(output={"trading_days": days, "company_name": company_name})
        return {"stock_df": stock_df, "company_name": company_name, "source_status": {"market": "ok"}}

    def _fetch_news(state: dict) -> dict:
        if state.get("error"):
            return {"source_status": {"news": "skipped"}}
        ticker = state["ticker"]
        company_name = state.get("company_name", "")
        trace = get_current_trace()
        span = trace.span(name="fetch_news", input={"ticker": ticker, "company_name": company_name}) if trace else None
        try:
            news_items = news.get_articles(ticker, company_name)
        except Exception as e:
            msg = f"Failed to fetch news data: {e}"
            if span: span.end(output={"error": str(e)}, level="ERROR")
            return {"error": msg, "source_status": {"news": msg}}
        count = len(news_items)
        logger.info("[%s] fetch_news: %d articles returned", ticker, count)
        if count == 0:
            logger.warning("[%s] fetch_news: 0 articles returned — NewsAPI may be unavailable or quota exceeded", ticker)
        elif count < 5:
            logger.warning("[%s] fetch_news: only %d articles returned, sentiment coverage may be sparse", ticker, count)
        if span: span.end(output={"count": count})
        return {"news_items": news_items, "source_status": {"news": "ok"}}

    def _fetch_reddit(state: dict) -> dict:
        if state.get("error"):
            return {"source_status": {"reddit": "skipped"}}
        ticker = state["ticker"]
        company_name = state.get("company_name", "")
        trace = get_current_trace()
        span = trace.span(name="fetch_reddit", input={"ticker": ticker, "company_name": company_name}) if trace else None
        try:
            reddit_items = social.get_posts(ticker, company_name)
        except Exception as e:
            msg = f"Failed to fetch Reddit data: {e}"
            if span: span.end(output={"error": str(e)}, level="ERROR")
            return {"error": msg, "source_status": {"reddit": msg}}
        count = len(reddit_items)
        logger.info("[%s] fetch_reddit: %d posts returned", ticker, count)
        for item in reddit_items:
            logger.info("[%s] fetch_reddit: post date=%s text=%s", ticker, item["date"], item["text"][:80])
        if count == 0:
            logger.warning("[%s] fetch_reddit: 0 posts returned — Reddit API may be unavailable or ticker has low social coverage", ticker)
        if span: span.end(output={"count": count})
        return {"reddit_items": reddit_items, "source_status": {"reddit": "ok"}}

    def _score_sentiment(state: dict) -> dict:
        if state.get("error"):
            return {"source_status": {"sentiment": "skipped"}}
        ticker = state["ticker"]
        news_items = state.get("news_items", [])
        reddit_items = state.get("reddit_items", [])
        items = news_items + reddit_items
        logger.info(
            "[%s] score_sentiment: received %d news + %d reddit = %d items from state",
            ticker, len(news_items), len(reddit_items), len(items),
        )
        if not items:
            msg = f"No news or social data found for '{ticker}'"
            return {"error": msg, "source_status": {"sentiment": msg}}
        trace = get_current_trace()
        span = trace.span(name="score_sentiment", input={"news": len(news_items), "social": len(reddit_items)}) if trace else None
        logger.info("[%s] score_sentiment: calling model.score on %d items", ticker, len(items))
        try:
            scores = model.score([item["text"] for item in items])
        except Exception as e:
            msg = f"Failed to score sentiment: {e}"
            if span: span.end(output={"error": str(e)}, level="ERROR")
            return {"error": msg, "source_status": {"sentiment": msg}}
        logger.info("[%s] score_sentiment: model.score returned", ticker)
        for idx, (item, score) in enumerate(zip(items, scores)):
            source_label = "news article" if item["source"] == "news" else "social post"
            if score is None:
                logger.info(
                    "[%s] score_sentiment: item %d/%d failed — %s: %s",
                    ticker, idx + 1, len(items), source_label, item["text"][:80],
                )
            else:
                logger.info(
                    "[%s] score_sentiment: item %d/%d — %.3f | %s: %s",
                    ticker, idx + 1, len(items), score, source_label, item["text"][:80],
                )
        # Filter out None scores (failed items). scores is always same length as items
        # so zip never truncates — reddit items at the tail are no longer silently dropped.
        scored = [
            {**item, "weighted_sentiment": score}
            for item, score in zip(items, scores)
            if score is not None
        ]
        news_scored = sum(1 for s in scored if s["source"] == "news")
        social_scored = sum(1 for s in scored if s["source"] == "social")
        logger.info(
            "[%s] score_sentiment: %d scored after filtering failures — %d news, %d social",
            ticker, len(scored), news_scored, social_scored,
        )
        valid_scores = [s["weighted_sentiment"] for s in scored]
        mean_score = sum(valid_scores) / len(valid_scores)
        logger.info(
            "[%s] score_sentiment: %d items scored — mean=%.3f, min=%.3f, max=%.3f",
            ticker, len(valid_scores), mean_score, min(valid_scores), max(valid_scores),
        )
        out_of_range = [s for s in valid_scores if not -1.0 <= s <= 1.0]
        if out_of_range:
            logger.warning(
                "[%s] score_sentiment: %d scores outside [-1, 1]: %s",
                ticker, len(out_of_range), out_of_range,
            )
        if abs(mean_score) < 0.02:
            logger.warning(
                "[%s] score_sentiment: mean score near zero (%.3f) — model may be returning all-neutral",
                ticker, mean_score,
            )
        if span: span.end(output={"scored": len(scored), "mean": round(mean_score, 4), "failed": len(items) - len(scored)})
        return {"scored_items": scored, "source_status": {"sentiment": "ok"}}

    def _aggregate(state: dict) -> dict:
        if state.get("error"):
            return {}

        ticker = state["ticker"]
        stock_df: pd.DataFrame = state["stock_df"]
        scored_items: list[dict] = state["scored_items"]

        trace = get_current_trace()
        span = trace.span(name="aggregate", input={"scored_items": len(scored_items)}) if trace else None

        df = pd.DataFrame(scored_items)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        news_daily = (
            df[df["source"] == "news"]
            .groupby("date")["weighted_sentiment"]
            .mean()
            .rename("news_sentiment")
        )
        social_daily = (
            df[df["source"] == "social"]
            .groupby("date")["weighted_sentiment"]
            .mean()
            .rename("social_sentiment")
        )

        # Forward-fill across calendar range so weekend/holiday articles
        # carry forward to the next trading day
        all_dates = pd.date_range(
            start=pd.Timestamp(min(stock_df.index)),
            end=pd.Timestamp(max(stock_df.index)),
        ).map(lambda d: d.date())

        total_days = len(stock_df)
        news_coverage = news_daily.reindex(stock_df.index).notna().sum()
        social_coverage = social_daily.reindex(stock_df.index).notna().sum()
        logger.info(
            "[%s] aggregate: news coverage %d/%d trading days, social coverage %d/%d trading days",
            ticker, news_coverage, total_days, social_coverage, total_days,
        )
        if news_coverage == 0:
            logger.warning("[%s] aggregate: no news sentiment aligned to any trading day", ticker)
        if social_coverage == 0:
            logger.warning("[%s] aggregate: no social sentiment aligned to any trading day", ticker)
            if not social_daily.empty:
                logger.warning(
                    "[%s] aggregate: social posts exist (dates: %s) but none overlap trading days (%s to %s)",
                    ticker,
                    sorted(social_daily.index.tolist()),
                    min(stock_df.index),
                    max(stock_df.index),
                )

        news_aligned = news_daily.reindex(all_dates).ffill().reindex(stock_df.index).fillna(0)
        news_aligned.name = "news_sentiment"
        social_aligned = social_daily.reindex(all_dates).ffill().reindex(stock_df.index).fillna(0)
        social_aligned.name = "social_sentiment"

        merged = stock_df.copy()
        merged["news_sentiment"] = news_aligned
        merged["social_sentiment"] = social_aligned

        # Overall scores
        news_mean = news_daily.mean()
        social_mean = social_daily.mean()
        overall_news = float(news_mean) if (not news_daily.empty and pd.notna(news_mean)) else 0.0
        overall_social = float(social_mean) if (not social_daily.empty and pd.notna(social_mean)) else 0.0
        overall_sentiment = round((overall_news + overall_social) / 2, 4)

        # Trend
        sorted_df = df.sort_values("date")
        recent_mean = sorted_df.tail(7)["weighted_sentiment"].mean()
        older_mean = sorted_df.head(7)["weighted_sentiment"].mean()
        if recent_mean > older_mean + 0.05:
            trend = "increasing"
        elif recent_mean < older_mean - 0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        result = {
            "ticker": ticker,
            "last_updated": datetime.now().isoformat(),
            "sentiment_score": safe(overall_sentiment),
            "data_points": len(scored_items),
            "breakdown": {
                "news_sentiment": safe(overall_news),
                "social_sentiment": safe(overall_social),
                "trend": trend,
            },
            "daily_data": [
                {
                    "date": str(idx),
                    "close": safe(row["Close"], 2),
                    "volume": int(row["Volume"]),
                    "ror": safe(row["ror"]),
                    "news_sentiment": safe(row["news_sentiment"]),
                    "social_sentiment": safe(row["social_sentiment"]),
                }
                for idx, row in merged.iterrows()
            ],
        }
        if span: span.end(output={"sentiment_score": safe(overall_sentiment), "trend": trend, "data_points": len(scored_items)})
        return {"result": result}

    graph = StateGraph(PipelineState)

    graph.add_node("fetch_market_data", _fetch_market_data)
    graph.add_node("fetch_news", _fetch_news)
    graph.add_node("fetch_reddit", _fetch_reddit)
    graph.add_node("score_sentiment", _score_sentiment)
    graph.add_node("aggregate", _aggregate)

    graph.set_entry_point("fetch_market_data")
    graph.add_edge("fetch_market_data", "fetch_news")
    graph.add_edge("fetch_news", "fetch_reddit")
    graph.add_edge("fetch_reddit", "score_sentiment")
    graph.add_edge("score_sentiment", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


pipeline = build_pipeline(
    market=YFinanceMarketData(),
    news=NewsAPIProvider(),
    social=RedditProvider(),
    model=FinBERTModel(),
)
