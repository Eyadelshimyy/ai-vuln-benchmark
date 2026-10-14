"""
Reddit social media provider.
"""
import os
from datetime import datetime, timedelta

import praw

from qsf.common.logging import get_current_trace, get_logger

logger = get_logger(__name__)

REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets", "Superstonk", "StockMarket", "QuantumComputing"]
REDDIT_MAX_COMMENTS = 5       # top N comments by upvotes to include per post
REDDIT_MAX_CHARS = 1800       # ~450 BERT tokens — hard cap on combined text per post


def _post_text(post) -> str:
    """Build a single text block from a Reddit post for FinBERT scoring.

    Combines title, selftext body, and top comments by upvote count into one
    call. This gives FinBERT richer context than the title alone, which is
    typically too short to produce non-neutral sentiment.

    replace_more(limit=0) discards MoreComments placeholders so we only
    iterate over real comment objects. Comments are sorted by upvote score
    descending — highest community-validated opinions first. If the combined
    text exceeds REDDIT_MAX_CHARS (~450 BERT tokens), the lowest-upvoted
    comment is dropped until it fits.
    """
    parts = [post.title]

    if post.selftext and post.selftext.strip():
        parts.append(post.selftext.strip())

    post.comments.replace_more(limit=0)
    top_comments = sorted(
        post.comments,
        key=lambda c: getattr(c, "score", 0),
        reverse=True,
    )[:REDDIT_MAX_COMMENTS]
    for comment in top_comments:
        body = getattr(comment, "body", "").strip()
        if body:
            parts.append(body)

    text = ". ".join(parts)
    # Drop lowest-upvoted comments until within character limit
    while len(text) > REDDIT_MAX_CHARS and len(parts) > 1:
        parts.pop()
        text = ". ".join(parts)
    logger.info(
        "_post_text: title=%s | parts=%d (title+selftext+comments) | chars=%d",
        post.title[:60], len(parts), len(text),
    )
    return text


class RedditProvider:
    def get_posts(self, ticker: str, company_name: str = "", days: int = 30) -> list[dict]:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "qsent/0.1"),
        )
        query = f'"{company_name}" OR "${ticker}"' if company_name else f"${ticker}"
        cutoff = datetime.now() - timedelta(days=days)
        trace = get_current_trace()
        posts = []
        for subreddit in REDDIT_SUBREDDITS:
            span = trace.span(name="reddit.search", input={"subreddit": subreddit, "query": query}) if trace else None
            count = 0
            for post in reddit.subreddit(subreddit).search(
                query, sort="new", time_filter="month", limit=50
            ):
                created = datetime.fromtimestamp(post.created_utc)
                if created >= cutoff:
                    posts.append({
                        "text": _post_text(post),
                        "date": created.strftime("%Y-%m-%d"),
                        "source": "social",
                    })
                    count += 1
            if span: span.end(output={"count": count})
        return posts
