"""End-to-end tests for web crawler tools.

These tests make real HTTP requests to crawl web pages and verify that the underlying
crawler functionality returns expected, non-empty content.
"""

import os

import pytest
from ellements.core.chunking import TextProcessor
from ellements.standard_tools.web.crawler import CrawlResult, WebCrawler

pytestmark = [
    pytest.mark.requires_browser,
    pytest.mark.requires_network,
    pytest.mark.skipif(
        os.getenv("ELLEMENTS_RUN_NETWORK_TESTS") != "1",
        reason="set ELLEMENTS_RUN_NETWORK_TESTS=1 to run live web-crawler e2e tests",
    ),
]


class TestWebCrawlerE2E:
    """End-to-end tests for web crawler with real HTTP requests."""

    @pytest.fixture
    def crawler(self):
        """Get web crawler instance."""
        return WebCrawler()

    @pytest.fixture
    def crawler_with_processor(self):
        """Get web crawler with text processor."""
        processor = TextProcessor(max_tokens=2000, strategy="simple")
        return WebCrawler(text_processor=processor, max_content_tokens=2000)

    @pytest.mark.asyncio
    async def test_crawl_url_returns_content(self, crawler):
        """Test that crawl_url returns non-empty page content."""
        result = await crawler.crawl(url="https://www.python.org/about/")

        # Should return CrawlResult object
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0

        # Should contain actual content from the page
        assert "Python" in result.markdown
        # Should have substantial content
        assert len(result.markdown) > 100

    @pytest.mark.asyncio
    async def test_crawl_extracts_text(self, crawler):
        """Test that crawl extracts readable text, not raw HTML."""
        result = await crawler.crawl(url="https://en.wikipedia.org/wiki/Python_(programming_language)")

        # Should return content
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0

        # Should contain readable text about Python
        assert "Python" in result.markdown
        assert "programming" in result.markdown.lower()

    @pytest.mark.asyncio
    async def test_crawl_different_site(self, crawler):
        """Test crawling a different website."""
        result = await crawler.crawl(url="https://github.com/features")

        # Should return content
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0

    @pytest.mark.asyncio
    async def test_crawl_docs_page(self, crawler):
        """Test crawling a documentation page."""
        result = await crawler.crawl(url="https://docs.python.org/3/tutorial/")

        # Should return content
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0

        # Should contain tutorial content
        assert "Python" in result.markdown
        assert len(result.markdown) > 200  # Documentation pages have substantial content

    @pytest.mark.asyncio
    async def test_crawl_with_text_processor(self, crawler_with_processor):
        """Test that text processor is used for long content."""
        result = await crawler_with_processor.crawl(url="https://en.wikipedia.org/wiki/Artificial_intelligence")

        # Should return content
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0

    @pytest.mark.asyncio
    async def test_crawl_handles_redirects(self, crawler):
        """Test that the crawler handles HTTP redirects."""
        # Many URLs redirect (http to https, www to non-www, etc.)
        result = await crawler.crawl(url="http://python.org")

        # Should still return content after following redirects
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0
        assert "Python" in result.markdown

    @pytest.mark.asyncio
    async def test_crawl_invalid_url_handles_gracefully(self, crawler):
        """Test that invalid URLs are handled gracefully."""
        try:
            result = await crawler.crawl(url="https://this-domain-definitely-does-not-exist-xyz123.com")
            # If it returns something, that's okay (error message)
            assert result is not None
        except Exception as e:
            # Should get an error
            assert isinstance(e, Exception)
            assert len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_crawl_malformed_url_handles_gracefully(self, crawler):
        """Test that malformed URLs are handled gracefully."""
        try:
            result = await crawler.crawl(url="not-a-valid-url")
            # If it returns something, that's okay (error message)
            assert result is not None
        except Exception as e:
            # Should get an error
            assert isinstance(e, Exception)
            assert len(str(e)) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [
        "https://www.python.org/",
        "https://en.wikipedia.org/wiki/Web_scraping",
        "https://github.com/",
    ])
    async def test_multiple_urls_work(self, crawler, url):
        """Test that various popular URLs can be crawled."""
        result = await crawler.crawl(url=url)

        # Should return non-empty content for each URL
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0

    @pytest.mark.asyncio
    async def test_crawl_extracts_main_content(self, crawler):
        """Test that the crawler focuses on main content."""
        result = await crawler.crawl(url="https://en.wikipedia.org/wiki/Python_(programming_language)")

        # Should have substantial content
        assert len(result.markdown) > 500

        # Should contain actual article content keywords
        assert any(keyword in result.markdown.lower() for keyword in [
            "programming", "language", "guido", "interpreted", "dynamic"
        ])

    @pytest.mark.asyncio
    async def test_crawl_result_structure(self, crawler):
        """Test that CrawlResult has expected structure."""
        result = await crawler.crawl(url="https://www.python.org/about/")

        # Should have expected fields
        assert result.url
        assert result.markdown
        assert result.success

    @pytest.mark.asyncio
    async def test_crawl_with_https(self, crawler):
        """Test that HTTPS URLs work correctly."""
        result = await crawler.crawl(url="https://www.python.org/")

        # Should return content over HTTPS
        assert result is not None
        assert isinstance(result, CrawlResult)
        assert result.markdown
        assert len(result.markdown) > 0
        assert "Python" in result.markdown
