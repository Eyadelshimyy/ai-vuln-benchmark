"""AI-optimized web crawling using Crawl4AI."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urljoin

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import CosineStrategy, LLMExtractionStrategy
from ellements.core import ToolRegistry
from ellements.core.exceptions import LLMError
from pydantic import BaseModel, Field


class CrawlResult(BaseModel):
    """Represents the result of crawling a single URL."""

    url: str = Field(description="The URL that was crawled")
    title: str = Field(default="", description="Page title")
    markdown: str = Field(default="", description="Clean markdown content")
    html: str = Field(default="", description="Raw HTML content")
    links: list[str] = Field(default_factory=list, description="Extracted links")
    images: list[str] = Field(default_factory=list, description="Extracted image URLs")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    success: bool = Field(default=True, description="Whether crawl was successful")
    error_message: str = Field(default="", description="Error message if failed")


class ExtractionResult(BaseModel):
    """Represents structured data extracted from crawled content."""

    url: str = Field(description="Source URL")
    extracted_data: Any = Field(description="Extracted structured data")
    extraction_method: str = Field(description="Method used for extraction")
    confidence: float | None = Field(
        default=None, description="Confidence score if available"
    )


class WebCrawler:
    """AI-optimized web crawler using Crawl4AI."""

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        user_agent: str | None = None,
        proxy: str | None = None,
        timeout: int = 30,
        delay: float = 1.0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the web crawler.

        Args:
            headless: Run browser in headless mode
            browser_type: Browser type ('chromium', 'firefox', 'webkit')
            user_agent: Custom user agent string
            proxy: Proxy configuration
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
            verbose: Show Crawl4AI progress output
            **kwargs: Additional configuration for AsyncWebCrawler
        """
        self.headless = headless
        self.browser_type = browser_type
        self.user_agent = user_agent
        self.proxy = proxy
        self.timeout = timeout
        self.delay = delay
        self.verbose = verbose
        self.config = kwargs
        self._crawler: AsyncWebCrawler | None = None

    async def __aenter__(self) -> WebCrawler:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def start(self) -> None:
        """Start the crawler."""
        if self._crawler is None:
            browser_config = BrowserConfig(
                headless=self.headless,
                browser_type=self.browser_type,
                proxy=self.proxy,
                verbose=self.verbose,
            )
            self._crawler = AsyncWebCrawler(config=browser_config, **self.config)
            await self._crawler.start()

    async def close(self) -> None:
        """Close the crawler."""
        if self._crawler is not None:
            await self._crawler.close()
            self._crawler = None

    async def crawl(
        self,
        url: str,
        include_raw_html: bool = False,
        extract_links: bool = True,
        extract_images: bool = True,
        css_selector: str | None = None,
        wait_for: str | None = None,
        screenshot: bool = False,
        **kwargs: Any,
    ) -> CrawlResult:
        """Crawl a single URL and extract content.

        Args:
            url: URL to crawl
            include_raw_html: Include raw HTML in result
            extract_links: Extract all links from the page
            extract_images: Extract all image URLs
            css_selector: CSS selector to extract specific content
            wait_for: CSS selector or JavaScript condition to wait for
            screenshot: Take a screenshot of the page
            **kwargs: Additional parameters for crawl4ai

        Returns:
            CrawlResult containing the crawled data
        """
        if self._crawler is None:
            await self.start()

        try:
            # Configure crawl parameters
            crawl_params = {
                "css_selector": css_selector,
                "wait_for": wait_for,
                "screenshot": screenshot,
                **kwargs,
            }

            # Remove None values
            crawl_params = {k: v for k, v in crawl_params.items() if v is not None}

            # Build run config with verbosity setting
            run_config = CrawlerRunConfig(
                verbose=self.verbose,
                log_console=self.verbose,
                **crawl_params,
            )

            # Perform the crawl
            assert self._crawler is not None
            result = await self._crawler.arun(url=url, config=run_config)

            # Extract links if requested
            links = []
            if extract_links and hasattr(result, "links") and result.links:
                for link in result.links.get("internal", []):
                    href = (
                        link.get("href", link) if isinstance(link, dict) else str(link)
                    )
                    if href:
                        links.append(urljoin(url, href))
                for link in result.links.get("external", []):
                    href = (
                        link.get("href", link) if isinstance(link, dict) else str(link)
                    )
                    if href:
                        links.append(href)

            # Extract images if requested
            images = []
            if extract_images and hasattr(result, "media") and result.media:
                for img in result.media.get("images", []):
                    src = img.get("src", img) if isinstance(img, dict) else str(img)
                    if src:
                        images.append(urljoin(url, src))

            # Build metadata
            metadata = {}
            if hasattr(result, "metadata") and result.metadata:
                metadata = result.metadata

            return CrawlResult(
                url=url,
                title=getattr(result, "title", "") or "",
                markdown=getattr(result, "markdown", "") or "",
                html=(getattr(result, "html", "") or "") if include_raw_html else "",
                links=links,
                images=images,
                metadata=metadata,
                success=True,
            )

        except Exception as e:
            return CrawlResult(url=url, success=False, error_message=str(e))

    async def crawl_multiple(
        self,
        urls: list[str],
        max_concurrent: int = 3,
        include_raw_html: bool = False,
        extract_links: bool = True,
        extract_images: bool = True,
        **kwargs: Any,
    ) -> list[CrawlResult]:
        """Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent crawls
            include_raw_html: Include raw HTML in results
            extract_links: Extract links from pages
            extract_images: Extract image URLs
            **kwargs: Additional crawl parameters

        Returns:
            List of CrawlResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                return await self.crawl(
                    url=url,
                    include_raw_html=include_raw_html,
                    extract_links=extract_links,
                    extract_images=extract_images,
                    **kwargs,
                )

        tasks = [crawl_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def extract_structured_data(
        self,
        url: str,
        extraction_strategy: str = "llm",
        schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        llm_model: str | None = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract structured data from a URL using AI.

        Args:
            url: URL to extract data from
            extraction_strategy: Strategy to use ('llm', 'cosine')
            schema: JSON schema for structured extraction
            instruction: Extraction instruction for LLM
            llm_model: LLM model to use for extraction
            **kwargs: Additional parameters

        Returns:
            ExtractionResult with structured data
        """
        if self._crawler is None:
            await self.start()

        try:
            # Configure extraction strategy
            if extraction_strategy == "llm" and (instruction or schema):
                strategy = LLMExtractionStrategy(
                    provider="ollama",  # Can be configured
                    api_token=None,
                    instruction=instruction,
                    schema=schema,
                    **kwargs,
                )
            elif extraction_strategy == "cosine":
                strategy = CosineStrategy(
                    semantic_filter=instruction or "relevant content", **kwargs
                )
            else:
                # Default to simple crawl
                strategy = None

            # Perform extraction
            assert self._crawler is not None
            if strategy:
                result = await self._crawler.arun(url=url, extraction_strategy=strategy)
                extracted_data = getattr(result, "extracted_content", None)
            else:
                result = await self._crawler.arun(url=url)
                extracted_data = getattr(result, "markdown", "")

            return ExtractionResult(
                url=url,
                extracted_data=extracted_data,
                extraction_method=extraction_strategy,
                confidence=getattr(result, "confidence", None),
            )

        except Exception as e:
            raise LLMError(f"Structured extraction failed for {url}: {e}") from e

    async def crawl_with_pagination(
        self,
        base_url: str,
        max_pages: int = 5,
        pagination_selector: str | None = None,
        **kwargs: Any,
    ) -> list[CrawlResult]:
        """Crawl a paginated website.

        Args:
            base_url: Starting URL
            max_pages: Maximum pages to crawl
            pagination_selector: CSS selector for next page link
            **kwargs: Additional crawl parameters

        Returns:
            List of results from all pages
        """
        results = []
        current_url = base_url

        for _page_num in range(max_pages):
            result = await self.crawl(current_url, **kwargs)
            results.append(result)

            if not result.success:
                break

            # Find next page URL if pagination selector provided
            if pagination_selector and result.links:
                # Simple logic - would need more sophisticated parsing
                # for complex pagination patterns
                next_urls = [
                    link
                    for link in result.links
                    if "next" in link.lower() or "page" in link.lower()
                ]
                if not next_urls:
                    break
                current_url = next_urls[0]
            else:
                break

        return results

    async def smart_crawl(
        self,
        url: str,
        content_type: str = "article",
        extract_main_content: bool = True,
        remove_noise: bool = True,
        **kwargs: Any,
    ) -> CrawlResult:
        """Intelligent crawling with content optimization.

        Args:
            url: URL to crawl
            content_type: Type of content expected ('article', 'product', 'blog')
            extract_main_content: Focus on main content area
            remove_noise: Remove ads, navigation, etc.
            **kwargs: Additional parameters

        Returns:
            Optimized CrawlResult
        """
        # Configure smart extraction based on content type
        smart_params: dict[str, Any] = {}

        if content_type == "article":
            smart_params["css_selector"] = "article, .content, .post, main"
        elif content_type == "product":
            smart_params["css_selector"] = ".product, .item, .listing"
        elif content_type == "blog":
            smart_params["css_selector"] = ".post, .entry, .blog-content"

        if remove_noise:
            smart_params["excluded_tags"] = ["nav", "footer", "aside", "advertisement"]

        return await self.crawl(url=url, **{**smart_params, **kwargs})


# Agent Tools
# ---------------------------------------------------------------------------


def web_crawler_tools(
    text_processor: Any | None = None,
    max_content_tokens: int | None = None,
) -> ToolRegistry:
    """Create web crawler tools for agents.

    These tools are returned as plain async callables. Framework-specific
    backends adapt them when needed.

    Args:
        text_processor: Optional TextProcessor for handling long crawled content.
                       If not provided but max_content_tokens is set,
                       a default processor will be created.
        max_content_tokens: Maximum tokens for crawled content. If None, no truncation.

    Returns:
        Dict of plain tool callables ready for adaptation by agent frameworks

    Example:
        >>> from ellements.core.chunking import TextProcessor
        >>> processor = TextProcessor(max_tokens=4000)
        >>> tools = web_crawler_tools(text_processor=processor)
        >>> agent = AgentBuilder("WebCrawler").with_tools(tools).build()
    """

    if text_processor is None and max_content_tokens is not None:
        from ellements.core.chunking import TextProcessor

        text_processor = TextProcessor(
            max_tokens=max_content_tokens,
            strategy="simple",
        )

    crawler = WebCrawler()

    async def crawl_url(url: str) -> str:
        """Crawl a web page and extract its main content as markdown.

        This function crawls the specified URL and returns clean markdown content,
        automatically handling long content according to the configured text processing strategy.

        Args:
            url: URL of the web page to crawl

        Returns:
            Markdown content from the web page (truncated if configured)
        """
        try:
            async with crawler:
                result = await crawler.crawl(url)
        except Exception as exc:
            return f"Failed to crawl {url}: {exc}"

        if not result.success:
            return f"Failed to crawl {url}: {result.error_message}"

        content = result.markdown

        if text_processor is not None and content:
            content = text_processor.process(content)

        return content

    return ToolRegistry.from_mapping(
        {
            "crawl_url": crawl_url,
        }
    )
