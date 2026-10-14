"""Tests for web_crawler_tools agent tool wrapper."""

import inspect

import pytest
from ellements.core import ToolRegistry
from ellements.core.chunking import TextProcessor
from ellements.standard_tools.web.crawler import web_crawler_tools


class TestWebCrawlerTools:
    """Test web_crawler_tools function and its tool wrappers."""

    @pytest.fixture
    def tools(self):
        """Get web crawler tools registry."""
        return web_crawler_tools()

    @pytest.fixture
    def tools_with_processor(self):
        """Get web crawler tools with text processor."""
        processor = TextProcessor(max_tokens=1000, strategy="simple")
        return web_crawler_tools(text_processor=processor)

    def test_web_crawler_tools_returns_registry(self):
        """Test that web_crawler_tools returns a ToolRegistry."""
        tools = web_crawler_tools()
        assert isinstance(tools, ToolRegistry)

    def test_web_crawler_tools_contains_expected_tools(self):
        """Test that all expected tools are present."""
        tools = web_crawler_tools()

        expected_tools = [
            "crawl_url",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Missing tool: {tool_name}"
            assert tools[tool_name].name == tool_name, f"Tool {tool_name} has wrong name"

    def test_tools_have_descriptions(self, tools):
        """Test that tools have descriptions."""
        assert "crawl" in tools["crawl_url"].description.lower()

    def test_tools_expose_canonical_spec_surface(self, tools):
        """Each tool is a canonical ToolSpec with name, description, schema, and invoke."""
        for tool_name, tool in tools.items():
            assert tool.name == tool_name
            assert isinstance(tool.description, str) and len(tool.description) > 0
            assert isinstance(tool.params_json_schema, dict)
            assert inspect.iscoroutinefunction(tool.invoke)

    def test_crawl_url_schema(self, tools):
        """Test crawl_url tool has correct parameter schema."""
        schema = tools["crawl_url"].params_json_schema
        assert "properties" in schema
        assert "url" in schema["properties"]

    def test_tools_with_max_content_tokens(self):
        """Test creating tools with max_content_tokens parameter."""
        tools = web_crawler_tools(max_content_tokens=500)
        assert "crawl_url" in tools
        assert tools["crawl_url"].name == "crawl_url"

    def test_tools_with_text_processor(self, tools_with_processor):
        """Test creating tools with explicit text processor."""
        assert "crawl_url" in tools_with_processor
        assert tools_with_processor["crawl_url"].name == "crawl_url"

    def test_tool_documentation(self, tools):
        """Test that all tools have proper docstrings."""
        for tool_name, tool in tools.items():
            assert len(tool.description) > 10, f"Tool {tool_name} has insufficient documentation"

    def test_tool_count(self, tools):
        """Test that we have the expected number of tools."""
        assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}"

    @pytest.mark.parametrize("tool_name", [
        "crawl_url",
    ])
    def test_individual_tool_structure(self, tools, tool_name):
        """Test each tool conforms to the canonical ToolSpec contract."""
        tool = tools[tool_name]
        assert tool.name == tool_name
        assert isinstance(tool.description, str) and len(tool.description) > 10
        assert isinstance(tool.params_json_schema, dict)
        assert "properties" in tool.params_json_schema
        assert inspect.iscoroutinefunction(tool.invoke)

    @pytest.mark.asyncio
    async def test_crawl_url_handles_runtime_loop_errors(self, monkeypatch):
        """Tool should return a failure string instead of crashing on crawler errors."""
        tools = web_crawler_tools()

        async def _raise_loop_error(self, url, **kwargs):
            raise RuntimeError("Can't patch loop of type <class 'uvloop.Loop'>")

        monkeypatch.setattr(
            "ellements.standard_tools.web.crawler.WebCrawler.crawl",
            _raise_loop_error,
        )

        async def _noop(self):
            return None

        monkeypatch.setattr(
            "ellements.standard_tools.web.crawler.WebCrawler.start",
            _noop,
        )
        monkeypatch.setattr(
            "ellements.standard_tools.web.crawler.WebCrawler.close",
            _noop,
        )

        result = await tools["crawl_url"].invoke(url="https://example.com")
        assert "Failed to crawl https://example.com" in str(result)
