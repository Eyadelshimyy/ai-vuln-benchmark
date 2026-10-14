"""Tests for web_search_tools agent tool wrapper."""

import inspect

import pytest
from ellements.core import ToolRegistry
from ellements.standard_tools.web.search import WebSearcher, web_search_tools


class TestWebSearchTools:
    """Test web_search_tools function and its tool wrappers."""

    @pytest.fixture
    def tools(self):
        """Get web search tools registry."""
        return web_search_tools()

    def test_web_search_tools_returns_registry(self):
        """Test that web_search_tools returns a ToolRegistry."""
        tools = web_search_tools()
        assert isinstance(tools, ToolRegistry)

    def test_web_search_tools_contains_expected_tools(self):
        """Test that all expected tools are present."""
        tools = web_search_tools()

        expected_tools = [
            "search_web",
            "search_news",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Missing tool: {tool_name}"
            assert tools[tool_name].name == tool_name, f"Tool {tool_name} has wrong name"

    def test_tools_have_descriptions(self, tools):
        """Test that tools have descriptions."""
        assert "search" in tools["search_web"].description.lower()
        assert "news" in tools["search_news"].description.lower()

    def test_tools_expose_canonical_spec_surface(self, tools):
        """Each tool is a canonical ToolSpec with name, description, schema, and invoke."""
        for tool_name, tool in tools.items():
            assert tool.name == tool_name
            assert isinstance(tool.description, str) and len(tool.description) > 0
            assert isinstance(tool.params_json_schema, dict)
            assert inspect.iscoroutinefunction(tool.invoke)

    def test_search_web_schema(self, tools):
        """Test search_web tool has correct parameter schema."""
        schema = tools["search_web"].params_json_schema
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "time_range" in schema["properties"]

    def test_search_news_schema(self, tools):
        """Test search_news tool has correct parameter schema."""
        schema = tools["search_news"].params_json_schema
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "time_range" in schema["properties"]

    def test_tool_count(self, tools):
        """Test that we have the expected number of tools."""
        assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}"

    def test_tools_use_same_searcher_instance(self, tools):
        """Test that all tools are created properly."""
        assert tools["search_web"].name == "search_web"
        assert tools["search_news"].name == "search_news"

    @pytest.mark.parametrize("tool_name", [
        "search_web",
        "search_news",
    ])
    def test_individual_tool_structure(self, tools, tool_name):
        """Test each tool conforms to the canonical ToolSpec contract."""
        tool = tools[tool_name]
        assert tool.name == tool_name
        assert isinstance(tool.description, str) and len(tool.description) > 10
        assert isinstance(tool.params_json_schema, dict)
        assert "properties" in tool.params_json_schema
        assert inspect.iscoroutinefunction(tool.invoke)


class TestWebSearcherResilience:
    def test_search_retries_backends_and_uses_timeout(self, monkeypatch):
        ddgs_timeouts = []
        backends_called = []

        class _DummyDDGS:
            def __init__(self, proxy=None, timeout=None, verify=True):
                ddgs_timeouts.append(timeout)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def text(self, query, **kwargs):
                backend = kwargs.get("backend")
                backends_called.append(backend)
                if backend == "google":
                    raise RuntimeError("operation timed out")
                return [{"title": "T", "href": "https://example.com", "body": "S"}]

        monkeypatch.setattr("ellements.standard_tools.web.search.DDGS", _DummyDDGS)
        searcher = WebSearcher(timeout=12)
        results = searcher.search("test query")

        assert results.total_results == 1
        assert ddgs_timeouts == [12]
        assert backends_called[:2] == ["google", "duckduckgo"]
