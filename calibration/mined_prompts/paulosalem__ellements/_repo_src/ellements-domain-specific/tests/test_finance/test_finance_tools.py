"""Tests for finance_tools agent tool wrapper."""

import inspect

import pytest
from ellements.core import ToolRegistry
from ellements.domain_specific.finance.yahoo_finance import finance_tools


class TestFinanceTools:
    """Test finance_tools function and its tool wrappers."""

    @pytest.fixture
    def tools(self):
        """Get finance tools registry."""
        return finance_tools()

    def test_finance_tools_returns_registry(self):
        """Test that finance_tools returns a ToolRegistry."""
        tools = finance_tools()
        assert isinstance(tools, ToolRegistry)

    def test_finance_tools_contains_expected_tools(self):
        """Test that all expected tools are present."""
        tools = finance_tools()

        expected_tools = [
            "search_asset",
            "get_asset_quote",
            "get_asset_profile",
            "get_financial_metrics",
            "get_income_statement",
            "get_balance_sheet",
            "get_cash_flow",
            "get_analyst_recommendations",
            "get_technical_indicators",
            "get_technical_chart",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Missing tool: {tool_name}"
            assert tools[tool_name].name == tool_name, f"Tool {tool_name} has wrong name"

    def test_tools_have_descriptions(self, tools):
        """Test that tools have descriptions."""
        assert "asset" in tools["search_asset"].description.lower()
        assert "quote" in tools["get_asset_quote"].description.lower()
        assert "profile" in tools["get_asset_profile"].description.lower()
        assert "metric" in tools["get_financial_metrics"].description.lower()

    def test_tools_expose_canonical_spec_surface(self, tools):
        """Each tool is a canonical ToolSpec with name, description, schema, and invoke."""
        for tool_name, tool in tools.items():
            assert tool.name == tool_name
            assert isinstance(tool.description, str) and len(tool.description) > 0
            assert isinstance(tool.params_json_schema, dict)
            assert inspect.iscoroutinefunction(tool.invoke)

    def test_search_asset_schema(self, tools):
        """Test search_asset tool has correct parameter schema."""
        schema = tools["search_asset"].params_json_schema
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]

    def test_get_asset_quote_schema(self, tools):
        """Test get_asset_quote tool has correct parameter schema."""
        schema = tools["get_asset_quote"].params_json_schema
        assert "properties" in schema
        assert "symbol" in schema["properties"]

    def test_get_financial_metrics_schema(self, tools):
        """Test get_financial_metrics tool has correct parameter schema."""
        schema = tools["get_financial_metrics"].params_json_schema
        assert "properties" in schema
        assert "symbol" in schema["properties"]

    def test_get_income_statement_schema(self, tools):
        """Test get_income_statement tool has correct parameter schema."""
        schema = tools["get_income_statement"].params_json_schema
        assert "properties" in schema
        assert "symbol" in schema["properties"]
        assert "quarterly" in schema["properties"]

    def test_tool_count(self, tools):
        """Test that we have the expected number of tools."""
        assert len(tools) == 10, f"Expected 10 tools, got {len(tools)}"

    def test_tools_use_same_searcher_instance(self, tools):
        """Test that all tools are created properly."""
        for name in (
            "search_asset",
            "get_asset_quote",
            "get_asset_profile",
            "get_financial_metrics",
            "get_income_statement",
            "get_balance_sheet",
            "get_cash_flow",
            "get_analyst_recommendations",
            "get_technical_indicators",
            "get_technical_chart",
        ):
            assert tools[name].name == name

    @pytest.mark.parametrize("tool_name", [
        "search_asset",
        "get_asset_quote",
        "get_asset_profile",
        "get_financial_metrics",
        "get_income_statement",
        "get_balance_sheet",
        "get_cash_flow",
        "get_analyst_recommendations",
        "get_technical_indicators",
        "get_technical_chart",
    ])
    def test_individual_tool_structure(self, tools, tool_name):
        """Test each tool conforms to the canonical ToolSpec contract."""
        tool = tools[tool_name]
        assert tool.name == tool_name
        assert isinstance(tool.description, str) and len(tool.description) > 10
        assert isinstance(tool.params_json_schema, dict)
        assert "properties" in tool.params_json_schema
        assert inspect.iscoroutinefunction(tool.invoke)
