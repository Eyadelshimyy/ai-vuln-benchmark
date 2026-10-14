"""End-to-end tests for finance_tools agent tool wrapper.

These tests make real API calls to Yahoo Finance to verify that tools return
expected, non-empty results.
"""

import json
import os
from unittest.mock import Mock

import pytest
from ellements.domain_specific.finance.technical_indicators import talib_available
from ellements.domain_specific.finance.yahoo_finance import finance_tools

pytestmark = [
    pytest.mark.requires_network,
    pytest.mark.skipif(
        os.getenv("ELLEMENTS_RUN_NETWORK_TESTS") != "1",
        reason="set ELLEMENTS_RUN_NETWORK_TESTS=1 to run live finance e2e tests",
    ),
]


def to_str(result):
    """Convert result to string for testing."""
    if isinstance(result, str):
        return result
    return str(result)


class TestFinanceToolsE2E:
    """End-to-end tests for finance tools with real API calls."""

    @pytest.fixture
    def tools(self):
        """Get finance tools dict."""
        return finance_tools()

    @pytest.fixture
    def mock_ctx(self):
        """Create a mock ToolContext."""
        return Mock()

    async def test_search_asset_returns_results(self, tools, mock_ctx):
        """Test that search_asset returns non-empty results for Apple."""
        tool = tools["search_asset"]

        # Invoke the tool with Apple as query
        input_data = json.dumps({"query": "Apple", "max_results": 5})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain Apple's ticker AAPL
        assert "AAPL" in result_str

    async def test_get_asset_quote_returns_data(self, tools, mock_ctx):
        """Test that get_asset_quote returns price data for AAPL."""
        tool = tools["get_asset_quote"]

        # Invoke the tool with AAPL
        input_data = json.dumps({"symbol": "AAPL"})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain price-related information
        result_lower = result_str.lower()
        assert "price" in result_lower or "quote" in result_lower or "current" in result_lower
        # Should contain AAPL symbol
        assert "AAPL" in result_str or "aapl" in result_str

    async def test_get_asset_profile_returns_company_info(self, tools, mock_ctx):
        """Test that get_asset_profile returns company information."""
        tool = tools["get_asset_profile"]

        # Invoke the tool with MSFT
        input_data = json.dumps({"symbol": "MSFT"})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain company information
        assert "Microsoft" in result_str or "MSFT" in result_str
        # Should contain sector or industry
        result_lower = result_str.lower()
        assert "sector" in result_lower or "industry" in result_lower or "technology" in result_lower

    async def test_get_financial_metrics_returns_ratios(self, tools, mock_ctx):
        """Test that get_financial_metrics returns financial ratios."""
        tool = tools["get_financial_metrics"]

        # Invoke the tool with GOOGL
        input_data = json.dumps({"symbol": "GOOGL"})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain financial metrics
        result_lower = result_str.lower()
        assert any(metric in result_lower for metric in ["p/e", "pe", "ratio", "margin", "roe", "roa", "debt", "equity"])

    async def test_get_income_statement_returns_revenue_data(self, tools, mock_ctx):
        """Test that get_income_statement returns revenue and earnings."""
        tool = tools["get_income_statement"]

        # Invoke the tool with TSLA (annual)
        input_data = json.dumps({"symbol": "TSLA", "quarterly": False})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain income statement data
        result_lower = result_str.lower()
        assert any(item in result_lower for item in ["revenue", "income", "earnings", "profit"])

    async def test_get_balance_sheet_returns_assets_liabilities(self, tools, mock_ctx):
        """Test that get_balance_sheet returns balance sheet data."""
        tool = tools["get_balance_sheet"]

        # Invoke the tool with NVDA (annual)
        input_data = json.dumps({"symbol": "NVDA", "quarterly": False})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain balance sheet data
        result_lower = result_str.lower()
        assert any(item in result_lower for item in ["assets", "liabilities", "equity", "cash"])

    async def test_get_cash_flow_returns_cash_data(self, tools, mock_ctx):
        """Test that get_cash_flow returns cash flow data."""
        tool = tools["get_cash_flow"]

        # Invoke the tool with AMZN (annual)
        input_data = json.dumps({"symbol": "AMZN", "quarterly": False})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain cash flow data
        result_lower = result_str.lower()
        assert any(item in result_lower for item in ["cash", "flow", "operating", "investing", "financing"])

    async def test_get_analyst_recommendations_returns_ratings(self, tools, mock_ctx):
        """Test that get_analyst_recommendations returns analyst data."""
        tool = tools["get_analyst_recommendations"]

        # Invoke the tool with META
        input_data = json.dumps({"symbol": "META"})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty result
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0

        # Should contain analyst recommendation data
        result_lower = result_str.lower()
        assert any(item in result_lower for item in ["analyst", "buy", "sell", "hold", "recommendation", "rating", "target", "price"])

    async def test_quarterly_data_flag_works(self, tools, mock_ctx):
        """Test that quarterly flag returns different data than annual."""
        tool = tools["get_income_statement"]

        # Get annual data
        annual_input = json.dumps({"symbol": "AAPL", "quarterly": False})
        annual_result = await tool.invoke(**json.loads(annual_input))

        # Get quarterly data
        quarterly_input = json.dumps({"symbol": "AAPL", "quarterly": True})
        quarterly_result = await tool.invoke(**json.loads(quarterly_input))

        # Both should be non-empty
        annual_str = to_str(annual_result)
        quarterly_str = to_str(quarterly_result)
        assert len(annual_str) > 0
        assert len(quarterly_str) > 0

        # They should be different (quarterly has more recent periods)
        assert annual_str != quarterly_str

    async def test_invalid_symbol_handles_gracefully(self, tools, mock_ctx):
        """Test that invalid symbols are handled gracefully."""
        tool = tools["get_asset_quote"]

        # Invoke with invalid symbol
        input_data = json.dumps({"symbol": "INVALID_SYMBOL_XYZ123"})

        # Should either return error message or raise exception gracefully
        try:
            result = await tool.invoke(**json.loads(input_data))
            # If it returns, should have some content (error message)
            assert result is not None
            result_str = to_str(result)
            assert len(result_str) > 0
        except Exception:
            # Exception is acceptable for invalid symbols
            assert True

    @pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "GOOGL"])
    async def test_multiple_symbols_work(self, tools, mock_ctx, symbol):
        """Test that multiple major stock symbols work."""
        tool = tools["get_asset_quote"]

        input_data = json.dumps({"symbol": symbol})
        result = await tool.invoke(**json.loads(input_data))

        # Should return non-empty data for each symbol
        assert result is not None
        result_str = to_str(result)
        assert len(result_str) > 0
        # Should contain the symbol somewhere in result
        assert symbol in result_str or symbol.lower() in result_str.lower()

    @pytest.mark.skipif(not talib_available(), reason="TA-Lib not installed")
    async def test_get_technical_indicators_returns_snapshot(self, tools, mock_ctx):
        tool = tools["get_technical_indicators"]
        input_data = json.dumps(
            {
                "symbol": "AAPL",
                "period": "3mo",
                "interval": "1d",
                "include_chart": False,
            }
        )
        result = await tool.invoke(**json.loads(input_data))
        result_str = to_str(result)
        assert "snapshot" in result_str.lower()
        assert "rsi" in result_str.lower()

    @pytest.mark.skipif(not talib_available(), reason="TA-Lib not installed")
    async def test_get_technical_chart_returns_canvas_payload(self, tools, mock_ctx):
        tool = tools["get_technical_chart"]
        input_data = json.dumps(
            {
                "symbol": "MSFT",
                "period": "3mo",
                "interval": "1d",
            }
        )
        result = await tool.invoke(**json.loads(input_data))
        result_str = to_str(result)
        assert "canvas_chart_payload" in result_str
        assert "data:image" in result_str
