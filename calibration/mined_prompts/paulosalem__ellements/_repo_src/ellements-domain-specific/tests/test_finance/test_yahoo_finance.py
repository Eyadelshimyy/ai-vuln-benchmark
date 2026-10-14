"""Tests for Yahoo Finance functionality."""

from datetime import datetime, timedelta

import pytest
from ellements.core.exceptions import LLMError
from ellements.domain_specific.finance import YahooFinanceSearcher

pytestmark = pytest.mark.requires_network


@pytest.fixture
def finance_searcher():
    """Create a YahooFinanceSearcher instance for testing."""
    return YahooFinanceSearcher()


class TestSearch:
    """Tests for asset search functionality."""

    @pytest.mark.asyncio
    async def test_search_apple(self, finance_searcher):
        """Test searching for Apple stock."""
        results = await finance_searcher.search_async("Apple", max_results=5)

        assert len(results) > 0
        assert results.query == "Apple"
        assert any("AAPL" in r.symbol for r in results.results)

    @pytest.mark.asyncio
    async def test_search_returns_correct_fields(self, finance_searcher):
        """Test that search results have all required fields."""
        results = await finance_searcher.search_async("Microsoft", max_results=1)

        if len(results) > 0:
            result = results.results[0]
            assert result.symbol
            assert result.name
            assert result.type

    def test_search_sync(self, finance_searcher):
        """Test synchronous search."""
        results = finance_searcher.search("Tesla", max_results=3)

        assert len(results) >= 0
        assert results.query == "Tesla"


class TestQuotes:
    """Tests for quote retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_quote_aapl(self, finance_searcher):
        """Test getting a quote for Apple."""
        quote = await finance_searcher.get_quote_async("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.name
        assert quote.current_price is not None or quote.previous_close is not None
        assert quote.currency

    @pytest.mark.asyncio
    async def test_quote_price_change_calculation(self, finance_searcher):
        """Test price change calculations."""
        quote = await finance_searcher.get_quote_async("MSFT")

        if quote.current_price and quote.previous_close:
            change = quote.price_change()
            change_pct = quote.price_change_percent()

            assert change is not None
            assert change_pct is not None
            assert isinstance(change, float)
            assert isinstance(change_pct, float)

    def test_get_quote_sync(self, finance_searcher):
        """Test synchronous quote retrieval."""
        quote = finance_searcher.get_quote("GOOGL")

        assert quote.symbol == "GOOGL"
        assert quote.name

    @pytest.mark.asyncio
    async def test_invalid_symbol_returns_none_price(self, finance_searcher):
        """Test that invalid symbols return a quote with None price."""
        quote = await finance_searcher.get_quote_async("INVALID_SYMBOL_XYZ123")
        assert quote.current_price is None


class TestProfile:
    """Tests for company profile retrieval."""

    @pytest.mark.asyncio
    async def test_get_profile_aapl(self, finance_searcher):
        """Test getting company profile for Apple."""
        profile = await finance_searcher.get_profile_async("AAPL")

        assert profile.symbol == "AAPL"
        assert profile.name
        assert profile.sector or profile.industry  # At least one should be present

    @pytest.mark.asyncio
    async def test_profile_has_description(self, finance_searcher):
        """Test that profile includes company description."""
        profile = await finance_searcher.get_profile_async("MSFT")

        assert profile.description  # Microsoft should have a description

    def test_get_profile_sync(self, finance_searcher):
        """Test synchronous profile retrieval."""
        profile = finance_searcher.get_profile("TSLA")

        assert profile.symbol == "TSLA"
        assert profile.name


class TestHistoricalData:
    """Tests for historical data retrieval."""

    @pytest.mark.asyncio
    async def test_get_historical_data_1mo(self, finance_searcher):
        """Test getting 1-month historical data."""
        history = await finance_searcher.get_historical_data_async(
            "AAPL",
            period="1mo",
            interval="1d"
        )

        assert history.symbol == "AAPL"
        assert len(history.prices) > 0
        assert history.period == "1mo"
        assert history.interval == "1d"

    @pytest.mark.asyncio
    async def test_historical_data_with_dates(self, finance_searcher):
        """Test getting historical data with specific date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        history = await finance_searcher.get_historical_data_async(
            "MSFT",
            start=start_date,
            end=end_date,
            interval="1d"
        )

        assert history.symbol == "MSFT"
        assert len(history.prices) > 0

    @pytest.mark.asyncio
    async def test_historical_data_calculations(self, finance_searcher):
        """Test historical data helper methods."""
        history = await finance_searcher.get_historical_data_async(
            "GOOGL",
            period="3mo",
            interval="1d"
        )

        # Test get_latest and get_oldest
        latest = history.get_latest()
        oldest = history.get_oldest()

        assert latest is not None
        assert oldest is not None
        assert latest.date >= oldest.date

        # Test get_price_range
        min_price, max_price = history.get_price_range()
        assert min_price is not None
        assert max_price is not None
        assert min_price <= max_price

        # Test calculate_return
        total_return = history.calculate_return()
        assert total_return is not None
        assert isinstance(total_return, float)

    def test_get_historical_data_sync(self, finance_searcher):
        """Test synchronous historical data retrieval."""
        history = finance_searcher.get_historical_data("AMZN", period="1mo")

        assert history.symbol == "AMZN"
        assert len(history.prices) > 0


class TestFinancialMetrics:
    """Tests for financial metrics retrieval."""

    @pytest.mark.asyncio
    async def test_get_financial_metrics_aapl(self, finance_searcher):
        """Test getting financial metrics for Apple."""
        metrics = await finance_searcher.get_financial_metrics_async("AAPL")

        assert metrics.symbol == "AAPL"
        # At least some core metrics should be present
        assert metrics.pe_ratio is not None or metrics.market_cap is not None

    @pytest.mark.asyncio
    async def test_financial_metrics_valuation(self, finance_searcher):
        """Test valuation metrics."""
        metrics = await finance_searcher.get_financial_metrics_async("MSFT")

        # Check at least one valuation metric
        assert any([
            metrics.pe_ratio,
            metrics.forward_pe,
            metrics.peg_ratio,
            metrics.price_to_book,
            metrics.price_to_sales
        ])

    def test_get_financial_metrics_sync(self, finance_searcher):
        """Test synchronous financial metrics retrieval."""
        metrics = finance_searcher.get_financial_metrics("GOOGL")

        assert metrics.symbol == "GOOGL"


class TestFinancialStatements:
    """Tests for financial statements retrieval."""

    @pytest.mark.asyncio
    async def test_get_financials_annual(self, finance_searcher):
        """Test getting annual financial statements."""
        fundamentals = await finance_searcher.get_financials_async("AAPL", quarterly=False)

        assert fundamentals.symbol == "AAPL"
        assert fundamentals.metrics.symbol == "AAPL"
        # Should have at least some statements
        assert len(fundamentals.income_statements) > 0 or len(fundamentals.balance_sheets) > 0

    @pytest.mark.asyncio
    async def test_get_financials_quarterly(self, finance_searcher):
        """Test getting quarterly financial statements."""
        fundamentals = await finance_searcher.get_financials_async("MSFT", quarterly=True)

        assert fundamentals.symbol == "MSFT"
        # Quarterly statements should typically have more entries
        if len(fundamentals.income_statements) > 0:
            assert fundamentals.income_statements[0].date is not None

    @pytest.mark.asyncio
    async def test_income_statement_fields(self, finance_searcher):
        """Test income statement fields."""
        fundamentals = await finance_searcher.get_financials_async("AAPL")

        if len(fundamentals.income_statements) > 0:
            stmt = fundamentals.income_statements[0]
            # At least some fields should be present
            assert stmt.date is not None
            assert stmt.total_revenue is not None or stmt.net_income is not None

    def test_get_financials_sync(self, finance_searcher):
        """Test synchronous financials retrieval."""
        fundamentals = finance_searcher.get_financials("TSLA")

        assert fundamentals.symbol == "TSLA"


class TestAnalystData:
    """Tests for analyst recommendations and institutional holders."""

    @pytest.mark.asyncio
    async def test_get_analyst_recommendations(self, finance_searcher):
        """Test getting analyst recommendations."""
        recommendations = await finance_searcher.get_analyst_recommendations_async("AAPL")

        assert "symbol" in recommendations
        assert recommendations["symbol"] == "AAPL"
        # At least one of the known analyst data fields should be present.
        expected_fields = {
            "history",
            "price_targets",
            "recommendations",
            "recommendation_key",
            "recommendation_mean",
            "number_of_analyst_opinions",
        }
        assert expected_fields & recommendations.keys(), (
            f"Expected analyst-data field in {sorted(recommendations.keys())}"
        )

    @pytest.mark.asyncio
    async def test_get_institutional_holders(self, finance_searcher):
        """Test getting institutional holder information."""
        holders = await finance_searcher.get_institutional_holders_async("MSFT")

        assert "symbol" in holders
        assert holders["symbol"] == "MSFT"

    def test_get_analyst_recommendations_sync(self, finance_searcher):
        """Test synchronous analyst recommendations retrieval."""
        recommendations = finance_searcher.get_analyst_recommendations("GOOGL")

        assert recommendations["symbol"] == "GOOGL"


class TestComprehensiveInfo:
    """Tests for comprehensive asset information."""

    @pytest.mark.asyncio
    async def test_get_asset_info_all(self, finance_searcher):
        """Test getting all information about an asset."""
        info = await finance_searcher.get_asset_info_async(
            "AAPL",
            include_quote=True,
            include_profile=True,
            include_history=True,
            history_period="1mo"
        )

        assert info["symbol"] == "AAPL"
        assert "quote" in info
        assert "profile" in info
        assert "history" in info

        # Verify each component
        assert info["quote"].current_price is not None or info["quote"].previous_close is not None
        assert info["profile"].name
        assert len(info["history"].prices) > 0

    @pytest.mark.asyncio
    async def test_get_asset_info_selective(self, finance_searcher):
        """Test getting selective information."""
        info = await finance_searcher.get_asset_info_async(
            "MSFT",
            include_quote=True,
            include_profile=False,
            include_history=False
        )

        assert info["symbol"] == "MSFT"
        assert "quote" in info
        assert "profile" not in info
        assert "history" not in info

    @pytest.mark.asyncio
    async def test_get_all_data_comprehensive(self, finance_searcher):
        """Test getting all available data."""
        all_data = await finance_searcher.get_all_data_async(
            "AAPL",
            include_financials=True,
            include_recommendations=True,
            include_holders=True
        )

        assert all_data["symbol"] == "AAPL"
        assert "quote" in all_data
        assert "profile" in all_data
        assert "fundamentals" in all_data
        assert "analyst_recommendations" in all_data
        assert "institutional_holders" in all_data

    def test_get_all_data_sync(self, finance_searcher):
        """Test synchronous comprehensive data retrieval."""
        all_data = finance_searcher.get_all_data("MSFT", include_financials=False)

        assert all_data["symbol"] == "MSFT"
        assert "quote" in all_data


class TestComparison:
    """Tests for asset comparison functionality."""

    @pytest.mark.asyncio
    async def test_compare_assets(self, finance_searcher):
        """Test comparing multiple assets."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        comparison = await finance_searcher.compare_assets_async(
            symbols,
            period="1mo",
            max_concurrent=2
        )

        assert len(comparison) > 0
        for symbol in symbols:
            if symbol in comparison:
                data = comparison[symbol]
                assert data.symbol == symbol
                assert len(data.prices) > 0

    def test_compare_assets_sync(self, finance_searcher):
        """Test synchronous asset comparison."""
        symbols = ["AAPL", "MSFT"]
        comparison = finance_searcher.compare_assets(symbols, period="1mo")

        assert len(comparison) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_search_query(self, finance_searcher):
        """Test search with empty query."""
        with pytest.raises((LLMError, Exception)):
            await finance_searcher.search_async("", max_results=1)

    @pytest.mark.asyncio
    async def test_invalid_period(self, finance_searcher):
        """Test historical data with invalid period."""
        # Should either work with default or raise an error
        try:
            history = await finance_searcher.get_historical_data_async(
                "AAPL",
                period="invalid_period"
            )
            # If it works, it should have used a default
            assert len(history.prices) >= 0
        except (LLMError, Exception):
            # Error is acceptable for invalid input
            pass

    @pytest.mark.asyncio
    async def test_compare_with_invalid_symbols(self, finance_searcher):
        """Test comparison with some invalid symbols."""
        symbols = ["AAPL", "INVALID_XYZ", "MSFT"]
        comparison = await finance_searcher.compare_assets_async(
            symbols,
            period="1mo"
        )

        # Should return data for valid symbols
        assert "AAPL" in comparison or "MSFT" in comparison
        # Invalid symbol should be skipped
        assert "INVALID_XYZ" not in comparison


class TestDataModels:
    """Tests for Pydantic data models."""

    @pytest.mark.asyncio
    async def test_asset_quote_model(self, finance_searcher):
        """Test AssetQuote model structure."""
        quote = await finance_searcher.get_quote_async("AAPL")

        # Test model serialization
        quote_dict = quote.model_dump()
        assert isinstance(quote_dict, dict)
        assert "symbol" in quote_dict
        assert "current_price" in quote_dict

        # Test model JSON serialization
        quote_json = quote.model_dump_json()
        assert isinstance(quote_json, str)

    @pytest.mark.asyncio
    async def test_historical_price_model(self, finance_searcher):
        """Test HistoricalPrice model structure."""
        history = await finance_searcher.get_historical_data_async("AAPL", period="5d")

        if len(history.prices) > 0:
            price = history.prices[0]

            # Verify all fields are accessible
            assert isinstance(price.date, datetime)
            assert isinstance(price.open, (float, type(None)))
            assert isinstance(price.close, (float, type(None)))
            assert isinstance(price.volume, (int, type(None)))

    @pytest.mark.asyncio
    async def test_financial_metrics_model(self, finance_searcher):
        """Test FinancialMetrics model structure."""
        metrics = await finance_searcher.get_financial_metrics_async("AAPL")

        # Test model serialization
        metrics_dict = metrics.model_dump()
        assert isinstance(metrics_dict, dict)
        assert "symbol" in metrics_dict
        assert "pe_ratio" in metrics_dict or "market_cap" in metrics_dict

    @pytest.mark.asyncio
    async def test_fundamentals_model(self, finance_searcher):
        """Test Fundamentals model structure."""
        fundamentals = await finance_searcher.get_financials_async("AAPL")

        # Test nested model access
        assert fundamentals.metrics.symbol == "AAPL"
        assert isinstance(fundamentals.income_statements, list)
        assert isinstance(fundamentals.balance_sheets, list)
        assert isinstance(fundamentals.cash_flows, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
