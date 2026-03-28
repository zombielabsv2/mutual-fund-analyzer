"""Tests for calculate_trailing_returns()."""
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def trailing_returns(app):
    return app.calculate_trailing_returns


class TestTrailingReturns:
    def test_empty_data(self, trailing_returns):
        assert trailing_returns([]) == {}

    def test_none_data(self, trailing_returns):
        assert trailing_returns(None) == {}

    def test_single_point(self, trailing_returns):
        assert trailing_returns([{"date": "01-01-2020", "nav": "100"}]) == {}

    def test_1y_return(self, trailing_returns):
        """NAV 100 → 115 over 1 year = 15% return."""
        start = datetime(2024, 1, 1)
        end = start + timedelta(days=365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "115"},
        ]
        result = trailing_returns(nav)
        assert "1Y" in result
        assert abs(result["1Y"] - 15.0) < 0.5

    def test_5y_cagr(self, trailing_returns):
        """NAV 100 → 200 over 5 years = ~14.87% CAGR."""
        start = datetime(2020, 1, 1)
        end = start + timedelta(days=5 * 365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "200"},
        ]
        result = trailing_returns(nav)
        assert "5Y" in result
        assert abs(result["5Y"] - 14.87) < 0.5

    def test_negative_return(self, trailing_returns):
        start = datetime(2024, 1, 1)
        end = start + timedelta(days=365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "90"},
        ]
        result = trailing_returns(nav)
        assert result["1Y"] < 0

    def test_insufficient_history_excluded(self, trailing_returns, nav_data_factory):
        """Only 2 years of data — should have 1Y but not 3Y or 5Y."""
        nav = nav_data_factory(2)
        result = trailing_returns(nav)
        assert "1Y" in result
        assert "3Y" not in result
        assert "5Y" not in result

    def test_all_periods_with_long_history(self, trailing_returns, nav_data_factory):
        """11 years of data should have 1Y, 3Y, 5Y, 10Y."""
        nav = nav_data_factory(11)
        result = trailing_returns(nav)
        assert "1Y" in result
        assert "3Y" in result
        assert "5Y" in result
        assert "10Y" in result

    def test_malformed_entries_skipped(self, trailing_returns):
        start = datetime(2024, 1, 1)
        end = start + timedelta(days=365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": "bad-date", "nav": "abc"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "115"},
        ]
        result = trailing_returns(nav)
        assert "1Y" in result

    def test_zero_nav_filtered(self, trailing_returns):
        start = datetime(2024, 1, 1)
        end = start + timedelta(days=365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "0"},
            {"date": (start + timedelta(days=1)).strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "115"},
        ]
        result = trailing_returns(nav)
        assert "1Y" in result

    def test_returns_are_rounded(self, trailing_returns, nav_data_factory):
        nav = nav_data_factory(6)
        result = trailing_returns(nav)
        for key, val in result.items():
            assert val == round(val, 2)
