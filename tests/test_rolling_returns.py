"""Tests for calculate_rolling_returns()."""
import pytest
from datetime import datetime, timedelta
import time


class TestEmptyAndInvalidInput:
    def test_empty_nav_data(self, calculate_rolling_returns):
        assert calculate_rolling_returns([]) == []

    def test_none_nav_data(self, calculate_rolling_returns):
        assert calculate_rolling_returns(None) == []

    def test_single_data_point(self, calculate_rolling_returns):
        nav = [{"date": "01-01-2020", "nav": "100"}]
        assert calculate_rolling_returns(nav) == []

    def test_two_points_insufficient_history(self, calculate_rolling_returns):
        nav = [
            {"date": "01-01-2020", "nav": "100"},
            {"date": "02-01-2020", "nav": "101"},
        ]
        assert calculate_rolling_returns(nav, years=5) == []


class TestMalformedEntries:
    def test_malformed_dates_skipped(self, calculate_rolling_returns, nav_data_factory):
        good = nav_data_factory(6)
        bad = [{"date": "not-a-date", "nav": "100"}, {"date": "99-99-9999", "nav": "50"}]
        result_good = calculate_rolling_returns(good, years=5)
        result_mixed = calculate_rolling_returns(good + bad, years=5)
        assert len(result_good) == len(result_mixed)

    def test_malformed_nav_skipped(self, calculate_rolling_returns, nav_data_factory):
        good = nav_data_factory(6)
        bad = [{"date": "01-06-2018", "nav": "abc"}]
        result_good = calculate_rolling_returns(good, years=5)
        result_mixed = calculate_rolling_returns(good + bad, years=5)
        assert len(result_good) == len(result_mixed)

    def test_missing_keys_skipped(self, calculate_rolling_returns, nav_data_factory):
        good = nav_data_factory(6)
        bad = [{"date": "01-06-2018"}, {"nav": "100"}, {}]
        result = calculate_rolling_returns(good + bad, years=5)
        assert len(result) > 0

    def test_zero_nav_filtered(self, calculate_rolling_returns, nav_data_factory):
        good = nav_data_factory(6)
        zeros = [{"date": "15-06-2018", "nav": "0"}, {"date": "16-06-2018", "nav": "0.0"}]
        result = calculate_rolling_returns(good + zeros, years=5)
        assert len(result) > 0  # zeros excluded, good data processed


class TestCAGRCalculation:
    def test_known_cagr_doubling_in_5_years(self, calculate_rolling_returns):
        """NAV 100 → 200 over exactly 5 years = 14.87% CAGR."""
        start = datetime(2015, 1, 1)
        end = start + timedelta(days=5 * 365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "200"},
        ]
        result = calculate_rolling_returns(nav, years=5)
        assert len(result) == 1
        assert abs(result[0]["return"] - 14.87) < 0.5

    def test_negative_returns(self, calculate_rolling_returns):
        """NAV 100 → 50 over 5 years = negative CAGR."""
        start = datetime(2015, 1, 1)
        end = start + timedelta(days=5 * 365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "50"},
        ]
        result = calculate_rolling_returns(nav, years=5)
        assert len(result) == 1
        assert result[0]["return"] < 0

    def test_flat_returns(self, calculate_rolling_returns):
        """NAV stays at 100 = 0% CAGR."""
        start = datetime(2015, 1, 1)
        end = start + timedelta(days=5 * 365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": end.strftime("%d-%m-%Y"), "nav": "100"},
        ]
        result = calculate_rolling_returns(nav, years=5)
        assert len(result) == 1
        assert abs(result[0]["return"]) < 0.5


class TestRollingPeriods:
    def test_three_year_rolling(self, calculate_rolling_returns, nav_data_factory):
        nav = nav_data_factory(4)
        result = calculate_rolling_returns(nav, years=3)
        assert len(result) > 0

    def test_five_year_rolling(self, calculate_rolling_returns, nav_data_factory):
        nav = nav_data_factory(6)
        result = calculate_rolling_returns(nav, years=5)
        assert len(result) > 0

    def test_ten_year_rolling(self, calculate_rolling_returns, nav_data_factory):
        nav = nav_data_factory(11)
        result = calculate_rolling_returns(nav, years=10)
        assert len(result) > 0

    def test_insufficient_for_period(self, calculate_rolling_returns, nav_data_factory):
        nav = nav_data_factory(4)
        result = calculate_rolling_returns(nav, years=5)
        assert result == []

    def test_returns_sorted_chronologically(self, calculate_rolling_returns, nav_data_factory):
        nav = nav_data_factory(7)
        result = calculate_rolling_returns(nav, years=5)
        dates = [r["date"] for r in result]
        assert dates == sorted(dates)


class TestWindowTolerance:
    def test_finds_nav_within_15_day_window(self, calculate_rolling_returns):
        """Should find a past NAV within 15-day tolerance."""
        start = datetime(2015, 1, 5)  # offset by 5 days
        exact_5y = start + timedelta(days=5 * 365)
        nav = [
            {"date": start.strftime("%d-%m-%Y"), "nav": "100"},
            {"date": exact_5y.strftime("%d-%m-%Y"), "nav": "200"},
        ]
        result = calculate_rolling_returns(nav, years=5)
        assert len(result) >= 1


class TestPerformance:
    def test_large_dataset_completes_quickly(self, calculate_rolling_returns, nav_data_factory):
        nav = nav_data_factory(8)  # ~2000 trading days
        start = time.time()
        result = calculate_rolling_returns(nav, years=5)
        elapsed = time.time() - start
        assert elapsed < 5.0
        assert len(result) > 0
