"""Tests for consolidate_holdings()."""
import pytest


class TestConsolidateHoldings:
    def test_empty_list(self, consolidate_holdings):
        assert consolidate_holdings([]) == []

    def test_single_holding(self, consolidate_holdings):
        result = consolidate_holdings([{"name": "Fund A", "invested": 10000}])
        assert len(result) == 1
        assert result[0]["invested"] == 10000

    def test_no_duplicates(self, consolidate_holdings):
        holdings = [
            {"name": "Fund A", "invested": 10000},
            {"name": "Fund B", "invested": 20000},
            {"name": "Fund C", "invested": 30000},
        ]
        result = consolidate_holdings(holdings)
        assert len(result) == 3

    def test_duplicate_names_merged(self, consolidate_holdings):
        holdings = [
            {"name": "Fund A", "invested": 10000},
            {"name": "Fund A", "invested": 5000},
        ]
        result = consolidate_holdings(holdings)
        assert len(result) == 1
        assert result[0]["invested"] == 15000

    def test_current_values_merged(self, consolidate_holdings):
        holdings = [
            {"name": "Fund A", "current": 50000},
            {"name": "Fund A", "current": 30000},
        ]
        result = consolidate_holdings(holdings)
        assert len(result) == 1
        assert result[0]["current"] == 80000

    def test_case_normalization(self, consolidate_holdings):
        holdings = [
            {"name": "Fund A", "invested": 10000},
            {"name": "fund a", "invested": 5000},
        ]
        result = consolidate_holdings(holdings)
        assert len(result) == 1

    def test_whitespace_normalization(self, consolidate_holdings):
        holdings = [
            {"name": "Fund  A", "invested": 10000},
            {"name": "Fund A", "invested": 5000},
        ]
        result = consolidate_holdings(holdings)
        assert len(result) == 1

    def test_three_way_merge(self, consolidate_holdings):
        holdings = [
            {"name": "Fund A", "invested": 10000},
            {"name": "Fund A", "invested": 5000},
            {"name": "Fund A", "invested": 3000},
        ]
        result = consolidate_holdings(holdings)
        assert len(result) == 1
        assert result[0]["invested"] == 18000

    def test_preserves_category(self, consolidate_holdings):
        holdings = [
            {"name": "Fund A", "category": "Large Cap", "invested": 10000},
            {"name": "Fund A", "invested": 5000},
        ]
        result = consolidate_holdings(holdings)
        assert result[0].get("category") == "Large Cap"
