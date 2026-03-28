"""Tests for normalize_category() and get_fine_category()."""
import pytest


class TestNormalizeCategory:
    @pytest.mark.parametrize("input_val,expected", [
        (None, "Other"),
        ("", "Other"),
        ("Equity Scheme - Large Cap Fund", "Large Cap"),
        ("EQUITY SCHEME - LARGE CAP FUND", "Large Cap"),
        ("Equity Scheme - Large & Mid Cap Fund", "Large & Mid Cap"),
        ("Equity Scheme - Mid Cap Fund", "Mid Cap"),
        ("Equity Scheme - Small Cap Fund", "Small Cap"),
        ("Equity Scheme - Flexi Cap Fund", "Flexi Cap"),
        ("Equity Scheme - Multi Cap Fund", "Multi Cap"),
        ("Equity Scheme - Value Fund", "Value / Contra"),
        ("Equity Scheme - Contra Fund", "Value / Contra"),
        ("Equity Scheme - Focused Fund", "Focused"),
        ("Equity Scheme - ELSS", "ELSS"),
        ("Equity Scheme - Sectoral/Thematic", "Sectoral / Thematic"),
        ("Equity Scheme - Index Fund", "Index Fund"),
        ("Equity Scheme - Nifty 50 Index", "Index Fund"),
        ("Hybrid Scheme - Balanced Advantage", "Hybrid"),
        ("Hybrid Scheme - Arbitrage Fund", "Hybrid"),
        ("Hybrid Scheme - Equity Savings", "Hybrid"),
        ("Debt Scheme - Corporate Bond Fund", "Debt"),
        ("Debt Scheme - Liquid Fund", "Debt"),
        ("Debt Scheme - Overnight Fund", "Debt"),
        ("Debt Scheme - Money Market", "Debt"),
        ("Other Scheme - Gold Fund", "Commodities"),
        ("Other Scheme - FoF Overseas", "International / FoF"),
        ("Something Completely Unknown", "Other"),
    ])
    def test_categories(self, normalize_category, input_val, expected):
        assert normalize_category(input_val) == expected

    def test_hybrid_before_equity(self, normalize_category):
        """Hybrid with 'equity' in name should still be Hybrid, not an equity category."""
        assert normalize_category("Hybrid Scheme - Equity Savings") == "Hybrid"

    def test_large_mid_not_large_cap(self, normalize_category):
        """'Large & Mid Cap' should not match 'Large Cap'."""
        result = normalize_category("Equity Scheme - Large & Mid Cap Fund")
        assert result == "Large & Mid Cap"
        assert result != "Large Cap"


class TestGetFineCategory:
    @pytest.mark.parametrize("name,cat,expected", [
        ("SBI Gold Fund Direct Growth", "", "Gold"),
        ("Invesco India Gold and Silver ETF", "", "Gold"),
        ("HDFC Liquid Fund Direct Growth", "", "Liquid"),
        ("Kotak Nifty Arbitrage Fund", "Hybrid", "Arbitrage"),
        ("ICICI Corporate Bond Fund", "Debt", "Debt - Corporate Bond"),
        ("HDFC Short Duration Fund", "Debt", "Debt - Short Duration"),
        ("SBI Dynamic Bond Fund", "Debt", "Debt - Medium/Long Duration"),
        ("ICICI Multi Asset Allocation Fund", "Hybrid", "Hybrid - Multi Asset"),
        ("HDFC Balanced Advantage Fund", "Hybrid", "Hybrid - Balanced Advantage"),
        ("ICICI Equity Savings Fund", "Hybrid Scheme", "Hybrid - Equity Savings"),
        ("Motilal Oswal Nasdaq 100 FoF", "Other", "International"),
        ("ICICI Prudential Technology Fund Direct", "Sectoral", "Sectoral - Technology"),
        ("SBI Banking & Financial Services", "Sectoral", "Sectoral - Banking & Financial"),
        ("HDFC Infrastructure Fund", "Sectoral", "Sectoral - Infrastructure"),
        ("Mirae Asset Healthcare Fund", "Thematic", "Sectoral - Healthcare"),
        ("Tata Manufacturing Fund", "Sectoral", "Sectoral - Manufacturing"),
        ("Parag Parikh Flexi Cap Fund", "Equity Scheme - Flexi Cap Fund", "Flexi Cap"),
    ])
    def test_fine_categories(self, get_fine_category, name, cat, expected):
        assert get_fine_category(name, cat) == expected

    def test_none_category_fallback(self, get_fine_category):
        result = get_fine_category("Some Random Fund", None)
        assert result == "Other"

    def test_energy_sector(self, get_fine_category):
        result = get_fine_category("Tata Resources & Energy Fund", "Sectoral")
        assert result == "Sectoral - Energy"

    def test_consumption_sector(self, get_fine_category):
        result = get_fine_category("ICICI Prudential FMCG Fund", "Sectoral")
        assert result == "Sectoral - Consumption"
