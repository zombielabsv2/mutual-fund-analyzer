"""Tests for match_fund_to_scheme()."""
import pytest


class TestFundMatching:
    def test_exact_match(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("HDFC Flexi Cap Fund Direct Plan Growth", mock_all_schemes)
        assert match is not None
        assert conf > 70

    def test_fuzzy_match(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("HDFC Flexi Cap", mock_all_schemes)
        assert match is not None
        assert "Direct" in match["schemeName"]

    def test_excludes_regular_plans(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("HDFC Flexi Cap Fund Growth", mock_all_schemes)
        if match:
            assert "Direct" in match["schemeName"]

    def test_excludes_idcw(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("HDFC Flexi Cap Fund", mock_all_schemes)
        if match:
            assert "IDCW" not in match["schemeName"]

    def test_excludes_dividend(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("SBI Large Cap Fund", mock_all_schemes)
        if match:
            assert "Dividend" not in match["schemeName"]
            assert "dividend" not in match["schemeName"].lower()

    def test_excludes_bonus(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("HDFC Flexi Cap Fund", mock_all_schemes)
        if match:
            assert "Bonus" not in match["schemeName"]

    def test_no_match_returns_none(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("XYZ Nonexistent Fund ABC", mock_all_schemes)
        assert match is None
        assert conf == 0

    def test_empty_scheme_list(self, match_fund_to_scheme):
        match, conf = match_fund_to_scheme("Any Fund", [])
        assert match is None
        assert conf == 0

    def test_case_insensitivity(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("hdfc flexi cap fund", mock_all_schemes)
        assert match is not None

    def test_special_characters(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("Parag Parikh (Direct) Flexi-Cap Growth", mock_all_schemes)
        assert match is not None

    def test_parag_parikh_match(self, match_fund_to_scheme, mock_all_schemes):
        match, conf = match_fund_to_scheme("Parag Parikh Flexi Cap Fund", mock_all_schemes)
        assert match is not None
        assert match["schemeCode"] == 100007
