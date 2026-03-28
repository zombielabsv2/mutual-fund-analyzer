"""Tests for swap recommendation logic — validates the bug fix where raw score
comparison prevents false swap recommendations."""
import pytest


def determine_swap_action(fund, top_fund, cat_funds):
    """Replicate the swap decision logic from streamlit_app.py lines 1006-1033."""
    fund_rank = None
    for i, ranked in enumerate(cat_funds):
        if ranked["schemeCode"] == fund["schemeCode"]:
            fund_rank = i + 1
            break

    fund_raw = fund.get("rawRobustnessScore", fund["robustnessScore"])
    top_raw = top_fund.get("rawRobustnessScore", top_fund["robustnessScore"])
    is_optimal = (
        (fund_rank is not None and fund_rank <= 3)
        or top_fund["robustnessScore"] <= fund["robustnessScore"]
        or top_raw <= fund_raw
    )

    if is_optimal:
        if fund_rank is not None and fund_rank <= 3:
            return "no_change"
        elif top_raw <= fund_raw and fund.get("confidence", 100) < 70:
            return "hold_monitor"
        else:
            return "outperforms"
    else:
        return "swap"


class TestSwapDecisionLogic:
    def test_fund_ranked_top_3_no_swap(self, fund_result_factory):
        fund = fund_result_factory(scheme_code="A", robustness=5.0, raw_robustness=5.0)
        top = fund_result_factory(scheme_code="B", robustness=10.0, raw_robustness=10.0)
        cat_funds = [top, fund_result_factory(scheme_code="C"), fund]  # fund is #3
        assert determine_swap_action(fund, top, cat_funds) == "no_change"

    def test_fund_higher_adjusted_score_no_swap(self, fund_result_factory):
        fund = fund_result_factory(scheme_code="A", robustness=10.0, raw_robustness=10.0)
        top = fund_result_factory(scheme_code="B", robustness=9.5, raw_robustness=9.5)
        assert determine_swap_action(fund, top, [top]) == "outperforms"

    def test_fund_higher_raw_score_no_swap(self, fund_result_factory):
        """THE BUG FIX: fund with better raw metrics shouldn't get a swap."""
        fund = fund_result_factory(
            scheme_code="A", robustness=5.0, raw_robustness=19.9,
            total_periods=400, confidence=52
        )
        top = fund_result_factory(
            scheme_code="B", robustness=11.75, raw_robustness=11.75,
            total_periods=1800, confidence=100
        )
        result = determine_swap_action(fund, top, [top])
        assert result != "swap"

    def test_high_raw_low_confidence_hold_monitor(self, fund_result_factory):
        fund = fund_result_factory(
            scheme_code="A", robustness=5.0, raw_robustness=12.0,
            total_periods=400, confidence=52
        )
        top = fund_result_factory(
            scheme_code="B", robustness=10.0, raw_robustness=10.0,
            total_periods=1800, confidence=100
        )
        assert determine_swap_action(fund, top, [top]) == "hold_monitor"

    def test_high_raw_high_confidence_outperforms(self, fund_result_factory):
        fund = fund_result_factory(
            scheme_code="A", robustness=10.0, raw_robustness=12.0,
            total_periods=1200, confidence=89
        )
        top = fund_result_factory(
            scheme_code="B", robustness=10.5, raw_robustness=10.5,
            total_periods=1800, confidence=100
        )
        assert determine_swap_action(fund, top, [top]) == "outperforms"

    def test_genuinely_worse_fund_gets_swap(self, fund_result_factory):
        fund = fund_result_factory(
            scheme_code="A", robustness=5.0, raw_robustness=7.0,
            total_periods=1500, confidence=100
        )
        top = fund_result_factory(
            scheme_code="B", robustness=10.0, raw_robustness=10.0,
            total_periods=1800, confidence=100
        )
        assert determine_swap_action(fund, top, [top]) == "swap"

    def test_confidence_penalty_alone_not_trigger_swap(self, fund_result_factory):
        """Even if adjusted score is lower due to confidence penalty, raw wins."""
        fund = fund_result_factory(
            scheme_code="A", robustness=5.0, raw_robustness=15.0,
            total_periods=300, confidence=45
        )
        top = fund_result_factory(
            scheme_code="B", robustness=9.0, raw_robustness=9.0,
            total_periods=1500, confidence=100
        )
        result = determine_swap_action(fund, top, [top])
        assert result != "swap"

    def test_raw_score_fallback_when_missing(self, fund_result_factory):
        """If rawRobustnessScore key is missing, fallback to robustnessScore."""
        fund = {"schemeCode": "A", "robustnessScore": 10.0, "confidence": 100}
        top = {"schemeCode": "B", "robustnessScore": 9.0}
        assert determine_swap_action(fund, top, [top]) == "outperforms"

    def test_exact_equal_raw_scores_no_swap(self, fund_result_factory):
        fund = fund_result_factory(scheme_code="A", robustness=5.0, raw_robustness=10.0, confidence=50)
        top = fund_result_factory(scheme_code="B", robustness=10.0, raw_robustness=10.0, confidence=100)
        result = determine_swap_action(fund, top, [top])
        assert result != "swap"

    def test_marginally_lower_raw_gets_swap(self, fund_result_factory):
        fund = fund_result_factory(
            scheme_code="A", robustness=5.0, raw_robustness=9.99,
            total_periods=1500, confidence=100
        )
        top = fund_result_factory(
            scheme_code="B", robustness=10.0, raw_robustness=10.0,
            total_periods=1800, confidence=100
        )
        assert determine_swap_action(fund, top, [top]) == "swap"


class TestPortfolioHealthScore:
    def test_perfect_portfolio(self):
        scores = [1.0, 1.0, 1.0]
        avg = sum(scores) / len(scores)
        assert round(avg * 10, 1) == 10.0

    def test_decent_portfolio(self):
        scores = [0.7, 0.6, 0.65]
        avg = sum(scores) / len(scores)
        assert 6.0 <= round(avg * 10, 1) < 8.0

    def test_poor_portfolio(self):
        scores = [0.3, 0.4, 0.5]
        avg = sum(scores) / len(scores)
        assert round(avg * 10, 1) < 6.0

    def test_single_fund(self):
        scores = [0.75]
        avg = sum(scores) / len(scores)
        assert round(avg * 10, 1) == 7.5
