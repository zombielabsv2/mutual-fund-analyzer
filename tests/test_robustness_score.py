"""Tests for robustness score formula and confidence discount."""
import math
import pytest


def raw_robustness(avg, pos_pct, std):
    """Replicate the formula from the app."""
    return (avg * (pos_pct / 100)) / (1 + std / 10)


def confidence(n):
    """Replicate the sqrt confidence discount."""
    return min(1.0, math.sqrt(n / 1500))


class TestRobustnessFormula:
    def test_basic(self):
        assert abs(raw_robustness(15, 95, 5) - 9.5) < 0.01

    def test_zero_stddev(self):
        assert abs(raw_robustness(12, 100, 0) - 12.0) < 0.01

    def test_high_stddev_penalized(self):
        score = raw_robustness(20, 90, 15)
        assert abs(score - 7.2) < 0.01

    def test_higher_avg_lower_score_with_volatility(self):
        """A volatile fund should score lower despite higher avg."""
        stable = raw_robustness(13, 100, 3)
        volatile = raw_robustness(15, 80, 12)
        assert stable > volatile

    def test_negative_avg(self):
        score = raw_robustness(-5, 30, 8)
        assert score < 0

    def test_zero_positive_pct(self):
        assert raw_robustness(10, 0, 5) == 0


class TestConfidenceDiscount:
    def test_at_1500_points(self):
        assert confidence(1500) == 1.0

    def test_above_1500_capped(self):
        assert confidence(2000) == 1.0

    def test_at_zero_points(self):
        assert confidence(0) == 0.0

    def test_at_375_is_50_percent(self):
        # sqrt(375/1500) = sqrt(0.25) = 0.5
        assert abs(confidence(375) - 0.5) < 0.001

    def test_sqrt_vs_linear(self):
        """Verify sqrt is gentler than linear."""
        n = 400
        sqrt_conf = confidence(n)
        linear_conf = n / 1500
        assert sqrt_conf > linear_conf  # sqrt is more generous
        assert abs(sqrt_conf - math.sqrt(400 / 1500)) < 0.001

    def test_at_100_points(self):
        expected = math.sqrt(100 / 1500)
        assert abs(confidence(100) - expected) < 0.001


class TestAdjustedScore:
    def test_adjusted_equals_raw_times_confidence(self):
        raw = 10.0
        conf = confidence(750)
        adjusted = raw * conf
        assert abs(adjusted - 10.0 * math.sqrt(0.5)) < 0.01

    def test_full_confidence_raw_equals_adjusted(self):
        raw = 15.0
        conf = confidence(2000)
        assert raw * conf == raw

    def test_zero_confidence_zeroes_score(self):
        raw = 15.0
        conf = confidence(0)
        assert raw * conf == 0.0


class TestFundResultFields(object):
    def test_load_rankings_has_raw_score(self, fund_result_factory):
        fund = fund_result_factory(total_periods=1500)
        assert "rawRobustnessScore" in fund
        assert fund["rawRobustnessScore"] == fund["robustnessScore"]

    def test_low_confidence_raw_higher_than_adjusted(self, fund_result_factory):
        fund = fund_result_factory(total_periods=400)
        assert fund["rawRobustnessScore"] > fund["robustnessScore"]

    def test_confidence_stored_as_percentage(self, fund_result_factory):
        fund = fund_result_factory(total_periods=750)
        expected = round(min(1.0, math.sqrt(750 / 1500)) * 100)
        assert fund["confidence"] == expected
