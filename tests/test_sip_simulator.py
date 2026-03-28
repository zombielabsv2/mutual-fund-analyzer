"""Tests for simulate_historical_sip()."""
import pytest
from datetime import datetime


@pytest.fixture
def simulate_sip(app):
    return app.simulate_historical_sip


class TestSimulateHistoricalSIP:
    def test_empty_data(self, simulate_sip):
        assert simulate_sip([]) is None

    def test_none_data(self, simulate_sip):
        assert simulate_sip(None) is None

    def test_single_point(self, simulate_sip):
        assert simulate_sip([{"date": "01-01-2020", "nav": "100"}]) is None

    def test_basic_sip(self, simulate_sip, nav_data_factory):
        """6 years of data, SIP from year 1 to year 5."""
        nav = nav_data_factory(6, annual_growth=0.12)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        assert 'monthly' in result
        assert 'summary' in result
        assert 'fd_comparison' in result

    def test_summary_fields(self, simulate_sip, nav_data_factory):
        nav = nav_data_factory(4, annual_growth=0.12)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        s = result['summary']
        assert 'total_invested' in s
        assert 'final_value' in s
        assert 'wealth_gained' in s
        assert 'xirr' in s
        assert 'absolute_return' in s
        assert 'total_months' in s

    def test_invested_amount_correct(self, simulate_sip, nav_data_factory):
        """Total invested should be monthly_amount * number of months."""
        nav = nav_data_factory(3, annual_growth=0.10)
        result = simulate_sip(nav, monthly_amount=5000)
        assert result is not None
        s = result['summary']
        expected_invested = 5000 * s['total_months']
        assert s['total_invested'] == expected_invested

    def test_positive_growth_positive_wealth(self, simulate_sip, nav_data_factory):
        """In a growing market, final value should exceed invested."""
        nav = nav_data_factory(5, annual_growth=0.15)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        assert result['summary']['final_value'] > result['summary']['total_invested']
        assert result['summary']['wealth_gained'] > 0

    def test_xirr_is_reasonable(self, simulate_sip, nav_data_factory):
        """XIRR should be close to the underlying growth rate."""
        nav = nav_data_factory(5, annual_growth=0.12)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        xirr_val = result['summary']['xirr']
        assert xirr_val is not None
        # SIP XIRR in 12% growth env should be roughly 8-16%
        assert 5 < xirr_val < 20

    def test_fd_comparison_exists(self, simulate_sip, nav_data_factory):
        nav = nav_data_factory(4, annual_growth=0.12)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        assert result['fd_comparison']['value_at_7pct'] > 0

    def test_fd_value_exceeds_invested(self, simulate_sip, nav_data_factory):
        """FD at 7% should always grow money."""
        nav = nav_data_factory(5, annual_growth=0.12)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        assert result['fd_comparison']['value_at_7pct'] > result['summary']['total_invested']

    def test_monthly_breakdown_length(self, simulate_sip, nav_data_factory):
        """Should have roughly 12 entries per year of data."""
        nav = nav_data_factory(3, annual_growth=0.10)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        months = len(result['monthly'])
        # ~36 months for 3 years, allow some tolerance
        assert 30 <= months <= 40

    def test_monthly_invested_is_cumulative(self, simulate_sip, nav_data_factory):
        nav = nav_data_factory(3, annual_growth=0.10)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        monthly = result['monthly']
        for i in range(1, len(monthly)):
            assert monthly[i]['invested'] >= monthly[i - 1]['invested']

    def test_monthly_units_cumulative(self, simulate_sip, nav_data_factory):
        nav = nav_data_factory(3, annual_growth=0.10)
        result = simulate_sip(nav, monthly_amount=10000)
        assert result is not None
        monthly = result['monthly']
        for i in range(1, len(monthly)):
            assert monthly[i]['total_units'] >= monthly[i - 1]['total_units']

    def test_custom_date_range(self, simulate_sip, nav_data_factory):
        """Should respect start_date and end_date."""
        nav = nav_data_factory(6, annual_growth=0.12, start_date=datetime(2015, 1, 1))
        result = simulate_sip(
            nav, monthly_amount=10000,
            start_date=datetime(2016, 1, 1),
            end_date=datetime(2019, 1, 1),
        )
        assert result is not None
        # Should be roughly 36 months
        assert 30 <= result['summary']['total_months'] <= 40

    def test_start_after_end_returns_none(self, simulate_sip, nav_data_factory):
        nav = nav_data_factory(5)
        result = simulate_sip(
            nav, monthly_amount=10000,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2019, 1, 1),
        )
        assert result is None

    def test_different_sip_amounts(self, simulate_sip, nav_data_factory):
        """Doubling the SIP amount should double invested and roughly double the value."""
        nav = nav_data_factory(4, annual_growth=0.12)
        r1 = simulate_sip(nav, monthly_amount=5000)
        r2 = simulate_sip(nav, monthly_amount=10000)
        assert r1 is not None and r2 is not None
        assert r2['summary']['total_invested'] == 2 * r1['summary']['total_invested']
        # Value should be roughly 2x (not exact due to rounding)
        ratio = r2['summary']['final_value'] / r1['summary']['final_value']
        assert 1.9 < ratio < 2.1
