"""Tests for SIP rolling returns (XIRR) and tax impact calculation."""
import pytest
import math
from datetime import datetime, timedelta


@pytest.fixture
def xirr_fn(app):
    return app.xirr


@pytest.fixture
def calculate_sip_rolling_returns(app):
    return app.calculate_sip_rolling_returns


class TestXIRR:
    def test_empty_cashflows(self, xirr_fn):
        assert xirr_fn([]) is None

    def test_single_cashflow(self, xirr_fn):
        assert xirr_fn([(datetime(2020, 1, 1), -10000)]) is None

    def test_simple_doubling_in_one_year(self, xirr_fn):
        """Invest 10000, get back 20000 after 1 year = 100% return."""
        cfs = [
            (datetime(2020, 1, 1), -10000),
            (datetime(2021, 1, 1), 20000),
        ]
        rate = xirr_fn(cfs)
        assert rate is not None
        assert abs(rate - 1.0) < 0.02  # ~100%

    def test_simple_10_percent(self, xirr_fn):
        """Invest 10000, get back 11000 after 1 year = ~10% return."""
        cfs = [
            (datetime(2020, 1, 1), -10000),
            (datetime(2021, 1, 1), 11000),
        ]
        rate = xirr_fn(cfs)
        assert rate is not None
        assert abs(rate - 0.10) < 0.02

    def test_monthly_sip_known_result(self, xirr_fn):
        """12 monthly SIPs of 10000 + final value. Basic sanity check."""
        cfs = []
        for m in range(1, 13):
            cfs.append((datetime(2020, m, 1), -10000))
        # Total invested: 120000. Final value: 132000 (~20% XIRR roughly)
        cfs.append((datetime(2020, 12, 31), 132000))
        rate = xirr_fn(cfs)
        assert rate is not None
        assert rate > 0.10  # Should be a positive return

    def test_negative_return(self, xirr_fn):
        """Invest 10000, get back 8000 after 1 year = -20%."""
        cfs = [
            (datetime(2020, 1, 1), -10000),
            (datetime(2021, 1, 1), 8000),
        ]
        rate = xirr_fn(cfs)
        assert rate is not None
        assert rate < 0

    def test_flat_return(self, xirr_fn):
        """Invest 10000, get back 10000 after 1 year = 0%."""
        cfs = [
            (datetime(2020, 1, 1), -10000),
            (datetime(2021, 1, 1), 10000),
        ]
        rate = xirr_fn(cfs)
        assert rate is not None
        assert abs(rate) < 0.02

    def test_multi_year_cagr(self, xirr_fn):
        """Invest 10000, get back 20000 after 5 years = ~14.87% CAGR."""
        cfs = [
            (datetime(2015, 1, 1), -10000),
            (datetime(2020, 1, 1), 20000),
        ]
        rate = xirr_fn(cfs)
        assert rate is not None
        assert abs(rate - 0.1487) < 0.02


class TestSIPRollingReturns:
    def test_empty_data(self, calculate_sip_rolling_returns):
        assert calculate_sip_rolling_returns([]) == []

    def test_none_data(self, calculate_sip_rolling_returns):
        assert calculate_sip_rolling_returns(None) == []

    def test_insufficient_history(self, calculate_sip_rolling_returns, nav_data_factory):
        """4 years of data can't produce 5-year SIP rolling returns."""
        nav = nav_data_factory(4)
        result = calculate_sip_rolling_returns(nav, years=5)
        assert result == []

    def test_sufficient_history_produces_results(self, calculate_sip_rolling_returns, nav_data_factory):
        """7 years of data should produce some 5-year SIP rolling returns."""
        nav = nav_data_factory(7, annual_growth=0.12)
        result = calculate_sip_rolling_returns(nav, years=5)
        assert len(result) > 0

    def test_returns_are_reasonable(self, calculate_sip_rolling_returns, nav_data_factory):
        """SIP returns in a 12% growth environment should be positive."""
        nav = nav_data_factory(7, annual_growth=0.12)
        result = calculate_sip_rolling_returns(nav, years=5)
        assert len(result) > 0
        for r in result:
            assert -50 < r['return'] < 100  # Sanity bounds

    def test_3_year_sip_rolling(self, calculate_sip_rolling_returns, nav_data_factory):
        """Should work with 3-year period too."""
        nav = nav_data_factory(5, annual_growth=0.15)
        result = calculate_sip_rolling_returns(nav, years=3)
        assert len(result) > 0

    def test_returns_sorted_by_date(self, calculate_sip_rolling_returns, nav_data_factory):
        nav = nav_data_factory(7)
        result = calculate_sip_rolling_returns(nav, years=5)
        dates = [r['date'] for r in result]
        assert dates == sorted(dates)


class TestTaxImpactCalculation:
    """Test the LTCG tax logic used in swap recommendations."""

    def _calc_tax(self, invested, current):
        """Replicate the tax calculation from streamlit_app.py."""
        if current <= invested:
            return 0, current
        gains = current - invested
        taxable_gains = max(0, gains - 125000)
        ltcg_tax = round(taxable_gains * 0.125)
        net_after_tax = round(current - ltcg_tax)
        return ltcg_tax, net_after_tax

    def test_no_gains_no_tax(self):
        tax, net = self._calc_tax(100000, 100000)
        assert tax == 0
        assert net == 100000

    def test_loss_no_tax(self):
        tax, net = self._calc_tax(100000, 80000)
        assert tax == 0
        assert net == 80000

    def test_gains_below_125k_exempt(self):
        tax, net = self._calc_tax(100000, 200000)
        # Gains = 100000, below 125000 threshold
        assert tax == 0
        assert net == 200000

    def test_gains_above_125k_taxed(self):
        tax, net = self._calc_tax(100000, 325000)
        # Gains = 225000, taxable = 225000 - 125000 = 100000
        # Tax = 100000 * 12.5% = 12500
        assert tax == 12500
        assert net == 325000 - 12500

    def test_large_gains(self):
        tax, net = self._calc_tax(500000, 1500000)
        # Gains = 1000000, taxable = 1000000 - 125000 = 875000
        # Tax = 875000 * 12.5% = 109375
        assert tax == 109375
        assert net == 1500000 - 109375

    def test_gains_exactly_125k(self):
        tax, net = self._calc_tax(100000, 225000)
        # Gains = 125000, taxable = 0
        assert tax == 0
        assert net == 225000

    def test_gains_just_above_125k(self):
        tax, net = self._calc_tax(100000, 226000)
        # Gains = 126000, taxable = 1000
        # Tax = 1000 * 0.125 = 125
        assert tax == 125
        assert net == 226000 - 125
