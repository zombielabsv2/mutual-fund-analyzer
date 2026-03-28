"""
Shared fixtures and Streamlit mock for testing streamlit_app.py without a Streamlit runtime.
"""
import sys
import types
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import pytest


def _make_streamlit_mock():
    """Create a mock streamlit module that allows importing streamlit_app."""
    mock_st = types.ModuleType("streamlit")

    # cache_data: pass-through decorator (supports both @cache_data and @cache_data(...))
    def cache_data(*args, **kwargs):
        def decorator(func):
            func.clear = lambda: None
            return func
        if args and callable(args[0]):
            args[0].clear = lambda: None
            return args[0]
        return decorator

    mock_st.cache_data = cache_data
    mock_st.set_page_config = lambda **kw: None

    # Session state as an attribute-accessible dict
    class SessionState(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            del self[key]

    mock_st.session_state = SessionState()

    # UI elements — return MagicMock context managers
    for attr in [
        "markdown", "write", "caption", "info", "success", "warning", "error",
        "subheader", "header", "title", "divider", "plotly_chart", "dataframe",
        "data_editor", "metric", "progress", "spinner", "text_input", "button",
        "selectbox", "multiselect", "radio", "file_uploader", "download_button",
        "popover", "expander", "container", "empty",
    ]:
        setattr(mock_st, attr, MagicMock())

    # tabs/columns return list of MagicMock context managers
    mock_st.tabs = lambda labels: [MagicMock() for _ in labels]
    mock_st.columns = lambda spec, **kw: [MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))]

    # column_config namespace
    mock_st.column_config = MagicMock()

    mock_st.rerun = lambda: None

    return mock_st


# Patch streamlit before importing the app module
sys.modules["streamlit"] = _make_streamlit_mock()

import importlib
_app = importlib.import_module("streamlit_app")

# --- Expose app functions as module-level for easy fixture access ---

@pytest.fixture
def app():
    """Return the imported streamlit_app module."""
    return _app


@pytest.fixture
def calculate_rolling_returns():
    return _app.calculate_rolling_returns


@pytest.fixture
def normalize_category():
    return _app.normalize_category


@pytest.fixture
def get_fine_category():
    return _app.get_fine_category


@pytest.fixture
def consolidate_holdings():
    return _app.consolidate_holdings


@pytest.fixture
def match_fund_to_scheme():
    return _app.match_fund_to_scheme


@pytest.fixture
def extract_holdings():
    return _app.extract_holdings


# --- Data Factories ---

@pytest.fixture
def nav_data_factory():
    """Generate synthetic NAV data in API format.

    Returns a function: nav_data_factory(years, initial_nav=100, annual_growth=0.12, start_date=None)
    """
    def _factory(years, initial_nav=100.0, annual_growth=0.12, start_date=None):
        if start_date is None:
            start_date = datetime(2015, 1, 1)
        end_date = start_date + timedelta(days=int(years * 365.25))
        daily_growth = (1 + annual_growth) ** (1 / 365) - 1

        nav_data = []
        current_date = start_date
        nav = initial_nav
        while current_date <= end_date:
            if current_date.weekday() < 5:  # skip weekends
                nav_data.append({
                    "date": current_date.strftime("%d-%m-%Y"),
                    "nav": str(round(nav, 4)),
                })
            nav *= (1 + daily_growth)
            current_date += timedelta(days=1)
        return nav_data
    return _factory


@pytest.fixture
def fund_result_factory():
    """Build a fund result dict with all required fields."""
    def _factory(
        scheme_code="100001",
        scheme_name="Test Fund Direct Plan-Growth",
        category="Flexi Cap",
        fine_category=None,
        fund_house="Test AMC Mutual Fund",
        avg_return=15.0,
        min_return=5.0,
        max_return=25.0,
        std_dev=5.0,
        positive_pct=95.0,
        total_periods=1500,
        confidence=None,
        raw_robustness=None,
        robustness=None,
    ):
        import math
        if raw_robustness is None:
            raw_robustness = (avg_return * (positive_pct / 100)) / (1 + std_dev / 10)
        if confidence is None:
            confidence = round(min(1.0, math.sqrt(total_periods / 1500)) * 100)
        if robustness is None:
            robustness = round(raw_robustness * (confidence / 100), 2)
        return {
            "schemeCode": scheme_code,
            "schemeName": scheme_name,
            "category": category,
            "fineCategory": fine_category or category,
            "fundHouse": fund_house,
            "avgReturn": avg_return,
            "minReturn": min_return,
            "maxReturn": max_return,
            "stdDev": std_dev,
            "positivePercentage": positive_pct,
            "totalPeriods": total_periods,
            "rawRobustnessScore": round(raw_robustness, 2),
            "confidence": confidence,
            "robustnessScore": robustness,
        }
    return _factory


@pytest.fixture
def mock_all_schemes():
    """A list of fake scheme dicts for match_fund_to_scheme testing."""
    return [
        {"schemeCode": 100001, "schemeName": "HDFC Flexi Cap Fund - Direct Plan - Growth"},
        {"schemeCode": 100002, "schemeName": "HDFC Flexi Cap Fund - Regular Plan - Growth"},
        {"schemeCode": 100003, "schemeName": "HDFC Flexi Cap Fund - Direct Plan - IDCW"},
        {"schemeCode": 100004, "schemeName": "SBI Large Cap Fund - Direct Plan - Growth"},
        {"schemeCode": 100005, "schemeName": "SBI Large Cap Fund - Direct Plan - Dividend"},
        {"schemeCode": 100006, "schemeName": "SBI Large Cap Fund - Regular Plan - Growth"},
        {"schemeCode": 100007, "schemeName": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"},
        {"schemeCode": 100008, "schemeName": "Axis Midcap Fund - Direct Plan - Growth"},
        {"schemeCode": 100009, "schemeName": "ICICI Prudential Technology Fund - Direct Plan - Growth"},
        {"schemeCode": 100010, "schemeName": "Quant Active Fund - Direct Plan - Growth"},
        {"schemeCode": 100011, "schemeName": "Kotak Emerging Equity Fund - Direct Plan - Growth"},
        {"schemeCode": 100012, "schemeName": "Mirae Asset Large Cap Fund - Direct Plan - Growth"},
        {"schemeCode": 100013, "schemeName": "DSP Value Fund - Direct Plan - Growth"},
        {"schemeCode": 100014, "schemeName": "HDFC Mid-Cap Opportunities Fund - Direct Plan - Growth"},
        {"schemeCode": 100015, "schemeName": "Nippon India Small Cap Fund - Direct Plan - Growth"},
        {"schemeCode": 100016, "schemeName": "HDFC Flexi Cap Fund - Direct Plan - Bonus"},
    ]
