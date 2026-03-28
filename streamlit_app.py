"""
Mutual Fund 5-Year Rolling Returns Analyzer — Streamlit Edition
Analyze, compare, and rank Indian mutual funds by rolling return robustness.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import pdfplumber
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time

# --- Page Config ---
st.set_page_config(
    page_title="MF Rolling Returns Analyzer",
    page_icon="📈",
    layout="wide",
)

MFAPI_BASE_URL = "https://api.mfapi.in/mf"

# --- Fund Universe (verified Direct Growth equity scheme codes) ---
RANKING_FUND_CODES = [
    # Large Cap
    "118269", "118479", "118531", "118617", "118632",
    "118825", "119018", "119160", "119250", "120465",
    # Large & Mid Cap
    "118278", "118510", "119202", "119218", "119436",
    "120357", "120596", "120665", "120826", "130498",
    # Mid Cap
    "118533", "118668", "118989", "119178", "119581",
    "119775", "120726", "120841", "125307", "140228",
    # Small Cap
    "118525", "118778", "119212", "119556", "119589",
    "120069", "120164", "120828", "125354", "125497", "130503",
    # Flexi Cap
    "118424", "118535", "118955", "119076", "120564",
    "120662", "120843", "122639", "129046", "133839",
    # Multi Cap
    "118650", "120823", "131164", "141226", "149303", "149368",
    # Value / Contra
    "103490", "118494", "118784", "118935", "119549",
    "119659", "119769", "119835", "120323", "120348",
    # Focused
    "118564", "118692", "118950", "119096", "119564",
    "119727", "120468", "120722", "120834", "122389",
    # ELSS / Tax Saving
    "118285", "118540", "118620", "118803", "119060",
    "119242", "119544", "119723", "119773", "120503",
    "120847", "135781",
    # Sectoral / Thematic
    "118267", "118537", "118589", "119028", "119597",
    "120175", "120587", "120837",
    # Index Funds
    "118266", "118482", "118741", "119648", "119827",
    "120716", "147622",
]


# --- Shared Calculation Logic ---

def calculate_rolling_returns(nav_data, years=5):
    """Calculate rolling returns from NAV data."""
    if not nav_data:
        return []
    parsed_data = []
    for item in nav_data:
        try:
            date = datetime.strptime(item['date'], '%d-%m-%Y')
            nav = float(item['nav'])
            if nav > 0:
                parsed_data.append((date, nav))
        except (ValueError, KeyError):
            continue
    parsed_data.sort(key=lambda x: x[0])
    if len(parsed_data) < 2:
        return []
    nav_lookup = {d: n for d, n in parsed_data}
    rolling_returns = []
    target_days = years * 365
    for current_date, current_nav in parsed_data:
        target_date = current_date - timedelta(days=target_days)
        past_nav = None
        past_date = None
        if target_date in nav_lookup:
            past_nav = nav_lookup[target_date]
            past_date = target_date
        else:
            for delta in range(0, 16):
                check_date = target_date + timedelta(days=delta)
                if check_date in nav_lookup:
                    past_nav = nav_lookup[check_date]
                    past_date = check_date
                    break
                check_date = target_date - timedelta(days=delta)
                if check_date in nav_lookup:
                    past_nav = nav_lookup[check_date]
                    past_date = check_date
                    break
        if past_nav and past_nav > 0:
            actual_years = (current_date - past_date).days / 365.25
            if actual_years >= years - 0.1:
                cagr = (pow(current_nav / past_nav, 1 / actual_years) - 1) * 100
                rolling_returns.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'return': round(cagr, 2)
                })
    return rolling_returns


def xirr(cashflows):
    """Calculate XIRR (annualized internal rate of return) using Newton-Raphson.

    Args:
        cashflows: list of (datetime, amount) tuples. Negative = outflow, positive = inflow.

    Returns:
        Annualized return as a decimal (e.g., 0.15 for 15%), or None if solver fails.
    """
    if not cashflows or len(cashflows) < 2:
        return None
    # Sort by date
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]

    def npv(rate):
        return sum(cf / ((1 + rate) ** ((dt - t0).days / 365.25)) for dt, cf in cashflows)

    def dnpv(rate):
        return sum(-cf * ((dt - t0).days / 365.25) / ((1 + rate) ** ((dt - t0).days / 365.25 + 1))
                    for dt, cf in cashflows)

    # Newton-Raphson with initial guess of 10%
    rate = 0.1
    for _ in range(100):
        val = npv(rate)
        deriv = dnpv(rate)
        if abs(deriv) < 1e-12:
            break
        new_rate = rate - val / deriv
        # Clamp to avoid divergence
        new_rate = max(-0.99, min(new_rate, 10.0))
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = new_rate
    else:
        return None
    # Validate: NPV should be near zero
    if abs(npv(rate)) > 1.0:
        return None
    return rate


def calculate_sip_rolling_returns(nav_data, years=5, monthly_amount=10000):
    """Calculate rolling SIP returns (XIRR) from NAV data.

    For each possible end date with sufficient history, simulates a monthly SIP
    of `monthly_amount` over `years` and calculates the XIRR.

    Returns:
        List of {'date': 'YYYY-MM-DD', 'return': percentage} dicts.
    """
    if not nav_data:
        return []

    # Parse and sort
    parsed = []
    for item in nav_data:
        try:
            dt = datetime.strptime(item['date'], '%d-%m-%Y')
            nav = float(item['nav'])
            if nav > 0:
                parsed.append((dt, nav))
        except (ValueError, KeyError):
            continue
    parsed.sort(key=lambda x: x[0])
    if len(parsed) < 2:
        return []

    nav_lookup = {d: n for d, n in parsed}
    target_days = years * 365

    # Build monthly SIP dates: for each end_date, find the start ~years ago
    # and simulate investing on the 1st of each month
    results = []
    # Sample every 5th data point to keep computation reasonable
    step = max(1, len(parsed) // 500)
    for idx in range(0, len(parsed), step):
        end_date, end_nav = parsed[idx]
        start_date = end_date - timedelta(days=target_days)

        # Need data going back `years`
        if start_date < parsed[0][0]:
            continue

        # Simulate monthly SIP
        cashflows = []
        current = datetime(start_date.year, start_date.month, 1)
        total_units = 0.0

        while current <= end_date:
            # Find nearest trading day NAV
            best_nav = None
            for delta in range(0, 8):
                check = current + timedelta(days=delta)
                if check in nav_lookup:
                    best_nav = nav_lookup[check]
                    break
            if best_nav:
                units = monthly_amount / best_nav
                total_units += units
                cashflows.append((current, -monthly_amount))

            # Next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        if not cashflows or total_units == 0:
            continue

        # Final redemption value
        final_value = total_units * end_nav
        cashflows.append((end_date, final_value))

        rate = xirr(cashflows)
        if rate is not None and -1 < rate < 5:
            results.append({
                'date': end_date.strftime('%Y-%m-%d'),
                'return': round(rate * 100, 2),
            })

    return results


def simulate_historical_sip(nav_data, monthly_amount=10000, start_date=None, end_date=None):
    """Simulate a historical SIP using actual NAV data.

    Returns dict with:
        monthly: list of {date, nav, units_bought, total_units, invested, value}
        summary: {total_invested, final_value, wealth_gained, xirr, total_months}
        fd_comparison: {value_at_7pct} — what a 7% FD would have returned
    """
    if not nav_data:
        return None

    parsed = []
    for item in nav_data:
        try:
            dt = datetime.strptime(item['date'], '%d-%m-%Y')
            nav = float(item['nav'])
            if nav > 0:
                parsed.append((dt, nav))
        except (KeyError, ValueError):
            continue
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: x[0])
    nav_lookup = {d: n for d, n in parsed}

    data_start = parsed[0][0]
    data_end = parsed[-1][0]

    if start_date is None:
        start_date = data_start
    if end_date is None:
        end_date = data_end

    # Clamp to available data
    start_date = max(start_date, data_start)
    end_date = min(end_date, data_end)

    if start_date >= end_date:
        return None

    # Simulate month by month
    monthly = []
    cashflows = []
    total_units = 0.0
    total_invested = 0.0
    current = datetime(start_date.year, start_date.month, 1)

    while current <= end_date:
        # Find nearest trading day NAV (within 8 days)
        best_nav = None
        best_date = None
        for delta in range(0, 8):
            check = current + timedelta(days=delta)
            if check in nav_lookup:
                best_nav = nav_lookup[check]
                best_date = check
                break

        if best_nav and best_date <= end_date:
            units = monthly_amount / best_nav
            total_units += units
            total_invested += monthly_amount
            current_value = total_units * best_nav
            cashflows.append((best_date, -monthly_amount))

            monthly.append({
                'date': best_date.strftime('%Y-%m-%d'),
                'nav': round(best_nav, 4),
                'units_bought': round(units, 4),
                'total_units': round(total_units, 4),
                'invested': round(total_invested),
                'value': round(current_value),
            })

        # Next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    if not monthly or total_units == 0:
        return None

    # Final value using last available NAV
    final_nav = parsed[-1][1]
    final_value = total_units * final_nav
    wealth_gained = final_value - total_invested

    # Update last month's value to reflect actual end NAV
    for m in monthly:
        m['value'] = round(float(m['total_units']) * final_nav)

    # Recalculate value at each month using that month's latest NAV
    # (we need the value progression chart, so use the NAV on each SIP date)
    running_units = 0.0
    for m in monthly:
        running_units += m['units_bought']
        # Find NAV on this date
        dt = datetime.strptime(m['date'], '%Y-%m-%d')
        m['value'] = round(running_units * nav_lookup.get(dt, m['nav']))

    # XIRR
    cashflows.append((data_end, final_value))
    rate = xirr(cashflows)

    # FD comparison: 7% annual compounding, monthly deposits
    fd_value = 0.0
    monthly_fd_rate = (1.07 ** (1 / 12)) - 1
    n_months = len(monthly)
    for i in range(n_months):
        months_remaining = n_months - i
        fd_value += monthly_amount * ((1 + monthly_fd_rate) ** months_remaining)

    # Nifty 50 comparison not computed here — caller can do it with the same function

    return {
        'monthly': monthly,
        'summary': {
            'total_invested': round(total_invested),
            'final_value': round(final_value),
            'wealth_gained': round(wealth_gained),
            'xirr': round(rate * 100, 2) if rate is not None else None,
            'absolute_return': round((final_value / total_invested - 1) * 100, 1) if total_invested > 0 else 0,
            'total_months': len(monthly),
        },
        'fd_comparison': {
            'value_at_7pct': round(fd_value),
        },
    }


def calculate_trailing_returns(nav_data):
    """Calculate 1Y, 3Y, 5Y point-to-point CAGR from NAV data."""
    if not nav_data:
        return {}
    parsed = []
    for item in nav_data:
        try:
            dt = datetime.strptime(item['date'], '%d-%m-%Y')
            nav = float(item['nav'])
            if nav > 0:
                parsed.append((dt, nav))
        except (KeyError, ValueError):
            continue
    if len(parsed) < 2:
        return {}
    parsed.sort(key=lambda x: x[0])
    latest_date, latest_nav = parsed[-1]
    results = {}
    for label, years in [('1Y', 1), ('3Y', 3), ('5Y', 5), ('10Y', 10)]:
        target = latest_date - timedelta(days=int(years * 365.25))
        best = None
        for dt, nav in parsed:
            if abs((dt - target).days) <= 15:
                if best is None or abs((dt - target).days) < abs((best[0] - target).days):
                    best = (dt, nav)
        if best:
            actual_years = (latest_date - best[0]).days / 365.25
            if actual_years >= years - 0.1:
                cagr = ((latest_nav / best[1]) ** (1 / actual_years) - 1) * 100
                results[label] = round(cagr, 2)
    return results


def normalize_category(scheme_category):
    """Map API scheme_category to a clean display category."""
    if not scheme_category:
        return "Other"
    cat = scheme_category.lower()
    # Non-equity categories first (must check before equity keywords)
    if "hybrid" in cat or "arbitrage" in cat or "multi asset" in cat or "balanced" in cat or "equity savings" in cat or "conservative" in cat:
        return "Hybrid"
    if "debt" in cat or "bond" in cat or "gilt" in cat or "liquid" in cat or "money market" in cat or "overnight" in cat or "duration" in cat or "credit" in cat or "income" in cat or "floating" in cat:
        return "Debt"
    if "gold" in cat or "silver" in cat or "commodit" in cat:
        return "Commodities"
    if "fof" in cat or "fund of fund" in cat or "international" in cat:
        return "International / FoF"
    # Equity categories
    if "large cap" in cat and "mid" not in cat:
        return "Large Cap"
    if "large" in cat and "mid" in cat:
        return "Large & Mid Cap"
    if "mid cap" in cat or "mid-cap" in cat:
        return "Mid Cap"
    if "small cap" in cat or "small-cap" in cat:
        return "Small Cap"
    if "flexi" in cat:
        return "Flexi Cap"
    if "multi" in cat:
        return "Multi Cap"
    if "value" in cat or "contra" in cat:
        return "Value / Contra"
    if "focused" in cat or "concentrate" in cat:
        return "Focused"
    if "elss" in cat or "tax" in cat:
        return "ELSS"
    if "thematic" in cat or "sectoral" in cat or "sector" in cat:
        return "Sectoral / Thematic"
    if "index" in cat or "nifty" in cat or "sensex" in cat:
        return "Index Fund"
    return "Other"


def get_fine_category(fund_name, scheme_category=''):
    """Return a granular category by analyzing the fund name and API category.

    This prevents unrelated funds (Gold vs Liquid vs Nasdaq) from being
    grouped together just because the API calls them all 'Other Scheme'.
    """
    name = fund_name.lower()
    cat = (scheme_category or '').lower()

    # --- Non-equity / Non-domestic (check name first, more reliable) ---
    if any(w in name for w in ['gold fund', 'gold etf', 'gold direct', 'gold and silver', 'silver fund', 'silver etf']):
        return 'Gold'
    if any(w in name for w in ['liquid fund', 'liquid plan', 'money market', 'overnight fund']):
        return 'Liquid'
    if 'arbitrage' in name:
        return 'Arbitrage'
    if any(w in name for w in ['corporate bond', 'credit risk', 'gilt fund', 'government sec', 'g-sec']):
        return 'Debt - Corporate Bond'
    if any(w in name for w in ['short term', 'short duration', 'low duration', 'ultra short']):
        return 'Debt - Short Duration'
    if any(w in name for w in ['medium duration', 'medium to long', 'long duration', 'dynamic bond']):
        return 'Debt - Medium/Long Duration'
    if 'income' in name and 'equity' not in name and ('fund' in name or 'plus' in name):
        return 'Debt'

    # Hybrid sub-types
    if 'multi asset' in name:
        return 'Hybrid - Multi Asset'
    if 'balanced advantage' in name or 'dynamic asset' in name:
        return 'Hybrid - Balanced Advantage'
    if 'equity savings' in name:
        return 'Hybrid - Equity Savings'
    if 'conservative' in name and ('hybrid' in name or 'hybrid' in cat):
        return 'Hybrid - Conservative'
    if 'aggressive' in name and 'hybrid' in cat:
        return 'Hybrid - Aggressive'

    # Catch remaining non-equity by API category
    if 'hybrid' in cat or 'arbitrage' in cat:
        return 'Hybrid'
    if 'debt' in cat or 'bond' in cat or 'gilt' in cat or 'liquid' in cat or 'money market' in cat or 'duration' in cat or 'floating' in cat:
        return 'Debt'
    if 'commodit' in cat or 'gold' in cat or 'silver' in cat:
        return 'Gold'

    # International / Global
    if any(w in name for w in ['nasdaq', 'nyse', 'fang+', 'international', 'global equity',
                                'us equity', 'us opportunities', 'america', 'china', 'japan',
                                'emerging market', 'world fund', 'global select']):
        return 'International'
    if 'fof' in cat and 'domestic' not in cat:
        return 'International'

    # --- Sectoral / Thematic — by actual sector ---
    if any(w in name for w in ['technology', 'tech fund', 'digital india', 'digital fund', 'it fund']):
        return 'Sectoral - Technology'
    if any(w in name for w in ['banking', 'financial services', 'financial service']):
        return 'Sectoral - Banking & Financial'
    if any(w in name for w in ['infrastructure', 'infra fund', 'economic reform']):
        return 'Sectoral - Infrastructure'
    if any(w in name for w in ['consumption', 'consumer', 'fmcg']):
        return 'Sectoral - Consumption'
    if any(w in name for w in ['healthcare', 'health fund', 'pharma']):
        return 'Sectoral - Healthcare'
    if any(w in name for w in ['energy', 'resources & energy', 'resources and energy', 'power fund']):
        return 'Sectoral - Energy'
    if any(w in name for w in ['manufacturing', 'make in india']):
        return 'Sectoral - Manufacturing'
    if 'thematic' in cat or 'sectoral' in cat:
        return 'Sectoral / Thematic'

    # --- Standard equity (broad category is fine) ---
    return normalize_category(scheme_category)


# --- Cached API Functions ---

def search_funds_api(query):
    """Search mutual funds using mfapi.in search endpoint."""
    try:
        response = requests.get(f"{MFAPI_BASE_URL}/search?q={query}", timeout=15)
        response.raise_for_status()
        data = response.json()
        return [{'schemeCode': str(f.get('schemeCode', '')), 'schemeName': f.get('schemeName', '')}
                for f in data[:20]]
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_fund_rolling_returns(scheme_code, years=5):
    """Fetch NAV data and compute rolling returns for one fund."""
    try:
        response = requests.get(f"{MFAPI_BASE_URL}/{scheme_code}", timeout=30)
        response.raise_for_status()
        data = response.json()
        nav_data = data.get('data', [])
        meta = data.get('meta', {})
        rolling = calculate_rolling_returns(nav_data, years)
        if not rolling:
            return None
        returns_values = [r['return'] for r in rolling]
        avg = sum(returns_values) / len(returns_values)
        std = math.sqrt(sum((x - avg) ** 2 for x in returns_values) / len(returns_values))
        stats = {
            'min': round(min(returns_values), 2),
            'max': round(max(returns_values), 2),
            'average': round(avg, 2),
            'stdDev': round(std, 2),
            'positivePercentage': round(
                len([r for r in returns_values if r > 0]) / len(returns_values) * 100, 2
            ),
            'totalPeriods': len(returns_values),
        }
        trailing = calculate_trailing_returns(nav_data)
        return {
            'meta': meta,
            'rollingReturns': rolling,
            'statistics': stats,
            'trailingReturns': trailing,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Analyzing 104 funds across all SEBI categories... This takes about a minute on first load.")
def load_all_rankings(years=5):
    """Fetch and rank all funds in the universe for a given rolling period."""
    def process_fund(scheme_code):
        try:
            response = requests.get(f"{MFAPI_BASE_URL}/{scheme_code}", timeout=20)
            if response.status_code != 200:
                return None
            data = response.json()
            nav_data = data.get('data', [])
            meta = data.get('meta', {})
            if not nav_data or not meta:
                return None
            rolling = calculate_rolling_returns(nav_data, years=years)
            if not rolling or len(rolling) < 10:
                return None
            returns_values = [r['return'] for r in rolling]
            avg = sum(returns_values) / len(returns_values)
            std = math.sqrt(sum((x - avg) ** 2 for x in returns_values) / len(returns_values))
            mn = min(returns_values)
            mx = max(returns_values)
            pos_pct = len([r for r in returns_values if r > 0]) / len(returns_values) * 100
            raw_robustness = (avg * (pos_pct / 100)) / (1 + std / 10)
            # Confidence discount: softer sqrt curve so newer funds aren't over-penalized
            confidence = min(1.0, math.sqrt(len(returns_values) / 1500))
            robustness = raw_robustness * confidence
            trailing = calculate_trailing_returns(nav_data)
            scheme_name = meta.get('scheme_name', 'Unknown')
            scheme_cat = meta.get('scheme_category', '')
            return {
                'schemeCode': scheme_code,
                'schemeName': scheme_name,
                'category': normalize_category(scheme_cat),
                'fineCategory': get_fine_category(scheme_name, scheme_cat),
                'fundHouse': meta.get('fund_house', 'Unknown'),
                'trailing1Y': trailing.get('1Y'),
                'trailing3Y': trailing.get('3Y'),
                'trailing5Y': trailing.get('5Y'),
                'avgReturn': round(avg, 2),
                'minReturn': round(mn, 2),
                'maxReturn': round(mx, 2),
                'stdDev': round(std, 2),
                'positivePercentage': round(pos_pct, 1),
                'totalPeriods': len(returns_values),
                'rawRobustnessScore': round(raw_robustness, 2),
                'confidence': round(confidence * 100),
                'robustnessScore': round(robustness, 2),
            }
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_fund, code): code for code in RANKING_FUND_CODES}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    results.sort(key=lambda x: x['robustnessScore'], reverse=True)
    return results


@st.cache_data(ttl=3600)
def analyze_portfolio_fund(scheme_code, years=5):
    """Fetch and fully analyze a single fund for portfolio review."""
    try:
        response = requests.get(f"{MFAPI_BASE_URL}/{scheme_code}", timeout=30)
        if response.status_code != 200:
            return None
        data = response.json()
        nav_data = data.get('data', [])
        meta = data.get('meta', {})
        if not nav_data or not meta:
            return None
        rolling = calculate_rolling_returns(nav_data, years=years)
        if not rolling or len(rolling) < 10:
            return None
        returns_values = [r['return'] for r in rolling]
        avg = sum(returns_values) / len(returns_values)
        std = math.sqrt(sum((x - avg) ** 2 for x in returns_values) / len(returns_values))
        mn = min(returns_values)
        mx = max(returns_values)
        pos_pct = len([r for r in returns_values if r > 0]) / len(returns_values) * 100
        raw_robustness = (avg * (pos_pct / 100)) / (1 + std / 10)
        confidence = min(1.0, math.sqrt(len(returns_values) / 1500))
        robustness = raw_robustness * confidence
        scheme_name = meta.get('scheme_name', 'Unknown')
        scheme_cat = meta.get('scheme_category', '')
        return {
            'schemeCode': scheme_code,
            'schemeName': scheme_name,
            'category': normalize_category(scheme_cat),
            'fineCategory': get_fine_category(scheme_name, scheme_cat),
            'fundHouse': meta.get('fund_house', 'Unknown'),
            'avgReturn': round(avg, 2),
            'minReturn': round(mn, 2),
            'maxReturn': round(mx, 2),
            'stdDev': round(std, 2),
            'positivePercentage': round(pos_pct, 1),
            'totalPeriods': len(returns_values),
            'rawRobustnessScore': round(raw_robustness, 2),
            'confidence': round(confidence * 100),
            'robustnessScore': round(robustness, 2),
        }
    except Exception:
        return None


# --- File Extraction & Fund Matching ---

def extract_holdings_from_pdf(pdf_file):
    """Extract mutual fund holdings from a PDF statement (Groww, Kuvera, MFCentral, etc.)."""
    holdings = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                # Find header row with "Scheme" or "Fund Name"
                header_idx = None
                for i, row in enumerate(table):
                    row_text = ' '.join(str(cell or '').lower() for cell in row)
                    if ('scheme' in row_text and 'name' in row_text) or ('fund' in row_text and 'amc' in row_text):
                        header_idx = i
                        break
                if header_idx is None:
                    continue

                header = table[header_idx]
                header_lower = [str(h or '').lower().strip() for h in header]

                def find_col(*keywords):
                    for idx, h in enumerate(header_lower):
                        if all(k in h for k in keywords):
                            return idx
                    return None

                name_col = find_col('scheme') or find_col('fund') or 0
                cat_col = find_col('category')
                subcat_col = find_col('sub')
                invested_col = find_col('invested')
                current_col = find_col('current')

                for row in table[header_idx + 1:]:
                    if not row or len(row) <= name_col or not row[name_col]:
                        continue
                    name = str(row[name_col]).strip()
                    if not name or 'scheme' in name.lower():
                        continue

                    holding = {'name': name}
                    if cat_col is not None and len(row) > cat_col and row[cat_col]:
                        holding['category'] = str(row[cat_col]).strip()
                    if subcat_col is not None and len(row) > subcat_col and row[subcat_col]:
                        holding['subcategory'] = str(row[subcat_col]).strip()
                    if invested_col is not None and len(row) > invested_col and row[invested_col]:
                        try:
                            holding['invested'] = float(str(row[invested_col]).replace(',', '').strip())
                        except (ValueError, TypeError):
                            pass
                    if current_col is not None and len(row) > current_col and row[current_col]:
                        try:
                            holding['current'] = float(str(row[current_col]).replace(',', '').strip())
                        except (ValueError, TypeError):
                            pass
                    holdings.append(holding)
    return holdings


def extract_holdings(uploaded_file):
    """Extract holdings from PDF, CSV, or Excel file."""
    file_type = uploaded_file.name.lower().split('.')[-1]

    if file_type == 'pdf':
        return extract_holdings_from_pdf(uploaded_file)

    # CSV or Excel
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type in ('xlsx', 'xls'):
        df = pd.read_excel(uploaded_file)
    else:
        return []

    df.columns = [str(c).strip().lower() for c in df.columns]
    name_col = next((c for c in df.columns if 'scheme' in c or 'fund' in c), df.columns[0])
    cat_col = next((c for c in df.columns if c == 'category'), None)
    subcat_col = next((c for c in df.columns if 'sub' in c and 'cat' in c), None)
    invested_col = next((c for c in df.columns if 'invested' in c), None)
    current_col = next((c for c in df.columns if 'current' in c), None)

    holdings = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        if not name or name == 'nan':
            continue
        holding = {'name': name}
        if cat_col and pd.notna(row.get(cat_col)):
            holding['category'] = str(row[cat_col]).strip()
        if subcat_col and pd.notna(row.get(subcat_col)):
            holding['subcategory'] = str(row[subcat_col]).strip()
        if invested_col and pd.notna(row.get(invested_col)):
            try:
                holding['invested'] = float(str(row[invested_col]).replace(',', ''))
            except (ValueError, TypeError):
                pass
        if current_col and pd.notna(row.get(current_col)):
            try:
                holding['current'] = float(str(row[current_col]).replace(',', ''))
            except (ValueError, TypeError):
                pass
        holdings.append(holding)
    return holdings


def consolidate_holdings(holdings):
    """Merge duplicate fund holdings (same fund, different folios)."""
    consolidated = {}
    for h in holdings:
        key = ' '.join(h['name'].lower().split())
        if key in consolidated:
            c = consolidated[key]
            if h.get('invested'):
                c['invested'] = (c.get('invested') or 0) + h.get('invested', 0)
            if h.get('current'):
                c['current'] = (c.get('current') or 0) + h.get('current', 0)
        else:
            consolidated[key] = h.copy()
    return list(consolidated.values())


@st.cache_data(ttl=3600)
def _fetch_all_schemes():
    """Fetch all schemes — raises on failure so st.cache_data won't cache errors."""
    response = requests.get(MFAPI_BASE_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError("Empty response from mfapi.in")
    return data


def get_all_schemes():
    """Fetch all mutual fund schemes. Returns [] on failure without caching the failure."""
    try:
        return _fetch_all_schemes()
    except Exception:
        return []


def match_fund_to_scheme(fund_name, all_schemes):
    """Find the best matching Direct Growth scheme for a portfolio fund name."""
    query = fund_name.lower().strip()

    # Extract key words (3+ chars, skip filler)
    filler = {'fund', 'plan', 'option', 'growth', 'direct', 'the', 'of', 'and', 'scheme'}
    words = [w for w in query.replace('-', ' ').replace('(', ' ').replace(')', ' ').split() if len(w) >= 3]
    key_words = [w for w in words if w not in filler]
    if not key_words:
        key_words = words[:3]

    # Pre-filter: must be Direct, not IDCW/dividend, and share at least 2 key words
    candidates = []
    for scheme in all_schemes:
        name = scheme.get('schemeName', '').lower()
        if 'direct' not in name:
            continue
        if 'idcw' in name or 'dividend' in name or 'bonus' in name:
            continue
        match_count = sum(1 for w in key_words if w in name)
        if match_count >= min(2, len(key_words)):
            candidates.append(scheme)

    # Relax to 1 key word if no candidates
    if not candidates:
        for scheme in all_schemes:
            name = scheme.get('schemeName', '').lower()
            if 'direct' not in name:
                continue
            if any(w in name for w in key_words):
                candidates.append(scheme)

    if not candidates:
        return None, 0

    # Score with SequenceMatcher
    best_match = None
    best_score = 0
    for scheme in candidates:
        name = scheme.get('schemeName', '').lower()
        score = SequenceMatcher(None, query, name).ratio()
        if 'growth' in name and 'growth' in query:
            score += 0.05
        if score > best_score:
            best_score = score
            best_match = scheme

    return (best_match, round(best_score * 100)) if best_score >= 0.35 else (None, 0)


# --- Score Breakdown Popover ---

def fund_link(fund, years=5, key_prefix="fl"):
    """Render a fund name as a clickable popover that shows the score breakdown."""
    name = fund['schemeName'].split(' -')[0].split(' Direct')[0]
    avg = fund['avgReturn']
    pos_pct = fund['positivePercentage']
    std = fund['stdDev']
    total = fund['totalPeriods']
    conf = fund.get('confidence', 100)
    raw_score = (avg * (pos_pct / 100)) / (1 + std / 10)
    final_score = raw_score * (conf / 100)

    with st.popover(name, use_container_width=True):
        st.markdown(f"**{name}**")
        st.caption(f"{fund.get('fineCategory', fund.get('category', ''))} · {fund.get('fundHouse', '').split(' Mutual')[0]}")
        st.markdown(
            f"| Metric | Value |\n|---|---|\n"
            f"| Average Return | {avg}% |\n"
            f"| Min / Max Return | {fund['minReturn']}% / {fund['maxReturn']}% |\n"
            f"| Std Deviation | {std} |\n"
            f"| Positive Periods | {pos_pct}% |\n"
            f"| Data Points | {total:,} |\n"
            f"| Confidence | {conf}% |"
        )
        st.markdown("**Score Calculation**")
        st.code(
            f"Raw   = ({avg} × {pos_pct / 100:.2f}) / (1 + {std}/10)\n"
            f"      = {avg * pos_pct / 100:.2f} / {1 + std / 10:.2f} = {raw_score:.2f}\n"
            f"Conf  = sqrt({total:,} / 1500) = {conf}%\n"
            f"Final = {raw_score:.2f} × {conf / 100:.2f} = {final_score:.2f}",
            language=None,
        )


# --- Session State ---
if 'selected_funds' not in st.session_state:
    st.session_state.selected_funds = []
if 'portfolio_funds' not in st.session_state:
    st.session_state.portfolio_funds = []
if 'portfolio_unmatched' not in st.session_state:
    st.session_state.portfolio_unmatched = []
if 'portfolio_file_id' not in st.session_state:
    st.session_state.portfolio_file_id = None


# --- Header ---
st.markdown(
    "<h1 style='text-align:center;'>📈 Mutual Fund 5-Year Rolling Returns Analyzer</h1>"
    "<p style='text-align:center;color:#666;'>Analyze and compare rolling returns of Indian mutual funds</p>",
    unsafe_allow_html=True,
)

# Style popover buttons as blue hyperlinks
st.markdown("""
<style>
[data-testid="stPopover"] > div > button {
    background: none !important;
    border: none !important;
    color: #1a73e8 !important;
    text-decoration: underline !important;
    padding: 0 !important;
    min-height: auto !important;
    cursor: pointer !important;
}
[data-testid="stPopover"] > div > button:hover {
    color: #0d47a1 !important;
}
[data-testid="stPopover"] > div > button p {
    color: #1a73e8 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Tabs ---
tab_analyzer, tab_rankings, tab_sip_sim, tab_portfolio, tab_pms, tab_methodology = st.tabs(["🔍 Analyzer", "🏆 Fund Rankings", "💰 SIP Simulator", "📋 Portfolio Review", "💎 PMS & AIF", "📐 Methodology"])


# ===================== ANALYZER TAB =====================
with tab_analyzer:
    st.subheader("Search & Compare Funds")

    query = st.text_input("Search by fund name or AMFI code", placeholder="e.g. Parag Parikh, HDFC Mid Cap, 122639...")

    if query and len(query) >= 2:
        results = search_funds_api(query)
        if not results:
            st.info("No funds found. Try a different search term.")
        else:
            fund_options = {r['schemeName']: r for r in results}
            selected_name = st.selectbox("Select a fund to add:", list(fund_options.keys()))

            if st.button("➕ Add to Comparison", type="primary"):
                fund = fund_options[selected_name]
                code, name = fund['schemeCode'], fund['schemeName']

                if any(f['code'] == code for f in st.session_state.selected_funds):
                    st.warning("This fund is already selected.")
                elif len(st.session_state.selected_funds) >= 5:
                    st.warning("Maximum 5 funds can be compared at once.")
                else:
                    with st.spinner(f"Loading rolling returns for {name[:40]}..."):
                        data = get_fund_rolling_returns(code)
                    if data and data.get('rollingReturns'):
                        st.session_state.selected_funds.append({
                            'code': code,
                            'name': name,
                            'data': data['rollingReturns'],
                            'stats': data['statistics'],
                            'trailing': data.get('trailingReturns', {}),
                        })
                        st.rerun()
                    else:
                        st.error("Not enough historical data for 5-year rolling returns.")

    # Show selected funds
    if st.session_state.selected_funds:
        st.markdown(f"**Selected Funds ({len(st.session_state.selected_funds)}/5)**")
        cols = st.columns(len(st.session_state.selected_funds))
        for i, fund in enumerate(st.session_state.selected_funds):
            with cols[i]:
                short_name = fund['name'].split(' -')[0].split(' Direct')[0][:25]
                if st.button(f"✕ {short_name}", key=f"rm_{fund['code']}"):
                    st.session_state.selected_funds.pop(i)
                    st.rerun()

        # --- Rolling Returns Chart ---
        col_title, col_mode, col_bench = st.columns([3, 2, 5])
        with col_title:
            st.subheader("Rolling Returns Chart")
        with col_mode:
            return_mode = st.radio("Mode", ["Lumpsum", "SIP (₹10K/mo)"], horizontal=True, key="return_mode")

        BENCHMARKS = {
            'Nifty 50': '120716',
            'Nifty Midcap 150': '147622',
            'Nifty 500': '120826',
        }
        with col_bench:
            selected_benchmarks = st.multiselect(
                "Overlay benchmarks",
                list(BENCHMARKS.keys()),
                default=[],
                key="benchmark_overlay",
            )

        is_sip = return_mode.startswith("SIP")

        # Load SIP data on demand
        if is_sip and 'sip_data' not in st.session_state.selected_funds[0]:
            with st.spinner("Computing SIP rolling returns (this may take a moment)..."):
                for fund in st.session_state.selected_funds:
                    sip_data = get_fund_rolling_returns(fund['code'], years=5)
                    if sip_data:
                        nav_response = requests.get(f"{MFAPI_BASE_URL}/{fund['code']}", timeout=30)
                        if nav_response.status_code == 200:
                            nav_raw = nav_response.json().get('data', [])
                            fund['sip_data'] = calculate_sip_rolling_returns(nav_raw, years=5)
                        else:
                            fund['sip_data'] = []
                    else:
                        fund['sip_data'] = []

        colors = ['#1a237e', '#c62828', '#2e7d32', '#f57c00', '#6a1b9a']
        benchmark_colors = ['#9e9e9e', '#bdbdbd', '#757575']
        fig = go.Figure()
        for i, fund in enumerate(st.session_state.selected_funds):
            if is_sip:
                chart_data = fund.get('sip_data', [])
            else:
                chart_data = fund['data']
            if not chart_data:
                continue
            dates = [d['date'] for d in chart_data]
            returns = [d['return'] for d in chart_data]
            label = fund['name'].split(' -')[0].split(' Direct')[0][:30]
            fig.add_trace(go.Scatter(
                x=dates, y=returns, name=label,
                mode='lines', line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='%{x}<br>%{y:.2f}%<extra>' + label + '</extra>',
            ))

        # Add benchmark traces (lumpsum only — SIP benchmark requires separate calc)
        if not is_sip:
            for bi, bname in enumerate(selected_benchmarks):
                bcode = BENCHMARKS[bname]
                bdata = get_fund_rolling_returns(bcode, years=5)
                if bdata and bdata.get('rollingReturns'):
                    bdates = [d['date'] for d in bdata['rollingReturns']]
                    breturns = [d['return'] for d in bdata['rollingReturns']]
                    fig.add_trace(go.Scatter(
                        x=bdates, y=breturns, name=bname,
                        mode='lines',
                        line=dict(color=benchmark_colors[bi % len(benchmark_colors)], width=1.5, dash='dash'),
                        hovertemplate='%{x}<br>%{y:.2f}%<extra>' + bname + '</extra>',
                    ))

        mode_label = "SIP XIRR" if is_sip else "CAGR"
        fig.update_layout(
            yaxis_title=f'{mode_label} (%)',
            xaxis_title='Date',
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(t=20, b=80),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        if is_sip:
            st.caption("SIP returns show the XIRR (annualized return) of a ₹10,000/month SIP over each rolling 5-year window.")

        # --- Trailing Returns ---
        st.subheader("Trailing Returns (Point-to-Point)")
        trailing_rows = []
        for fund in st.session_state.selected_funds:
            t = fund.get('trailing', {})
            trailing_rows.append({
                'Fund Name': fund['name'].split(' -')[0].split(' Direct')[0],
                '1Y (%)': t.get('1Y', '—'),
                '3Y (%)': t.get('3Y', '—'),
                '5Y (%)': t.get('5Y', '—'),
                '10Y (%)': t.get('10Y', '—'),
            })
        st.dataframe(pd.DataFrame(trailing_rows), use_container_width=True, hide_index=True)

        # --- Rolling Return Statistics ---
        st.subheader("Rolling Return Statistics")
        stats_rows = []
        for fund in st.session_state.selected_funds:
            s = fund['stats']
            stats_rows.append({
                'Fund Name': fund['name'].split(' -')[0].split(' Direct')[0],
                'Min (%)': s['min'],
                'Max (%)': s['max'],
                'Average (%)': s['average'],
                'Std Dev': s['stdDev'],
                'Positive Periods (%)': s['positivePercentage'],
                'Total Periods': s['totalPeriods'],
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

        # --- CSV Export ---
        all_dates = sorted(set(d['date'] for f in st.session_state.selected_funds for d in f['data']))
        csv_rows = []
        for date in all_dates:
            row = {'Date': date}
            for fund in st.session_state.selected_funds:
                lookup = {d['date']: d['return'] for d in fund['data']}
                short = fund['name'].split(' -')[0][:30]
                row[short] = lookup.get(date, '')
            csv_rows.append(row)
        csv_df = pd.DataFrame(csv_rows)
        st.download_button(
            "📥 Export Rolling Returns CSV",
            csv_df.to_csv(index=False),
            f"rolling_returns_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )
    else:
        st.info("Search and select funds above to analyze their 5-year rolling returns.")


# ===================== RANKINGS TAB =====================
with tab_rankings:
    col_period, col_top, col_spacer = st.columns([2, 2, 6])
    with col_period:
        rolling_years = st.selectbox("Rolling Period", [3, 5, 10], index=1, key="rolling_years", format_func=lambda x: f"{x}-Year Rolling")
    with col_top:
        top_n = st.selectbox("Show", [10, 20, 30, 50, 100], index=0, key="top_n_select")

    st.subheader(f"Top Funds by {rolling_years}-Year Rolling Return Robustness")
    st.caption("Funds ranked by a robustness score that rewards high average returns, consistency, and downside protection.")

    if st.button("🔄 Load / Refresh Rankings", type="primary"):
        load_all_rankings.clear()

    rankings = load_all_rankings(years=rolling_years)

    if rankings:
        # Category filter
        categories = sorted(set(f['category'] for f in rankings))
        selected_cats = st.multiselect("Filter by category", categories, default=[], placeholder="All categories")

        # Filter
        filtered = rankings
        if selected_cats:
            filtered = [f for f in rankings if f['category'] in selected_cats]
        filtered = filtered[:top_n]

        # --- Insights ---
        avg_ret = sum(f['avgReturn'] for f in rankings) / len(rankings)
        best_cat, best_score = '', 0
        cat_groups = {}
        for f in rankings:
            cat_groups.setdefault(f['category'], []).append(f['robustnessScore'])
        for cat, scores in cat_groups.items():
            cat_avg = sum(scores) / len(scores)
            if cat_avg > best_score:
                best_cat, best_score = cat, cat_avg
        decent = [f for f in rankings if f['avgReturn'] >= avg_ret]
        most_consistent = min(decent, key=lambda f: f['stdDev']) if decent else rankings[0]

        c1, c2 = st.columns(2)
        c1.metric("Funds Analyzed", len(rankings))
        c2.metric("Top Fund", rankings[0]['schemeName'].split(' -')[0].split(' Direct')[0][:22])
        c3, c4 = st.columns(2)
        c3.metric("Strongest Category", best_cat)
        c4.metric("Most Consistent", most_consistent['schemeName'].split(' -')[0].split(' Direct')[0][:22])

        # --- Rankings Table ---
        display_df = pd.DataFrame([{
            '#': i + 1,
            'Fund Name': f['schemeName'].split(' -')[0].split(' Direct')[0],
            'Category': f['category'],
            '1Y %': f.get('trailing1Y', ''),
            '3Y %': f.get('trailing3Y', ''),
            '5Y %': f.get('trailing5Y', ''),
            'Avg %': f['avgReturn'],
            'Std Dev': f['stdDev'],
            'Score': f['robustnessScore'],
        } for i, f in enumerate(filtered)])

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '#': st.column_config.NumberColumn(width="small"),
                '1Y %': st.column_config.NumberColumn(format="%.1f"),
                '3Y %': st.column_config.NumberColumn(format="%.1f"),
                '5Y %': st.column_config.NumberColumn(format="%.1f"),
                'Avg %': st.column_config.NumberColumn(format="%.1f"),
                'Std Dev': st.column_config.NumberColumn(format="%.1f"),
                'Score': st.column_config.NumberColumn(format="%.1f"),
            },
        )

        # Score breakdown selector
        fund_names = [f['schemeName'].split(' -')[0].split(' Direct')[0] for f in filtered]
        selected_fund_name = st.selectbox(
            "View score breakdown for:",
            [""] + fund_names,
            format_func=lambda x: "Select a fund..." if x == "" else x,
            key="ranking_breakdown_select",
        )
        if selected_fund_name:
            idx = fund_names.index(selected_fund_name)
            fund_link(filtered[idx], years=rolling_years, key_prefix=f"rk_{filtered[idx]['schemeCode']}")

        # --- Download ---
        download_df = pd.DataFrame([{
            '#': i + 1,
            'Fund Name': f['schemeName'].split(' -')[0].split(' Direct')[0],
            'Fund House': f.get('fundHouse', '').split(' Mutual')[0],
            'Category': f['category'],
            '1Y Return %': f.get('trailing1Y', ''),
            '3Y Return %': f.get('trailing3Y', ''),
            '5Y Return %': f.get('trailing5Y', ''),
            'Rolling Avg %': f['avgReturn'],
            'Min %': f['minReturn'],
            'Max %': f['maxReturn'],
            'Std Dev': f['stdDev'],
            'Positive %': f['positivePercentage'],
            'Data Pts': f['totalPeriods'],
            'Confidence %': f.get('confidence', 100),
            'Robustness': f['robustnessScore'],
        } for i, f in enumerate(filtered)])
        csv_data = download_df.to_csv(index=False)
        st.download_button(
            "📥 Download Rankings CSV",
            csv_data,
            f"fund_rankings_{rolling_years}Y_top{top_n}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )
    else:
        st.info("Click **Load / Refresh Rankings** to analyze funds.")


# ===================== SIP SIMULATOR TAB =====================
with tab_sip_sim:
    st.subheader("What If I Had Invested via SIP?")
    st.caption("See what your money would have grown to with a monthly SIP in any mutual fund, using actual historical NAV data.")

    sip_query = st.text_input("Search for a fund", placeholder="e.g. Parag Parikh Flexi Cap, HDFC Mid Cap, 122639...", key="sip_search")

    sip_fund_code = None
    sip_fund_name = None

    if sip_query and len(sip_query) >= 2:
        sip_results = search_funds_api(sip_query)
        if sip_results:
            sip_options = {r['schemeName']: r['schemeCode'] for r in sip_results}
            sip_selected = st.selectbox("Select fund:", list(sip_options.keys()), key="sip_fund_select")
            sip_fund_code = sip_options[sip_selected]
            sip_fund_name = sip_selected
        else:
            st.info("No funds found. Try a different search term.")

    col_amt, col_start, col_end = st.columns(3)
    with col_amt:
        sip_amount = st.number_input("Monthly SIP (₹)", min_value=500, max_value=1000000, value=10000, step=500, key="sip_amount")
    with col_start:
        sip_start_year = st.number_input("Start Year", min_value=2000, max_value=2026, value=2015, key="sip_start")
    with col_end:
        sip_end_year = st.number_input("End Year", min_value=2001, max_value=2026, value=2026, key="sip_end")

    if sip_fund_code and sip_start_year < sip_end_year:
        if st.button("🚀 Simulate SIP", type="primary"):
            with st.spinner(f"Simulating ₹{sip_amount:,}/month SIP from {sip_start_year} to {sip_end_year}..."):
                try:
                    response = requests.get(f"{MFAPI_BASE_URL}/{sip_fund_code}", timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    nav_data = data.get('data', [])

                    start_dt = datetime(sip_start_year, 1, 1)
                    end_dt = datetime(sip_end_year, 12, 31)

                    result = simulate_historical_sip(nav_data, sip_amount, start_dt, end_dt)

                    # Also simulate Nifty 50 for comparison
                    nifty_response = requests.get(f"{MFAPI_BASE_URL}/120716", timeout=30)
                    nifty_result = None
                    if nifty_response.status_code == 200:
                        nifty_nav = nifty_response.json().get('data', [])
                        nifty_result = simulate_historical_sip(nifty_nav, sip_amount, start_dt, end_dt)

                    if result:
                        st.session_state.sip_sim_result = result
                        st.session_state.sip_sim_nifty = nifty_result
                        st.session_state.sip_sim_fund_name = sip_fund_name
                        st.session_state.sip_sim_amount = sip_amount
                        st.rerun()
                    else:
                        st.error("Not enough NAV data for the selected period. Try different dates.")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    # --- Display Results ---
    if st.session_state.get('sip_sim_result'):
        result = st.session_state.sip_sim_result
        nifty_result = st.session_state.get('sip_sim_nifty')
        fund_name = st.session_state.get('sip_sim_fund_name', 'Fund').split(' -')[0].split(' Direct')[0]
        sip_amt = st.session_state.get('sip_sim_amount', 10000)
        s = result['summary']
        fd = result['fd_comparison']

        # --- Hero metrics ---
        st.markdown(f"### {fund_name}")
        c1, c2 = st.columns(2)
        c1.metric("Total Invested", f"₹{s['total_invested']:,.0f}")
        c2.metric("Current Value", f"₹{s['final_value']:,.0f}")
        c3, c4 = st.columns(2)
        c3.metric("Wealth Gained", f"₹{s['wealth_gained']:,.0f}",
                   delta=f"{s['absolute_return']:+.1f}%")
        c4.metric("SIP XIRR", f"{s['xirr']}%" if s['xirr'] is not None else "—")

        # --- Comparison bar ---
        st.markdown("### How does it compare?")
        comp_cols = st.columns(3)
        comp_cols[0].metric(f"Your SIP ({fund_name[:20]})", f"₹{s['final_value']:,.0f}")
        comp_cols[1].metric("FD @ 7%", f"₹{fd['value_at_7pct']:,.0f}",
                            delta=f"₹{s['final_value'] - fd['value_at_7pct']:+,.0f} vs Fund")
        if nifty_result:
            ns = nifty_result['summary']
            comp_cols[2].metric("Nifty 50 SIP", f"₹{ns['final_value']:,.0f}",
                                delta=f"₹{s['final_value'] - ns['final_value']:+,.0f} vs Fund")

        # --- Growth Chart ---
        st.markdown("### Growth Over Time")
        monthly = result['monthly']
        fig = go.Figure()

        # Invested line
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in monthly],
            y=[m['invested'] for m in monthly],
            name='Total Invested',
            mode='lines',
            line=dict(color='#9e9e9e', width=2, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(158,158,158,0.1)',
            hovertemplate='%{x}<br>Invested: ₹%{y:,.0f}<extra></extra>',
        ))

        # Fund value line
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in monthly],
            y=[m['value'] for m in monthly],
            name=fund_name[:25],
            mode='lines',
            line=dict(color='#1a237e', width=2.5),
            fill='tonexty',
            fillcolor='rgba(26,35,126,0.1)',
            hovertemplate='%{x}<br>Value: ₹%{y:,.0f}<extra></extra>',
        ))

        # Nifty 50 comparison
        if nifty_result and nifty_result.get('monthly'):
            nm = nifty_result['monthly']
            fig.add_trace(go.Scatter(
                x=[m['date'] for m in nm],
                y=[m['value'] for m in nm],
                name='Nifty 50 SIP',
                mode='lines',
                line=dict(color='#c62828', width=1.5, dash='dash'),
                hovertemplate='%{x}<br>Nifty 50: ₹%{y:,.0f}<extra></extra>',
            ))

        fig.update_layout(
            yaxis_title='Value (₹)',
            xaxis_title='Date',
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.12),
            margin=dict(t=20, b=60),
            height=450,
            yaxis_tickformat=',',
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Monthly Breakdown ---
        with st.expander("Monthly Breakdown", expanded=False):
            df_monthly = pd.DataFrame(monthly)
            df_monthly.columns = ['Date', 'NAV', 'Units Bought', 'Total Units', 'Invested (₹)', 'Value (₹)']
            st.dataframe(df_monthly, use_container_width=True, hide_index=True)

        # --- Download ---
        csv_data = pd.DataFrame(monthly).to_csv(index=False)
        st.download_button(
            "📥 Download SIP Breakdown CSV",
            csv_data,
            f"sip_simulation_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )

        # Clear button
        if st.button("🗑️ Clear & Simulate Another"):
            for key in ['sip_sim_result', 'sip_sim_nifty', 'sip_sim_fund_name', 'sip_sim_amount']:
                st.session_state.pop(key, None)
            st.rerun()

    elif not sip_fund_code:
        st.info("Search for a fund above, set your SIP amount and date range, then hit **Simulate SIP** to see what your investment would have grown to.")


# ===================== PORTFOLIO REVIEW TAB =====================
with tab_portfolio:
    st.subheader("Portfolio Review & Swap Recommendations")
    st.caption("Upload your mutual fund statement (PDF, CSV, or Excel) from Groww, Kuvera, MFCentral, or any broker. We'll auto-identify your funds and recommend more robust alternatives in each category.")

    col_upload, col_period = st.columns([6, 2])
    with col_period:
        portfolio_years = st.selectbox("Rolling Period", [3, 5, 10], index=1, key="portfolio_rolling_years", format_func=lambda x: f"{x}-Year Rolling")
    with col_upload:
        uploaded = st.file_uploader(
            "Upload your holdings statement",
            type=["pdf", "csv", "xlsx", "xls"],
            help="Supported: PDF/CSV/Excel from Groww, Kuvera, MFCentral, CAMS, Karvy, or any broker.",
        )

    if uploaded:
        # Detect new file — clear old results automatically
        file_id = f"{uploaded.name}_{uploaded.size}"
        if file_id != st.session_state.portfolio_file_id:
            st.session_state.portfolio_funds = []
            st.session_state.portfolio_unmatched = []
            st.session_state.portfolio_file_id = file_id

        if not st.session_state.portfolio_funds:
            # Extract holdings from file
            with st.spinner("Extracting holdings from your statement..."):
                raw_holdings = extract_holdings(uploaded)

            if not raw_holdings:
                st.error("Could not extract holdings from this file. Please make sure it contains a table with fund names.")
            else:
                holdings = consolidate_holdings(raw_holdings)
                st.success(f"Found **{len(raw_holdings)} entries** → **{len(holdings)} unique funds** after consolidating duplicate folios.")

                if st.button("🔍 Analyze Portfolio & Get Recommendations", type="primary"):
                    all_schemes = get_all_schemes()
                    portfolio_results = []
                    unmatched_funds = []
                    progress = st.progress(0, text="Matching and analyzing your funds...")

                    for i, h in enumerate(holdings):
                        scheme, confidence = match_fund_to_scheme(h['name'], all_schemes)
                        if scheme:
                            code = str(scheme['schemeCode'])
                            data = analyze_portfolio_fund(code, years=portfolio_years)
                            if data:
                                data['amount'] = h.get('invested')
                                data['current'] = h.get('current')
                                data['original_name'] = h['name']
                                data['match_confidence'] = confidence
                                data['original_category'] = h.get('category', '')
                                data['original_subcategory'] = h.get('subcategory', '')
                                portfolio_results.append(data)
                            else:
                                unmatched_funds.append(h['name'])
                        else:
                            unmatched_funds.append(h['name'])
                        progress.progress((i + 1) / len(holdings), text=f"Analyzing fund {i + 1}/{len(holdings)}...")

                    progress.empty()
                    st.session_state.portfolio_funds = portfolio_results
                    st.session_state.portfolio_unmatched = unmatched_funds
                    st.rerun()

    # --- Portfolio Analysis & Recommendations ---
    if st.session_state.portfolio_funds:
        portfolio = st.session_state.portfolio_funds

        # Clear button
        if st.button("🗑️ Clear & Upload New Statement"):
            st.session_state.portfolio_funds = []
            st.session_state.portfolio_unmatched = []
            st.session_state.portfolio_file_id = None
            st.rerun()

        # Show unmatched funds
        if st.session_state.portfolio_unmatched:
            with st.expander(f"⚠️ {len(st.session_state.portfolio_unmatched)} funds could not be matched (insufficient history or not found)"):
                for name in st.session_state.portfolio_unmatched:
                    st.markdown(f"- {name}")

        # Portfolio summary table
        st.markdown("### Your Portfolio")
        port_rows = []
        for f in portfolio:
            row = {
                'Fund Name': f['schemeName'].split(' -')[0].split(' Direct')[0],
                'Category': f.get('fineCategory', f['category']),
                'Avg %': f['avgReturn'],
                'Std Dev': f['stdDev'],
                '+ve %': f['positivePercentage'],
                'Score': f['robustnessScore'],
            }
            if f.get('amount'):
                row['Invested ₹'] = f"{f['amount']:,.0f}"
            if f.get('current'):
                row['Current ₹'] = f"{f['current']:,.0f}"
            port_rows.append(row)
        st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)

        # Score breakdown selector
        port_fund_names = [f['schemeName'].split(' -')[0].split(' Direct')[0] for f in portfolio]
        selected_port_fund = st.selectbox(
            "View score breakdown for:",
            [""] + port_fund_names,
            format_func=lambda x: "Select a fund..." if x == "" else x,
            key="portfolio_breakdown_select",
        )
        if selected_port_fund:
            pidx = port_fund_names.index(selected_port_fund)
            fund_link(portfolio[pidx], years=portfolio_years, key_prefix=f"pf_{portfolio[pidx]['schemeCode']}")

        # Load rankings for comparison — index by both broad and fine category
        rankings = load_all_rankings(years=portfolio_years)
        rankings_by_cat = {}
        rankings_by_fine = {}
        for f in rankings:
            rankings_by_cat.setdefault(f['category'], []).append(f)
            fine = f.get('fineCategory', f['category'])
            rankings_by_fine.setdefault(fine, []).append(f)

        # --- Recommendations ---
        st.markdown("### Recommendations")
        health_scores = []
        swaps_needed = 0
        already_optimal = 0

        for fund in portfolio:
            fine_cat = fund.get('fineCategory', fund['category'])
            cat = fund['category']
            fund_name = fund['schemeName'].split(' -')[0].split(' Direct')[0]

            # Skip non-equity / non-domestic categories
            NON_COMPARABLE = ('Hybrid', 'Debt', 'Gold', 'Liquid', 'Arbitrage', 'International', 'Commodities')
            if fine_cat.startswith(NON_COMPARABLE) or fine_cat == 'Other':
                already_optimal += 1
                health_scores.append(1.0)
                st.info(f"**ℹ️ {fund_name}** — **{fine_cat}** fund. Our rankings cover domestic equity funds only, so no like-for-like comparison available.")
                continue

            # Find comparable funds: try fine category first, then broad category
            cat_funds = rankings_by_fine.get(fine_cat, [])
            if not cat_funds:
                cat_funds = rankings_by_cat.get(cat, [])

            if not cat_funds:
                already_optimal += 1
                health_scores.append(1.0)
                st.info(f"**ℹ️ {fund_name}** — **{fine_cat}**. No comparable ranked funds in this specific category.")
                continue

            top_fund = cat_funds[0]

            # Check if portfolio fund is in the top 3 for its category
            fund_rank = None
            for i, ranked in enumerate(cat_funds):
                if ranked['schemeCode'] == fund['schemeCode']:
                    fund_rank = i + 1
                    break

            # Health ratio
            health = min(fund['robustnessScore'] / top_fund['robustnessScore'], 1.0) if top_fund['robustnessScore'] > 0 else 0
            health_scores.append(health)

            # Only recommend swap if the top fund is genuinely better on raw metrics
            # (not just benefiting from confidence penalty on the user's fund)
            fund_raw = fund.get('rawRobustnessScore', fund['robustnessScore'])
            top_raw = top_fund.get('rawRobustnessScore', top_fund['robustnessScore'])
            is_optimal = (fund_rank and fund_rank <= 3) or top_fund['robustnessScore'] <= fund['robustnessScore'] or top_raw <= fund_raw

            if is_optimal:
                already_optimal += 1
                if fund_rank and fund_rank <= 3:
                    st.success(f"**✅ {fund_name}** — Ranked **#{fund_rank} in {cat}**. No change needed. (Robustness: {fund['robustnessScore']})")
                elif top_raw <= fund_raw and fund.get('confidence', 100) < 70:
                    st.info(f"**📊 {fund_name}** — Strong raw metrics (Score: {fund_raw}) but limited history ({fund.get('confidence', 100)}% confidence). Hold and monitor as more data becomes available.")
                else:
                    st.success(f"**✅ {fund_name}** — Already outperforms the top ranked fund in **{cat}**. (Robustness: {fund['robustnessScore']} vs {top_fund['robustnessScore']})")
            else:
                swaps_needed += 1
                top_name = top_fund['schemeName'].split(' -')[0].split(' Direct')[0]

                with st.expander(f"🔄 **{fund_name}** → **{top_name}**", expanded=True):
                    st.caption("Data comparison only — not investment advice. Consult a SEBI-registered advisor.")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"##### Your Fund")
                        fund_link(fund, years=portfolio_years, key_prefix=f"sw_your_{fund['schemeCode']}")
                        st.caption(f"{fine_cat}")
                        st.metric("Avg Return", f"{fund['avgReturn']}%")
                        st.metric("Min Return", f"{fund['minReturn']}%")
                        st.metric("Std Dev", f"{fund['stdDev']}")
                        st.metric("Positive Periods", f"{fund['positivePercentage']}%")
                        st.metric("Robustness Score", f"{fund['robustnessScore']}")

                    with col2:
                        st.markdown(f"##### Higher-ranked alternative (#{1} in {fine_cat})")
                        fund_link(top_fund, years=portfolio_years, key_prefix=f"sw_rec_{top_fund['schemeCode']}")
                        st.caption(f"{top_fund['fundHouse'].split(' Mutual')[0]}")
                        ret_delta = round(top_fund['avgReturn'] - fund['avgReturn'], 1)
                        st.metric("Avg Return", f"{top_fund['avgReturn']}%", delta=f"{ret_delta:+.1f}%")
                        min_delta = round(top_fund['minReturn'] - fund['minReturn'], 1)
                        st.metric("Min Return", f"{top_fund['minReturn']}%", delta=f"{min_delta:+.1f}%")
                        std_delta = round(top_fund['stdDev'] - fund['stdDev'], 1)
                        st.metric("Std Dev", f"{top_fund['stdDev']}", delta=f"{std_delta:.1f}", delta_color="inverse")
                        pos_delta = round(top_fund['positivePercentage'] - fund['positivePercentage'], 1)
                        st.metric("Positive Periods", f"{top_fund['positivePercentage']}%", delta=f"{pos_delta:+.1f}%")
                        rob_delta = round(top_fund['robustnessScore'] - fund['robustnessScore'], 1)
                        st.metric("Robustness Score", f"{top_fund['robustnessScore']}", delta=f"{rob_delta:+.1f}")

                    # Rationale
                    st.markdown("---")
                    st.markdown("**Why switch?**")
                    rationale = []
                    if fund['robustnessScore'] > 0:
                        rob_pct = round((top_fund['robustnessScore'] / fund['robustnessScore'] - 1) * 100)
                        if rob_pct > 0:
                            rationale.append(f"**{rob_pct}% higher robustness score** ({top_fund['robustnessScore']} vs {fund['robustnessScore']}) — more reliable wealth creation")
                    if top_fund['avgReturn'] > fund['avgReturn']:
                        rationale.append(f"**{ret_delta:+.1f}% higher average return** over rolling {portfolio_years}-year periods")
                    if top_fund['minReturn'] > fund['minReturn']:
                        rationale.append(f"**Better downside protection** — worst {portfolio_years}-year return: {top_fund['minReturn']}% vs {fund['minReturn']}%")
                    if top_fund['stdDev'] < fund['stdDev']:
                        rationale.append(f"**More consistent** — standard deviation of {top_fund['stdDev']} vs {fund['stdDev']}")
                    if top_fund['positivePercentage'] > fund['positivePercentage']:
                        rationale.append(f"**Positive in {top_fund['positivePercentage']}%** of {portfolio_years}-year periods vs {fund['positivePercentage']}%")
                    if not rationale:
                        rationale.append(f"Top-ranked fund in {cat} category with robustness score of {top_fund['robustnessScore']}")
                    for point in rationale:
                        st.markdown(f"- {point}")

                    # Tax impact estimate
                    if fund.get('amount') and fund.get('current') and fund['current'] > fund['amount']:
                        gains = fund['current'] - fund['amount']
                        # LTCG: 12.5% on gains above Rs 1.25L (assuming >1 year holding)
                        taxable_gains = max(0, gains - 125000)
                        ltcg_tax = round(taxable_gains * 0.125)
                        net_after_tax = round(fund['current'] - ltcg_tax)
                        st.markdown("---")
                        st.markdown("**Tax impact of switching** (assuming >1 year holding)")
                        tc1, tc2, tc3, tc4 = st.columns(4)
                        tc1.metric("Gains", f"₹{gains:,.0f}")
                        tc2.metric("LTCG Tax (12.5%)", f"₹{ltcg_tax:,.0f}")
                        tc3.metric("Net After Tax", f"₹{net_after_tax:,.0f}")
                        tc4.metric("Tax as % of Value", f"{ltcg_tax / fund['current'] * 100:.1f}%")
                        if ltcg_tax > 0:
                            st.caption(f"LTCG of 12.5% applies on gains above ₹1.25L. First ₹1.25L of gains is tax-free. STCG (<1 year) is 20%.")

                    # Alternatives
                    alts = [f for f in cat_funds[1:4] if f['schemeCode'] != fund['schemeCode']]
                    if alts:
                        st.caption(f"Other strong options in {cat}:")
                        for af in alts:
                            fund_link(af, years=portfolio_years, key_prefix=f"alt_{fund['schemeCode']}_{af['schemeCode']}")

        # --- Portfolio Health Score ---
        if health_scores:
            st.divider()
            avg_health = sum(health_scores) / len(health_scores)
            score = round(avg_health * 10, 1)

            st.markdown("### Portfolio Health Score")
            col_score, col_detail = st.columns([1, 3])
            with col_score:
                color = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                st.markdown(f"## {color} {score} / 10")
            with col_detail:
                st.metric("Funds Analyzed", len(portfolio))
                st.metric("Already Optimal", f"{already_optimal}/{len(portfolio)}")
                st.metric("Swaps Suggested", swaps_needed)

            if score >= 8:
                st.success("Your portfolio is **well-optimized** — most funds are top-ranked in their categories.")
            elif score >= 6:
                st.info("Your portfolio is **decent** but has room for improvement. Consider the swap recommendations above.")
            else:
                st.warning("Your portfolio has **significant room for optimization**. The recommended swaps could meaningfully improve your risk-adjusted returns.")

        # --- Consolidation Analysis ---
        # Group portfolio funds by FINE category to avoid comparing unrelated funds
        from collections import defaultdict
        cat_holdings = defaultdict(list)
        for fund in portfolio:
            cat_holdings[fund.get('fineCategory', fund['category'])].append(fund)

        # Find categories with 2+ funds
        over_diversified = {cat: funds for cat, funds in cat_holdings.items() if len(funds) >= 2}

        if over_diversified:
            st.divider()
            st.markdown("### Consolidation Opportunities")
            st.caption("Categories where you hold multiple funds. Consolidating to the strongest 1–2 funds per category reduces overlap, simplifies tracking, and concentrates capital in the best performers.")

            total_current = sum(len(funds) for funds in over_diversified.values())
            total_after = 0
            total_freed = 0

            for cat in sorted(over_diversified.keys()):
                funds = over_diversified[cat]
                # Sort by robustness descending
                funds_sorted = sorted(funds, key=lambda f: f['robustnessScore'], reverse=True)

                # For the recommended "keep" list: also consider if a ranked fund is better than all held funds
                cat_ranked = rankings_by_fine.get(cat, rankings_by_cat.get(cat, []))
                best_ranked = cat_ranked[0] if cat_ranked else None

                # Decide how many to keep: 1 for most categories, 2 if large allocation
                keep_count = 1
                keep = funds_sorted[:keep_count]
                exit_funds = funds_sorted[keep_count:]

                # If the #1 ranked fund in this category beats all held funds, suggest it as the consolidation target
                upgrade_to = None
                if best_ranked and best_ranked['robustnessScore'] > keep[0]['robustnessScore']:
                    # Check if user already holds the ranked fund
                    holds_best = any(f['schemeCode'] == best_ranked['schemeCode'] for f in funds)
                    if not holds_best:
                        upgrade_to = best_ranked

                total_after += keep_count
                freed_amount = sum(f.get('current') or f.get('amount') or 0 for f in exit_funds)
                total_freed += freed_amount

                keep_name = keep[0]['schemeName'].split(' -')[0].split(' Direct')[0]

                with st.expander(f"**{cat}** — {len(funds)} funds → keep {keep_count}", expanded=len(funds) >= 3):
                    # Show all held funds ranked
                    st.markdown("**Your holdings in this category (ranked by robustness):**")
                    for i, f in enumerate(funds_sorted):
                        fname = f['schemeName'].split(' -')[0].split(' Direct')[0]
                        amt_str = f" · ₹{f['current']:,.0f}" if f.get('current') else (f" · ₹{f['amount']:,.0f}" if f.get('amount') else "")
                        action = "Keep" if i < keep_count else "Consider exiting"
                        icon = "✅" if i < keep_count else "🔻"
                        st.markdown(f"{icon} **{fname}** — Score: {f['robustnessScore']}, Avg: {f['avgReturn']}%{amt_str} → *{action}*")
                        fund_link(f, years=portfolio_years, key_prefix=f"con_{cat}_{f['schemeCode']}")

                    if upgrade_to:
                        up_name = upgrade_to['schemeName'].split(' -')[0].split(' Direct')[0]
                        st.markdown(f"💡 **Even better:** Consolidate into **{up_name}** (Score: {upgrade_to['robustnessScore']}, Avg: {upgrade_to['avgReturn']}%) — ranked #1 in {cat}.")
                        fund_link(upgrade_to, years=portfolio_years, key_prefix=f"up_{cat}_{upgrade_to['schemeCode']}")

                    # Rationale
                    st.markdown("---")
                    if len(funds) >= 3:
                        st.markdown(f"**Why consolidate?** Holding **{len(funds)} funds in {cat}** creates significant portfolio overlap — "
                                    f"these funds likely hold many of the same underlying stocks. "
                                    f"Consolidating into the strongest performer simplifies your portfolio and concentrates capital where returns are best.")
                    else:
                        rob_diff = round(funds_sorted[0]['robustnessScore'] - funds_sorted[-1]['robustnessScore'], 1)
                        st.markdown(f"**Why consolidate?** Your strongest {cat} fund has a robustness score of **{funds_sorted[0]['robustnessScore']}** "
                                    f"vs **{funds_sorted[-1]['robustnessScore']}** for the weakest — a gap of {rob_diff}. "
                                    f"Consolidating removes the drag of the weaker fund.")

                    if freed_amount > 0:
                        st.markdown(f"**Capital freed up:** ₹{freed_amount:,.0f} — can be redeployed into the keeper fund or a missing category.")

            # Summary
            total_can_exit = total_current - total_after
            st.markdown(f"---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Overlapping Funds", total_current)
            col_b.metric("Can Be Consolidated", total_can_exit)
            if total_freed > 0:
                col_c.metric("Capital to Redeploy", f"₹{total_freed:,.0f}")

        # --- Missing Category Opportunities ---
        # Find equity categories present in rankings but absent from portfolio
        equity_categories = {"Large Cap", "Large & Mid Cap", "Mid Cap", "Small Cap",
                             "Flexi Cap", "Multi Cap", "Value / Contra", "Focused",
                             "ELSS", "Sectoral / Thematic", "Index Fund"}
        portfolio_cats = set(f['category'] for f in portfolio)
        missing_cats = sorted(equity_categories - portfolio_cats)

        if missing_cats and rankings:
            st.divider()
            st.markdown("### Categories Missing From Your Portfolio")
            st.caption("These equity categories have strong performers in our rankings but are not represented in your portfolio.")

            # Compute portfolio average robustness (equity funds only)
            equity_funds = [f for f in portfolio if f['category'] in equity_categories]
            portfolio_avg_rob = (sum(f['robustnessScore'] for f in equity_funds) / len(equity_funds)) if equity_funds else 0
            portfolio_avg_ret = (sum(f['avgReturn'] for f in equity_funds) / len(equity_funds)) if equity_funds else 0

            opportunities = []
            for cat in missing_cats:
                cat_funds = rankings_by_cat.get(cat, [])
                if not cat_funds:
                    continue
                top = cat_funds[0]
                cat_avg_rob = sum(f['robustnessScore'] for f in cat_funds) / len(cat_funds)
                cat_avg_ret = sum(f['avgReturn'] for f in cat_funds) / len(cat_funds)
                opportunities.append({
                    'category': cat,
                    'top_fund': top,
                    'cat_avg_rob': round(cat_avg_rob, 1),
                    'cat_avg_ret': round(cat_avg_ret, 1),
                    'num_funds': len(cat_funds),
                    'beats_portfolio': cat_avg_rob > portfolio_avg_rob,
                })

            # Sort: categories that beat portfolio average first, then by avg robustness
            opportunities.sort(key=lambda x: (not x['beats_portfolio'], -x['cat_avg_rob']))

            for opp in opportunities:
                top = opp['top_fund']
                top_name = top['schemeName'].split(' -')[0].split(' Direct')[0]
                cat = opp['category']

                if opp['beats_portfolio'] and portfolio_avg_rob > 0:
                    icon = "🟢"
                    badge = f"Category avg robustness **{opp['cat_avg_rob']}** vs your portfolio avg **{round(portfolio_avg_rob, 1)}**"
                else:
                    icon = "🔵"
                    badge = f"Category avg robustness: **{opp['cat_avg_rob']}**"

                with st.expander(f"{icon} **{cat}** — Top pick: {top_name}"):
                    st.markdown(badge)
                    st.markdown("**Top pick:**")
                    fund_link(top, years=portfolio_years, key_prefix=f"miss_top_{cat}")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Avg Return", f"{top['avgReturn']}%")
                    c2.metric("Min Return", f"{top['minReturn']}%")
                    c3.metric("Std Dev", f"{top['stdDev']}")
                    c4.metric("Positive %", f"{top['positivePercentage']}%")
                    c5.metric("Robustness", f"{top['robustnessScore']}")

                    # Show top 3 in category
                    if len(rankings_by_cat.get(cat, [])) > 1:
                        st.markdown("**Other top funds in this category:**")
                        for i, f in enumerate(rankings_by_cat[cat][1:3]):
                            fname = f['schemeName'].split(' -')[0].split(' Direct')[0]
                            st.markdown(f"{i+2}. **{fname}** — Avg: {f['avgReturn']}%, Score: {f['robustnessScore']}")
                            fund_link(f, years=portfolio_years, key_prefix=f"miss_{cat}_{f['schemeCode']}")

                    if opp['beats_portfolio']:
                        ret_uplift = round(opp['cat_avg_ret'] - portfolio_avg_ret, 1)
                        st.markdown(f"**Why consider?** This category averages **{opp['cat_avg_ret']}% returns** "
                                    f"({'**' + str(ret_uplift) + '%** higher' if ret_uplift > 0 else 'comparable'} "
                                    f"to your portfolio's {round(portfolio_avg_ret, 1)}%) with a robustness score of {opp['cat_avg_rob']}.")

            # Download recommendations
            rec_rows = []
            for fund in portfolio:
                cat = fund['category']
                cat_funds = rankings_by_cat.get(cat, [])
                top = cat_funds[0] if cat_funds else None
                fund_rank = None
                if cat_funds:
                    for i, r in enumerate(cat_funds):
                        if r['schemeCode'] == fund['schemeCode']:
                            fund_rank = i + 1
                            break
                rec_rows.append({
                    'Current Fund': fund['schemeName'].split(' -')[0].split(' Direct')[0],
                    'Category': fund.get('fineCategory', cat),
                    'Current Robustness': fund['robustnessScore'],
                    'Current Avg Return %': fund['avgReturn'],
                    'Action': ("Keep" if (fund_rank and fund_rank <= 3) or (top and top['robustnessScore'] <= fund['robustnessScore']) else "Consider Swap"),
                    'Recommended Fund': top['schemeName'].split(' -')[0].split(' Direct')[0] if top and top['robustnessScore'] > fund['robustnessScore'] else '—',
                    'Recommended Robustness': top['robustnessScore'] if top and top['robustnessScore'] > fund['robustnessScore'] else '—',
                    'Recommended Avg Return %': top['avgReturn'] if top and top['robustnessScore'] > fund['robustnessScore'] else '—',
                })
            rec_df = pd.DataFrame(rec_rows)
            st.download_button(
                "📥 Download Recommendations CSV",
                rec_df.to_csv(index=False),
                f"portfolio_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )
    elif not uploaded:
        st.info("Upload your mutual fund statement (PDF from Groww/Kuvera, CSV, or Excel) to get personalized swap recommendations.")


# ===================== PMS & AIF REFERENCE TAB =====================
with tab_pms:
    st.subheader("PMS & AIF Reference Guide — India")
    st.caption("Portfolio Management Services (min ₹50L) and Alternative Investment Funds (min ₹1Cr). Data from public sources — verify with official factsheets before investing.")

    pms_view = st.radio("View", ["PMS Strategies", "AIF Category III", "Fee Comparison", "PMS vs Mutual Funds"], horizontal=True)

    if pms_view == "PMS Strategies":
        # --- PMS Data ---
        # Data sourced from pmsaifworld.com, pmsbazaar.com, provider websites (Mar 2026)
        # SI CAGR = Since Inception CAGR (TWRR). Verify with latest factsheets.
        pms_data = [
            {"Provider": "Carnelian", "Strategy": "Shift Strategy", "Approach": "Mid & Small Cap", "Inception": 2020, "SI CAGR %": 36.0, "5Y CAGR %": None, "3Y CAGR %": 29.3, "Fee Model": "2.5% + 15-20% > 8% hurdle", "AUM (₹ Cr)": 4400},
            {"Provider": "Stallion Asset", "Strategy": "Core Fund", "Approach": "Multicap", "Inception": 2018, "SI CAGR %": 29.9, "5Y CAGR %": 33.4, "3Y CAGR %": 37.4, "Fee Model": "1% + 15% > 10% hurdle", "AUM (₹ Cr)": 5915},
            {"Provider": "Abakkus", "Strategy": "Emerging Opportunities", "Approach": "Mid & Small Cap", "Inception": 2018, "SI CAGR %": 30.3, "5Y CAGR %": None, "3Y CAGR %": 24.4, "Fee Model": "1.5% + 20% > 10% hurdle", "AUM (₹ Cr)": 5382},
            {"Provider": "Wallfort", "Strategy": "Diversified Fund", "Approach": "Multicap", "Inception": 2020, "SI CAGR %": 28.7, "5Y CAGR %": None, "3Y CAGR %": 32.6, "Fee Model": "2% + 20% > 10% hurdle", "AUM (₹ Cr)": 406},
            {"Provider": "SageOne", "Strategy": "Core Portfolio (SCP)", "Approach": "Multicap", "Inception": 2012, "SI CAGR %": 26.3, "5Y CAGR %": 26.6, "3Y CAGR %": 22.7, "Fee Model": "13.75% profit share, no fixed fee", "AUM (₹ Cr)": 4320},
            {"Provider": "Green Lantern", "Strategy": "Growth Fund", "Approach": "Mid & Small Cap", "Inception": 2018, "SI CAGR %": 24.0, "5Y CAGR %": 48.8, "3Y CAGR %": 40.2, "Fee Model": "2.5% + 20% > 10% hurdle", "AUM (₹ Cr)": 1139},
            {"Provider": "Equitree", "Strategy": "Emerging Opportunities", "Approach": "Small & Micro Cap", "Inception": 2017, "SI CAGR %": 23.0, "5Y CAGR %": 43.0, "3Y CAGR %": None, "Fee Model": "2% + 20% on portfolio doubling/5yr", "AUM (₹ Cr)": 1123},
            {"Provider": "Sameeksha", "Strategy": "Equity Fund", "Approach": "Multicap", "Inception": 2015, "SI CAGR %": 22.2, "5Y CAGR %": 29.1, "3Y CAGR %": 26.8, "Fee Model": "1.5% + 20% > 10% hurdle", "AUM (₹ Cr)": 1112},
            {"Provider": "Alchemy", "Strategy": "High Growth", "Approach": "Flexicap", "Inception": 2002, "SI CAGR %": 19.9, "5Y CAGR %": 12.4, "3Y CAGR %": 21.0, "Fee Model": "2.5% + 15-20% > 8-10% hurdle", "AUM (₹ Cr)": 957},
            {"Provider": "2Point2 Capital", "Strategy": "Long Term Value", "Approach": "Concentrated Value", "Inception": 2016, "SI CAGR %": 19.7, "5Y CAGR %": 19.1, "3Y CAGR %": 24.7, "Fee Model": "Value-oriented, max 15 stocks", "AUM (₹ Cr)": 1816},
            {"Provider": "ICICI Pru PMS", "Strategy": "Contra Strategy", "Approach": "Multicap Contra", "Inception": 2006, "SI CAGR %": 19.6, "5Y CAGR %": 26.7, "3Y CAGR %": 21.8, "Fee Model": "2.5% fixed or 1% + 20%", "AUM (₹ Cr)": None},
            {"Provider": "Marcellus", "Strategy": "Consistent Compounders", "Approach": "Large Cap Quality", "Inception": 2018, "SI CAGR %": 18.6, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "0% + 20% > 8% (or 1% + 15% > 12%)", "AUM (₹ Cr)": 2476},
            {"Provider": "Negen Capital", "Strategy": "Special Situations", "Approach": "Multicap", "Inception": 2017, "SI CAGR %": 18.6, "5Y CAGR %": 37.5, "3Y CAGR %": 24.3, "Fee Model": "2% + 20% > 10% hurdle", "AUM (₹ Cr)": 1196},
            {"Provider": "ASK", "Strategy": "Growth Portfolio", "Approach": "Multicap", "Inception": 2001, "SI CAGR %": 17.4, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% fixed or 1.5% + 20% > 10%", "AUM (₹ Cr)": 26869},
            {"Provider": "Ambit", "Strategy": "Coffee Can Portfolio", "Approach": "Large Cap Quality", "Inception": 2017, "SI CAGR %": 16.5, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% + 15% profit share", "AUM (₹ Cr)": 1700},
            {"Provider": "ASK", "Strategy": "Indian Entrepreneur (IEP)", "Approach": "Multicap", "Inception": 2010, "SI CAGR %": 15.8, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% fixed or 1.5% + 20% > 10%", "AUM (₹ Cr)": 26869},
            {"Provider": "Renaissance", "Strategy": "India Next Portfolio", "Approach": "Multicap", "Inception": 2014, "SI CAGR %": 15.5, "5Y CAGR %": 33.3, "3Y CAGR %": 20.2, "Fee Model": "2% + 20% > 10% hurdle", "AUM (₹ Cr)": 892},
            {"Provider": "Motilal Oswal", "Strategy": "NTDOP", "Approach": "Multicap", "Inception": 2003, "SI CAGR %": 15.4, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% + 20% > 8% hurdle", "AUM (₹ Cr)": 14483},
            {"Provider": "Buoyant Capital", "Strategy": "Opportunities Multicap", "Approach": "Multicap", "Inception": 2016, "SI CAGR %": 15.0, "5Y CAGR %": 37.7, "3Y CAGR %": None, "Fee Model": "Target 15% CAGR, 18-25 stocks", "AUM (₹ Cr)": None},
            {"Provider": "White Oak", "Strategy": "India Pioneers Equity", "Approach": "Multicap", "Inception": 2018, "SI CAGR %": None, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% fixed or 1.5% + 20% > 10%", "AUM (₹ Cr)": 5457},
            {"Provider": "Unifi", "Strategy": "BCAD", "Approach": "Multi-Strategy", "Inception": 2018, "SI CAGR %": None, "5Y CAGR %": 22.2, "3Y CAGR %": 24.5, "Fee Model": "2% + 1.5% variable", "AUM (₹ Cr)": 20934},
            {"Provider": "360 ONE", "Strategy": "Multicap PMS", "Approach": "Multicap + Hedging", "Inception": 2013, "SI CAGR %": None, "5Y CAGR %": 20.0, "3Y CAGR %": 19.8, "Fee Model": "2.5% fixed (tiered by AUM)", "AUM (₹ Cr)": 3085},
            {"Provider": "Marcellus", "Strategy": "Kings of Capital", "Approach": "Financial Sector", "Inception": 2020, "SI CAGR %": 10.5, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% fixed, 10% hurdle", "AUM (₹ Cr)": 205},
            {"Provider": "Invesco", "Strategy": "DAWN / RISE", "Approach": "Multi-Strategy", "Inception": 2016, "SI CAGR %": None, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "Flat 10% profit share (lowest in industry)", "AUM (₹ Cr)": 20178},
            {"Provider": "Valentis", "Strategy": "Rising Star Opportunity", "Approach": "Small Cap", "Inception": 2016, "SI CAGR %": None, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% + 15% > 10% hurdle", "AUM (₹ Cr)": 1685},
            {"Provider": "Kotak", "Strategy": "Special Situations Value", "Approach": "Special Situations", "Inception": None, "SI CAGR %": None, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "2.5% or 1.5% + 15% > 10%", "AUM (₹ Cr)": 1767},
            {"Provider": "Counter Cyclical", "Strategy": "Diversified Long Term Value", "Approach": "Small Cap Value", "Inception": 2019, "SI CAGR %": None, "5Y CAGR %": None, "3Y CAGR %": None, "Fee Model": "0% fixed + 20% > 10% hurdle (no mgmt fee)", "AUM (₹ Cr)": 793},
        ]

        df_pms = pd.DataFrame(pms_data)
        df_pms = df_pms.sort_values('SI CAGR %', ascending=False, na_position='last').reset_index(drop=True)
        df_pms.index = df_pms.index + 1
        df_pms.index.name = '#'

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            approach_filter = st.multiselect("Filter by approach", sorted(df_pms['Approach'].unique()), default=[], placeholder="All approaches")
        with col_f2:
            provider_filter = st.multiselect("Filter by provider", sorted(df_pms['Provider'].unique()), default=[], placeholder="All providers")

        filtered = df_pms.copy()
        if approach_filter:
            filtered = filtered[filtered['Approach'].isin(approach_filter)]
        if provider_filter:
            filtered = filtered[filtered['Provider'].isin(provider_filter)]

        st.dataframe(filtered, use_container_width=True, column_config={
            'SI CAGR %': st.column_config.NumberColumn(format="%.1f"),
            '5Y CAGR %': st.column_config.NumberColumn(format="%.1f"),
            '3Y CAGR %': st.column_config.NumberColumn(format="%.1f"),
            'AUM (₹ Cr)': st.column_config.NumberColumn(format="%,.0f"),
        })

        # Insights
        has_cagr = df_pms.dropna(subset=['SI CAGR %'])
        if len(has_cagr) > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Strategies", len(df_pms))
            c2.metric("Avg SI CAGR", f"{has_cagr['SI CAGR %'].mean():.1f}%")
            c3.metric("Top Performer", f"{has_cagr.iloc[0]['Provider']} {has_cagr.iloc[0]['Strategy']}"[:30])
            c4.metric("Min Investment", "₹50 Lakh")

        st.warning("**Data reliability:** Returns sourced from [PMS AIF World](https://www.pmsaifworld.com/pms/best-pms-in-india/), [PMSBazaar](https://pmsbazaar.com/PMSRanking), and provider websites. SI CAGR is Time-Weighted (TWRR) and varies by measurement date. Longer track records (Alchemy 2002, ASK 2001, Motilal 2003) are more meaningful. **Always verify with the latest official factsheet before investing.** Returns shown are pre-tax and net of fees unless stated otherwise.")

    elif pms_view == "AIF Category III":
        aif_data = [
            {"Provider": "A9 Finsight", "Fund": "Finavenue Growth Fund", "Type": "Long-Only", "FY25 Return %": 46.66, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": None},
            {"Provider": "Negen Capital", "Fund": "Undiscovered Value Fund", "Type": "Long-Only", "FY25 Return %": 33.61, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": None},
            {"Provider": "Swyom Advisors", "Fund": "India Alpha Fund", "Type": "Long-Short", "FY25 Return %": 31.79, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": None},
            {"Provider": "Helios Capital", "Fund": "India Long/Short", "Type": "Long-Short", "FY25 Return %": None, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": 1347},
            {"Provider": "Avendus", "Fund": "Enhanced Return Fund II", "Type": "70% Long + 30% L/S", "FY25 Return %": None, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": None},
            {"Provider": "Nuvama", "Fund": "MARS (Multi-Asset Return)", "Type": "Market Neutral", "FY25 Return %": 13.0, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": 11307},
            {"Provider": "WhiteSpace Alpha", "Fund": "Equity Plus Fund 1", "Type": "Market Neutral / Quant", "FY25 Return %": 14.5, "Min Invest": "₹1 Cr", "AUM (₹ Cr)": None},
        ]

        df_aif = pd.DataFrame(aif_data)
        df_aif = df_aif.sort_values('FY25 Return %', ascending=False, na_position='last').reset_index(drop=True)
        df_aif.index = df_aif.index + 1
        df_aif.index.name = '#'

        st.dataframe(df_aif, use_container_width=True, column_config={
            'FY25 Return %': st.column_config.NumberColumn(format="%.1f"),
            'AUM (₹ Cr)': st.column_config.NumberColumn(format="%,.0f"),
        })

        c1, c2, c3 = st.columns(3)
        c1.metric("Min Investment", "₹1 Crore")
        c2.metric("Typical Fees", "2% + 20% above hurdle")
        c3.metric("Top FY25 Return", f"{aif_data[0]['FY25 Return %']}%")

        st.markdown("""
**Key differences from Mutual Funds:**
- **Higher minimum** — ₹1 Cr vs no minimum for MFs
- **Less liquidity** — lock-in periods common (1-3 years)
- **Can go short** — Long-Short and Market Neutral strategies can profit in falling markets
- **Less regulated disclosure** — no daily NAV requirement like mutual funds
- **Tax** — Category III AIFs are taxed at fund level (not pass-through), resulting in higher effective tax
""")

        st.warning("**Avendus Absolute Return Fund** — one of India's most popular AIFs — was wound down in January 2025 due to SEBI regulatory changes on derivative usage.")

    elif pms_view == "Fee Comparison":
        st.markdown("### Fee Structure Comparison")
        st.caption("How different PMS providers charge — and how it impacts your net returns")

        fee_data = [
            {"Provider": "Invesco", "Fixed Fee": "0%", "Performance Fee": "10% on all profits", "Hurdle": "None", "Model": "Pure profit share"},
            {"Provider": "SageOne", "Fixed Fee": "0%", "Performance Fee": "13.75% on all profits", "Hurdle": "None", "Model": "Pure profit share"},
            {"Provider": "Marcellus CCP", "Fixed Fee": "0%", "Performance Fee": "20%", "Hurdle": "8%", "Model": "Performance only"},
            {"Provider": "Marcellus CCP Alt", "Fixed Fee": "1.0%", "Performance Fee": "15%", "Hurdle": "12%", "Model": "Hybrid"},
            {"Provider": "Marcellus LCP", "Fixed Fee": "1.5%", "Performance Fee": "20%", "Hurdle": "10%", "Model": "Hybrid + exit load"},
            {"Provider": "ASK / Kotak", "Fixed Fee": "1.5%", "Performance Fee": "20%", "Hurdle": "10%", "Model": "Hybrid"},
            {"Provider": "ASK / Motilal", "Fixed Fee": "2.5%", "Performance Fee": "None", "Hurdle": "—", "Model": "Fixed only"},
            {"Provider": "Alchemy", "Fixed Fee": "2.5%", "Performance Fee": "15-20%", "Hurdle": "8-10%", "Model": "Hybrid (expensive)"},
            {"Provider": "Motilal / Carnelian", "Fixed Fee": "2.5%", "Performance Fee": "20%", "Hurdle": "8%", "Model": "Hybrid (expensive)"},
            {"Provider": "Nippon India", "Fixed Fee": "2.5%", "Performance Fee": "None", "Hurdle": "—", "Model": "Fixed only"},
        ]

        st.dataframe(pd.DataFrame(fee_data), use_container_width=True, hide_index=True)

        # Fee impact illustration
        st.markdown("### Fee Impact on ₹50 Lakh Over 5 Years")
        st.caption("Assuming 20% gross return — how different fee structures eat into your wealth")

        gross = 50_00_000
        scenarios = [
            ("Direct MF (0.5% TER)", 0.005, 0, 0),
            ("Invesco PMS (0% + 10%)", 0, 0.10, 0),
            ("SageOne (0% + 13.75%)", 0, 0.1375, 0),
            ("Marcellus CCP (0% + 20% > 8%)", 0, 0.20, 0.08),
            ("Hybrid (1.5% + 20% > 10%)", 0.015, 0.20, 0.10),
            ("Expensive (2.5% + 20% > 8%)", 0.025, 0.20, 0.08),
            ("Fixed Only (2.5%)", 0.025, 0, 0),
        ]

        impact_rows = []
        for label, fixed, perf, hurdle in scenarios:
            gross_rate = 0.20
            net_rate = gross_rate - fixed
            perf_drag = max(0, net_rate - hurdle) * perf
            final_rate = net_rate - perf_drag
            final_val = gross * ((1 + final_rate) ** 5)
            fees_paid = gross * ((1 + gross_rate) ** 5) - final_val
            impact_rows.append({
                "Fee Model": label,
                "Net CAGR %": round(final_rate * 100, 1),
                "Final Value (₹)": f"₹{final_val:,.0f}",
                "Fees Paid (₹)": f"₹{fees_paid:,.0f}",
            })

        st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
        st.info("This is a simplified illustration. Actual fees depend on high-water marks, crystallization periods, and market conditions.")

    else:  # PMS vs Mutual Funds
        st.markdown("### When Does PMS Make Sense Over Mutual Funds?")

        comparison = [
            {"Feature": "Minimum Investment", "Mutual Fund": "₹100 (SIP)", "PMS": "₹50 Lakh", "AIF Cat III": "₹1 Crore"},
            {"Feature": "Regulation", "Mutual Fund": "SEBI (strict, daily NAV)", "PMS": "SEBI (moderate)", "AIF Cat III": "SEBI (flexible)"},
            {"Feature": "Transparency", "Mutual Fund": "Daily NAV, monthly portfolio", "PMS": "Demat holdings visible", "AIF Cat III": "Periodic reporting"},
            {"Feature": "Customization", "Mutual Fund": "None (pooled)", "PMS": "Some (separate account)", "AIF Cat III": "Strategy-level"},
            {"Feature": "Liquidity", "Mutual Fund": "T+1 to T+3 redemption", "PMS": "T+3, possible exit load", "AIF Cat III": "Lock-in (1-3 years)"},
            {"Feature": "Taxation", "Mutual Fund": "LTCG 12.5% > ₹1.25L", "PMS": "Same as direct equity", "AIF Cat III": "Taxed at fund level (MMR)"},
            {"Feature": "Typical Fees", "Mutual Fund": "0.3-1.5% TER", "PMS": "1.5-2.5% + 15-20% perf", "AIF Cat III": "2% + 20% perf"},
            {"Feature": "Can Short Sell", "Mutual Fund": "No", "PMS": "No", "AIF Cat III": "Yes"},
            {"Feature": "Track Record Data", "Mutual Fund": "Public (mfapi.in)", "PMS": "Limited (SEBI monthly)", "AIF Cat III": "Very limited"},
            {"Feature": "Best For", "Mutual Fund": "Everyone", "PMS": "HNIs wanting quality + control", "AIF Cat III": "Sophisticated investors"},
        ]

        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

        st.markdown("""
### Key Takeaways

**PMS is worth considering when:**
- You have ₹50L+ to allocate to equities
- You want a specific investment philosophy (e.g., Marcellus quality, SageOne deep value)
- You value seeing individual stock holdings in your demat
- The PMS has a **5+ year track record** beating comparable MF categories after fees

**Stick with Direct Mutual Funds when:**
- Investment amount is below ₹50L
- You want daily liquidity and simple taxation
- A comparable MF category delivers similar returns at 0.3-1% TER vs 2.5%+ PMS fees
- The PMS track record is less than 3 years

**Consider AIF Cat III when:**
- You have ₹1Cr+ and want absolute return / market-neutral strategies
- You want exposure to long-short or derivatives-based strategies unavailable in MFs
- You can accept 1-3 year lock-in periods
""")

        # Quick comparison: Top PMS vs Top MF
        st.markdown("### Head-to-Head: Top PMS vs Top Mutual Funds")
        st.caption("Do PMS strategies justify their higher fees?")

        h2h = [
            {"Category": "Multicap", "Top PMS": "White Oak Pioneers (25.7%)", "Top MF (5Y Rolling Avg)": "Check Rankings tab", "Fee Gap": "~2% fixed + 20% perf vs 0.5% TER"},
            {"Category": "Small Cap", "Top PMS": "SageOne SSP (40.2%)", "Top MF (5Y Rolling Avg)": "Check Rankings tab", "Fee Gap": "13.75% profit share vs 0.5% TER"},
            {"Category": "Large Cap Quality", "Top PMS": "Marcellus CCP (18.6%)", "Top MF (5Y Rolling Avg)": "Check Rankings tab", "Fee Gap": "0% + 20% > 8% vs 0.5% TER"},
            {"Category": "Flexicap", "Top PMS": "Alchemy High Growth (19.9%)", "Top MF (5Y Rolling Avg)": "Check Rankings tab", "Fee Gap": "2.5% + 15% > 10% vs 0.5% TER"},
        ]
        st.dataframe(pd.DataFrame(h2h), use_container_width=True, hide_index=True)
        st.info("Use the **Fund Rankings** tab to see actual mutual fund rolling return data for direct comparison. PMS returns shown are since-inception CAGR (not rolling) and vary by measurement date.")


# ===================== METHODOLOGY TAB =====================
with tab_methodology:
    st.markdown("""
## How This Works

This tool analyzes Indian equity mutual funds by computing their **5-year rolling returns** and ranking them by a **robustness score** that captures not just how high the returns are, but how consistent and reliable they have been over time.

---

### 1. Fund Universe

We analyze **104 Direct Growth equity mutual funds** across 11 SEBI-defined categories:

| Category Group | Categories | Count |
|---|---|---|
| **Market Cap Based** | Large Cap, Large & Mid Cap, Mid Cap, Small Cap | 41 funds |
| **Diversified / Strategy** | Flexi Cap, Multi Cap, Value/Contra, Focused | 36 funds |
| **Tax Saving & Others** | ELSS, Sectoral/Thematic, Index Funds | 27 funds |

**Selection criteria:**
- Only **Direct Growth** plans (no regular plans, no IDCW/dividend)
- Funds from **25+ top AMCs** (SBI, HDFC, ICICI, Axis, Kotak, Mirae, Nippon, DSP, UTI, Canara Robeco, Franklin, Parag Parikh, Quant, Motilal Oswal, Tata, Aditya Birla, Edelweiss, Sundaram, PGIM, Bandhan, HSBC, Invesco, etc.)
- Only funds with **sufficient NAV history** (≥ 5 years of data) are included in rankings

---

### 2. Rolling Returns Calculation

For each fund, we compute the **5-year CAGR** at every available date:

> **CAGR = (NAV_today / NAV_5_years_ago)^(1/5) − 1**

This is computed for **every trading day** where a matching NAV exists ~5 years prior (within a 15-day window).

> **Why rolling returns?** A single-point return ("this fund gave 18% over 5 years") depends entirely on entry and exit dates. Rolling returns show what the return would have been if you invested on *any* date and held for 5 years — revealing consistency, not just peak performance.

---

### 3. Statistical Metrics

| Metric | What It Tells You |
|---|---|
| **Average Return** | Mean 5-year CAGR across all windows. Higher = better wealth creation. |
| **Minimum Return** | Worst 5-year return ever. A positive min means the fund *never* lost money over any 5-year period. |
| **Maximum Return** | Best 5-year return. Shows upside potential. |
| **Standard Deviation** | How much returns vary. Lower = more predictable. |
| **Positive Periods %** | % of all 5-year windows with positive returns. 100% = never negative over 5 years. |

---

### 4. Robustness Score

> **Raw Score = (Avg Return × Positive%) / (1 + StdDev / 10)**
>
> **Confidence = min(1, sqrt(Data Points / 1500))**
>
> **Final Robustness Score = Raw Score × Confidence**

This rewards:
- **High average returns** — wealth creation over time
- **High positive period %** — rarely loses money over 5-year windows
- **Low standard deviation** — consistent, not wildly swinging
- **More data points** — a fund that has been through multiple market cycles (bull runs, corrections, crashes) gets full credit, while a newer fund with limited history is discounted

**Why the confidence adjustment?** A fund launched 5.5 years ago would have only ~400 rolling 5-year data points, all starting from a similar market period. If that period happened to be a market low (e.g., COVID crash), every window looks great — but that tells you about the market timing, not the fund quality. A fund with 2,000+ data points spanning 10+ years has proven itself across diverse market regimes. The confidence factor uses a square root curve so the discount is gentle — a fund with 400 data points gets ~52% confidence (not a harsh 27%). Swap recommendations also compare raw scores to avoid penalizing your fund for limited history when it's genuinely better on every metric.

A fund with 15% avg but wild swings will score *lower* than a fund with 13% avg that is rock-solid consistent. The score captures **reliability of wealth creation**.

---

### 5. Data Source

All NAV data from [mfapi.in](https://www.mfapi.in/) — historical NAV data for all AMFI-registered Indian mutual funds, updated daily.

Fund categorization uses the **SEBI mutual fund categorization framework** (October 2017).

---

### 6. Limitations

- **Survivorship bias** — Only active funds are analyzed. Merged/closed funds (likely poor performers) are excluded.
- **Inception date bias** — Funds launched just before a bull market will have inflated rolling returns. The confidence discount mitigates this but does not eliminate it — check the "Data Pts" and "Confidence %" columns when evaluating newer funds.
- **Past performance** — Rolling returns are backward-looking. High robustness does not guarantee future returns.
- **Fund manager changes** — Track record may reflect a previous manager's skill.
- **AUM & liquidity** — Not accounted for; can impact future performance (especially small cap).
- **Expense ratios** — NAV is net of expenses, so implicitly included. Direct plans have lower costs than Regular.
- **Tax impact** — Swap comparisons do not account for capital gains tax, exit loads, or stamp duty on switching.
""")

# --- Disclaimer (all tabs) ---
st.divider()
st.caption(
    "**Disclaimer:** This tool provides data and analytics for informational and educational purposes only. "
    "It does not constitute investment advice, a recommendation, or an offer to buy or sell any mutual fund. "
    "Past performance does not guarantee future results. Rankings and comparisons are based on historical data "
    "and quantitative metrics — they should not be the sole basis for investment decisions. "
    "Please consult a SEBI-registered investment advisor before making any investment decisions."
)
