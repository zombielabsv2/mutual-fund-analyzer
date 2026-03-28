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

@st.cache_data(ttl=3600)
def search_funds_api(query):
    """Search mutual funds by name or code."""
    try:
        response = requests.get(f"{MFAPI_BASE_URL}", timeout=30)
        response.raise_for_status()
        all_funds = response.json()
        q = query.lower()
        results = []
        for fund in all_funds:
            code = str(fund.get('schemeCode', ''))
            name = fund.get('schemeName', '')
            if q in name.lower() or q in code:
                results.append({'schemeCode': code, 'schemeName': name})
            if len(results) >= 20:
                break
        return results
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
        return {
            'meta': meta,
            'rollingReturns': rolling,
            'statistics': stats,
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
            robustness = (avg * (pos_pct / 100)) / (1 + std / 10)
            # Confidence discount: penalize funds with fewer data points
            # ~1500 points ≈ 6+ years of rolling windows, which spans diverse market regimes
            confidence = min(1.0, len(returns_values) / 1500)
            robustness *= confidence
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
        robustness = (avg * (pos_pct / 100)) / (1 + std / 10)
        confidence = min(1.0, len(returns_values) / 1500)
        robustness *= confidence
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
def get_all_schemes():
    """Fetch all mutual fund schemes from mfapi.in."""
    try:
        response = requests.get(MFAPI_BASE_URL, timeout=30)
        response.raise_for_status()
        return response.json()
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


# --- Score Breakdown ---

def render_score_breakdown(fund, years=5):
    """Render a detailed, transparent robustness score breakdown for a fund."""
    name = fund['schemeName'].split(' -')[0].split(' Direct')[0]
    avg = fund['avgReturn']
    pos_pct = fund['positivePercentage']
    std = fund['stdDev']
    total = fund['totalPeriods']
    conf = fund.get('confidence', 100)

    raw_score = (avg * (pos_pct / 100)) / (1 + std / 10)
    final_score = raw_score * (conf / 100)

    st.markdown("---")
    st.markdown(f"#### Score Breakdown: {name}")

    col_formula, col_chart = st.columns([2, 3])

    with col_formula:
        st.markdown(f"**Raw Metrics** — {total:,} rolling {years}-year windows")
        st.markdown(
            f"| Metric | Value |\n|---|---|\n"
            f"| Average Return | {avg}% |\n"
            f"| Min Return | {fund['minReturn']}% |\n"
            f"| Max Return | {fund['maxReturn']}% |\n"
            f"| Std Deviation | {std} |\n"
            f"| Positive Periods | {pos_pct}% |"
        )
        st.markdown("**Robustness Score Calculation**")
        st.code(
            f"Raw Score  = (Avg × Positive%) / (1 + StdDev/10)\n"
            f"           = ({avg} × {pos_pct / 100:.3f}) / (1 + {std}/10)\n"
            f"           = {avg * pos_pct / 100:.2f} / {1 + std / 10:.3f}\n"
            f"           = {raw_score:.2f}\n"
            f"\n"
            f"Confidence = min(1, {total:,} / 1,500) = {conf}%\n"
            f"\n"
            f"Final Score = {raw_score:.2f} × {conf / 100:.2f} = {final_score:.2f}",
            language=None,
        )

    with col_chart:
        data = get_fund_rolling_returns(fund['schemeCode'], years=years)
        if data and data.get('rollingReturns'):
            returns_data = data['rollingReturns']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[d['date'] for d in returns_data],
                y=[d['return'] for d in returns_data],
                mode='lines',
                line=dict(color='#1a237e', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(26,35,126,0.1)',
                hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>',
            ))
            fig.add_hline(y=avg, line_dash="dash", line_color="#c62828",
                          annotation_text=f"Avg: {avg}%")
            fig.add_hline(y=0, line_color="gray", line_width=0.5)
            fig.update_layout(
                title=f"{years}-Year Rolling Returns",
                yaxis_title='CAGR (%)',
                height=350,
                margin=dict(t=40, b=20, l=40, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Each point = the {years}-year CAGR ending on that date. "
                f"The spread drives the avg, std dev, and positive % above."
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

# --- Tabs ---
tab_analyzer, tab_rankings, tab_portfolio, tab_pms, tab_methodology = st.tabs(["🔍 Analyzer", "🏆 Fund Rankings", "📋 Portfolio Review", "💎 PMS & AIF", "📐 Methodology"])


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
        st.subheader("5-Year Rolling Returns Chart")
        colors = ['#1a237e', '#c62828', '#2e7d32', '#f57c00', '#6a1b9a']
        fig = go.Figure()
        for i, fund in enumerate(st.session_state.selected_funds):
            dates = [d['date'] for d in fund['data']]
            returns = [d['return'] for d in fund['data']]
            label = fund['name'].split(' -')[0].split(' Direct')[0][:30]
            fig.add_trace(go.Scatter(
                x=dates, y=returns, name=label,
                mode='lines', line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='%{x}<br>%{y:.2f}%<extra>' + label + '</extra>',
            ))
        fig.update_layout(
            yaxis_title='CAGR (%)',
            xaxis_title='Date',
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(t=20, b=80),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Statistics Table ---
        st.subheader("Statistics Summary")
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
        st.cache_data.clear()

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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Funds Analyzed", len(rankings))
        c2.metric("Top Fund", rankings[0]['schemeName'].split(' -')[0].split(' Direct')[0][:22])
        c3.metric("Strongest Category", best_cat)
        c4.metric("Most Consistent", most_consistent['schemeName'].split(' -')[0].split(' Direct')[0][:22])

        # --- Rankings Table ---
        df = pd.DataFrame(filtered)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        df['Fund Name'] = df['schemeName'].apply(lambda x: x.split(' -')[0].split(' Direct')[0])
        df['Fund House'] = df['fundHouse'].apply(lambda x: x.split(' Mutual')[0])

        display_df = df[['Rank', 'Fund Name', 'Fund House', 'category', 'avgReturn', 'minReturn',
                         'maxReturn', 'stdDev', 'positivePercentage', 'totalPeriods', 'confidence', 'robustnessScore']].copy()
        display_df.columns = ['#', 'Fund Name', 'Fund House', 'Category', 'Avg Return %',
                              'Min %', 'Max %', 'Std Dev', 'Positive %', 'Data Pts', 'Confidence %', 'Robustness']

        st.caption("Click any row to see how its robustness score is calculated.")
        ranking_event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                '#': st.column_config.NumberColumn(width="small"),
                'Avg Return %': st.column_config.NumberColumn(format="%.1f"),
                'Min %': st.column_config.NumberColumn(format="%.1f"),
                'Max %': st.column_config.NumberColumn(format="%.1f"),
                'Std Dev': st.column_config.NumberColumn(format="%.1f"),
                'Positive %': st.column_config.NumberColumn(format="%.0f"),
                'Data Pts': st.column_config.NumberColumn(format="%d"),
                'Confidence %': st.column_config.NumberColumn(format="%d"),
                'Robustness': st.column_config.NumberColumn(format="%.1f"),
            },
        )

        if ranking_event.selection.rows:
            render_score_breakdown(filtered[ranking_event.selection.rows[0]], years=rolling_years)

        # --- Download ---
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Rankings CSV",
            csv_data,
            f"fund_rankings_{rolling_years}Y_top{top_n}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )
    else:
        st.info("Click **Load / Refresh Rankings** to analyze funds.")


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
                'Avg Return %': f['avgReturn'],
                'Min %': f['minReturn'],
                'Std Dev': f['stdDev'],
                'Positive %': f['positivePercentage'],
                'Robustness': f['robustnessScore'],
            }
            if f.get('amount'):
                row['Invested (₹)'] = f"{f['amount']:,.0f}"
            if f.get('current'):
                row['Current (₹)'] = f"{f['current']:,.0f}"
            if f.get('match_confidence'):
                row['Match %'] = f"{f['match_confidence']}%"
            port_rows.append(row)
        st.caption("Click any row to see how its robustness score is calculated.")
        portfolio_event = st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")

        if portfolio_event.selection.rows:
            render_score_breakdown(portfolio[portfolio_event.selection.rows[0]], years=portfolio_years)

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

            # Only recommend swap if the top fund is actually better
            is_optimal = (fund_rank and fund_rank <= 3) or top_fund['robustnessScore'] <= fund['robustnessScore']

            if is_optimal:
                already_optimal += 1
                if fund_rank and fund_rank <= 3:
                    st.success(f"**✅ {fund_name}** — Ranked **#{fund_rank} in {cat}**. No change needed. (Robustness: {fund['robustnessScore']})")
                else:
                    st.success(f"**✅ {fund_name}** — Already outperforms the top ranked fund in **{cat}**. (Robustness: {fund['robustnessScore']} vs {top_fund['robustnessScore']})")
            else:
                swaps_needed += 1
                top_name = top_fund['schemeName'].split(' -')[0].split(' Direct')[0]

                with st.expander(f"🔄 **{fund_name}** → **{top_name}**", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"##### Your Fund")
                        st.markdown(f"**{fund_name}**")
                        st.caption(f"{fine_cat}")
                        st.metric("Avg Return", f"{fund['avgReturn']}%")
                        st.metric("Min Return", f"{fund['minReturn']}%")
                        st.metric("Std Dev", f"{fund['stdDev']}")
                        st.metric("Positive Periods", f"{fund['positivePercentage']}%")
                        st.metric("Robustness Score", f"{fund['robustnessScore']}")
                        _raw = (fund['avgReturn'] * fund['positivePercentage'] / 100) / (1 + fund['stdDev'] / 10)
                        _conf = fund.get('confidence', 100)
                        st.caption(f"= ({fund['avgReturn']} × {fund['positivePercentage']/100:.2f}) / (1+{fund['stdDev']}/10) × {_conf}% conf")

                    with col2:
                        st.markdown(f"##### Recommended (#{1} in {fine_cat})")
                        st.markdown(f"**{top_name}**")
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
                        _raw_t = (top_fund['avgReturn'] * top_fund['positivePercentage'] / 100) / (1 + top_fund['stdDev'] / 10)
                        _conf_t = top_fund.get('confidence', 100)
                        st.caption(f"= ({top_fund['avgReturn']} × {top_fund['positivePercentage']/100:.2f}) / (1+{top_fund['stdDev']}/10) × {_conf_t}% conf")

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

                    # Alternatives
                    alts = [f for f in cat_funds[1:4] if f['schemeCode'] != fund['schemeCode']]
                    if alts:
                        alt_text = ", ".join(
                            f"{f['schemeName'].split(' -')[0].split(' Direct')[0]} (Score: {f['robustnessScore']})"
                            for f in alts
                        )
                        st.caption(f"Other strong options in {cat}: {alt_text}")

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
                        if i < keep_count:
                            st.markdown(f"✅ **{i+1}. {fname}** — Robustness: {f['robustnessScore']}, Avg: {f['avgReturn']}%{amt_str} → **Keep**")
                        else:
                            st.markdown(f"🔻 {i+1}. {fname} — Robustness: {f['robustnessScore']}, Avg: {f['avgReturn']}%{amt_str} → **Consider exiting**")

                    if upgrade_to:
                        up_name = upgrade_to['schemeName'].split(' -')[0].split(' Direct')[0]
                        st.markdown(f"💡 **Even better:** Consolidate all into **{up_name}** (Robustness: {upgrade_to['robustnessScore']}, Avg: {upgrade_to['avgReturn']}%) — ranked #1 in {cat}.")

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
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Avg Return", f"{top['avgReturn']}%")
                    c2.metric("Min Return", f"{top['minReturn']}%")
                    c3.metric("Std Dev", f"{top['stdDev']}")
                    c4.metric("Positive %", f"{top['positivePercentage']}%")
                    c5.metric("Robustness", f"{top['robustnessScore']}")

                    # Show top 3 in category
                    if len(rankings_by_cat.get(cat, [])) > 1:
                        st.markdown("**Top funds in this category:**")
                        for i, f in enumerate(rankings_by_cat[cat][:3]):
                            fname = f['schemeName'].split(' -')[0].split(' Direct')[0]
                            st.markdown(f"{i+1}. **{fname}** — Avg: {f['avgReturn']}%, Robustness: {f['robustnessScore']}")

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
> **Confidence = min(1, Data Points / 1500)**
>
> **Final Robustness Score = Raw Score × Confidence**

This rewards:
- **High average returns** — wealth creation over time
- **High positive period %** — rarely loses money over 5-year windows
- **Low standard deviation** — consistent, not wildly swinging
- **More data points** — a fund that has been through multiple market cycles (bull runs, corrections, crashes) gets full credit, while a newer fund with limited history is discounted

**Why the confidence adjustment?** A fund launched 5.5 years ago would have only ~400 rolling 5-year data points, all starting from a similar market period. If that period happened to be a market low (e.g., COVID crash), every window looks great — but that tells you about the market timing, not the fund quality. A fund with 2,000+ data points spanning 10+ years has proven itself across diverse market regimes. The confidence factor ensures newer funds must earn their ranking over time.

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
""")
