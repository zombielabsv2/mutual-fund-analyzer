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
    if "hybrid" in cat:
        return "Hybrid"
    return "Other"


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
def load_all_rankings():
    """Fetch and rank all funds in the universe."""
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
            rolling = calculate_rolling_returns(nav_data, years=5)
            if not rolling or len(rolling) < 10:
                return None
            returns_values = [r['return'] for r in rolling]
            avg = sum(returns_values) / len(returns_values)
            std = math.sqrt(sum((x - avg) ** 2 for x in returns_values) / len(returns_values))
            mn = min(returns_values)
            mx = max(returns_values)
            pos_pct = len([r for r in returns_values if r > 0]) / len(returns_values) * 100
            robustness = (avg * (pos_pct / 100)) / (1 + std / 10)
            return {
                'schemeCode': scheme_code,
                'schemeName': meta.get('scheme_name', 'Unknown'),
                'category': normalize_category(meta.get('scheme_category', '')),
                'fundHouse': meta.get('fund_house', 'Unknown'),
                'avgReturn': round(avg, 2),
                'minReturn': round(mn, 2),
                'maxReturn': round(mx, 2),
                'stdDev': round(std, 2),
                'positivePercentage': round(pos_pct, 1),
                'totalPeriods': len(returns_values),
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
def analyze_portfolio_fund(scheme_code):
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
        rolling = calculate_rolling_returns(nav_data, years=5)
        if not rolling or len(rolling) < 10:
            return None
        returns_values = [r['return'] for r in rolling]
        avg = sum(returns_values) / len(returns_values)
        std = math.sqrt(sum((x - avg) ** 2 for x in returns_values) / len(returns_values))
        mn = min(returns_values)
        mx = max(returns_values)
        pos_pct = len([r for r in returns_values if r > 0]) / len(returns_values) * 100
        robustness = (avg * (pos_pct / 100)) / (1 + std / 10)
        return {
            'schemeCode': scheme_code,
            'schemeName': meta.get('scheme_name', 'Unknown'),
            'category': normalize_category(meta.get('scheme_category', '')),
            'fundHouse': meta.get('fund_house', 'Unknown'),
            'avgReturn': round(avg, 2),
            'minReturn': round(mn, 2),
            'maxReturn': round(mx, 2),
            'stdDev': round(std, 2),
            'positivePercentage': round(pos_pct, 1),
            'totalPeriods': len(returns_values),
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


# --- Session State ---
if 'selected_funds' not in st.session_state:
    st.session_state.selected_funds = []
if 'portfolio_funds' not in st.session_state:
    st.session_state.portfolio_funds = []
if 'portfolio_unmatched' not in st.session_state:
    st.session_state.portfolio_unmatched = []


# --- Header ---
st.markdown(
    "<h1 style='text-align:center;'>📈 Mutual Fund 5-Year Rolling Returns Analyzer</h1>"
    "<p style='text-align:center;color:#666;'>Analyze and compare rolling returns of Indian mutual funds</p>",
    unsafe_allow_html=True,
)

# --- Tabs ---
tab_analyzer, tab_rankings, tab_portfolio, tab_methodology = st.tabs(["🔍 Analyzer", "🏆 Fund Rankings", "📋 Portfolio Review", "📐 Methodology"])


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
    st.subheader("Top Funds by 5-Year Rolling Return Robustness")
    st.caption("Funds ranked by a robustness score that rewards high average returns, consistency, and downside protection.")

    col_top, col_spacer = st.columns([2, 8])
    with col_top:
        top_n = st.selectbox("Show", [10, 20, 30, 50, 100], index=0, key="top_n_select")

    if st.button("🔄 Load / Refresh Rankings", type="primary"):
        st.cache_data.clear()

    rankings = load_all_rankings()

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
                         'maxReturn', 'stdDev', 'positivePercentage', 'robustnessScore']].copy()
        display_df.columns = ['#', 'Fund Name', 'Fund House', 'Category', 'Avg Return %',
                              'Min %', 'Max %', 'Std Dev', 'Positive %', 'Robustness']

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '#': st.column_config.NumberColumn(width="small"),
                'Avg Return %': st.column_config.NumberColumn(format="%.1f"),
                'Min %': st.column_config.NumberColumn(format="%.1f"),
                'Max %': st.column_config.NumberColumn(format="%.1f"),
                'Std Dev': st.column_config.NumberColumn(format="%.1f"),
                'Positive %': st.column_config.NumberColumn(format="%.0f"),
                'Robustness': st.column_config.NumberColumn(format="%.1f"),
            },
        )

        # --- Download ---
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Rankings CSV",
            csv_data,
            f"fund_rankings_top{top_n}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )
    else:
        st.info("Click **Load / Refresh Rankings** to analyze funds.")


# ===================== PORTFOLIO REVIEW TAB =====================
with tab_portfolio:
    st.subheader("Portfolio Review & Swap Recommendations")
    st.caption("Upload your mutual fund statement (PDF, CSV, or Excel) from Groww, Kuvera, MFCentral, or any broker. We'll auto-identify your funds and recommend more robust alternatives in each category.")

    uploaded = st.file_uploader(
        "Upload your holdings statement",
        type=["pdf", "csv", "xlsx", "xls"],
        help="Supported: PDF/CSV/Excel from Groww, Kuvera, MFCentral, CAMS, Karvy, or any broker.",
    )

    if uploaded and not st.session_state.portfolio_funds:
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
                        data = analyze_portfolio_fund(code)
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
                'Category': f['category'],
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
        st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)

        # Load rankings for comparison
        rankings = load_all_rankings()
        rankings_by_cat = {}
        for f in rankings:
            rankings_by_cat.setdefault(f['category'], []).append(f)

        # --- Recommendations ---
        st.markdown("### Recommendations")
        health_scores = []
        swaps_needed = 0
        already_optimal = 0

        for fund in portfolio:
            cat = fund['category']
            cat_funds = rankings_by_cat.get(cat, [])
            fund_name = fund['schemeName'].split(' -')[0].split(' Direct')[0]

            if not cat_funds:
                st.warning(f"**{fund_name}** — No ranked funds in **{cat}** for comparison.")
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

            if fund_rank and fund_rank <= 3:
                already_optimal += 1
                st.success(f"**✅ {fund_name}** — Ranked **#{fund_rank} in {cat}**. No change needed. (Robustness: {fund['robustnessScore']})")
            else:
                swaps_needed += 1
                top_name = top_fund['schemeName'].split(' -')[0].split(' Direct')[0]

                with st.expander(f"🔄 **{fund_name}** → **{top_name}**", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"##### Your Fund")
                        st.markdown(f"**{fund_name}**")
                        st.caption(f"{cat}")
                        st.metric("Avg Return", f"{fund['avgReturn']}%")
                        st.metric("Min Return", f"{fund['minReturn']}%")
                        st.metric("Std Dev", f"{fund['stdDev']}")
                        st.metric("Positive Periods", f"{fund['positivePercentage']}%")
                        st.metric("Robustness Score", f"{fund['robustnessScore']}")

                    with col2:
                        st.markdown(f"##### Recommended (#{1} in {cat})")
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

                    # Rationale
                    st.markdown("---")
                    st.markdown("**Why switch?**")
                    rationale = []
                    if fund['robustnessScore'] > 0:
                        rob_pct = round((top_fund['robustnessScore'] / fund['robustnessScore'] - 1) * 100)
                        if rob_pct > 0:
                            rationale.append(f"**{rob_pct}% higher robustness score** ({top_fund['robustnessScore']} vs {fund['robustnessScore']}) — more reliable wealth creation")
                    if top_fund['avgReturn'] > fund['avgReturn']:
                        rationale.append(f"**{ret_delta:+.1f}% higher average return** over rolling 5-year periods")
                    if top_fund['minReturn'] > fund['minReturn']:
                        rationale.append(f"**Better downside protection** — worst 5-year return: {top_fund['minReturn']}% vs {fund['minReturn']}%")
                    if top_fund['stdDev'] < fund['stdDev']:
                        rationale.append(f"**More consistent** — standard deviation of {top_fund['stdDev']} vs {fund['stdDev']}")
                    if top_fund['positivePercentage'] > fund['positivePercentage']:
                        rationale.append(f"**Positive in {top_fund['positivePercentage']}%** of 5-year periods vs {fund['positivePercentage']}%")
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
                    'Category': cat,
                    'Current Robustness': fund['robustnessScore'],
                    'Current Avg Return %': fund['avgReturn'],
                    'Action': f"Keep (Rank #{fund_rank})" if fund_rank and fund_rank <= 3 else "Consider Swap",
                    'Recommended Fund': top['schemeName'].split(' -')[0].split(' Direct')[0] if top and (not fund_rank or fund_rank > 3) else '—',
                    'Recommended Robustness': top['robustnessScore'] if top and (not fund_rank or fund_rank > 3) else '—',
                    'Recommended Avg Return %': top['avgReturn'] if top and (not fund_rank or fund_rank > 3) else '—',
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

> **Score = (Avg Return × Positive%) / (1 + StdDev / 10)**

This rewards:
- **High average returns** — wealth creation over time
- **High positive period %** — rarely loses money over 5-year windows
- **Low standard deviation** — consistent, not wildly swinging

A fund with 15% avg but wild swings will score *lower* than a fund with 13% avg that is rock-solid consistent. The score captures **reliability of wealth creation**.

---

### 5. Data Source

All NAV data from [mfapi.in](https://www.mfapi.in/) — historical NAV data for all AMFI-registered Indian mutual funds, updated daily.

Fund categorization uses the **SEBI mutual fund categorization framework** (October 2017).

---

### 6. Limitations

- **Survivorship bias** — Only active funds are analyzed. Merged/closed funds (likely poor performers) are excluded.
- **Past performance** — Rolling returns are backward-looking. High robustness does not guarantee future returns.
- **Fund manager changes** — Track record may reflect a previous manager's skill.
- **AUM & liquidity** — Not accounted for; can impact future performance (especially small cap).
- **Expense ratios** — NAV is net of expenses, so implicitly included. Direct plans have lower costs than Regular.
""")
