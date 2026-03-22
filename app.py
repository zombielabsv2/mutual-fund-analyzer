"""
Mutual Fund 5-Year Rolling Returns Analyzer
Flask backend for analyzing Indian mutual fund rolling returns using mfapi.in
"""

from flask import Flask, render_template, jsonify, request
import requests
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time

app = Flask(__name__)

MFAPI_BASE_URL = "https://api.mfapi.in/mf"

# Popular equity mutual funds (Direct Growth plans) for top funds analysis
POPULAR_EQUITY_FUNDS = [
    # Large Cap
    "120503",  # Axis Bluechip Fund Direct Growth
    "120505",  # Mirae Asset Large Cap Fund Direct Growth
    "118834",  # ICICI Prudential Bluechip Fund Direct Growth
    "100033",  # SBI Blue Chip Fund Direct Growth
    # Flexi Cap
    "118955",  # HDFC Flexi Cap Fund Direct Growth
    "125497",  # Parag Parikh Flexi Cap Fund Direct Growth
    "120465",  # UTI Flexi Cap Fund Direct Growth
    "106235",  # Kotak Flexicap Fund Direct Growth
    # Mid Cap
    "120586",  # Axis Midcap Fund Direct Growth
    "118989",  # HDFC Mid-Cap Opportunities Fund Direct Growth
    "101539",  # Kotak Emerging Equity Fund Direct Growth
    "119775",  # DSP Midcap Fund Direct Growth
    # Small Cap
    "125354",  # Axis Small Cap Fund Direct Growth
    "119775",  # SBI Small Cap Fund Direct Growth
    "118778",  # HDFC Small Cap Fund Direct Growth
    "125307",  # Nippon India Small Cap Fund Direct Growth
    # Multi Cap / Focused
    "119598",  # SBI Focused Equity Fund Direct Growth
    "120716",  # Mirae Asset Focused Fund Direct Growth
    "118632",  # ICICI Prudential Multicap Fund Direct Growth
    "145552",  # Quant Active Fund Direct Growth
    # Value / Contra
    "118825",  # ICICI Prudential Value Discovery Fund Direct Growth
    "106252",  # Kotak India EQ Contra Fund Direct Growth
    "118988",  # HDFC Capital Builder Value Fund Direct Growth
    "119028",  # Invesco India Contra Fund Direct Growth
    # ELSS
    "120847",  # Axis Long Term Equity Fund Direct Growth
    "120503",  # Mirae Asset Tax Saver Fund Direct Growth
    "119773",  # DSP Tax Saver Fund Direct Growth
    "100516",  # SBI Long Term Equity Fund Direct Growth
    # Thematic / Sectoral
    "120837",  # ICICI Prudential Technology Fund Direct Growth
    "120175",  # Tata Digital India Fund Direct Growth
    "118812",  # ICICI Prudential Banking and Financial Services Fund Direct Growth
    "106250",  # Kotak Infrastructure and Economic Reform Fund Direct Growth
    # Index Funds
    "120684",  # UTI Nifty 50 Index Fund Direct Growth
    "118778",  # HDFC Index Fund Nifty 50 Plan Direct Growth
    "119714",  # Nippon India Index Fund Nifty Plan Direct Growth
    "145200",  # Motilal Oswal Nifty Midcap 150 Index Fund Direct Growth
]

# Comprehensive fund universe for rankings — verified Direct Growth equity scheme codes
RANKING_FUND_CODES = [
    # Large Cap (10)
    "118269",  # Canara Robeco Large Cap
    "118479",  # Bandhan Large Cap
    "118531",  # Franklin India Large Cap
    "118617",  # Edelweiss Large Cap
    "118632",  # Nippon India Large Cap
    "118825",  # Mirae Asset Large Cap
    "119018",  # HDFC Large Cap
    "119160",  # Tata Large Cap
    "119250",  # DSP Large Cap
    "120465",  # Axis Large Cap
    # Large & Mid Cap (10)
    "118278",  # Canara Robeco Large & Mid Cap
    "118510",  # Franklin India Large & Mid Cap
    "119202",  # Tata Large & Mid Cap
    "119218",  # DSP Large & Mid Cap
    "119436",  # Aditya Birla Sun Life Large & Mid Cap
    "120357",  # Invesco India Large & Mid Cap
    "120596",  # ICICI Prudential Large & Mid Cap
    "120665",  # UTI Large & Mid Cap
    "120826",  # Quant Large & Mid Cap
    "130498",  # HDFC Large & Mid Cap
    # Mid Cap (10)
    "118533",  # Franklin India Mid Cap
    "118668",  # Nippon India Growth Mid Cap
    "118989",  # HDFC Mid Cap
    "119178",  # Tata Mid Cap
    "119581",  # Sundaram Mid Cap
    "119775",  # Kotak Midcap
    "120726",  # UTI Mid Cap
    "120841",  # Quant Mid Cap
    "125307",  # PGIM India Midcap
    "140228",  # Edelweiss Mid Cap
    # Small Cap (11)
    "118525",  # Franklin India Small Cap
    "118778",  # Nippon India Small Cap
    "119212",  # DSP Small Cap
    "119556",  # Aditya Birla Sun Life Small Cap
    "119589",  # Sundaram Small Cap
    "120069",  # HSBC Small Cap
    "120164",  # Kotak Small Cap
    "120828",  # Quant Small Cap
    "125354",  # Axis Small Cap
    "125497",  # SBI Small Cap
    "130503",  # HDFC Small Cap
    # Flexi Cap (10)
    "118424",  # Bandhan Flexi Cap
    "118535",  # Franklin India Flexi Cap
    "118955",  # HDFC Flexi Cap
    "119076",  # DSP Flexi Cap
    "120564",  # Aditya Birla Sun Life Flexi Cap
    "120662",  # UTI Flexi Cap
    "120843",  # Quant Flexi Cap
    "122639",  # Parag Parikh Flexi Cap
    "129046",  # Motilal Oswal Flexi Cap
    "133839",  # PGIM India Flexi Cap
    # Multi Cap (6)
    "118650",  # Nippon India Multi Cap
    "120823",  # Quant Multi Cap
    "131164",  # UTI Multi Cap
    "141226",  # Mahindra Manulife Multi Cap
    "149303",  # Bandhan Multi Cap
    "149368",  # HDFC Multi Cap
    # Value / Contra (10)
    "103490",  # Quantum Value Fund
    "118494",  # Templeton India Value Fund
    "118784",  # Nippon India Value Fund
    "118935",  # HDFC Value Fund
    "119549",  # Sundaram Value Fund
    "119659",  # Aditya Birla Sun Life Value Fund
    "119769",  # Kotak Contra Fund
    "119835",  # SBI Contra Fund
    "120323",  # ICICI Prudential Value Fund
    "120348",  # Invesco India Contra Fund
    # Focused (10)
    "118564",  # Franklin India Focused Equity
    "118692",  # Nippon India Focused
    "118950",  # HDFC Focused
    "119096",  # DSP Focused
    "119564",  # Aditya Birla Sun Life Focused
    "119727",  # SBI Focused Fund
    "120468",  # Axis Focused Fund
    "120722",  # ICICI Prudential Focused Equity
    "120834",  # Quant Focused Fund
    "122389",  # Motilal Oswal Focused Fund
    # ELSS / Tax Saving (12)
    "118285",  # Canara Robeco ELSS Tax Saver
    "118540",  # Franklin India ELSS Tax Saver
    "118620",  # Edelweiss ELSS Tax Saver
    "118803",  # Nippon India ELSS Tax Saver
    "119060",  # HDFC ELSS Tax Saver
    "119242",  # DSP ELSS Tax Saver
    "119544",  # Aditya Birla Sun Life ELSS Tax Saver
    "119723",  # SBI ELSS Tax Saver
    "119773",  # Kotak ELSS Tax Saver
    "120503",  # Axis ELSS Tax Saver
    "120847",  # Quant ELSS Tax Saver
    "135781",  # Mirae Asset ELSS Tax Saver
    # Sectoral / Thematic (8)
    "118267",  # Canara Robeco Infrastructure
    "118537",  # Franklin India Technology
    "118589",  # Nippon India Banking & Financial Services
    "119028",  # DSP Natural Resources & New Energy
    "119597",  # Sundaram Financial Services Opportunities
    "120175",  # Tata Digital India
    "120587",  # ICICI Prudential FMCG
    "120837",  # ICICI Prudential Technology
    # Index Funds (7)
    "118266",  # Canara Robeco Nifty Index
    "118482",  # Bandhan Nifty 50 Index
    "118741",  # Nippon India Index Fund Nifty 50
    "119648",  # Aditya Birla Sun Life Nifty 50 Index
    "119827",  # SBI Nifty Index Fund
    "120716",  # UTI Nifty 50 Index Fund
    "147622",  # Motilal Oswal Nifty Midcap 150 Index
]

# Cache for fund rankings
_rankings_cache = {}
_rankings_cache_time = 0
RANKINGS_CACHE_TTL = 3600  # 1 hour


def normalize_category(scheme_category):
    """Map API scheme_category to a clean display category"""
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


@lru_cache(maxsize=100)
def get_all_funds():
    """Fetch all mutual fund schemes from mfapi.in"""
    try:
        response = requests.get(MFAPI_BASE_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching fund list: {e}")
        return []


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/search')
def search_funds():
    """Search mutual funds by name or scheme code"""
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify([])

    all_funds = get_all_funds()
    results = []

    for fund in all_funds:
        scheme_code = str(fund.get('schemeCode', ''))
        scheme_name = fund.get('schemeName', '')

        if query in scheme_name.lower() or query in scheme_code:
            results.append({
                'schemeCode': scheme_code,
                'schemeName': scheme_name
            })

        if len(results) >= 20:
            break

    return jsonify(results)


@app.route('/api/fund/<scheme_code>')
def get_fund_data(scheme_code):
    """Get fund NAV history"""
    try:
        response = requests.get(f"{MFAPI_BASE_URL}/{scheme_code}", timeout=30)
        response.raise_for_status()
        data = response.json()

        return jsonify({
            'meta': data.get('meta', {}),
            'data': data.get('data', [])
        })
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500


def calculate_rolling_returns(nav_data, years=5):
    """
    Calculate rolling returns from NAV data.

    Args:
        nav_data: List of {'date': 'DD-MM-YYYY', 'nav': 'value'} dicts
        years: Rolling period in years (default 5)

    Returns:
        List of {'date': 'YYYY-MM-DD', 'return': percentage} dicts
    """
    if not nav_data:
        return []

    # Convert to list of (date, nav) tuples, sorted by date ascending
    parsed_data = []
    for item in nav_data:
        try:
            date = datetime.strptime(item['date'], '%d-%m-%Y')
            nav = float(item['nav'])
            if nav > 0:
                parsed_data.append((date, nav))
        except (ValueError, KeyError):
            continue

    # Sort by date ascending
    parsed_data.sort(key=lambda x: x[0])

    if len(parsed_data) < 2:
        return []

    # Create a date-to-nav lookup for faster access
    nav_lookup = {d: n for d, n in parsed_data}
    dates = [d for d, _ in parsed_data]

    rolling_returns = []
    target_days = years * 365

    for current_date, current_nav in parsed_data:
        # Find the date approximately 'years' years ago
        target_date = current_date - timedelta(days=target_days)

        # Find the closest available date to target
        past_nav = None
        past_date = None

        # Look for exact match first
        if target_date in nav_lookup:
            past_nav = nav_lookup[target_date]
            past_date = target_date
        else:
            # Find closest date within a 15-day window
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
            # Calculate actual years between dates
            actual_years = (current_date - past_date).days / 365.25

            if actual_years >= years - 0.1:  # Allow small tolerance
                # CAGR formula: ((end/start)^(1/years) - 1) * 100
                cagr = (pow(current_nav / past_nav, 1 / actual_years) - 1) * 100
                rolling_returns.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'return': round(cagr, 2)
                })

    return rolling_returns


@app.route('/api/rolling-returns/<scheme_code>')
def get_rolling_returns(scheme_code):
    """Calculate and return 5-year rolling returns for a fund"""
    years = request.args.get('years', 5, type=int)

    try:
        response = requests.get(f"{MFAPI_BASE_URL}/{scheme_code}", timeout=30)
        response.raise_for_status()
        data = response.json()

        nav_data = data.get('data', [])
        meta = data.get('meta', {})

        rolling_returns = calculate_rolling_returns(nav_data, years)

        # Calculate statistics
        if rolling_returns:
            returns_values = [r['return'] for r in rolling_returns]
            stats = {
                'min': round(min(returns_values), 2),
                'max': round(max(returns_values), 2),
                'average': round(sum(returns_values) / len(returns_values), 2),
                'stdDev': round(
                    math.sqrt(sum((x - sum(returns_values)/len(returns_values))**2
                             for x in returns_values) / len(returns_values)), 2
                ),
                'positivePercentage': round(
                    len([r for r in returns_values if r > 0]) / len(returns_values) * 100, 2
                ),
                'totalPeriods': len(returns_values)
            }
        else:
            stats = None

        return jsonify({
            'meta': meta,
            'rollingReturns': rolling_returns,
            'statistics': stats
        })

    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/top-funds')
def get_top_funds():
    """
    Find top funds with average 5-year rolling returns above threshold.
    Scans popular equity funds and returns those meeting criteria.
    """
    min_return = request.args.get('min_return', 18, type=float)
    limit = request.args.get('limit', 10, type=int)

    results = []
    processed = set()  # Avoid duplicates

    for scheme_code in POPULAR_EQUITY_FUNDS:
        if scheme_code in processed:
            continue
        processed.add(scheme_code)

        try:
            response = requests.get(f"{MFAPI_BASE_URL}/{scheme_code}", timeout=15)
            if response.status_code != 200:
                continue

            data = response.json()
            nav_data = data.get('data', [])
            meta = data.get('meta', {})

            if not nav_data or not meta:
                continue

            rolling_returns = calculate_rolling_returns(nav_data, years=5)

            if not rolling_returns:
                continue

            returns_values = [r['return'] for r in rolling_returns]
            avg_return = sum(returns_values) / len(returns_values)

            if avg_return >= min_return:
                results.append({
                    'schemeCode': scheme_code,
                    'schemeName': meta.get('scheme_name', 'Unknown'),
                    'category': meta.get('scheme_category', 'Unknown'),
                    'fundHouse': meta.get('fund_house', 'Unknown'),
                    'averageReturn': round(avg_return, 2),
                    'minReturn': round(min(returns_values), 2),
                    'maxReturn': round(max(returns_values), 2),
                    'stdDev': round(
                        math.sqrt(sum((x - avg_return)**2 for x in returns_values) / len(returns_values)), 2
                    ),
                    'positivePercentage': round(
                        len([r for r in returns_values if r > 0]) / len(returns_values) * 100, 2
                    ),
                    'totalPeriods': len(returns_values)
                })

        except Exception as e:
            print(f"Error processing fund {scheme_code}: {e}")
            continue

    # Sort by average return descending and limit results
    results.sort(key=lambda x: x['averageReturn'], reverse=True)
    results = results[:limit]

    return jsonify({
        'funds': results,
        'minReturnFilter': min_return,
        'totalFound': len(results)
    })


@app.route('/api/fund-rankings')
def get_fund_rankings():
    """
    Analyze all funds in RANKING_FUND_CODES and return ranked results
    grouped by SEBI category, sorted by robustness score.
    """
    global _rankings_cache, _rankings_cache_time

    # Return cached results if fresh
    if _rankings_cache and (time.time() - _rankings_cache_time) < RANKINGS_CACHE_TTL:
        return jsonify(_rankings_cache)

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
            std = math.sqrt(
                sum((x - avg) ** 2 for x in returns_values) / len(returns_values)
            )
            mn = min(returns_values)
            mx = max(returns_values)
            pos_pct = len([r for r in returns_values if r > 0]) / len(returns_values) * 100

            # Robustness score: rewards high avg, consistency, downside protection
            robustness = (avg * (pos_pct / 100)) / (1 + std / 10)

            return {
                'schemeCode': scheme_code,
                'schemeName': meta.get('scheme_name', 'Unknown'),
                'category': normalize_category(meta.get('scheme_category', '')),
                'rawCategory': meta.get('scheme_category', 'Unknown'),
                'fundHouse': meta.get('fund_house', 'Unknown'),
                'avgReturn': round(avg, 2),
                'minReturn': round(mn, 2),
                'maxReturn': round(mx, 2),
                'stdDev': round(std, 2),
                'positivePercentage': round(pos_pct, 1),
                'totalPeriods': len(returns_values),
                'robustnessScore': round(robustness, 2),
            }
        except Exception as e:
            print(f"Error processing fund {scheme_code}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_fund, code): code
            for code in RANKING_FUND_CODES
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Sort by robustness score descending
    results.sort(key=lambda x: x['robustnessScore'], reverse=True)

    # Group by category
    by_category = {}
    for fund in results:
        cat = fund['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(fund)

    # Sort within each category by robustness
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x['robustnessScore'], reverse=True)

    response_data = {
        'allFunds': results,
        'byCategory': by_category,
        'categories': sorted(by_category.keys()),
        'totalFunds': len(results),
    }

    _rankings_cache = response_data
    _rankings_cache_time = time.time()

    return jsonify(response_data)


if __name__ == '__main__':
    print("Starting Mutual Fund Rolling Returns Analyzer...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
