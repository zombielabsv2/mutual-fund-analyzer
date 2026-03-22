/**
 * Mutual Fund Rolling Returns Analyzer
 * Frontend logic for chart rendering, data management, and fund rankings
 */

// Chart colors for different funds
const CHART_COLORS = [
    '#1a237e',  // Deep blue
    '#c62828',  // Red
    '#2e7d32',  // Green
    '#f57c00',  // Orange
    '#6a1b9a',  // Purple
];

// State management
const state = {
    selectedFunds: [],
    fundData: {},
    chart: null,
    // Rankings state
    rankingsData: null,
    rankingsLoaded: false,
    activeCategory: 'All',
    sortField: 'robustnessScore',
    sortAsc: false,
    topN: 10,
};

// DOM Elements
const elements = {
    searchInput: document.getElementById('searchInput'),
    searchResults: document.getElementById('searchResults'),
    selectedFundsList: document.getElementById('selectedFundsList'),
    fundCount: document.getElementById('fundCount'),
    noFundsMessage: document.getElementById('noFundsMessage'),
    chartSection: document.getElementById('chartSection'),
    statsSection: document.getElementById('statsSection'),
    statsTableBody: document.getElementById('statsTableBody'),
    exportBtn: document.getElementById('exportBtn'),
    downloadChartBtn: document.getElementById('downloadChartBtn'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    // Rankings elements
    loadRankingsBtn: document.getElementById('loadRankingsBtn'),
    refreshRankingsBtn: document.getElementById('refreshRankingsBtn'),
    rankingsLoading: document.getElementById('rankingsLoading'),
    rankingsContent: document.getElementById('rankingsContent'),
    rankingsInsights: document.getElementById('rankingsInsights'),
    categoryFilters: document.getElementById('categoryFilters'),
    rankingsTableBody: document.getElementById('rankingsTableBody'),
    topNSelect: document.getElementById('topNSelect'),
    downloadRankingsBtn: document.getElementById('downloadRankingsBtn'),
};

// ===== Tab Navigation =====
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab + 'Panel').classList.add('active');
    });
});

// ===== Utility Functions =====

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

function showLoading(show = true) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}

function truncateName(name, maxLength) {
    if (name.length <= maxLength) return name;
    return name.substring(0, maxLength - 3) + '...';
}

// ===== Analyzer Tab Functions =====

async function searchFunds(query) {
    if (!query || query.length < 2) {
        elements.searchResults.classList.remove('active');
        return;
    }
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const results = await response.json();
        if (results.length === 0) {
            elements.searchResults.innerHTML = '<div class="search-result-item">No funds found</div>';
        } else {
            elements.searchResults.innerHTML = results.map(fund => `
                <div class="search-result-item" data-code="${fund.schemeCode}" data-name="${fund.schemeName}">
                    <div class="result-name">${fund.schemeName}</div>
                    <div class="result-code">Code: ${fund.schemeCode}</div>
                </div>
            `).join('');
        }
        elements.searchResults.classList.add('active');
    } catch (error) {
        console.error('Search error:', error);
    }
}

async function addFund(schemeCode, schemeName) {
    if (state.selectedFunds.some(f => f.schemeCode === schemeCode)) {
        alert('This fund is already selected');
        return;
    }
    if (state.selectedFunds.length >= 5) {
        alert('Maximum 5 funds can be compared at once');
        return;
    }

    showLoading(true);
    try {
        const response = await fetch(`/api/rolling-returns/${schemeCode}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        if (!data.rollingReturns || data.rollingReturns.length === 0) {
            alert('This fund does not have enough historical data for 5-year rolling returns');
            showLoading(false);
            return;
        }

        const colorIndex = state.selectedFunds.length;
        const fund = {
            schemeCode,
            schemeName,
            color: CHART_COLORS[colorIndex],
            data: data.rollingReturns,
            statistics: data.statistics
        };
        state.selectedFunds.push(fund);
        state.fundData[schemeCode] = data;
        updateUI();
    } catch (error) {
        console.error('Error fetching fund data:', error);
        alert('Error fetching fund data. Please try again.');
    } finally {
        showLoading(false);
    }
}

function removeFund(schemeCode) {
    state.selectedFunds = state.selectedFunds.filter(f => f.schemeCode !== schemeCode);
    delete state.fundData[schemeCode];
    state.selectedFunds.forEach((fund, index) => {
        fund.color = CHART_COLORS[index];
    });
    updateUI();
}

function updateUI() {
    updateFundsList();
    updateChart();
    updateStatistics();
}

function updateFundsList() {
    const count = state.selectedFunds.length;
    elements.fundCount.textContent = count;
    if (count === 0) {
        elements.selectedFundsList.innerHTML = '';
        elements.noFundsMessage.style.display = 'block';
        elements.chartSection.style.display = 'none';
        elements.statsSection.style.display = 'none';
    } else {
        elements.noFundsMessage.style.display = 'none';
        elements.chartSection.style.display = 'block';
        elements.statsSection.style.display = 'block';
        elements.selectedFundsList.innerHTML = state.selectedFunds.map(fund => `
            <div class="fund-tag">
                <span class="color-dot" style="background-color: ${fund.color}"></span>
                <span class="fund-name">${truncateName(fund.schemeName, 40)}</span>
                <button class="remove-btn" data-code="${fund.schemeCode}" title="Remove">&times;</button>
            </div>
        `).join('');
    }
}

function updateChart() {
    if (state.selectedFunds.length === 0) {
        if (state.chart) { state.chart.destroy(); state.chart = null; }
        return;
    }
    const datasets = state.selectedFunds.map(fund => ({
        label: truncateName(fund.schemeName, 30),
        data: fund.data.map(d => ({ x: d.date, y: d.return })),
        borderColor: fund.color,
        backgroundColor: fund.color + '20',
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 10,
        tension: 0.1,
        fill: false
    }));

    const ctx = document.getElementById('rollingReturnsChart').getContext('2d');
    if (state.chart) {
        state.chart.data.datasets = datasets;
        state.chart.update();
    } else {
        state.chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top', labels: { usePointStyle: true, padding: 15 } },
                    tooltip: {
                        callbacks: {
                            title: ctx => ctx[0].raw.x,
                            label: ctx => `${ctx.dataset.label}: ${ctx.raw.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'category',
                        title: { display: true, text: 'Date' },
                        ticks: {
                            maxTicksLimit: 10,
                            callback: function(value, index) {
                                const date = this.getLabelForValue(value);
                                if (index % Math.ceil(this.chart.data.datasets[0].data.length / 10) === 0) return date;
                                return '';
                            }
                        }
                    },
                    y: {
                        title: { display: true, text: 'Rolling Returns (CAGR %)' },
                        ticks: { callback: v => v + '%' }
                    }
                }
            }
        });
    }
}

function updateStatistics() {
    if (state.selectedFunds.length === 0) { elements.statsTableBody.innerHTML = ''; return; }
    elements.statsTableBody.innerHTML = state.selectedFunds.map(fund => {
        const stats = fund.statistics;
        if (!stats) return '';
        return `
            <tr>
                <td><div class="fund-name-cell"><span class="color-dot" style="background-color: ${fund.color}"></span>${truncateName(fund.schemeName, 50)}</div></td>
                <td>${stats.min.toFixed(2)}</td>
                <td>${stats.max.toFixed(2)}</td>
                <td>${stats.average.toFixed(2)}</td>
                <td>${stats.stdDev.toFixed(2)}</td>
                <td>${stats.positivePercentage.toFixed(1)}</td>
                <td>${stats.totalPeriods}</td>
            </tr>
        `;
    }).join('');
}

function exportToCSV() {
    if (state.selectedFunds.length === 0) { alert('No funds selected to export'); return; }
    const allDates = new Set();
    state.selectedFunds.forEach(fund => fund.data.forEach(d => allDates.add(d.date)));
    const sortedDates = Array.from(allDates).sort();
    const fundLookups = state.selectedFunds.map(fund => {
        const lookup = {};
        fund.data.forEach(d => { lookup[d.date] = d.return; });
        return { name: fund.schemeName, lookup };
    });
    let csv = 'Date,' + fundLookups.map(f => `"${f.name}"`).join(',') + '\n';
    sortedDates.forEach(date => {
        const row = [date];
        fundLookups.forEach(f => row.push(f.lookup[date] !== undefined ? f.lookup[date] : ''));
        csv += row.join(',') + '\n';
    });
    csv += '\n\nStatistics Summary\n';
    csv += 'Fund Name,Min (%),Max (%),Average (%),Std Dev,Positive Periods (%),Total Periods\n';
    state.selectedFunds.forEach(fund => {
        const stats = fund.statistics;
        if (stats) csv += `"${fund.schemeName}",${stats.min},${stats.max},${stats.average},${stats.stdDev},${stats.positivePercentage},${stats.totalPeriods}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rolling_returns_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function downloadChart() {
    if (!state.chart) { alert('No chart to download'); return; }
    const canvas = document.getElementById('rollingReturnsChart');
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    const padding = 20;
    tempCanvas.width = canvas.width + padding * 2;
    tempCanvas.height = canvas.height + padding * 2;
    tempCtx.fillStyle = '#ffffff';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.drawImage(canvas, padding, padding);
    const link = document.createElement('a');
    link.download = `rolling_returns_chart_${new Date().toISOString().split('T')[0]}.png`;
    link.href = tempCanvas.toDataURL('image/png', 1.0);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ===== Fund Rankings Tab Functions =====

async function loadRankings() {
    elements.loadRankingsBtn.style.display = 'none';
    elements.refreshRankingsBtn.style.display = 'none';
    elements.rankingsLoading.style.display = 'flex';
    elements.rankingsContent.style.display = 'none';

    try {
        const response = await fetch('/api/fund-rankings');
        const data = await response.json();
        state.rankingsData = data;
        state.rankingsLoaded = true;
        state.activeCategory = 'All';

        renderCategoryFilters();
        renderRankingsInsights();
        renderRankingsTable();

        elements.rankingsLoading.style.display = 'none';
        elements.rankingsContent.style.display = 'block';
        elements.refreshRankingsBtn.style.display = 'inline-flex';
    } catch (error) {
        console.error('Error loading rankings:', error);
        elements.rankingsLoading.style.display = 'none';
        elements.loadRankingsBtn.style.display = 'inline-flex';
        alert('Error loading fund rankings. Please try again.');
    }
}

function renderCategoryFilters() {
    const categories = ['All', ...state.rankingsData.categories];
    const counts = {};
    counts['All'] = state.rankingsData.totalFunds;
    for (const [cat, funds] of Object.entries(state.rankingsData.byCategory)) {
        counts[cat] = funds.length;
    }

    elements.categoryFilters.innerHTML = categories.map(cat => `
        <button class="category-pill ${cat === state.activeCategory ? 'active' : ''}" data-category="${cat}">
            ${cat} <span class="pill-count">${counts[cat] || 0}</span>
        </button>
    `).join('');
}

function renderRankingsInsights() {
    const data = state.rankingsData;
    if (!data || data.totalFunds === 0) {
        elements.rankingsInsights.innerHTML = '<p>No fund data available.</p>';
        return;
    }

    const allFunds = data.allFunds;
    const topFund = allFunds[0];
    const avgReturn = (allFunds.reduce((s, f) => s + f.avgReturn, 0) / allFunds.length).toFixed(1);

    // Find best category by average robustness
    let bestCat = '', bestCatScore = 0;
    for (const [cat, funds] of Object.entries(data.byCategory)) {
        const catAvg = funds.reduce((s, f) => s + f.robustnessScore, 0) / funds.length;
        if (catAvg > bestCatScore) { bestCatScore = catAvg; bestCat = cat; }
    }

    // Most consistent fund (lowest std dev among funds with decent returns)
    const decentFunds = allFunds.filter(f => f.avgReturn >= avgReturn);
    const mostConsistent = decentFunds.length > 0
        ? decentFunds.reduce((best, f) => f.stdDev < best.stdDev ? f : best)
        : allFunds[0];

    elements.rankingsInsights.innerHTML = `
        <div class="insights-grid">
            <div class="insight-card">
                <div class="insight-value">${data.totalFunds}</div>
                <div class="insight-label">Funds Analyzed</div>
            </div>
            <div class="insight-card">
                <div class="insight-value">${topFund.schemeName.split(' -')[0].split(' Direct')[0]}</div>
                <div class="insight-label">Top Ranked Fund (Score: ${topFund.robustnessScore})</div>
            </div>
            <div class="insight-card">
                <div class="insight-value">${bestCat}</div>
                <div class="insight-label">Strongest Category (Avg Score: ${bestCatScore.toFixed(1)})</div>
            </div>
            <div class="insight-card">
                <div class="insight-value">${mostConsistent.schemeName.split(' -')[0].split(' Direct')[0]}</div>
                <div class="insight-label">Most Consistent (Std Dev: ${mostConsistent.stdDev})</div>
            </div>
        </div>
    `;
}

function getFilteredFunds() {
    if (!state.rankingsData) return [];
    if (state.activeCategory === 'All') return [...state.rankingsData.allFunds];
    return [...(state.rankingsData.byCategory[state.activeCategory] || [])];
}

function renderRankingsTable() {
    let funds = getFilteredFunds();

    // Sort
    funds.sort((a, b) => {
        const va = a[state.sortField], vb = b[state.sortField];
        return state.sortAsc ? va - vb : vb - va;
    });

    // Apply top N filter
    funds = funds.slice(0, state.topN);

    // Update sort indicators
    document.querySelectorAll('.rankings-table .sortable').forEach(th => {
        th.classList.remove('active-sort', 'sort-asc', 'sort-desc');
        if (th.dataset.sort === state.sortField) {
            th.classList.add('active-sort', state.sortAsc ? 'sort-asc' : 'sort-desc');
        }
    });

    elements.rankingsTableBody.innerHTML = funds.map((fund, i) => {
        const scoreClass = fund.robustnessScore >= 10 ? 'score-high'
            : fund.robustnessScore >= 6 ? 'score-mid' : 'score-low';
        const returnClass = fund.avgReturn >= 15 ? 'return-good'
            : fund.avgReturn >= 10 ? 'return-ok' : 'return-low';

        return `
            <tr class="ranking-row" data-code="${fund.schemeCode}" data-name="${fund.schemeName}">
                <td class="rank-cell">${i + 1}</td>
                <td class="name-cell">
                    <div class="fund-name-primary">${fund.schemeName.split(' -')[0].split(' Direct')[0]}</div>
                    <div class="fund-house-label">${fund.fundHouse.split(' Mutual')[0]}</div>
                </td>
                <td><span class="category-badge">${fund.category}</span></td>
                <td class="num-cell ${returnClass}">${fund.avgReturn.toFixed(1)}%</td>
                <td class="num-cell">${fund.minReturn.toFixed(1)}%</td>
                <td class="num-cell">${fund.maxReturn.toFixed(1)}%</td>
                <td class="num-cell">${fund.stdDev.toFixed(1)}</td>
                <td class="num-cell">${fund.positivePercentage.toFixed(0)}%</td>
                <td class="num-cell"><span class="score-badge ${scoreClass}">${fund.robustnessScore.toFixed(1)}</span></td>
            </tr>
        `;
    }).join('');
}

// ===== Event Listeners =====

// Analyzer tab
elements.searchInput.addEventListener('input', debounce(e => searchFunds(e.target.value), 300));
elements.searchInput.addEventListener('focus', () => {
    if (elements.searchInput.value.length >= 2) elements.searchResults.classList.add('active');
});
document.addEventListener('click', e => {
    if (!elements.searchInput.contains(e.target) && !elements.searchResults.contains(e.target))
        elements.searchResults.classList.remove('active');
});
elements.searchResults.addEventListener('click', e => {
    const item = e.target.closest('.search-result-item');
    if (item && item.dataset.code) {
        addFund(item.dataset.code, item.dataset.name);
        elements.searchInput.value = '';
        elements.searchResults.classList.remove('active');
    }
});
elements.selectedFundsList.addEventListener('click', e => {
    if (e.target.classList.contains('remove-btn')) removeFund(e.target.dataset.code);
});
elements.exportBtn.addEventListener('click', exportToCSV);
elements.downloadChartBtn.addEventListener('click', downloadChart);

// Rankings tab
elements.loadRankingsBtn.addEventListener('click', loadRankings);
elements.refreshRankingsBtn.addEventListener('click', loadRankings);

elements.topNSelect.addEventListener('change', e => {
    state.topN = parseInt(e.target.value);
    renderRankingsTable();
});

elements.downloadRankingsBtn.addEventListener('click', downloadRankingsCSV);

function downloadRankingsCSV() {
    let funds = getFilteredFunds();
    funds.sort((a, b) => state.sortAsc ? a[state.sortField] - b[state.sortField] : b[state.sortField] - a[state.sortField]);
    funds = funds.slice(0, state.topN);

    let csv = 'Rank,Fund Name,Fund House,Category,Avg Return (%),Min Return (%),Max Return (%),Std Dev,Positive %,Robustness Score\n';
    funds.forEach((fund, i) => {
        csv += `${i + 1},"${fund.schemeName}","${fund.fundHouse}","${fund.category}",${fund.avgReturn},${fund.minReturn},${fund.maxReturn},${fund.stdDev},${fund.positivePercentage},${fund.robustnessScore}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fund_rankings_top${state.topN}_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

elements.categoryFilters.addEventListener('click', e => {
    const pill = e.target.closest('.category-pill');
    if (!pill) return;
    state.activeCategory = pill.dataset.category;
    document.querySelectorAll('.category-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    renderRankingsTable();
});

// Column sort
document.querySelector('.rankings-table thead').addEventListener('click', e => {
    const th = e.target.closest('.sortable');
    if (!th) return;
    const field = th.dataset.sort;
    if (state.sortField === field) {
        state.sortAsc = !state.sortAsc;
    } else {
        state.sortField = field;
        state.sortAsc = false;
    }
    renderRankingsTable();
});

// Click ranking row to load in Analyzer
document.getElementById('rankingsTableBody').addEventListener('click', e => {
    const row = e.target.closest('.ranking-row');
    if (!row) return;
    const code = row.dataset.code;
    const name = row.dataset.name;
    // Switch to analyzer tab and add fund
    document.querySelector('.tab-btn[data-tab="analyzer"]').click();
    addFund(code, name);
});

console.log('Mutual Fund Rolling Returns Analyzer initialized');
