# Technical Reference — Stock Growth Potential Analyser

## Table of Contents
1. [Tech Stack](#tech-stack)
2. [Architecture Overview](#architecture-overview)
3. [Data Pipeline](#data-pipeline)
4. [Scoring Methodology](#scoring-methodology)
   - [Piotroski F-Score](#1-piotroski-f-score-0--9)
   - [Valuation Score](#2-valuation-score-0--100)
   - [Growth Score](#3-growth-score-0--100)
   - [Momentum Score](#4-momentum-score-0--100)
   - [Composite Score](#5-composite-score-weighted-average)
5. [Rating Derivation](#rating-derivation)
6. [Known Data Limitations](#known-data-limitations)

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| **Data** | `yfinance 1.3.0` | Yahoo Finance API — prices, fundamentals, analyst targets |
| **Data** | `curl_cffi 0.15+` | TLS fingerprint spoofing (Chrome impersonation) — prevents Yahoo 429 rate limits |
| **Computation** | `pandas 2.2` | DataFrame manipulation for financial statements |
| **Computation** | `numpy 1.26` | Numerical operations (RSI, momentum) |
| **Computation** | `scipy` | Statistical helpers |
| **Terminal UI** | `rich 13.9` | Formatted tables, progress bars, styled console output |
| **Web UI** | `streamlit ≥1.35` | Browser-based dashboard with dropdown, progress bar, results table |
| **Concurrency** | `ThreadPoolExecutor` | Parallel ticker fetching (default `MAX_WORKERS = 1` for rate-limit safety) |

### Python version requirement
Python **≥ 3.10** is required. `curl_cffi` (a `yfinance 1.3.0` dependency) does not support Python 3.9.

---

## Architecture Overview

```
main.py  (CLI entry point)
│
├── config/sectors.py       — Sector → ticker mappings + display names
│
├── src/scorer.py           — Core scoring engine
│   ├── analyse_ticker()    — Full pipeline for one ticker
│   ├── compute_piotroski() — 9-point financial health check
│   └── compute_scores()    — Valuation / Growth / Momentum → Composite
│
├── src/display.py          — Rich terminal rendering
│   ├── results_table()     — Full ranked table
│   ├── top_picks_panel()   — Deep-dive cards for top N stocks
│   └── summary_stats()     — Sector-level summary panel
│
└── streamlit_app.py        — Web UI (runs independently of CLI)
```

Data flows strictly **downward**: `sectors.py` → `scorer.py` → `display.py` / `streamlit_app.py`. The scorer has no knowledge of how results are displayed.

---

## Data Pipeline

### Per-ticker flow (`analyse_ticker`)

```
yf.Ticker(ticker)
       │
       ├── stock.info           → price, P/E, P/B, margins, ROE, analyst target
       │                           (wrapped in _yf_fetch for YFRateLimitError retries)
       │
       ├── stock.history(1y)    → daily OHLCV for RSI, momentum, MAs
       │
       ├── stock.balance_sheet  ─┐
       ├── stock.income_stmt    ─┤→ compute_piotroski()
       └── stock.cashflow       ─┘
```

### Rate-limit handling

`yfinance 1.3.0` uses `curl_cffi` internally to impersonate Chrome's TLS fingerprint, which is the primary mechanism preventing Yahoo Finance from identifying and throttling the client.

`_yf_fetch()` wraps every data call with up to 3 retries, catching `YFRateLimitError` with exponential backoff (2s, 4s, 8s + jitter). `REQUEST_DELAY = 1.2s` is applied between tickers.

---

## Scoring Methodology

All sub-scores are normalised to **0 – 100**. Missing data causes that metric to be skipped rather than scored as zero, preserving accuracy for stocks with partial disclosure.

---

### 1. Piotroski F-Score (0 – 9)

A binary signal for each of 9 criteria drawn from annual financial statements (current year vs. prior year). Originally published by Joseph Piotroski (2000) as a screen for financially healthy value stocks.

#### Profitability (F1 – F4)

| Signal | Criterion | Source |
|---|---|---|
| **F1** — ROA positive | Net Income / Total Assets > 0 | Income stmt + Balance sheet |
| **F2** — OCF positive | Operating Cash Flow > 0 | Cash flow stmt |
| **F3** — ROA improving | ROA(t) > ROA(t-1) | Prior year comparison |
| **F4** — Accruals low | OCF / Total Assets > ROA | Cash earnings quality |

> F4 measures earnings quality: if cash flow exceeds accrual-based income, the company is generating real cash rather than paper profits.

#### Leverage & Liquidity (F5 – F7)

| Signal | Criterion | Source |
|---|---|---|
| **F5** — Leverage decreasing | Long-term Debt / Total Assets (t) ≤ (t-1) | Balance sheet |
| **F6** — Liquidity improving | Current Assets / Current Liabilities (t) > (t-1) | Balance sheet |
| **F7** — No share dilution | Shares outstanding (t) ≤ shares(t-1) × 1.02 | Income stmt / Balance sheet |

> F7 allows a 2% tolerance to avoid penalising minor stock-based compensation rounding.

#### Operating Efficiency (F8 – F9)

| Signal | Criterion | Source |
|---|---|---|
| **F8** — Gross margin improving | Gross Profit / Revenue (t) > (t-1) | Income stmt |
| **F9** — Asset turnover improving | Revenue / Total Assets (t) > (t-1) | Income stmt + Balance sheet |

#### Interpretation

| Score | Badge | Meaning |
|---|---|---|
| 7 – 9 | ▲ Strong | Financially healthy — multiple positive signals |
| 5 – 6 | ◆ Neutral | Mixed signals |
| 3 – 4 | ▼ Weak | Several red flags |
| 0 – 2 | ✗ Bearish | Significant financial deterioration |

A Piotroski score ≤ 4 is used as a **hard gate** — such stocks are ineligible for a Strong Buy rating regardless of other scores.

---

### 2. Valuation Score (0 – 100)

Higher score = cheaper relative to its earnings / book / cash flow. Each metric is linearly mapped to 0 – 100 with a cap at the "overvalued" extreme.

| Metric | Weight | Cap / Formula |
|---|---|---|
| **Forward P/E** | 1.0× | `max(0, 100 − (fwd_PE / 60) × 100)` — caps at 60× |
| **Trailing P/E** | 0.7× | Same formula, discounted vs. forward PE |
| **Price / Book** | 1.0× | `max(0, 100 − (P/B / 10) × 100)` — caps at 10× |
| **Price / Sales** | 1.0× | `max(0, 100 − (P/S / 8) × 100)` — caps at 8× |
| **EV / EBITDA** | 1.5× | `max(0, 100 − (EV_EBITDA / 30) × 100)` — caps at 30× |
| **PEG ratio** | 1.5× | `max(0, 100 − (PEG / 3) × 100)` — PEG < 1 is excellent |
| **FCF yield** | 1.0× | `min(100, FCF_yield% × 10)` — 10% yield → full marks |

Final score = weighted average of available metrics. EV/EBITDA and PEG are double-weighted because they are more robust cross-sector valuation tools.

> **Note on Forward P/E:** Yahoo Finance's `forwardPE` field uses GAAP-based consensus estimates. Company-guided non-GAAP EPS (common in software) can be materially higher, causing the displayed forward P/E to understate the actual non-GAAP multiple by 10–20%.

---

### 3. Growth Score (0 – 100)

Rewards companies with accelerating revenue, improving margins, and strong returns on equity.

| Metric | Weight | Formula |
|---|---|---|
| **Revenue growth YoY** | 1.0× | `clamp(50 + growth% × 100, 0, 100)` — 0% growth → 50 pts |
| **Earnings growth YoY** | 1.5× | `clamp(50 + growth% × 50, 0, 100)` — weighted higher |
| **Gross margin** | 1.0× | `min(100, gross_margin% × 150)` — high margins rewarded |
| **Operating margin** | 1.0× | `clamp(50 + op_margin% × 200, 0, 100)` |
| **ROE** | 1.0× | `min(100, ROE% × 300)` — 33%+ ROE → full marks |
| **Piotroski score** | 1.5× | `(piotroski / 9) × 100` — quality of earnings signal |

The inclusion of Piotroski in growth scoring means financial health influences both the quality gate (Strong Buy eligibility) and the underlying composite score.

---

### 4. Momentum Score (0 – 100)

Technical indicators assessing price trajectory and trend strength.

#### Price vs 52-week low

| Ratio (price / 52w low) | Points | Rationale |
|---|---|---|
| 1.05 – 1.50 (up 5–50%) | 80 | Sweet spot: recovering but not extended |
| 1.50 – 2.00 (up 50–100%) | 60 | Extended but still rising |
| < 1.05 (near 52w low) | 40 | Potential value or value trap |
| > 2.00 (doubled+) | 30 | Overextended |

#### 6-month price momentum

```
score = clamp(50 + momentum_6m% × 100, 0, 100)
```
A flat stock scores 50. Academic finance research (Jegadeesh & Titman, 1993) demonstrates that 6-month momentum is a persistent cross-sectional predictor of 3–12 month returns.

#### RSI (14-day, Wilder's smoothed)

Computed inline using EWM (exponentially weighted moving average) with `com=13`:

```
avg_gain = gains.ewm(com=13, min_periods=14).mean()
avg_loss = losses.ewm(com=13, min_periods=14).mean()
RSI = 100 − (100 / (1 + avg_gain / avg_loss))
```

| RSI range | Points | Interpretation |
|---|---|---|
| 40 – 65 | 80 | Healthy trend — neither overbought nor oversold |
| 30 – 40 | 65 | Oversold — potential mean reversion |
| 65 – 75 | 55 | Mildly overbought |
| < 30 or > 75 | 25 | Extreme — overbought/oversold |

#### Moving average alignment

| Signal | Points | Weight |
|---|---|---|
| Above 200-day MA | 70 | 1.0× |
| Below 200-day MA | 30 | 1.0× |
| Above 50-day MA | 65 | 0.5× |
| Below 50-day MA | 35 | 0.5× |

---

### 5. Composite Score (weighted average)

```
Composite = Valuation × 0.35 + Growth × 0.40 + Momentum × 0.25
```

| Dimension | Weight | Rationale |
|---|---|---|
| **Valuation** | 35% | Price paid matters — overpaying destroys long-run returns |
| **Growth** | 40% | Primary driver of equity value over a 3–5 year horizon |
| **Momentum** | 25% | Captures sentiment and near-term price trend |

---

## Rating Derivation

Ratings are **algo-derived** — they are not a pass-through of Yahoo Finance's consensus. The Street consensus is preserved separately in `yahoo_consensus` and used only for the Contrarian flag.

```
if composite ≥ 56  AND  analyst_upside ≥ 30%  AND  piotroski ≥ 5:
    → Strong Buy

elif composite < 43  OR  (piotroski ≤ 4 AND analyst_upside < 20%):
    → Hold          # weak composite OR bearish health + limited upside

elif composite ≥ 35  AND  analyst_upside ≥ 5%:
    → Buy

elif composite ≥ 20:
    → Hold

else:
    → Avoid
```

### Contrarian Strong Buy flag

When the algo assigns **Strong Buy** but the Yahoo Finance consensus is `hold`, `neutral`, `underperform`, or `sell`, the rating is relabelled **"Contrarian Strong Buy"** and surfaced with a warning in both the table and the deep-dive card. This surfaces value-vs-momentum divergence without suppressing the signal.

### Threshold rationale

| Threshold | Why |
|---|---|
| Composite ≥ 56 (not 55) | Prevents identical-composite stocks from getting different labels due to a sub-1% upside difference |
| Piotroski ≥ 5 | Excludes financially deteriorating stocks (bearish flag) from the top label |
| Upside ≥ 30% | Ensures Street consensus supports meaningful appreciation potential |
| Hold gate: composite < 43 | Prevents low-quality stocks with misleadingly high upside estimates from being labelled Buy |
| Hold gate: Piotroski ≤ 4 + upside < 20% | Catches weak-fundamental stocks with limited near-term analyst support |

---

## Known Data Limitations

| Issue | Impact | Status |
|---|---|---|
| **Yahoo Finance rate limiting** | 429 errors if requests are too frequent | Mitigated by `curl_cffi` Chrome impersonation + 1.2s inter-ticker delay |
| **Forward P/E is GAAP-based** | Software companies' non-GAAP guided EPS can make fwd P/E appear 10–20% lower than company guidance | Known, noted in code — no reliable free-data fix |
| **Annual financials only** | Piotroski uses annual statements; recent quarterly deterioration may not be reflected | Accepted trade-off for stability |
| **Analyst targets are mean, not median** | Outlier price targets can skew average upside | Use directional signal, not precise forecast |
| **Survivorship bias in sector lists** | Pre-screened tickers exclude micro-caps and recently delisted stocks | By design — focus is large/mid-cap liquid names |
| **Yahoo data latency** | `stock.info` data may lag by 1–2 trading days | Acceptable for multi-week holding horizon |
