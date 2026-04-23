# Stock Growth Potential Analyser

A multi-factor stock screening and AI validation tool for NYSE-listed stocks.  
Select a sector, get a ranked list scored on Piotroski F-Score, valuation multiples, growth signals, and momentum — then optionally run an AI layer to pressure-test the algo's ratings against the raw numbers.

**Live app:** [stock-analyzer-csvkpywthyb2lsjbf7lbwg.streamlit.app](https://stock-analyzer-csvkpywthyb2lsjbf7lbwg.streamlit.app/)  
**GitHub:** [github.com/nishantkhimesra/stock-analyzer](https://github.com/nishantkhimesra/stock-analyzer)

---

## Table of Contents

1. [What the App Does](#what-the-app-does)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [How Scoring Works](#how-scoring-works)
   - [Piotroski F-Score](#1-piotroski-f-score-09)
   - [Valuation Score](#2-valuation-score-0100)
   - [Growth Score](#3-growth-score-0100)
   - [Momentum Score](#4-momentum-score-0100)
   - [Composite Score](#5-composite-score)
   - [Rating Gates](#6-rating-gates)
5. [AI Validation](#ai-validation)
   - [What it checks](#what-it-checks)
   - [Validation rules](#validation-rules)
   - [Azure OpenAI setup](#azure-openai-setup)
6. [Review with AI](#review-with-ai)
   - [How candidates are selected](#how-candidates-are-selected)
   - [What the agent does](#what-the-agent-does)
   - [Verdicts](#verdicts)
7. [Download Report](#download-report)
8. [CLI Mode](#cli-mode)
9. [Known Data Limitations](#known-data-limitations)
10. [Disclaimer](#disclaimer)

---

## What the App Does

### Streamlit Web UI

The primary interface is a **Streamlit dashboard** with three phases:

1. **Sector scan** — select one of 12 built-in sectors (Tech, EV, Pharma, Semiconductor, Fintech, Energy, Mining, Chemical, Defense, REIT, Consumer, Biotech), click **🔍 Analyse Sector**, and the app fetches live data from Yahoo Finance for every ticker in that sector.

2. **Ranked results** — stocks are scored on 4 dimensions and sorted by composite score. The page shows:
   - A summary bar (avg composite, avg analyst upside, rating distribution)
   - Full results table with all metrics
   - Deep-dive cards for the top N picks (configurable in the sidebar)
   - A failed-data log for tickers that could not be fetched

3. **AI Validation** — after a scan, click **▶ Run Validation** to call an LLM (Azure OpenAI or OpenAI) that independently reviews every rating against the numeric data, flags disagreements, and produces an overall sector assessment.

4. **Review with AI** — after validation, click **🔬 Review with AI** to send the top Strong Buy picks (with high AI validation confidence) to a research agent that searches for recent news, identifies catalysts and risks, and delivers a buy/caution/avoid verdict with written reasoning.

All scan, validation, and review results are cached in `st.session_state` — switching tabs or adjusting the deep-dive slider does **not** re-fetch data. Running a new sector scan clears the previous validation and review automatically.

### CLI Mode

A Rich-formatted terminal interface (`python main.py`) is also available for scripted use. See [CLI Mode](#cli-mode) below.

---

## Project Structure

```
stock-analyzer/
│
├── streamlit_app.py        ← Streamlit web UI (primary interface)
├── main.py                 ← CLI entry point (terminal Rich UI)
│
├── src/
│   ├── scorer.py           ← Core scoring engine
│   │                           analyse_ticker(), compute_piotroski(),
│   │                           compute_scores(), StockScore dataclass
│   └── display.py          ← Rich terminal rendering
│                               results_table(), top_picks_panel(), summary_stats()
│
├── eval/
│   └── evaluate.py         ← AI validation + review pipeline
│                               call_openai(), call_openai_grounded(),
│                               call_review_agent(),
│                               VALIDATION_PROMPT, GROUNDING_PROMPT,
│                               REVIEW_AGENT_PROMPT
│
├── config/
│   └── sectors.py          ← Sector → ticker mappings + display names
│
├── .env.example            ← Environment variable template
├── requirements.txt
└── README.md
```

Data flows strictly downward: `sectors.py` → `scorer.py` → `streamlit_app.py` / `display.py`. The scorer has no knowledge of how results are rendered.

---

## Quick Start

### Local setup

```bash
git clone https://github.com/nishantkhimesra/stock-analyzer.git
cd stock-analyzer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Copy and fill in your credentials (OpenAI or Azure OpenAI)
cp .env.example .env

streamlit run streamlit_app.py
```

### Streamlit Cloud (deployed)

Secrets are managed via the Streamlit Cloud dashboard → **Settings → Secrets**. Paste in TOML format:

```toml
AZURE_OPENAI_API_KEY = "your-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
```

The app auto-syncs these into `os.environ` at startup so the validation pipeline picks them up without any code changes.

---

## How Scoring Works

All sub-scores are normalised to **0 – 100**. Missing data causes that metric to be skipped rather than scored as zero, preserving accuracy for stocks with partial disclosure.

---

### 1. Piotroski F-Score (0–9)

A binary signal for each of 9 criteria drawn from annual financial statements. Originally published by Joseph Piotroski (2000) as a screen for financially healthy value stocks.

#### Profitability (F1–F4)

| # | Signal | Criterion |
|---|---|---|
| F1 | ROA positive | Net Income / Total Assets > 0 |
| F2 | OCF positive | Operating Cash Flow > 0 |
| F3 | ROA improving | ROA(t) > ROA(t−1) |
| F4 | Low accruals | OCF / Total Assets > ROA — cash-backed earnings |

#### Leverage & Liquidity (F5–F7)

| # | Signal | Criterion |
|---|---|---|
| F5 | Leverage decreasing | Long-term Debt / Total Assets (t) ≤ (t−1) |
| F6 | Liquidity improving | Current Ratio (t) > Current Ratio (t−1) |
| F7 | No share dilution | Shares (t) ≤ shares (t−1) × 1.02 |

#### Operating Efficiency (F8–F9)

| # | Signal | Criterion |
|---|---|---|
| F8 | Gross margin improving | Gross Profit / Revenue (t) > (t−1) |
| F9 | Asset turnover improving | Revenue / Total Assets (t) > (t−1) |

#### Piotroski interpretation

| Score | Badge | Meaning |
|---|---|---|
| 7–9 | ▲ Strong | Financially healthy |
| 5–6 | ◆ Neutral | Mixed signals |
| 3–4 | ▼ Weak | Several red flags |
| 0–2 | ✗ Bearish | Significant deterioration |

**Hyper-growth exception:** Stocks with Piotroski ≤ 5 but revenue growth ≥ 50% **and** positive operating margin are flagged `piotroski_context = "hyper_growth"`. Rapid capex expansion often causes ROA to lag and accruals to spike — these are measurement artefacts, not financial distress.

---

### 2. Valuation Score (0–100)

Higher score = cheaper relative to earnings / book / cash flow. Each metric is linearly mapped to 0–100 with a cap at the overvalued extreme.

| Metric | Weight | Formula |
|---|---|---|
| Forward P/E | 1.0× | `max(0, 100 − (fwd_PE / 60) × 100)` |
| Trailing P/E | 0.7× | Same formula, discounted vs. forward PE |
| Price / Book | 1.0× | `max(0, 100 − (P/B / 10) × 100)` |
| Price / Sales | 1.0× | `max(0, 100 − (P/S / 8) × 100)` |
| EV / EBITDA | **1.5×** | `max(0, 100 − (EV_EBITDA / 30) × 100)` |
| PEG ratio | **1.5×** | `max(0, 100 − (PEG / 3) × 100)` |
| FCF yield | 1.0× | `min(100, FCF_yield% × 10)` |

EV/EBITDA and PEG are double-weighted as the most robust cross-sector tools.

> **Note:** Yahoo Finance `forwardPE` uses GAAP-based consensus estimates. Software companies' non-GAAP guided EPS can make forward P/E appear 10–20% lower than company guidance. If Piotroski = 0, valuation score is capped at 20 to prevent post-bankruptcy accounting artefacts from inflating the score.

---

### 3. Growth Score (0–100)

| Metric | Weight | Formula |
|---|---|---|
| Revenue growth YoY | 1.0× | `clamp(50 + growth% × 100, 0, 100)` — 0% growth → 50 pts |
| Earnings growth YoY | **1.5×** | `clamp(50 + growth% × 50, 0, 100)` |
| Gross margin | 1.0× | `min(100, gross_margin% × 150)` |
| Operating margin | 1.0× | `clamp(50 + op_margin% × 200, 0, 100)` |
| ROE | 1.0× | `min(100, ROE% × 300)` — 33%+ ROE = full marks |
| Piotroski score | **1.5×** | `(piotroski / 9) × 100` |

Piotroski's inclusion here means financial health influences both the quality gate (Strong Buy eligibility) and the underlying composite score.

---

### 4. Momentum Score (0–100)

#### Price vs 52-week low

| Ratio (price / 52w low) | Points | Rationale |
|---|---|---|
| 1.05–1.50 (up 5–50%) | 80 | Sweet spot — recovering but not extended |
| 1.50–2.00 (up 50–100%) | 60 | Extended but still rising |
| < 1.05 (near 52w low) | 40 | Potential value or value trap |
| > 2.00 (doubled+) | 30 | Overextended — unless Piotroski ≥ 7 + rev growth ≥ 50% |

#### 6-month price momentum

```
score = clamp(50 + momentum_6m% × 100, 0, 100)
```

Based on the Jegadeesh & Titman (1993) cross-sectional momentum factor — 6-month momentum is a persistent predictor of 3–12 month returns.

#### RSI (14-day, Wilder's smoothed EWM)

| RSI range | Points | Interpretation |
|---|---|---|
| 40–65 | 80 | Healthy trend |
| 30–40 | 65 | Oversold — potential mean reversion |
| 65–75 | 55 | Mildly overbought |
| < 30 or > 75 | 25 | Extreme — overbought or very weak |

#### Moving average alignment

| Signal | Points | Weight |
|---|---|---|
| Above 200-day MA | 70 | 1.0× |
| Below 200-day MA | 30 | 1.0× |
| Above 50-day MA | 65 | 0.5× |
| Below 50-day MA | 35 | 0.5× |

---

### 5. Composite Score

```
Composite = Valuation × 0.35 + Growth × 0.40 + Momentum × 0.25
```

| Dimension | Weight | Rationale |
|---|---|---|
| Valuation | 35% | Price paid matters — overpaying destroys long-run returns |
| Growth | 40% | Primary driver of equity value over a 3–5 year horizon |
| Momentum | 25% | Captures sentiment and near-term price trend |

---

### 6. Rating Gates

Ratings are **algo-derived** — they are not a pass-through of Yahoo Finance consensus. The Street consensus is preserved separately in `yahoo_consensus` and used only for the Contrarian flag.

```
if composite == 0  AND  piotroski == 0:
    → Avoid  (no positive signals at all — covers bankrupt/delisted stocks)

elif (composite ≥ 56 AND analyst_upside ≥ 30% AND piotroski ≥ 5)
  OR (composite ≥ 65 AND piotroski ≥ 7):
    → Strong Buy  (second clause: exceptional quality, upside gate relaxed)

elif composite < 43  OR  piotroski ≤ 4:
    → Hold  (weak composite OR poor financial health — upside does NOT override)

elif composite ≥ 35  AND  analyst_upside ≥ 5%:
    → Buy

elif composite ≥ 20:
    → Hold

else:
    → Avoid
```

**Contrarian Strong Buy:** When the algo assigns Strong Buy but the Yahoo consensus is `hold`, `neutral`, `underperform`, or `sell`, the label is changed to **"Contrarian Strong Buy"** and surfaced with a warning. This flags value-vs-Street divergence without suppressing the signal.

#### Why these thresholds?

| Threshold | Reason |
|---|---|
| Composite ≥ 56 (not 55) | Prevents two stocks with identical composites getting different labels from a sub-1% upside difference |
| Piotroski ≥ 5 for Strong Buy | Excludes financially deteriorating stocks from the top label |
| Upside ≥ 30% for Strong Buy | Ensures Street consensus supports meaningful appreciation potential |
| Hold gate: composite < 43 | Prevents low-quality stocks with misleadingly high upside from being labelled Buy |
| cs == 0 and p == 0 → Avoid | Catches bankrupt/post-restructuring stocks whose metrics produce zero without triggering `data_quality = "failed"` |

---

## AI Validation

After a sector scan, click **▶ Run Validation** to call the AI validation pipeline. This uses an LLM (Azure OpenAI or direct OpenAI) to independently review every rating — not to generate a new opinion, but to pressure-test the algo's math.

### What it checks

For every stock the AI must follow these steps **in order**:

**Step 1 — Upside sign**  
`analyst_upside` is the % gap between current price and mean analyst price target. A negative value means the stock already trades **above** its target — it cannot be a bullish pick regardless of revenue growth.

**Step 2 — Piotroski gate**  
Strong Buy requires Piotroski ≥ 5. Piotroski ≤ 2 is a distress signal that must be flagged on any Buy or Strong Buy. The `hyper_growth` context flag is an exception — fast-growing profitable companies sometimes score low due to measurement artefacts.

**Step 3 — Composite threshold**  
The AI checks the actual number: Strong Buy needs composite ≥ 56, Buy needs composite ≥ 35. If the composite doesn't support the rating, the AI disagrees and cites the specific gap.

**Step 4 — Internal consistency**  
Metrics are cross-checked — a stock with fwd P/E > 60 AND negative upside AND Piotroski ≤ 3 should not be bullish regardless of story. The AI is forbidden from inferring a rating from company name or revenue growth narrative alone.

**Step 5 — Data quality flags**  
Implausibly low forward P/E on a loss-making company (post-bankruptcy accounting artefact), or analyst upside > 150% with Piotroski ≤ 3 (likely stale data), are flagged.

### Output

The validation returns:

- **Per-stock table** — algo rating vs AI suggested rating, agreement level (✅ agree / ⚠️ partial / ❌ disagree), confidence (🟢 high / 🟡 medium / 🔴 low), and notes citing specific numbers
- **Overall assessment** — sector summary, top conviction picks (must have positive upside + Piotroski ≥ 5 + composite ≥ 56), most contested ratings, systemic issues, data quality flags
- **Disagreement warning** — orange bar listing any tickers the AI explicitly disagrees on

### Azure OpenAI setup

The app auto-detects credentials. If `AZURE_OPENAI_ENDPOINT` is present in the environment, Azure is used; otherwise direct OpenAI is used.

**`.env` (local development):**

```bash
# Option A — OpenAI direct
OPENAI_API_KEY=your-openai-key

# Option B — Azure OpenAI (takes priority if AZURE_OPENAI_ENDPOINT is set)
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# AZURE_OPENAI_API_VERSION=2025-01-01-preview  # optional, this is the default
```

**Streamlit Cloud:** Add the same keys via **Settings → Secrets** in TOML format (no `.env` file needed).

The validation is also available as a CLI command:

```bash
python eval/evaluate.py --sector tech --mode opinion
python eval/evaluate.py --sector tech --mode grounded   # web-search fact-check (OpenAI)
python eval/evaluate.py --sector tech --mode both
python eval/evaluate.py --sector tech --mode opinion --save  # saves JSON to disk
```

---

## Review with AI

After running AI Validation, click **🔬 Review with AI** to launch a research agent that performs a final due-diligence pass on the strongest picks. The agent is designed to be quantitatively anchored — it derives a baseline verdict from the numbers first and only deviates from it for concrete, stock-specific news.

### How candidates are selected

The agent only reviews stocks that meet **all three** of the following criteria:

1. `analyst_rating` is `"Strong Buy"` or `"Contrarian Strong Buy"`
2. AI Validation `agreement` is `"agree"` — the algo's gate logic was confirmed correct
3. AI Validation `confidence` is `"high"` — no numeric ambiguity or partial data

Up to 3 candidates are selected (sorted by composite score, highest first).

### What the agent does

The agent follows a **fixed four-step process** for every candidate:

**Step 1 — Quantitative baseline (mandatory first)**

The agent reads `composite_score`, `piotroski`, `analyst_upside`, and `fwd_pe` verbatim from the data and classifies the stock:

| Classification | Criteria |
|---|---|
| ▲ **STRONG** | composite ≥ 65 AND piotroski ≥ 7 (Path A) — OR — composite ≥ 56 AND piotroski ≥ 5 AND upside ≥ 30% (Path B) |
| ◆ **MODERATE** | composite 43–55 AND piotroski 5–6 AND upside ≥ 5% |
| ▼ **WEAK** | anything else — flagged as unexpected for a review candidate |

The agent also reads the `ai_validation_notes` and `ai_validation_concerns` passed from the validation run and explicitly addresses any flags raised there.

**Step 2 — Stock-specific news research**

The agent searches for news published in the **last 4 weeks** for that specific stock: earnings results, guidance updates, pipeline approvals, analyst upgrades/downgrades, management changes, or macro events that affect this company specifically.

Generic sector-wide risks — "patent expirations", "increased competition", "regulatory hurdles" — are **prohibited** as key risks unless a concrete recent event ties them to this specific stock. Every news item cited uses hedged attribution ("According to [source]…" or "Recent reports indicate…"). If nothing relevant is found, the agent states it explicitly and falls back to the quantitative baseline.

**Step 3 — Verdict determination**

The verdict starts from the quantitative baseline and only deviates for concrete evidence:

| Quant baseline | News signal | Verdict |
|---|---|---|
| ▲ STRONG | Neutral or positive | ✅ Confirm Buy |
| ▲ STRONG | One specific near-term risk | ⚠️ Proceed with Caution |
| ◆ MODERATE | Positive news | ✅ Confirm Buy |
| ◆ MODERATE | Neutral or no news | ⚠️ Proceed with Caution |
| Any | Concrete hard negative event | ❌ Avoid for Now |

**"Avoid for Now" requires a concrete hard event:** earnings miss + guidance cut, fraud allegation, insolvency signal, product recall, or major contract loss. Sector-level concerns alone never justify it. The "Contrarian" label on its own never justifies "Proceed with Caution" — see below.

**Step 4 — Confidence calibration**

| Confidence | Meaning |
|---|---|
| `high` | Specific recent news found + quant baseline is unambiguous (▲ STRONG) |
| `medium` | News inconclusive, or quant baseline is ◆ MODERATE, or concerns noted |
| `low` | No recent news found — verdict based on quant data alone |

### Understanding "Contrarian Strong Buy"

A **Contrarian Strong Buy** means the quantitative model scores the stock as Strong Buy on fundamentals (high composite + Piotroski ≥ 7), but Wall Street consensus is bearish or neutral. The "Contrarian" label captures Street divergence — it is **not a warning**. The agent is explicitly instructed that a stock with composite ≥ 65 and Piotroski = 9 is quantitatively exceptional and should default to "Confirm Buy" unless specific negative news overrides that.

### Verdicts

| Verdict | Meaning | UI colour |
|---|---|---|
| ✅ **Confirm Buy** | Quant baseline is strong; news is neutral-to-positive | Green |
| ⚠️ **Proceed with Caution** | Quant case holds but a specific near-term risk was found, or baseline is only MODERATE | Amber |
| ❌ **Avoid for Now** | A concrete hard negative event contradicts the bullish rating | Red |

### What each verdict card shows

Each card displays (in order): **quant baseline** (the verbatim numeric anchor) · **recent news summary** with source attribution · **key catalysts** · **key risks** · **reasoning** (must cite ≥ 2 numbers from the data and specific news found).

### Web search capability

On Azure OpenAI, the agent uses **Bing Search grounding** if `BING_SEARCH_KEY` is set, enabling live web retrieval. On direct OpenAI, `gpt-4o-mini-search-preview` (built-in web search) is used automatically. Without a search key on Azure, the agent falls back to model knowledge with reduced recency.

> **Note:** The Review with AI agent is an informational tool, not a trading signal. It relies on news available at the time of the run and can miss important context. Always conduct independent research before any investment decision.

---

## Download Report

After a scan (with or without validation and review), click **⬇️ Download Report** — available in both the sidebar and the bottom of the page.

The report is a `.md` file named `{sector}_analysis_{YYYYMMDD_HHMM}.md`. The download button caption tells you exactly what is included:

| What was run | Report contents |
|---|---|
| Sector scan only | Summary + full results table |
| Scan + AI Validation | + Validation table + overall assessment |
| Scan + Validation + Review with AI | + Per-stock review cards (verdict, catalysts, risks, reasoning) |

Placeholder notes are inserted for any stage that was not run, so the report structure is always consistent.

---

## CLI Mode

```bash
# Interactive — prompts for sector
python main.py

# Direct sector
python main.py --sector tech
python main.py --sector ev --top 10

# Custom tickers added to the sector scan
python main.py --sector tech --tickers ARM TSM AMAT

# Scan only custom tickers
python main.py --sector custom --tickers CDE KGC FCX

# List available sectors
python main.py --list-sectors
```

Pre-built Windsurf launch configs are in `.vscode/launch.json` (F5 → select config).

---

## Known Data Limitations

| Issue | Impact |
|---|---|
| **Yahoo Finance rate limiting** | 429 errors if requests are too frequent — mitigated by `curl_cffi` Chrome impersonation + 1.2s inter-ticker delay |
| **Forward P/E is GAAP-based** | Software companies' non-GAAP guided EPS can make fwd P/E appear 10–20% lower than company guidance |
| **Annual financials only** | Piotroski uses annual statements — recent quarterly deterioration may not be reflected |
| **Analyst targets are mean, not median** | Outlier price targets can skew average upside — use as a directional signal, not a precise forecast |
| **Yahoo data latency** | `stock.info` may lag by 1–2 trading days — acceptable for multi-week holding horizon |
| **Survivorship bias** | Pre-screened sector lists exclude micro-caps and recently delisted stocks — by design, focus is large/mid-cap liquid names |

---

## Disclaimer

This tool is for **informational and educational purposes only**. It is not financial advice.  
Always conduct your own research and consult a qualified financial advisor before making any investment decisions.
