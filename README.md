# Stock Growth Potential Analyser

A multi-factor stock screening tool for NYSE-listed stocks, built for **Windsurf IDE**.  
Enter a sector, get a ranked list of undervalued, high-growth-potential stocks — scored using Piotroski F-Score, valuation multiples, momentum signals, and a composite growth score.

---

## Features

- **12 sectors built-in** — Chemical, Mining, Semiconductors, Energy, Pharma, Fintech, EV, Defense, REIT, Consumer, Tech, Biotech
- **Piotroski F-Score** (0–9) — proven academic signal for separating winners from value traps
- **Valuation score** — forward P/E, P/B, P/S, EV/EBITDA, PEG ratio, FCF yield
- **Growth score** — revenue & earnings YoY growth, gross/operating margins, ROE
- **Momentum score** — 6-month price momentum, RSI(14), 50/200-day MA alignment
- **Composite ranking** (0–100) — weighted combination of all factors
- **Rich terminal UI** — colour-coded tables, progress bar, per-stock deep-dive cards
- **Parallel fetching** — 6 concurrent API calls for fast results
- **Custom tickers** — add any stock to any sector scan

---

## Quick Start

### 1. Clone / open in Windsurf

```bash
# Open the folder in Windsurf
# File → Open Folder → select stock-analyzer/
```

### 2. Create virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
# Interactive mode — will prompt you for a sector
python main.py

# Direct sector
python main.py --sector chemical
python main.py --sector mining
python main.py --sector semiconductor

# Custom top-N
python main.py --sector chemical --top 10

# Custom tickers (add to sector scan)
python main.py --sector chemical --tickers CE HUN RPM

# Scan only custom tickers (no default sector list)
python main.py --sector custom --tickers CDE KGC FCX OLN FMC

# List all sectors
python main.py --list-sectors
```

### 5. Windsurf debug panel (F5)

Pre-built launch configs are in `.vscode/launch.json`:
- **▶ Run Analyser (interactive)** — prompts for sector
- **▶ Chemical Sector** — runs chemical sector directly
- **▶ Mining Sector** — runs mining sector directly
- **▶ Custom Tickers** — example with manual ticker list
- **▶ List Sectors** — prints available sectors

---

## Project Structure

```
stock-analyzer/
│
├── main.py                    ← Entry point — CLI + orchestration
│
├── src/
│   ├── scorer.py              ← Core scoring engine
│   │                              Piotroski F-Score
│   │                              Valuation / Growth / Momentum scores
│   │                              Composite scoring (weighted)
│   └── display.py             ← Rich terminal UI
│                                  Results table, top-picks cards, summary
│
├── config/
│   └── sectors.py             ← Sector → ticker mappings (edit to customise)
│
├── .vscode/
│   ├── launch.json            ← Windsurf debug configurations
│   ├── settings.json          ← Python interpreter, formatter settings
│   └── extensions.json        ← Recommended extensions
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## How Scoring Works

### Piotroski F-Score (0–9)

| Signal | Category | Meaning |
|---|---|---|
| ROA > 0 | Profitability | Company is profitable |
| OCF > 0 | Profitability | Positive operating cash flow |
| ROA improving YoY | Profitability | Earnings quality improving |
| OCF > ROA (low accruals) | Profitability | Cash-backed earnings |
| Leverage decreasing | Leverage | Debt shrinking relative to assets |
| Current ratio improving | Liquidity | Balance sheet getting stronger |
| No share dilution | Dilution | Shares not increasing |
| Gross margin improving | Efficiency | Margins expanding |
| Asset turnover improving | Efficiency | Assets used more productively |

**Score ≥ 7** → Strong signal. **Score ≤ 3** → Potential value trap.

### Composite Score Weights

| Component | Weight | Key drivers |
|---|---|---|
| Growth Score | 40% | Revenue growth, EPS growth, margins, ROE, Piotroski |
| Valuation Score | 35% | Forward P/E, P/B, EV/EBITDA, PEG, FCF yield |
| Momentum Score | 25% | 6m price momentum, RSI, MA alignment |

### Rating Scale

| Score | Label |
|---|---|
| 65–100 | ★ Strong Buy |
| 50–64  | ▲ Buy |
| 35–49  | → Hold |
| 0–34   | ✗ Avoid |

---

## Customising Sectors

Edit `config/sectors.py` to:
- Add new sectors
- Add/remove tickers from existing sectors
- Change sector display names

```python
SECTOR_TICKERS = {
    "my_sector": ["AAPL", "MSFT", "GOOGL"],
    ...
}
SECTOR_DISPLAY = {
    "my_sector": "My Custom Sector",
    ...
}
```

---

## Data Source

All data is fetched via **Yahoo Finance** (`yfinance`) — completely free, no API key needed.  
Rate limiting: Yahoo Finance allows ~2,000 requests/hour. With 6 parallel workers and ~20 tickers per sector, a full scan typically completes in 20–40 seconds.

---

## Disclaimer

This tool is for informational and educational purposes only. It is **not** financial advice.  
Always conduct your own research and consult a qualified financial advisor before making investment decisions.
