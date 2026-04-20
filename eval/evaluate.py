"""
eval/evaluate.py — AI-powered validation agent for stock screening results.

Usage:
    python eval/evaluate.py --sector tech
    python eval/evaluate.py --sector semiconductor --model gpt-4o
    python eval/evaluate.py --sector fintech --save fintech_validation.json

Requires:
    OPENAI_API_KEY set in environment (or .env file)
"""

import sys
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box
from rich.panel import Panel

from config.sectors import SECTOR_TICKERS, SECTOR_DISPLAY
from src.scorer import analyse_ticker, StockScore

load_dotenv()

console = Console()

DEFAULT_MODEL = "gpt-4o-mini"  # fast & cheap; use gpt-4o for deeper analysis
MAX_WORKERS   = 3

SCORING_CONTEXT = """
## Scoring Methodology

**Composite Score** = Valuation×0.35 + Growth×0.40 + Momentum×0.25  (0–100 scale)

**Valuation Score** (0–100): Forward P/E capped at 60×, P/B capped at 10×,
P/S capped at 8×, EV/EBITDA capped at 30×, PEG capped at 3×, FCF yield.
EV/EBITDA and PEG are double-weighted. Lower multiples = higher score.

**Growth Score** (0–100): Revenue growth YoY (50 pts at 0% growth, linear),
earnings growth (1.5× weight), gross margin (×1.5), operating margin,
ROE (>33% = full marks), Piotroski score (1.5× weight for earnings quality).

**Momentum Score** (0–100): 6-month price momentum (academic factor),
RSI-14 (healthy range 40–65 = 80 pts, overbought/oversold penalised),
50-day and 200-day MA alignment, price vs 52-week low.

**Piotroski F-Score** (0–9): 9 binary signals across profitability (F1–F4),
leverage/liquidity (F5–F7), operating efficiency (F8–F9).

**Rating Gates:**
- Strong Buy: (composite ≥ 56 AND analyst upside ≥ 30% AND Piotroski ≥ 5)
              OR (composite ≥ 65 AND Piotroski ≥ 7) [fundamental bypass for
              stocks like MU where price already moved but quality is exceptional]
- Hold gate:  composite < 43 OR (Piotroski ≤ 4 AND upside < 20%)
- Buy:        composite ≥ 35 AND analyst upside ≥ 5%
- Hold:       composite ≥ 20
- Avoid:      composite < 20
- Contrarian Strong Buy: Strong Buy by algo but Street consensus is Hold/Neutral

**Known data limitations:**
- Forward P/E from Yahoo is GAAP-based; non-GAAP guided EPS (common in software)
  can make fwd P/E appear 10–20% lower than company guidance.
- Piotroski scores ≤ 5 on stocks with rev growth ≥ 50% and positive operating
  margin are flagged as "hyper_growth" — likely measurement artefacts, not distress.
"""

VALIDATION_PROMPT = """
You are a senior quantitative analyst validating algorithmic stock screening output.

{scoring_context}

Below are the screening results for the **{sector}** sector. Your job is to:
1. Validate whether each rating is defensible given the metrics.
2. Flag data anomalies, edge cases, or stocks where you'd assign a different rating.
3. Identify any systemic patterns or calibration issues across the sector.

---

## Screening Results

```json
{results_json}
```

---

## Your Task

Respond with **valid JSON only** (no markdown fences, no preamble). Use this exact schema:

{{
  "stock_validations": [
    {{
      "ticker": "AAPL",
      "rating_given": "Buy",
      "rating_suggested": "Buy",
      "agreement": "agree|partial|disagree",
      "confidence": "high|medium|low",
      "concerns": [],
      "notes": "one-sentence rationale"
    }}
  ],
  "overall_assessment": {{
    "sector_summary": "2-3 sentence overview of the sector's health based on results",
    "systemic_issues": ["any patterns or calibration concerns across the whole run"],
    "top_conviction_picks": ["up to 3 tickers you most agree with"],
    "most_contested_ratings": ["tickers where you'd assign a different rating"],
    "data_quality_flags": ["tickers with suspicious or inconsistent metrics"]
  }}
}}
"""


def serialise_result(r: StockScore) -> dict:
    """Convert a StockScore to a compact dict for the Claude prompt."""
    return {
        "ticker":           r.ticker,
        "company":          r.company_name,
        "rating":           r.analyst_rating,
        "yahoo_consensus":  r.yahoo_consensus or "N/A",
        "composite":        r.composite_score,
        "valuation":        r.valuation_score,
        "growth":           r.growth_score,
        "momentum":         r.momentum_score,
        "piotroski":        r.piotroski_score,
        "piotroski_context": r.piotroski_detail.get("_context", "") if r.piotroski_detail else "",
        "fwd_pe":           round(r.forward_pe, 1)           if r.forward_pe           else None,
        "revenue_growth":   f"{r.revenue_growth_yoy*100:.1f}%" if r.revenue_growth_yoy else None,
        "gross_margin":     f"{r.gross_margin*100:.1f}%"     if r.gross_margin         else None,
        "roe":              f"{r.roe*100:.1f}%"              if r.roe                  else None,
        "rsi_14":           round(r.rsi_14, 0)               if r.rsi_14               else None,
        "analyst_upside":   f"{r.upside_to_target:.1f}%"     if r.upside_to_target     else None,
        "analyst_pt":       round(r.analyst_target, 2)       if r.analyst_target       else None,
        "data_quality":     r.data_quality,
    }


def run_analysis(sector_key: str) -> list[StockScore]:
    tickers = SECTOR_TICKERS[sector_key]
    console.print(f"\n[dim]Scanning {len(tickers)} tickers in [bold]{SECTOR_DISPLAY[sector_key]}[/bold]…[/dim]")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(analyse_ticker, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            results.append(r)
            status = "[green]✓[/green]" if r.data_quality != "failed" else "[red]✗[/red]"
            console.print(f"  {status} {r.ticker} ({done}/{len(tickers)})", end="\r")

    console.print()
    results.sort(key=lambda r: (r.data_quality == "failed", -r.composite_score))
    return results


def call_openai(results: list[StockScore], sector_key: str, model: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    valid = [r for r in results if r.data_quality != "failed"]
    payload = [serialise_result(r) for r in valid]

    prompt = VALIDATION_PROMPT.format(
        scoring_context=SCORING_CONTEXT,
        sector=SECTOR_DISPLAY[sector_key],
        results_json=json.dumps(payload, indent=2),
    )

    console.print(f"\n[dim]Calling OpenAI ({model}) for validation…[/dim]")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def render_report(results: list[StockScore], validation: dict, sector_key: str):
    sector_name = SECTOR_DISPLAY[sector_key]
    ov = validation.get("overall_assessment", {})
    stock_vals = {v["ticker"]: v for v in validation.get("stock_validations", [])}

    console.print()
    console.print(Rule(f"[bold white] Claude Validation Report — {sector_name} [/bold white]", style="cyan"))

    # ── Per-stock table ───────────────────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", show_lines=False, padding=(0, 1))
    table.add_column("Ticker",      style="bold white",  width=7)
    table.add_column("Algo Rating", width=22)
    table.add_column("Claude Says", width=22)
    table.add_column("Agreement",   width=10)
    table.add_column("Confidence",  width=10)
    table.add_column("Notes",       width=52, no_wrap=False)

    agreement_colour = {"agree": "green", "partial": "yellow", "disagree": "red"}
    confidence_colour = {"high": "bright_green", "medium": "yellow", "low": "orange1"}

    valid = [r for r in results if r.data_quality != "failed"]
    for r in valid:
        v = stock_vals.get(r.ticker, {})
        agree  = v.get("agreement", "—")
        conf   = v.get("confidence", "—")
        rating_s = v.get("rating_suggested", "—")
        notes    = v.get("notes", "")
        concerns = v.get("concerns", [])
        if concerns:
            notes += "  [dim]⚠ " + " · ".join(concerns) + "[/dim]"

        table.add_row(
            r.ticker,
            r.analyst_rating,
            rating_s,
            f"[{agreement_colour.get(agree, 'white')}]{agree}[/]",
            f"[{confidence_colour.get(conf, 'white')}]{conf}[/]",
            notes,
        )

    console.print(table)

    # ── Overall assessment ────────────────────────────────────────────────────
    console.print(Rule("[bold white] Overall Assessment [/bold white]", style="cyan"))

    summary = ov.get("sector_summary", "")
    if summary:
        console.print(Panel(summary, title="Sector Summary", border_style="dim"))

    def bullet_list(title: str, items: list, colour: str = "white"):
        if not items:
            return
        console.print(f"\n[bold {colour}]{title}[/bold {colour}]")
        for item in items:
            console.print(f"  • {item}")

    bullet_list("Systemic Issues",       ov.get("systemic_issues", []),       "yellow")
    bullet_list("Top Conviction Picks",  ov.get("top_conviction_picks", []),  "bright_green")
    bullet_list("Most Contested Ratings",ov.get("most_contested_ratings", []),"orange1")
    bullet_list("Data Quality Flags",    ov.get("data_quality_flags", []),    "red")

    console.print()


def main():
    parser = argparse.ArgumentParser(description="AI validation agent for stock screening results")
    parser.add_argument("--sector", required=True, choices=list(SECTOR_TICKERS.keys()),
                        help="Sector to analyse and validate")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--save",   metavar="FILE",
                        help="Save raw Claude JSON response to a file")
    args = parser.parse_args()

    results    = run_analysis(args.sector)
    validation = call_openai(results, args.sector, args.model)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(validation, f, indent=2)
        console.print(f"[dim]Raw validation saved to {args.save}[/dim]")

    render_report(results, validation, args.sector)

    # Exit non-zero if OpenAI flagged any disagreements
    disagreements = [
        v for v in validation.get("stock_validations", [])
        if v.get("agreement") == "disagree"
    ]
    if disagreements:
        tickers = ", ".join(v["ticker"] for v in disagreements)
        console.print(f"[yellow]⚠  OpenAI disagrees on: {tickers}[/yellow]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
