"""
eval/evaluate.py — AI-powered validation agent for stock screening results.

Modes:
  opinion  (default) — GPT reviews rating logic as a second opinion
  grounded            — GPT uses live web search to verify price / analyst PT / revenue data
  both                — runs opinion first, then grounded fact-check

Usage:
    python eval/evaluate.py --sector tech
    python eval/evaluate.py --sector semiconductor --model gpt-4o
    python eval/evaluate.py --sector fintech --mode grounded
    python eval/evaluate.py --sector tech --mode both --save run.json

Requires:
    OPENAI_API_KEY set in environment (or .env file)
    openai >= 1.66.0 for grounded mode (Responses API + web search)
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

DEFAULT_MODEL   = "gpt-4o-mini"          # opinion mode — fast & cheap
GROUNDED_MODEL  = "gpt-4o-search-preview"  # grounded mode — web search enabled via Responses API
MAX_WORKERS     = 3

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
- Hold gate:  composite < 43 OR Piotroski ≤ 4  (absolute floor — high upside does not override)
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


GROUNDING_PROMPT = """
You are a financial data accuracy agent. Use web search to verify the following
stock data points that were pulled from Yahoo Finance earlier today.

For each stock, search and verify:
1. **Current price** — is the stated price within ±10% of what you find?
2. **Analyst consensus PT** — does the stated target match the median Street target?
3. **Revenue growth YoY** — does the stated growth match the most recent reported figures?

Stocks to verify:
{stocks_json}

Rules:
- Search for each stock individually; do not skip any.
- If a data point cannot be found or verified, mark it as "unverified".
- Flag any discrepancy > 10% on price or > 15% on analyst PT as a concern.
- Revenue growth discrepancy > 3 percentage points is worth flagging.

Respond with ONLY valid JSON (no markdown fences, no preamble). Schema:
{{
  "data_checks": [
    {{
      "ticker": "CRM",
      "price_stated": 185.0,
      "price_found": "183.50",
      "price_ok": true,
      "pt_stated": 268.0,
      "pt_found": "230.00",
      "pt_ok": false,
      "rev_growth_stated": "12.1%",
      "rev_growth_found": "11.8%",
      "rev_growth_ok": true,
      "flags": ["analyst PT may be stale — found $230 vs stated $268"],
      "data_ok": true
    }}
  ],
  "grounding_summary": "2-sentence overall data accuracy assessment"
}}
"""


def serialise_result(r: StockScore) -> dict:
    """Convert a StockScore to a compact dict for the opinion prompt."""
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


def call_openai_grounded(results: list[StockScore], top_n: int = 8) -> dict:
    """Fact-check top_n stocks via OpenAI web search (Responses API)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    if not hasattr(client, "responses"):
        console.print(
            "[red]Error:[/red] openai SDK >= 1.66.0 required for grounded mode.\n"
            "Run: pip install -U openai"
        )
        sys.exit(1)

    valid = [r for r in results if r.data_quality != "failed"][:top_n]
    payload = [
        {
            "ticker":          r.ticker,
            "company":         r.company_name,
            "price_stated":    round(r.current_price, 2) if r.current_price else None,
            "analyst_pt":      round(r.analyst_target, 2) if r.analyst_target else None,
            "rev_growth":      f"{r.revenue_growth_yoy*100:.1f}%" if r.revenue_growth_yoy else None,
        }
        for r in valid
    ]

    prompt = GROUNDING_PROMPT.format(stocks_json=json.dumps(payload, indent=2))
    console.print(
        f"\n[dim]Calling OpenAI web search ({GROUNDED_MODEL}) "
        f"to fact-check top {len(valid)} stocks…[/dim]"
    )

    try:
        response = client.responses.create(
            model=GROUNDED_MODEL,
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            console.print(
                f"[red]Error:[/red] Model '{GROUNDED_MODEL}' not available on your API tier.\n"
                "OpenAI web search requires a Tier 1+ account (first billing payment made).\n"
                "Check access at: https://platform.openai.com/docs/models/gpt-4o-search-preview"
            )
            sys.exit(1)
        raise

    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def render_grounding_report(grounding: dict, sector_key: str):
    """Render the web-search data accuracy report."""
    sector_name = SECTOR_DISPLAY[sector_key]
    checks = {c["ticker"]: c for c in grounding.get("data_checks", [])}

    console.print()
    console.print(Rule(
        f"[bold white] Web-Search Grounding Report — {sector_name} [/bold white]",
        style="magenta",
    ))

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", show_lines=False, padding=(0, 1))
    table.add_column("Ticker",       style="bold white", width=7)
    table.add_column("Price (stated→found)",   width=22)
    table.add_column("Analyst PT (stated→found)", width=26)
    table.add_column("Rev Growth (stated→found)", width=26)
    table.add_column("Flags",        width=44, no_wrap=False)

    ok_colour   = {True: "green", False: "red"}
    unverified  = "[dim]unverified[/dim]"

    for ticker, c in checks.items():
        def fmt_pair(stated, found, ok):
            if found in (None, "", "unverified"):
                return f"{stated} → {unverified}"
            colour = ok_colour.get(ok, "white")
            return f"{stated} → [{colour}]{found}[/{colour}]"

        price_cell = fmt_pair(c.get("price_stated"), c.get("price_found"), c.get("price_ok"))
        pt_cell    = fmt_pair(c.get("pt_stated"),    c.get("pt_found"),    c.get("pt_ok"))
        rev_cell   = fmt_pair(c.get("rev_growth_stated"), c.get("rev_growth_found"), c.get("rev_growth_ok"))
        flags      = "  ".join(f"[yellow]⚠ {f}[/yellow]" for f in c.get("flags", []))

        table.add_row(ticker, price_cell, pt_cell, rev_cell, flags or "[dim green]clean[/dim green]")

    console.print(table)

    summary = grounding.get("grounding_summary", "")
    if summary:
        console.print(Panel(summary, title="Grounding Summary", border_style="dim magenta"))

    dirty = [t for t, c in checks.items() if not c.get("data_ok", True)]
    if dirty:
        console.print(f"[red]⚠  Potentially stale data for: {', '.join(dirty)}[/red]\n")
    else:
        console.print("[green]✓  All verified data points look accurate.[/green]\n")


def render_report(results: list[StockScore], validation: dict, sector_key: str):
    sector_name = SECTOR_DISPLAY[sector_key]
    ov = validation.get("overall_assessment", {})
    stock_vals = {v["ticker"]: v for v in validation.get("stock_validations", [])}

    console.print()
    console.print(Rule(f"[bold white] OpenAI Validation Report — {sector_name} [/bold white]", style="cyan"))

    # ── Per-stock table ───────────────────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", show_lines=False, padding=(0, 1))
    table.add_column("Ticker",      style="bold white",  width=7)
    table.add_column("Algo Rating", width=22)
    table.add_column("OpenAI Says", width=22)
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
                        help=f"OpenAI model for opinion mode (default: {DEFAULT_MODEL})")
    parser.add_argument("--mode",   default="opinion",
                        choices=["opinion", "grounded", "both"],
                        help="opinion: logic review | grounded: web fact-check | both: run both")
    parser.add_argument("--grounded-top", type=int, default=8, metavar="N",
                        help="Number of top stocks to fact-check in grounded mode (default: 8)")
    parser.add_argument("--save",   metavar="FILE",
                        help="Save raw JSON output(s) to a file")
    args = parser.parse_args()

    results = run_analysis(args.sector)
    output  = {}

    if args.mode in ("opinion", "both"):
        validation = call_openai(results, args.sector, args.model)
        output["opinion"] = validation
        render_report(results, validation, args.sector)

        disagreements = [
            v for v in validation.get("stock_validations", [])
            if v.get("agreement") == "disagree"
        ]
        if disagreements:
            tickers = ", ".join(v["ticker"] for v in disagreements)
            console.print(f"[yellow]⚠  OpenAI disagrees on: {tickers}[/yellow]")

    if args.mode in ("grounded", "both"):
        grounding = call_openai_grounded(results, top_n=args.grounded_top)
        output["grounded"] = grounding
        render_grounding_report(grounding, args.sector)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"[dim]Output saved to {args.save}[/dim]")

    # Exit non-zero if any data accuracy issues were found
    if "grounded" in output:
        dirty = [
            c["ticker"] for c in output["grounded"].get("data_checks", [])
            if not c.get("data_ok", True)
        ]
        if dirty:
            sys.exit(1)


if __name__ == "__main__":
    main()
