"""
eval/evaluate.py — AI-powered validation agent for stock screening results.

Modes:
  opinion  (default) — GPT reviews rating logic as a second opinion
  grounded            — GPT uses live web search to verify price / analyst PT / revenue data
  both                — runs opinion first, then grounded fact-check

Providers (auto-detected from .env):
  OpenAI  — set OPENAI_API_KEY
  Azure   — set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_DEPLOYMENT
             (optional) BING_SEARCH_KEY enables live web search in grounded mode

Usage:
    python eval/evaluate.py --sector tech
    python eval/evaluate.py --sector semiconductor --model gpt-4o
    python eval/evaluate.py --sector fintech --mode grounded
    python eval/evaluate.py --sector tech --mode both --save run.json
"""

import sys
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box
from rich.panel import Panel

from config.sectors import SECTOR_TICKERS, SECTOR_DISPLAY
from config.dynamic_fetch import fetch_sector_tickers
from src.scorer import analyse_ticker, StockScore

load_dotenv()

console = Console()

DEFAULT_MODEL   = "gpt-4o-mini"           # opinion mode — OpenAI model name
GROUNDED_MODEL  = "gpt-4o-mini-search-preview"  # grounded mode — OpenAI only
AZURE_API_VER   = "2025-01-01-preview"    # Azure OpenAI API version
MAX_WORKERS     = 3


def is_azure() -> bool:
    """True when Azure credentials are present in the environment."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))


def build_client():
    """Return an OpenAI or AzureOpenAI client based on available env vars."""
    if is_azure():
        key      = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        version  = os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VER)
        if not key or not endpoint:
            console.print(
                "[red]Error:[/red] Azure mode requires AZURE_OPENAI_API_KEY "
                "and AZURE_OPENAI_ENDPOINT in your .env file."
            )
            sys.exit(1)
        return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=version)

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        console.print(
            "[red]Error:[/red] No API key found. Set OPENAI_API_KEY (OpenAI) "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (Azure) in your .env file."
        )
        sys.exit(1)
    return OpenAI(api_key=key)


def opinion_model(cli_model: str) -> str:
    """Resolve model/deployment name for opinion mode."""
    if cli_model != DEFAULT_MODEL:
        return cli_model  # explicit --model flag always wins
    if is_azure():
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    return DEFAULT_MODEL

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

**Rating Decision Tree — evaluate IN ORDER, stop at the FIRST match:**

  Let: cs  = composite (integer from JSON)
       p   = piotroski (integer from JSON)
       up  = analyst_upside stripped of %, e.g. "43.9%" → 43.9
       fcf = fcf_yield as a number (0 if absent)

  [1] IF cs == 0 AND p == 0                                        → "Avoid"
  [2] IF (cs >= 56 AND up >= 30.0 AND p >= 5)                      → "Strong Buy"  # standard
       OR (cs >= 65 AND p >= 7)                                     → "Strong Buy"  # quality
       OR (fcf > 8.0 AND p >= 5 AND cs >= 52)                      → "Strong Buy"  # FCF bypass
  [3] IF cs < 43 OR p <= 4                                         → "Hold"        # absolute floor
  [4] IF cs >= 35 AND up >= 5.0                                     → "Buy"
  [5] IF cs >= 20                                                   → "Hold"
  [6] (default)                                                     → "Avoid"

**Operator semantics — ALL comparisons are INCLUSIVE on the boundary:**
  ">= 5"  means 5 OR HIGHER  — piotroski of exactly 5 PASSES the >= 5 test
  "<= 4"  means 4 OR LOWER   — piotroski of exactly 4 TRIGGERS the Hold floor
  "< 43"  means 42 or lower  — composite of exactly 43 does NOT trigger Hold floor
  ">= 30" means 30.0 or higher — upside of exactly 30.0% PASSES the Strong Buy gate

**Known data limitations:**
- Forward P/E from Yahoo is GAAP-based; non-GAAP guided EPS (common in software)
  can make fwd P/E appear 10–20% lower than company guidance.
- Piotroski scores ≤ 5 on stocks with rev growth ≥ 50% and positive operating
  margin are flagged as "hyper_growth" — likely measurement artefacts, not distress.
"""

VALIDATION_SYSTEM_MSG = (
    "You are a quantitative data validator. "
    "You reason EXCLUSIVELY from the numeric values present in the JSON payload "
    "provided by the user. "
    "You NEVER use your training-time knowledge of any stock's composite score, "
    "Piotroski score, analyst upside, or analyst price target — those numbers "
    "change daily and your internal memory of them is unreliable. "
    "If the JSON says composite=58 for MSFT, the composite is 58. Full stop. "
    "Before judging any stock, transcribe composite, piotroski, and "
    "analyst_upside verbatim from the JSON. Writing a value not present "
    "in the JSON for that ticker is a critical error."
)

VALIDATION_PROMPT = """
Below are the screening results for the **{sector}** sector.

{scoring_context}

## ⚠ CRITICAL — Ground every judgment in the JSON below

The JSON contains the ONLY authoritative values for composite, piotroski, and
analyst_upside for this run. Do NOT use your memory of what these numbers
"should be" for well-known stocks. The numbers in the JSON are current;
your training data is not.

For every stock, the `notes` field MUST begin with:
  `composite=<value>, piotroski=<value>, upside=<value> —`
(copy these three values verbatim from the JSON before writing any judgment).
If you write a number that does not appear in the JSON for that ticker, that is
a validation error.

Below are the screening results.

---

## Screening Results

```json
{results_json}
```

---

## Validation Rules — follow these IN ORDER for every stock

**Step 0 — Data quality gate (check this FIRST, before any numeric reasoning)**
- Each stock has a `data_quality` field: `"full"`, `"partial"`, or `"failed"`.
- `"partial"` means the price was available but several key metrics were missing
  (e.g. no P/E, no revenue growth). Composite scores built on partial data are
  unreliable — missing valuation inputs default to 0, deflating the score.
- `"failed"` means no price was available at all. All scores are zero artefacts.
- For any stock where `data_quality != "full"`, you MUST output:
  - `"rating_suggested": "Insufficient data"`
  - `"agreement": "partial"`
  - `"confidence": "low"`
  - `"notes": "data_quality=<value> — scores not reliable; rating withheld"`
  Do NOT attempt to validate or disagree with a rating derived from partial/failed data.

**Step 1 — Upside check (read the number, do not infer direction from narrative)**
- `analyst_upside` is the % gap between current price and mean analyst price target.
- If `analyst_upside` is negative or null, the stock trades AT or ABOVE its target.
  A negative-upside stock CANNOT be a top conviction pick or a Strong Buy.
  Treat negative upside as a bearish signal regardless of revenue growth or story.

**Step 2 — Walk the decision tree (do NOT check gates independently)**

For each stock, read cs, p, up, and fcf from the JSON, then evaluate the
decision tree above in order [1]→[2]→[3]→[4]→[5]→[6], stopping at the first
branch that fires. The outcome of that walk is the correct rating.

Common mistakes to avoid:
- Do NOT jump from "fails Strong Buy" directly to Avoid — Buy and Hold fire first.
- Do NOT apply the p >= 7 rule except on the cs >= 65 quality-bypass path.
- Do NOT trigger Hold floor [3] when cs = 43 exactly (< 43 means 42 or lower).
- piotroski of exactly 5 PASSES >= 5. Do not write "5 is below the required 5".
- A stock with cs >= 35 AND up >= 5% gets Buy at step [4] regardless of narrative.

If the walk outcome matches rating_given → agree.
If it differs → disagree, cite which step number produced a different result and
show the exact values: e.g. "step [3] fires: cs=38 < 43 → Hold, not Buy".

**Step 3 — Distress flags (independent of rating agreement)**
- piotroski ≤ 2 on any Buy or Strong Buy → add concern, keep agreement if gates pass.
- piotroski_context = "hyper_growth" overrides low-p flags for fast-growth stocks.

**Step 4 — Internal consistency**
- Check that `revenue_growth`, `fwd_pe`, `rsi_14`, and `analyst_upside` are mutually
  consistent with the rating. A stock with fwd_pe > 60 AND negative upside AND
  piotroski ≤ 3 should not be Buy or Strong Buy regardless of revenue growth.
- Do NOT infer rating from company name, sector narrative, or revenue growth alone.

**Step 5 — Flag stale or suspicious data**
- Flag any `fwd_pe` that looks implausibly low for a loss-making company (post-bankruptcy
  accounting can produce artificially positive forward EPS estimates).
- Flag any `analyst_upside` > 150% on a stock with piotroski ≤ 3 as possibly stale data.

---

## Output

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
      "notes": "one-sentence rationale citing specific numbers (upside %, piotroski, composite)"
    }}
  ],
  "overall_assessment": {{
    "sector_summary": "2-3 sentence overview citing avg composite and upside distribution",
    "systemic_issues": ["any patterns or calibration concerns across the whole run"],
    "top_conviction_picks": ["up to 3 tickers: positive upside + piotroski ≥ 5 + composite ≥ 56"],
    "most_contested_ratings": ["tickers where numeric gates contradict the given rating"],
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


def run_analysis(sector_key: str, live: bool = False) -> list[StockScore]:
    if live:
        tickers, source = fetch_sector_tickers(sector_key)
    else:
        tickers, source = SECTOR_TICKERS[sector_key], "curated"
    src_tag = "[cyan](live)[/cyan]" if source == "live" else "[dim](curated)[/dim]"
    console.print(
        f"\n[dim]Scanning {len(tickers)} tickers in "
        f"[bold]{SECTOR_DISPLAY[sector_key]}[/bold] {src_tag}…[/dim]"
    )

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
    provider = "Azure" if is_azure() else "OpenAI"
    client   = build_client()

    valid   = [r for r in results if r.data_quality != "failed"]
    payload = [serialise_result(r) for r in valid]

    prompt = VALIDATION_PROMPT.format(
        scoring_context=SCORING_CONTEXT,
        sector=SECTOR_DISPLAY[sector_key],
        results_json=json.dumps(payload, indent=2),
    )

    console.print(f"\n[dim]Calling {provider} ({model}) for validation…[/dim]")

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VALIDATION_SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def call_openai_grounded(results: list[StockScore], top_n: int = 8) -> dict:
    """Fact-check top_n stocks using web search.

    OpenAI: uses gpt-4o-mini-search-preview (web search built-in).
    Azure:  uses the configured deployment + Bing Search grounding
            if BING_SEARCH_KEY is set, otherwise falls back to model
            knowledge only (no live search) with a warning.
    """
    client = build_client()

    valid = [r for r in results if r.data_quality != "failed"][:top_n]
    payload = [
        {
            "ticker":       r.ticker,
            "company":      r.company_name,
            "price_stated": round(r.current_price, 2) if r.current_price else None,
            "analyst_pt":   round(r.analyst_target, 2) if r.analyst_target else None,
            "rev_growth":   f"{r.revenue_growth_yoy*100:.1f}%"
                            if r.revenue_growth_yoy else None,
        }
        for r in valid
    ]

    prompt = GROUNDING_PROMPT.format(stocks_json=json.dumps(payload, indent=2))

    if is_azure():
        model      = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        bing_key   = os.environ.get("BING_SEARCH_KEY")
        bing_ep    = os.environ.get(
            "BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/"
        )
        extra: dict = {}
        if bing_key:
            console.print(
                f"\n[dim]Calling Azure ({model} + Bing grounding) "
                f"to fact-check top {len(valid)} stocks…[/dim]"
            )
            extra = {
                "extra_body": {
                    "data_sources": [{
                        "type": "bing_search",
                        "parameters": {
                            "endpoint": bing_ep,
                            "key": bing_key,
                            "search_top_n": 5,
                        },
                    }]
                }
            }
        else:
            console.print(
                f"\n[yellow]⚠ BING_SEARCH_KEY not set — Azure grounded mode will use "
                f"model knowledge only (no live web search).[/yellow]\n"
                f"[dim]Calling Azure ({model}) to fact-check top {len(valid)} stocks…[/dim]"
            )
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
    else:
        # OpenAI: gpt-4o-mini-search-preview has web search built into the model.
        console.print(
            f"\n[dim]Calling OpenAI web search ({GROUNDED_MODEL}) "
            f"to fact-check top {len(valid)} stocks…[/dim]"
        )
        try:
            response = client.chat.completions.create(
                model=GROUNDED_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                console.print(
                    f"[red]Error:[/red] Model '{GROUNDED_MODEL}' not found. "
                    "Verify at: https://platform.openai.com/account/rate-limits"
                )
                sys.exit(1)
            raise

    raw = response.choices[0].message.content.strip()
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
        validation = call_openai(results, args.sector, opinion_model(args.model))
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
