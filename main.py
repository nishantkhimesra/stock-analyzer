#!/usr/bin/env python3
"""
Stock Growth Potential Analyser
================================
Analyses stocks in a given sector for undervaluation and growth potential.
Uses: Piotroski F-Score · Valuation multiples · Momentum · ML-style composite scoring

Usage:
    python main.py                        # interactive prompt
    python main.py --sector chemical      # direct sector
    python main.py --sector mining --top 10
    python main.py --sector tech --tickers AAPL MSFT NVDA
    python main.py --list-sectors
"""

import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.table import Table
from rich import box

# Local imports
sys.path.insert(0, ".")
from config.sectors import SECTOR_TICKERS, SECTOR_DISPLAY
from src.scorer import analyse_ticker, StockScore
from src.display import (
    console, sector_banner, results_table,
    top_picks_panel, failed_tickers_note, summary_stats
)


MAX_WORKERS = 1   # sequential — avoids Yahoo Finance 429 rate limits
DEFAULT_TOP = 10  # number of top picks to deep-dive


def list_sectors():
    """Print available sectors."""
    table = Table(title="Available Sectors", box=box.SIMPLE_HEAVY,
                  header_style="bold cyan", padding=(0, 2))
    table.add_column("Key",          style="bold yellow", width=16)
    table.add_column("Display Name", style="white",       width=32)
    table.add_column("Tickers",      style="dim",         width=8, justify="right")

    for key, name in SECTOR_DISPLAY.items():
        count = len(SECTOR_TICKERS.get(key, []))
        table.add_row(key, name, str(count))

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Run:[/dim] [cyan]python main.py --sector <key>[/cyan]\n"
        "[dim]Or add custom tickers:[/dim] [cyan]python main.py --sector chemical "
        "--tickers CE HUN RPM[/cyan]\n"
    )


def resolve_sector(raw: str) -> tuple[str, str]:
    """
    Match user input to a sector key.
    Supports partial matches, e.g. 'chem' → 'chemical'.
    Returns (key, display_name).
    """
    raw = raw.strip().lower()

    # Exact match
    if raw in SECTOR_TICKERS:
        return raw, SECTOR_DISPLAY.get(raw, raw.title())

    # Partial match
    for key in SECTOR_TICKERS:
        if raw in key or key in raw:
            return key, SECTOR_DISPLAY.get(key, key.title())

    # Match against display name
    for key, name in SECTOR_DISPLAY.items():
        if raw in name.lower():
            return key, name

    return None, None


def fetch_all(tickers: list[str]) -> list[StockScore]:
    """Fetch and score all tickers with a progress bar."""
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[current]}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Fetching market data…",
            total=len(tickers),
            current="",
        )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(analyse_ticker, t): t for t in tickers}

            for future in as_completed(futures):
                ticker = futures[future]
                progress.update(task, advance=1, current=ticker)
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    s = StockScore(ticker=ticker)
                    s.error = str(e)
                    s.data_quality = "failed"
                    results[ticker] = s

    # Return in original ticker order
    return [results[t] for t in tickers if t in results]


def rank_results(results: list[StockScore]) -> list[StockScore]:
    """Sort by composite score descending; failed tickers go to the bottom."""
    valid  = [r for r in results if r.data_quality != "failed"]
    failed = [r for r in results if r.data_quality == "failed"]
    valid.sort(key=lambda r: r.composite_score, reverse=True)
    return valid + failed


def run(sector_input: str, extra_tickers: list[str] = None, top: int = DEFAULT_TOP):
    """Main analysis pipeline."""

    # ── Resolve sector ────────────────────────────────────────
    key, display_name = resolve_sector(sector_input)

    if key is None and not extra_tickers:
        console.print(
            f"\n[bold red]Unknown sector:[/bold red] '{sector_input}'\n"
            "Run [cyan]python main.py --list-sectors[/cyan] to see options.\n"
        )
        sys.exit(1)

    if key is None:
        # User only provided custom tickers with no matching sector
        key = sector_input
        display_name = sector_input.title()
        tickers = []
    else:
        tickers = list(SECTOR_TICKERS.get(key, []))

    # Merge extra tickers (deduped, uppercased)
    if extra_tickers:
        for t in extra_tickers:
            tu = t.upper()
            if tu not in tickers:
                tickers.append(tu)

    if not tickers:
        console.print(
            "\n[bold red]No tickers found.[/bold red] "
            "Provide tickers with [cyan]--tickers[/cyan] or choose a known sector.\n"
        )
        sys.exit(1)

    # ── Banner ────────────────────────────────────────────────
    sector_banner(display_name, len(tickers))

    # ── Fetch + Score ─────────────────────────────────────────
    t0 = time.time()
    raw_results = fetch_all(tickers)
    elapsed = time.time() - t0

    console.print(
        f"\n[dim]Fetched {len(raw_results)} stocks in {elapsed:.1f}s[/dim]"
    )

    # ── Rank ──────────────────────────────────────────────────
    ranked = rank_results(raw_results)

    # ── Summary stats ─────────────────────────────────────────
    summary_stats(ranked, display_name)

    # ── Main results table ────────────────────────────────────
    console.print(results_table(ranked))

    # ── Top picks deep dive ───────────────────────────────────
    top_picks_panel(ranked, n=min(top, 5))

    # ── Failed tickers note ───────────────────────────────────
    failed_tickers_note(ranked)

    console.print(
        "\n[dim italic]Not financial advice. Data via Yahoo Finance. "
        "Always conduct independent research.[/dim italic]\n"
    )


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stock Growth Potential Analyser — sector-based multi-factor screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                             # interactive prompt
  python main.py --sector chemical           # analyse chemical sector
  python main.py --sector mining --top 10    # top 10 mining picks
  python main.py --sector tech --tickers NVDA AMD INTC
  python main.py --list-sectors              # show all available sectors
        """
    )
    parser.add_argument("--sector",  "-s", type=str, help="Sector to analyse (e.g. chemical, mining, tech)")
    parser.add_argument("--tickers", "-t", nargs="+", help="Additional/custom tickers to include")
    parser.add_argument("--top",          type=int, default=DEFAULT_TOP, help=f"Number of top picks (default: {DEFAULT_TOP})")
    parser.add_argument("--list-sectors", action="store_true", help="List all available sectors and exit")
    args = parser.parse_args()

    if args.list_sectors:
        list_sectors()
        return

    sector_input = args.sector
    if not sector_input:
        # Interactive prompt
        console.print("\n[bold cyan]Stock Growth Potential Analyser[/bold cyan]")
        console.print("[dim]Sectors:[/dim] " + ", ".join(SECTOR_DISPLAY.keys()) + "\n")
        sector_input = Prompt.ask(
            "[bold]Enter sector[/bold]",
            default="chemical"
        )

    run(
        sector_input=sector_input,
        extra_tickers=args.tickers,
        top=args.top,
    )


if __name__ == "__main__":
    main()
