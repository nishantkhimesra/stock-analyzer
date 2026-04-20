# src/display.py — Rich terminal UI for stock analysis output

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.align import Align
from rich.layout import Layout
import time

console = Console()


def sector_banner(sector_name: str, ticker_count: int):
    """Display a header banner for the sector analysis."""
    console.print()
    console.print(Rule(f"[bold white] Stock Growth Potential Analyser [/bold white]", style="blue"))
    console.print(
        Panel(
            f"[bold cyan]{sector_name.upper()}[/bold cyan]\n"
            f"[dim]Screening {ticker_count} stocks · Piotroski F-Score · Valuation · Growth · Momentum[/dim]",
            style="blue",
            padding=(0, 2),
        )
    )
    console.print()


def score_color(score: float) -> str:
    if score >= 70: return "bright_green"
    if score >= 55: return "green"
    if score >= 40: return "yellow"
    if score >= 25: return "orange1"
    return "red"


def score_bar(score: float, width: int = 10) -> str:
    filled = int((score / 100) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{score_color(score)}]{bar}[/]"


def piotroski_badge(score: int, detail: dict | None = None) -> str:
    note = " [dim](growth artefact)[/dim]" if (detail or {}).get("_context") == "hyper_growth" else ""
    if score >= 7: return f"[bold bright_green]{score}/9 ▲[/bold bright_green]{note}"
    if score >= 5: return f"[yellow]{score}/9 ◆[/yellow]{note}"
    if score >= 3: return f"[orange1]{score}/9 ▼[/orange1]{note}"
    return f"[red]{score}/9 ✗[/red]{note}"


def rating_badge(rating: str) -> str:
    r = rating.lower()
    if "contrarian" in r:   return "[bold bright_green]⬆ Strong Buy[/bold bright_green] [dim yellow](Contrarian)[/dim yellow]"
    if "strong buy" in r:   return "[bold bright_green]⬆ Strong Buy[/bold bright_green]"
    if "buy" in r:          return "[green]↑ Buy[/green]"
    if "hold" in r:         return "[yellow]→ Hold[/yellow]"
    if "underperform" in r: return "[orange1]↓ Underperform[/orange1]"
    if "sell" in r:         return "[red]⬇ Sell[/red]"
    if "avoid" in r:        return "[red]✗ Avoid[/red]"
    return f"[dim]{rating or 'N/A'}[/dim]"


def fmt(val, fmt_str=".1f", suffix="", prefix="", none_str="N/A"):
    if val is None: return f"[dim]{none_str}[/dim]"
    try:
        return f"{prefix}{val:{fmt_str}}{suffix}"
    except Exception:
        return f"[dim]{none_str}[/dim]"


def fmt_pct(val, none_str="N/A"):
    if val is None: return f"[dim]{none_str}[/dim]"
    color = "bright_green" if val >= 10 else "green" if val >= 0 else "red"
    sign  = "+" if val >= 0 else ""
    return f"[{color}]{sign}{val:.1f}%[/{color}]"


def results_table(results: list, top_n: int = None) -> Table:
    """Build the main results table."""
    title = f"Ranked Results — Top {top_n}" if top_n else "Full Results"
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold white on grey23",
        border_style="blue",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("Rank",        style="dim",         width=4,  justify="right")
    table.add_column("Ticker",      style="bold cyan",   width=6)
    table.add_column("Company",     style="white",       width=22, no_wrap=True)
    table.add_column("Price",       style="white",       width=8,  justify="right")
    table.add_column("Composite",   width=14,            justify="center")
    table.add_column("Valuation",   width=12,            justify="center")
    table.add_column("Growth",      width=12,            justify="center")
    table.add_column("Momentum",    width=12,            justify="center")
    table.add_column("Piotroski",   width=10,            justify="center")
    table.add_column("Fwd P/E",     style="dim white",   width=8,  justify="right")
    table.add_column("Rev Growth",  width=9,             justify="right")
    table.add_column("RSI",         width=6,             justify="right")
    table.add_column("Analyst PT",  width=10,            justify="right")
    table.add_column("Upside",      width=8,             justify="right")
    table.add_column("Rating",      width=16)

    display = results[:top_n] if top_n else results

    for i, r in enumerate(display, 1):
        if r.data_quality == "failed":
            table.add_row(
                str(i), r.ticker, "[dim]Data unavailable[/dim]",
                *["[dim]—[/dim]"] * 13
            )
            continue

        name = r.company_name[:20] + "…" if len(r.company_name) > 21 else r.company_name
        comp_bar = score_bar(r.composite_score, 8)
        comp_num = f"[{score_color(r.composite_score)}]{r.composite_score:.0f}[/]"

        rsi_str = "N/A"
        if r.rsi_14:
            rsi_color = "bright_green" if 40 <= r.rsi_14 <= 60 \
                        else "green" if 30 <= r.rsi_14 < 40 \
                        else "yellow" if r.rsi_14 < 30 \
                        else "orange1"
            rsi_str = f"[{rsi_color}]{r.rsi_14:.0f}[/{rsi_color}]"

        table.add_row(
            str(i),
            r.ticker,
            name,
            f"${r.current_price:.2f}" if r.current_price else "N/A",
            f"{comp_bar} {comp_num}",
            f"{score_bar(r.valuation_score, 6)} [{score_color(r.valuation_score)}]{r.valuation_score:.0f}[/]",
            f"{score_bar(r.growth_score, 6)} [{score_color(r.growth_score)}]{r.growth_score:.0f}[/]",
            f"{score_bar(r.momentum_score, 6)} [{score_color(r.momentum_score)}]{r.momentum_score:.0f}[/]",
            piotroski_badge(r.piotroski_score, r.piotroski_detail),
            fmt(r.forward_pe, ".1f", "x") if r.forward_pe else fmt(r.pe_ratio, ".1f", "x"),
            fmt_pct(r.revenue_growth_yoy * 100 if r.revenue_growth_yoy else None),
            rsi_str,
            f"${r.analyst_target:.2f}" if r.analyst_target else "[dim]N/A[/dim]",
            fmt_pct(r.upside_to_target),
            rating_badge(r.analyst_rating),
        )

    return table


def top_picks_panel(results: list, n: int = 5):
    """Display detailed cards for the top N picks."""
    console.print()
    console.print(Rule(f"[bold white] Top {n} Picks — Deep Dive [/bold white]", style="green"))

    top = [r for r in results[:n] if r.data_quality != "failed"]

    cards = []
    for r in top:
        _rl = r.analyst_rating.lower()
        tier = (
            "[bold bright_green]★ STRONG BUY[/bold bright_green] [dim yellow](Contrarian)[/dim yellow]" if "contrarian" in _rl else
            "[bold bright_green]★ STRONG BUY[/bold bright_green]" if "strong buy" in _rl else
            "[bold green]▲ BUY[/bold green]"                       if "buy" in _rl else
            "[yellow]→ HOLD[/yellow]"                              if "hold" in _rl else
            "[red]✗ AVOID[/red]"
        )

        # Piotroski breakdown
        p_bullets = []
        for k, v in r.piotroski_detail.items():
            if k == "error": continue
            label = k.replace("F", "").replace("_", " ").title()
            icon  = "[bright_green]✓[/bright_green]" if v == 1 else "[dim red]✗[/dim red]"
            p_bullets.append(f"  {icon} {label}")
        p_text = "\n".join(p_bullets[:5]) or "  [dim]No detail[/dim]"

        # Key metrics block
        pe_str  = f"{r.forward_pe:.1f}x fwd" if r.forward_pe else (f"{r.pe_ratio:.1f}x trail" if r.pe_ratio else "N/A")
        gm_str  = f"{r.gross_margin*100:.1f}%" if r.gross_margin else "N/A"
        roe_str = f"{r.roe*100:.1f}%" if r.roe else "N/A"
        ma_str  = (
            "[bright_green]Above both MAs[/bright_green]" if (r.above_50dma and r.above_200dma) else
            "[yellow]Mixed MA signals[/yellow]"           if (r.above_50dma or r.above_200dma) else
            "[red]Below both MAs[/red]"
        )

        content = (
            f"{tier}\n"
            f"[bold white]{r.company_name}[/bold white]\n"
            f"[dim]{r.industry}[/dim]\n\n"
            f"[bold]Composite Score:[/bold] [{score_color(r.composite_score)}]{r.composite_score:.0f}/100[/]\n"
            f"[bold]Piotroski:[/bold]       {piotroski_badge(r.piotroski_score, r.piotroski_detail)}\n\n"
            f"[bold]Price:[/bold]    ${r.current_price:.2f}\n"
            f"[bold]P/E:[/bold]      {pe_str}\n"
            f"[bold]Gross Margin:[/bold] {gm_str}\n"
            f"[bold]ROE:[/bold]      {roe_str}\n"
            f"[bold]Rev Growth:[/bold] {fmt_pct(r.revenue_growth_yoy*100 if r.revenue_growth_yoy else None)}\n"
            f"[bold]RSI(14):[/bold]  {r.rsi_14:.0f}" + ("\n" if r.rsi_14 else "N/A\n") +
            f"[bold]MA Status:[/bold] {ma_str}\n\n"
            f"[bold]Analyst PT:[/bold] " +
            (f"[bold bright_green]${r.analyst_target:.2f}[/bold bright_green] ({fmt_pct(r.upside_to_target)})\n"
             if r.analyst_target else "[dim]N/A[/dim]\n") +
            f"[bold]Rating:[/bold]    {rating_badge(r.analyst_rating)}\n" +
            (
                "[dim yellow]⚠ High Street upside held back by weak\n"
                f"  Piotroski ({r.piotroski_score}/9) or low composite\n"
                f"  ({r.composite_score:.0f}/100) — upside alone doesn't\n"
                "  justify a Buy.[/dim yellow]\n\n"
                if r.analyst_rating == "Hold"
                and (r.upside_to_target or 0) >= 60
                else "\n"
            ) +
            f"[dim]— Piotroski detail (top 5) —[/dim]\n{p_text}"
        )

        cards.append(Panel(
            content,
            title=f"[bold cyan]#{results.index(r)+1}  {r.ticker}[/bold cyan]",
            border_style="green" if r.composite_score >= 65 else "blue",
            width=40,
            padding=(0, 1),
        ))

    console.print(Columns(cards, equal=True))


def failed_tickers_note(results: list):
    """Note any tickers that failed to fetch."""
    failed = [r.ticker for r in results if r.data_quality == "failed"]
    if failed:
        console.print(
            f"\n[dim]⚠  Could not fetch data for: {', '.join(failed)}[/dim]"
        )


def summary_stats(results: list, sector_name: str):
    """Print summary statistics panel."""
    valid = [r for r in results if r.data_quality != "failed"]
    if not valid: return

    def _rating(r): return r.analyst_rating.lower()
    strong_buys = sum(1 for r in valid if "strong buy" in _rating(r) or "contrarian" in _rating(r))
    buys        = sum(1 for r in valid if "buy" in _rating(r) and "strong" not in _rating(r))
    holds       = sum(1 for r in valid if "hold" in _rating(r))
    avoids      = sum(1 for r in valid if "sell" in _rating(r) or "underperform" in _rating(r) or "avoid" in _rating(r))
    avg_score   = sum(r.composite_score for r in valid) / len(valid)

    avg_upside  = [r.upside_to_target for r in valid if r.upside_to_target is not None]
    avg_up_str  = f"{sum(avg_upside)/len(avg_upside):.1f}%" if avg_upside else "N/A"

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
    table.add_column(style="dim")
    table.add_column(style="bold white")

    table.add_row("Sector",         sector_name)
    table.add_row("Stocks analysed", str(len(valid)))
    table.add_row("Avg composite score", f"{avg_score:.1f}/100")
    table.add_row("Avg analyst upside",  avg_up_str)
    table.add_row("", "")
    table.add_row("[bright_green]★ Strong Buy[/bright_green]", str(strong_buys))
    table.add_row("[green]▲ Buy[/green]",       str(buys))
    table.add_row("[yellow]→ Hold[/yellow]",    str(holds))
    table.add_row("[red]✗ Avoid[/red]",         str(avoids))

    console.print()
    console.print(Panel(table, title="[bold]Analysis Summary[/bold]", border_style="dim", padding=(0, 1)))
    console.print()
