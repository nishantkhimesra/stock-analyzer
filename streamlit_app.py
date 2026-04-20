import sys
import os
import datetime
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.sectors import SECTOR_TICKERS, SECTOR_DISPLAY
from src.scorer import analyse_ticker

try:
    from eval.evaluate import call_openai, is_azure, opinion_model, DEFAULT_MODEL
    EVAL_AVAILABLE = True
except Exception:
    EVAL_AVAILABLE = False

# ── Sync Streamlit Cloud secrets → os.environ ─────────────────────────────────
# Streamlit Cloud injects secrets via st.secrets, not os.environ.
# This one-time sync makes all existing os.environ.get() calls work unchanged.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # local dev: no st.secrets, .env handles it via load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Growth Potential Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
.stExpander > details > summary { font-size: 1.05rem; font-weight: 600; }
.block-container { padding-top: 1.5rem; }
div[data-testid="stSidebarContent"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Growth Potential Analyser")
st.caption(
    "Multi-factor scoring: Piotroski F-Score · Valuation · Growth · Momentum  "
    "— *Not financial advice. Data via Yahoo Finance.*"
)
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────
def rating_label(rating: str) -> str:
    rl = rating.lower()
    if "contrarian" in rl:
        return "⬆ Strong Buy ⚡"
    if "strong buy" in rl:
        return "⬆ Strong Buy"
    if "buy" in rl:
        return "↑ Buy"
    if "hold" in rl:
        return "→ Hold"
    if "avoid" in rl or "sell" in rl:
        return "✗ Avoid"
    return rating or "N/A"


def pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def dollar(val):
    if val is None:
        return "N/A"
    return f"${val:.2f}"


def ma_status(r):
    if r.above_50dma and r.above_200dma:
        return "✅ Above both MAs"
    if r.above_50dma or r.above_200dma:
        return "⚠️ Mixed signals"
    return "❌ Below both MAs"


def generate_report(sector_key: str, results: list, validation: dict | None) -> str:
    """Build a complete markdown report for download."""
    display    = SECTOR_DISPLAY[sector_key]
    now        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    valid_res  = [r for r in results if r.data_quality != "failed"]
    failed_res = [r for r in results if r.data_quality == "failed"]

    lines = [
        f"# 📈 Stock Analysis Report — {display}",
        f"Generated: {now}  |  Not financial advice. Data via Yahoo Finance.",
        "", "---", "", "## Summary",
    ]

    avg_cs = sum(r.composite_score for r in valid_res) / len(valid_res) if valid_res else 0
    ups    = [r.upside_to_target for r in valid_res if r.upside_to_target]
    avg_up = sum(ups) / len(ups) if ups else 0
    def _rl(r): return r.analyst_rating.lower()
    sb = sum(1 for r in valid_res if "strong buy" in _rl(r) or "contrarian" in _rl(r))
    b  = sum(1 for r in valid_res if "buy" in _rl(r)
             and "strong" not in _rl(r) and "contrarian" not in _rl(r))
    h  = sum(1 for r in valid_res if "hold" in _rl(r))
    av = sum(1 for r in valid_res if "avoid" in _rl(r) or "sell" in _rl(r))

    lines += [
        f"- **Stocks analysed:** {len(valid_res)}",
        f"- **Avg composite score:** {avg_cs:.1f}/100",
        f"- **Avg analyst upside:** {avg_up:.1f}%",
        f"- **Rating distribution:** {sb} Strong Buy · {b} Buy · {h} Hold · {av} Avoid",
        "", "---", "", "## Full Results", "",
        "| Rank | Ticker | Company | Price | Composite | Val | Growth"
        " | Mom | Piotroski | Fwd P/E | Rev Growth | Upside % | Rating |",
        "|------|--------|---------|-------|-----------|-----|-------"
        "|-----|-----------|---------|-----------|----------|--------|",
    ]

    for i, r in enumerate(valid_res, 1):
        price  = f"${r.current_price:.2f}"           if r.current_price       else "N/A"
        fpe    = f"{r.forward_pe:.1f}x"              if r.forward_pe          else "N/A"
        revg   = f"{r.revenue_growth_yoy*100:.1f}%"  if r.revenue_growth_yoy  else "N/A"
        upside = f"{r.upside_to_target:.1f}%"        if r.upside_to_target    else "N/A"
        lines.append(
            f"| {i} | {r.ticker} | {r.company_name[:28]} | {price}"
            f" | {r.composite_score:.0f} | {r.valuation_score:.0f}"
            f" | {r.growth_score:.0f} | {r.momentum_score:.0f}"
            f" | {r.piotroski_score}/9 | {fpe} | {revg} | {upside}"
            f" | {r.analyst_rating} |"
        )

    if failed_res:
        lines += ["",
                  f"*Data fetch failed for: {', '.join(r.ticker for r in failed_res)}*"]

    if validation:
        stock_vals = {v["ticker"]: v for v in validation.get("stock_validations", [])}
        ov         = validation.get("overall_assessment", {})
        lines += [
            "", "---", "", "## 🤖 AI Validation", "",
            "| Ticker | Algo Rating | AI Says | Agreement | Confidence | Notes |",
            "|--------|-------------|---------|-----------|------------|-------|",
        ]
        for r in valid_res:
            v        = stock_vals.get(r.ticker, {})
            agree    = v.get("agreement", "—")
            conf     = v.get("confidence", "—")
            notes    = v.get("notes", "")
            concerns = v.get("concerns", [])
            if concerns:
                notes += " ⚠ " + " · ".join(concerns)
            lines.append(
                f"| {r.ticker} | {r.analyst_rating}"
                f" | {v.get('rating_suggested', '—')}"
                f" | {agree} | {conf} | {notes} |"
            )
        lines += ["", "### Overall Assessment", ""]
        if ov.get("sector_summary"):
            lines += [ov["sector_summary"], ""]
        if ov.get("top_conviction_picks"):
            lines.append("**Top Conviction Picks:** "
                         + ", ".join(ov["top_conviction_picks"]))
        if ov.get("most_contested_ratings"):
            lines.append("**Most Contested:** "
                         + ", ".join(ov["most_contested_ratings"]))
        if ov.get("systemic_issues"):
            lines += ["", "**Systemic Issues**"]
            lines += [f"- {s}" for s in ov["systemic_issues"]]
        if ov.get("data_quality_flags"):
            lines += ["", "**Data Quality Flags:** "
                      + ", ".join(ov["data_quality_flags"])]
    else:
        lines += ["", "---", "",
                  "*AI Validation not run. Click ▶ Run Validation in the app.*"]

    lines += ["", "---", "*Generated by Stock Growth Potential Analyser*"]
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    sector_key = st.selectbox(
        "Select Sector",
        options=list(SECTOR_DISPLAY.keys()),
        format_func=lambda k: SECTOR_DISPLAY[k],
        index=list(SECTOR_DISPLAY.keys()).index("tech"),
    )
    display_name = SECTOR_DISPLAY[sector_key]
    tickers = SECTOR_TICKERS[sector_key]
    st.caption(f"🔎 {len(tickers)} stocks in this sector")

    top_n = st.slider("Deep-dive picks to show", min_value=3, max_value=10, value=5)

    st.divider()
    run_btn = st.button("🔍 Analyse Sector", type="primary", use_container_width=True)

    st.divider()
    st.info(
        f"⏱ Approx. **{len(tickers) * 1.2:.0f}–{len(tickers) * 2:.0f}s** "
        "per sector scan (1 request/s rate limit)."
    )

    st.markdown("---")
    st.markdown(
        "**Scoring weights**\n"
        "- Valuation 35 %\n"
        "- Growth 40 %\n"
        "- Momentum 25 %"
    )

    if "results" in st.session_state and st.session_state.get("scan_sector") == sector_key:
        st.divider()
        _rpt = generate_report(
            sector_key,
            st.session_state["results"],
            st.session_state.get("validation")
            if st.session_state.get("validation_sector") == sector_key
            else None,
        )
        _fname = f"{sector_key}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
        st.download_button(
            "⬇️ Download Report",
            data=_rpt,
            file_name=_fname,
            mime="text/markdown",
            use_container_width=True,
        )


# ── Landing state ─────────────────────────────────────────────────────────────
if not run_btn and "results" not in st.session_state:
    st.info("👈 Select a sector from the sidebar and click **Analyse Sector** to begin.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 📊 Piotroski F-Score")
        st.markdown("9-point financial health check — profitability, leverage, efficiency.")
    with c2:
        st.markdown("### 💰 Valuation")
        st.markdown("P/E, P/B, P/S, EV/EBITDA, PEG, FCF yield — sector-relative scoring.")
    with c3:
        st.markdown("### 🚀 Growth")
        st.markdown("Revenue & earnings growth, gross margin, ROE, operating margin.")
    with c4:
        st.markdown("### 📉 Momentum")
        st.markdown("6-month price momentum, RSI(14), 50-day & 200-day moving averages.")
    st.stop()


# ── Run analysis (fresh scan or use cache) ────────────────────────────────────
if run_btn or st.session_state.get("scan_sector") != sector_key:
    st.subheader(f"🔄 Scanning {display_name}")
    progress_bar = st.progress(0, text="Starting…")
    status_placeholder = st.empty()

    _res = []
    for i, ticker in enumerate(tickers):
        status_placeholder.caption(f"Fetching **{ticker}** ({i + 1}/{len(tickers)})…")
        progress_bar.progress((i + 1) / len(tickers), text=f"{ticker} ({i + 1}/{len(tickers)})")
        _res.append(analyse_ticker(ticker))

    progress_bar.empty()
    status_placeholder.empty()

    _res.sort(key=lambda r: (r.data_quality == "failed", -r.composite_score))
    st.session_state["results"]     = _res
    st.session_state["scan_sector"] = sector_key
    st.session_state.pop("validation", None)  # clear stale validation on new scan

results = st.session_state["results"]
valid   = [r for r in results if r.data_quality != "failed"]
failed  = [r for r in results if r.data_quality == "failed"]

if not valid:
    st.error("No data could be retrieved. Check your internet connection or try again later.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader(f"📊 {display_name} — Summary")

avg_composite = sum(r.composite_score for r in valid) / len(valid)
upsides = [r.upside_to_target for r in valid if r.upside_to_target is not None]
avg_upside = sum(upsides) / len(upsides) if upsides else 0

def _rl(r): return r.analyst_rating.lower()
strong_buys = sum(1 for r in valid if "strong buy" in _rl(r) or "contrarian" in _rl(r))
buys        = sum(1 for r in valid if "buy" in _rl(r) and "strong" not in _rl(r) and "contrarian" not in _rl(r))
holds       = sum(1 for r in valid if "hold" in _rl(r))
avoids      = sum(1 for r in valid if "avoid" in _rl(r) or "sell" in _rl(r))

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Stocks analysed", len(valid))
m2.metric("Avg composite", f"{avg_composite:.1f}/100")
m3.metric("Avg analyst upside", f"{avg_upside:.1f}%")
m4.metric("⬆ Strong Buy", strong_buys)
m5.metric("↑ Buy", buys)
m6.metric("→ Hold / ✗ Avoid", f"{holds} / {avoids}")

st.divider()

# ── Full results table ────────────────────────────────────────────────────────
st.subheader("📋 Full Results")

rows = []
for i, r in enumerate(valid, 1):
    rows.append({
        "Rank":        i,
        "Ticker":      r.ticker,
        "Company":     r.company_name[:28],
        "Price":       r.current_price,
        "Composite":   r.composite_score,
        "Valuation":   r.valuation_score,
        "Growth":      r.growth_score,
        "Momentum":    r.momentum_score,
        "Piotroski":   r.piotroski_score,
        "Fwd P/E":     r.forward_pe,
        "Rev Growth %": round(r.revenue_growth_yoy * 100, 1) if r.revenue_growth_yoy else None,
        "RSI":         round(r.rsi_14, 0) if r.rsi_14 else None,
        "Analyst PT":  r.analyst_target,
        "Upside %":    round(r.upside_to_target, 1) if r.upside_to_target else None,
        "Rating":      rating_label(r.analyst_rating),
    })

df = pd.DataFrame(rows)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price":      st.column_config.NumberColumn("Price",      format="$%.2f"),
        "Composite":  st.column_config.ProgressColumn("Composite",  min_value=0, max_value=100, format="%.0f"),
        "Valuation":  st.column_config.ProgressColumn("Valuation",  min_value=0, max_value=100, format="%.0f"),
        "Growth":     st.column_config.ProgressColumn("Growth",     min_value=0, max_value=100, format="%.0f"),
        "Momentum":   st.column_config.ProgressColumn("Momentum",   min_value=0, max_value=100, format="%.0f"),
        "Piotroski":  st.column_config.NumberColumn("Piotroski (0-9)"),
        "Fwd P/E":    st.column_config.NumberColumn("Fwd P/E",    format="%.1fx"),
        "Rev Growth %": st.column_config.NumberColumn("Rev Growth %", format="%.1f%%"),
        "Analyst PT": st.column_config.NumberColumn("Analyst PT", format="$%.2f"),
        "Upside %":   st.column_config.NumberColumn("Upside %",   format="%.1f%%"),
    },
)

st.divider()

# ── Top picks — deep dive ─────────────────────────────────────────────────────
picks_n = min(top_n, len(valid))
st.subheader(f"🏆 Top {picks_n} Picks — Deep Dive")

for r in valid[:picks_n]:
    rank = valid.index(r) + 1
    rl   = r.analyst_rating.lower()

    if "contrarian" in rl:
        badge = "⬆ STRONG BUY ⚡ (Contrarian)"
        border = "🟣"
    elif "strong buy" in rl:
        badge = "⬆ STRONG BUY"
        border = "🟢"
    elif "buy" in rl:
        badge = "↑ BUY"
        border = "🔵"
    elif "hold" in rl:
        badge = "→ HOLD"
        border = "🟡"
    else:
        badge = "✗ AVOID"
        border = "🔴"

    expanded = rank <= 3
    with st.expander(
        f"{border} #{rank}  **{r.ticker}** — {r.company_name}  |  {badge}  |  Composite: **{r.composite_score:.0f}/100**",
        expanded=expanded,
    ):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**💲 Valuation**")
            st.metric("Price",         dollar(r.current_price))
            st.metric("Fwd P/E",       f"{r.forward_pe:.1f}x"  if r.forward_pe  else "N/A")
            st.metric("Gross Margin",  pct(r.gross_margin))
            st.metric("ROE",           pct(r.roe))

        with c2:
            st.markdown("**📈 Growth & Technical**")
            st.metric("Rev Growth",    pct(r.revenue_growth_yoy))
            st.metric("RSI (14)",      f"{r.rsi_14:.0f}" if r.rsi_14 else "N/A")
            st.metric("MA Status",     ma_status(r))
            is_hg = (r.piotroski_detail or {}).get("_context") == "hyper_growth"
            p_label = f"{r.piotroski_score}/9" + (" ⚡" if is_hg else "")
            p_help  = "Low score may reflect hyper-growth artefacts (ROA lag, deferred revenue accruals) — not financial distress." if is_hg else None
            st.metric("Piotroski", p_label, help=p_help)

        with c3:
            st.markdown("**🎯 Analyst View**")
            st.metric("Analyst PT",    dollar(r.analyst_target))
            st.metric("Upside",        f"{r.upside_to_target:.1f}%" if r.upside_to_target else "N/A")
            st.metric("Street consensus", r.yahoo_consensus or "N/A")
            if "contrarian" in rl:
                st.warning("⚡ Algo says Strong Buy — Street says Hold/Neutral.")
            if r.analyst_rating == "Hold" and (r.upside_to_target or 0) >= 60:
                st.warning(
                    f"⚠️ High Street upside ({r.upside_to_target:.0f}%) but algo holds — "
                    f"composite {r.composite_score:.0f}/100 and Piotroski {r.piotroski_score}/9 "
                    "are below the Buy threshold. Upside alone doesn't justify a Buy."
                )

        # Piotroski detail
        if r.piotroski_detail:
            if (r.piotroski_detail or {}).get("_context") == "hyper_growth":
                st.info("⚡ **Growth artefact** — low Piotroski likely reflects ROA lag / accrual spikes from rapid expansion, not financial distress.")
            st.markdown("**Piotroski breakdown (top 5)**")
            items = [(k, v) for k, v in r.piotroski_detail.items() if k not in ("error", "_context")][:5]
            p_cols = st.columns(len(items))
            for col, (k, v) in zip(p_cols, items):
                label = k.replace("F", "").replace("_", " ").title()
                col.metric(label, "✓" if v == 1 else "✗", delta=None)

# ── Failed tickers ────────────────────────────────────────────────────────────
if failed:
    st.caption(f"⚠️ Could not fetch data for: {', '.join(r.ticker for r in failed)}")

# ── AI Validation ─────────────────────────────────────────────────────────────
st.divider()
v_left, v_right = st.columns([4, 1])
with v_left:
    st.subheader("🤖 AI Validation")
    if EVAL_AVAILABLE:
        provider = "Azure OpenAI" if is_azure() else "OpenAI"
        st.caption(
            f"Uses **{provider}** to review whether each algo rating is defensible. "
            "Runs independently of the sector scan — click once after analysis."
        )
    else:
        st.caption("⚠️ Eval module unavailable — check OPENAI_API_KEY / Azure credentials in .env")
with v_right:
    val_btn = st.button(
        "▶ Run Validation",
        type="primary",
        use_container_width=True,
        disabled=not EVAL_AVAILABLE,
    )

if val_btn and EVAL_AVAILABLE:
    with st.spinner("Calling AI validator — this takes ~15-30 s…"):
        try:
            _val = call_openai(results, sector_key, opinion_model(DEFAULT_MODEL))
            st.session_state["validation"]        = _val
            st.session_state["validation_sector"] = sector_key
        except Exception as _e:
            st.session_state["validation_error"] = str(_e)
    st.rerun()  # Force clean render — prevents ghost duplicate of this section

if "validation_error" in st.session_state:
    st.error(f"Validation failed: {st.session_state.pop('validation_error')}")

if (
    "validation" in st.session_state
    and st.session_state.get("validation_sector") == sector_key
    and not val_btn
):
    _val = st.session_state["validation"]
    stock_vals = {v["ticker"]: v for v in _val.get("stock_validations", [])}
    ov = _val.get("overall_assessment", {})

    # ── Per-stock table ───────────────────────────────────────────────────────
    agree_icon  = {"agree": "✅", "partial": "⚠️", "disagree": "❌"}
    conf_icon   = {"high": "🟢", "medium": "🟡", "low": "🔴"}

    rows = []
    for r in valid:
        v = stock_vals.get(r.ticker, {})
        agree = v.get("agreement", "—")
        conf  = v.get("confidence", "—")
        notes = v.get("notes", "")
        concerns = v.get("concerns", [])
        if concerns:
            notes += "  ⚠ " + " · ".join(concerns)
        rows.append({
            "Ticker":      r.ticker,
            "Algo Rating": r.analyst_rating,
            "AI Says":     v.get("rating_suggested", "—"),
            "Agreement":   f"{agree_icon.get(agree, '')} {agree}",
            "Confidence":  f"{conf_icon.get(conf, '')} {conf}",
            "Notes":       notes,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Notes": st.column_config.TextColumn("Notes", width="large"),
        },
    )

    # ── Overall assessment ────────────────────────────────────────────────────
    with st.expander("📋 Overall Assessment", expanded=True):
        summary = ov.get("sector_summary", "")
        if summary:
            st.info(summary)

        oa1, oa2 = st.columns(2)
        with oa1:
            if ov.get("top_conviction_picks"):
                st.markdown("**🏆 Top Conviction Picks**")
                for t in ov["top_conviction_picks"]:
                    st.markdown(f"- {t}")
            if ov.get("most_contested_ratings"):
                st.markdown("**⚖️ Most Contested Ratings**")
                for t in ov["most_contested_ratings"]:
                    st.markdown(f"- {t}")
        with oa2:
            if ov.get("systemic_issues"):
                st.markdown("**⚠️ Systemic Issues**")
                for item in ov["systemic_issues"]:
                    st.markdown(f"- {item}")
            if ov.get("data_quality_flags"):
                st.markdown("**🚩 Data Quality Flags**")
                for t in ov["data_quality_flags"]:
                    st.markdown(f"- {t}")

    disagreements = [
        v["ticker"] for v in _val.get("stock_validations", [])
        if v.get("agreement") == "disagree"
    ]
    if disagreements:
        st.warning(f"AI disagrees on: {', '.join(disagreements)}")

# ── Download report ───────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state.get("scan_sector") == sector_key:
    st.divider()
    _has_val = (
        "validation" in st.session_state
        and st.session_state.get("validation_sector") == sector_key
    )
    _rpt = generate_report(
        sector_key,
        st.session_state["results"],
        st.session_state["validation"] if _has_val else None,
    )
    _fname = f"{sector_key}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
    dl1, dl2 = st.columns([3, 1])
    with dl1:
        _label = (
            "📄 Report includes sector analysis + AI validation"
            if _has_val else
            "📄 Report includes sector analysis only (run validation to include AI results)"
        )
        st.caption(_label)
    with dl2:
        st.download_button(
            "⬇️ Download Report",
            data=_rpt,
            file_name=_fname,
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )
