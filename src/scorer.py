# src/scorer.py — Multi-factor stock scoring engine
# Combines: Piotroski F-Score + Valuation + Momentum + Growth

import time
import random
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

REQUEST_DELAY = 1.2  # seconds between ticker requests


def _yf_fetch(fn, retries: int = 3, base_delay: float = 2.0):
    from yfinance.exceptions import YFRateLimitError
    for attempt in range(retries):
        try:
            return fn()
        except YFRateLimitError:
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
            else:
                raise
        except Exception:
            raise
    return None


@dataclass
class StockScore:
    ticker: str
    company_name: str = ""
    current_price: float = 0.0
    sector: str = ""
    industry: str = ""

    # Piotroski (0-9)
    piotroski_score: int = 0
    piotroski_detail: dict = field(default_factory=dict)

    # Valuation
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    peg_ratio: Optional[float] = None
    fcf_yield: Optional[float] = None

    # Growth
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None

    # Momentum & Technical
    price_vs_52w_low: Optional[float] = None
    price_vs_52w_high: Optional[float] = None
    momentum_6m: Optional[float] = None
    rsi_14: Optional[float] = None
    above_50dma: Optional[bool] = None
    above_200dma: Optional[bool] = None

    # Composite score (0-100)
    valuation_score: float = 0.0
    growth_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0

    # Analyst data
    analyst_target: Optional[float] = None
    upside_to_target: Optional[float] = None
    analyst_rating: str = ""
    yahoo_consensus: str = ""  # raw Yahoo recommendationKey, preserved for divergence check

    # Flags
    error: Optional[str] = None
    data_quality: str = "full"  # full | partial | failed


def safe_get(df, key, col=0):
    """Safely retrieve a value from a DataFrame."""
    try:
        if key in df.index:
            val = df.loc[key].iloc[col]
            if pd.notna(val) and val != 0:
                return float(val)
    except Exception:
        pass
    return None


def compute_piotroski(ticker_obj: yf.Ticker) -> tuple[int, dict]:
    """
    Piotroski F-Score (0–9):
    Profitability (4): ROA, OCF, ROA_delta, Accruals
    Leverage/Liquidity (3): Delta_Leverage, Delta_Liquidity, No_Dilution
    Operating Efficiency (2): Delta_Gross_Margin, Delta_Asset_Turnover
    """
    score = 0
    detail = {}

    try:
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt
        cf = ticker_obj.cashflow

        if bs is None or inc is None or cf is None:
            return 0, {"error": "Missing financial statements"}
        if bs.shape[1] < 2 or inc.shape[1] < 2:
            return 0, {"error": "Insufficient historical data"}

        # Helper: get current (0) and prior (1) year values
        def get(df, key, yr=0):
            return safe_get(df, key, yr)

        ta_curr = get(bs, "Total Assets", 0)
        ta_prev = get(bs, "Total Assets", 1)
        ni_curr = get(inc, "Net Income", 0)
        ocf_curr = get(cf, "Operating Cash Flow", 0)

        # F1 – ROA > 0
        if ta_curr and ni_curr:
            roa = ni_curr / ta_curr
            f1 = 1 if roa > 0 else 0
            score += f1
            detail["F1_ROA_positive"] = f1

        # F2 – Operating cash flow > 0
        f2 = 1 if (ocf_curr and ocf_curr > 0) else 0
        score += f2
        detail["F2_OCF_positive"] = f2

        # F3 – ROA improving
        if ta_curr and ta_prev and ni_curr:
            ni_prev = get(inc, "Net Income", 1)
            if ni_prev and ta_prev:
                roa_curr = ni_curr / ta_curr
                roa_prev = ni_prev / ta_prev
                f3 = 1 if roa_curr > roa_prev else 0
                score += f3
                detail["F3_ROA_improving"] = f3

        # F4 – Accruals: OCF > ROA
        if ta_curr and ni_curr and ocf_curr:
            roa = ni_curr / ta_curr
            ocf_ratio = ocf_curr / ta_curr
            f4 = 1 if ocf_ratio > roa else 0
            score += f4
            detail["F4_accruals_low"] = f4

        # F5 – Leverage decreasing
        ltd_curr = get(bs, "Long Term Debt", 0) or 0
        ltd_prev = get(bs, "Long Term Debt", 1) or 0
        if ta_curr and ta_prev:
            lev_curr = ltd_curr / ta_curr
            lev_prev = ltd_prev / ta_prev
            f5 = 1 if lev_curr <= lev_prev else 0
            score += f5
            detail["F5_leverage_lower"] = f5

        # F6 – Liquidity improving (current ratio)
        ca_curr = get(bs, "Current Assets", 0) or 0
        cl_curr = get(bs, "Current Liabilities", 0) or 0
        ca_prev = get(bs, "Current Assets", 1) or 0
        cl_prev = get(bs, "Current Liabilities", 1) or 0
        if cl_curr > 0 and cl_prev > 0:
            cr_curr = ca_curr / cl_curr
            cr_prev = ca_prev / cl_prev
            f6 = 1 if cr_curr > cr_prev else 0
            score += f6
            detail["F6_liquidity_improved"] = f6

        # F7 – No share dilution
        shares_curr = get(inc, "Basic Average Shares", 0) or \
                      get(bs, "Ordinary Shares Number", 0)
        shares_prev = get(inc, "Basic Average Shares", 1) or \
                      get(bs, "Ordinary Shares Number", 1)
        if shares_curr and shares_prev:
            f7 = 1 if shares_curr <= shares_prev * 1.02 else 0
            score += f7
            detail["F7_no_dilution"] = f7

        # F8 – Gross margin improving
        rev_curr = get(inc, "Total Revenue", 0)
        gp_curr  = get(inc, "Gross Profit", 0)
        rev_prev = get(inc, "Total Revenue", 1)
        gp_prev  = get(inc, "Gross Profit", 1)
        if rev_curr and gp_curr and rev_prev and gp_prev:
            gm_curr = gp_curr / rev_curr
            gm_prev = gp_prev / rev_prev
            f8 = 1 if gm_curr > gm_prev else 0
            score += f8
            detail["F8_gross_margin_improved"] = f8

        # F9 – Asset turnover improving
        if ta_curr and ta_prev and rev_curr and rev_prev:
            at_curr = rev_curr / ta_curr
            at_prev = rev_prev / ta_prev
            f9 = 1 if at_curr > at_prev else 0
            score += f9
            detail["F9_asset_turnover_improved"] = f9

    except Exception as e:
        detail["error"] = str(e)

    return score, detail


def compute_scores(result: StockScore) -> StockScore:
    """
    Derive composite score components from raw metrics.
    Each component is 0–100.
    """

    # ── Valuation Score ──────────────────────────────────────
    val_pts = 0.0
    val_cnt = 0

    # Lower P/E = better (cap at 60x)
    if result.forward_pe and result.forward_pe > 0:
        score = max(0, 100 - (result.forward_pe / 60) * 100)
        val_pts += score; val_cnt += 1
    elif result.pe_ratio and result.pe_ratio > 0:
        score = max(0, 100 - (result.pe_ratio / 60) * 100)
        val_pts += score * 0.7; val_cnt += 0.7  # discount trailing PE

    # Lower P/B = better (cap at 10x)
    if result.pb_ratio and 0 < result.pb_ratio < 10:
        val_pts += max(0, 100 - (result.pb_ratio / 10) * 100); val_cnt += 1

    # Lower P/S = better (cap at 8x)
    if result.ps_ratio and 0 < result.ps_ratio < 8:
        val_pts += max(0, 100 - (result.ps_ratio / 8) * 100); val_cnt += 1

    # Lower EV/EBITDA = better (cap at 30x)
    if result.ev_ebitda and 0 < result.ev_ebitda < 30:
        val_pts += max(0, 100 - (result.ev_ebitda / 30) * 100); val_cnt += 1.5

    # PEG < 1 = highly valued
    if result.peg_ratio and 0 < result.peg_ratio:
        score = max(0, 100 - (result.peg_ratio / 3) * 100)
        val_pts += score; val_cnt += 1.5

    # FCF yield > 5% is excellent
    if result.fcf_yield and result.fcf_yield > 0:
        val_pts += min(100, result.fcf_yield * 10); val_cnt += 1

    result.valuation_score = round(val_pts / val_cnt, 1) if val_cnt else 0.0

    # ── Growth Score ─────────────────────────────────────────
    grw_pts = 0.0
    grw_cnt = 0

    if result.revenue_growth_yoy is not None:
        grw_pts += min(100, max(0, 50 + result.revenue_growth_yoy * 100))
        grw_cnt += 1

    if result.earnings_growth_yoy is not None:
        grw_pts += min(100, max(0, 50 + result.earnings_growth_yoy * 50))
        grw_cnt += 1.5  # earnings growth weighted higher

    if result.gross_margin and result.gross_margin > 0:
        grw_pts += min(100, result.gross_margin * 100 * 1.5); grw_cnt += 1

    if result.operating_margin is not None:
        om_score = 50 + result.operating_margin * 200
        grw_pts += min(100, max(0, om_score)); grw_cnt += 1

    if result.roe and result.roe > 0:
        grw_pts += min(100, result.roe * 300); grw_cnt += 1

    # Piotroski contributes to growth quality
    grw_pts += (result.piotroski_score / 9) * 100; grw_cnt += 1.5

    result.growth_score = round(grw_pts / grw_cnt, 1) if grw_cnt else 0.0

    # ── Momentum Score ───────────────────────────────────────
    mom_pts = 0.0
    mom_cnt = 0

    # Price vs 52w low (higher = more recovered, could be overvalued or confirming)
    if result.price_vs_52w_low:
        # Reward stocks that have risen 10-50% from 52w low (not too far, not at bottom)
        ratio = result.price_vs_52w_low
        if 1.05 <= ratio <= 1.5:
            mom_pts += 80
        elif 1.5 < ratio <= 2.0:
            mom_pts += 60
        elif ratio < 1.05:
            mom_pts += 40  # still at bottom — could mean value or value trap
        else:
            mom_pts += 30  # very extended
        mom_cnt += 1

    # 6-month price momentum
    if result.momentum_6m is not None:
        # Academic momentum factor: positive 6m momentum = buy signal
        mom_pts += min(100, max(0, 50 + result.momentum_6m * 100))
        mom_cnt += 1

    # RSI: 40-65 is "healthy" range
    if result.rsi_14:
        if 40 <= result.rsi_14 <= 65:
            mom_pts += 80
        elif 30 <= result.rsi_14 < 40:
            mom_pts += 65  # oversold — potential buy
        elif 65 < result.rsi_14 <= 75:
            mom_pts += 55
        else:
            mom_pts += 25  # overbought or very weak
        mom_cnt += 1

    # Moving average alignment
    if result.above_200dma is not None:
        mom_pts += 70 if result.above_200dma else 30; mom_cnt += 1
    if result.above_50dma is not None:
        mom_pts += 65 if result.above_50dma else 35; mom_cnt += 0.5

    result.momentum_score = round(mom_pts / mom_cnt, 1) if mom_cnt else 0.0

    # ── Composite (weighted) ──────────────────────────────────
    # Weights: Valuation 35% | Growth 40% | Momentum 25%
    result.composite_score = round(
        result.valuation_score * 0.35
        + result.growth_score   * 0.40
        + result.momentum_score * 0.25,
        1
    )

    return result


def analyse_ticker(ticker: str) -> StockScore:
    """Full pipeline for a single ticker."""
    result = StockScore(ticker=ticker.upper())
    time.sleep(REQUEST_DELAY)

    try:
        stock = yf.Ticker(ticker)
        info = _yf_fetch(lambda: stock.info) or {}

        # Basic info
        result.company_name = info.get("longName") or info.get("shortName") or ticker
        result.sector = info.get("sector", "")
        result.industry = info.get("industry", "")
        result.current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0

        # Valuation multiples
        result.pe_ratio     = info.get("trailingPE")
        result.forward_pe   = info.get("forwardPE")
        result.pb_ratio     = info.get("priceToBook")
        result.ps_ratio     = info.get("priceToSalesTrailingTwelveMonths")
        result.ev_ebitda    = info.get("enterpriseToEbitda")
        result.peg_ratio    = info.get("trailingPegRatio") or info.get("pegRatio")

        # FCF yield
        fcf = info.get("freeCashflow")
        mc  = info.get("marketCap")
        if fcf and mc and mc > 0:
            result.fcf_yield = (fcf / mc) * 100

        # Growth metrics
        result.revenue_growth_yoy   = info.get("revenueGrowth")
        result.earnings_growth_yoy  = info.get("earningsGrowth")
        result.gross_margin         = info.get("grossMargins")
        result.operating_margin     = info.get("operatingMargins")
        result.roe                  = info.get("returnOnEquity")

        # 52-week range
        low52  = info.get("fiftyTwoWeekLow")
        high52 = info.get("fiftyTwoWeekHigh")
        price  = result.current_price
        if low52 and price:
            result.price_vs_52w_low  = price / low52
        if high52 and price:
            result.price_vs_52w_high = price / high52

        # Analyst data
        result.analyst_target = info.get("targetMeanPrice")
        if result.analyst_target and price and price > 0:
            result.upside_to_target = ((result.analyst_target - price) / price) * 100
        result.yahoo_consensus = info.get("recommendationKey", "").replace("_", " ").title()
        result.analyst_rating  = result.yahoo_consensus

        # Historical price data for momentum & MAs
        hist = _yf_fetch(lambda: stock.history(period="1y", interval="1d"))
        if hist is not None and not hist.empty and len(hist) >= 50:
            close = hist["Close"]

            # 6-month momentum (roughly 126 trading days)
            if len(close) >= 126:
                result.momentum_6m = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126]
            elif len(close) >= 60:
                result.momentum_6m = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]

            # Moving averages
            ma50  = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
            result.above_50dma  = bool(close.iloc[-1] > ma50)
            result.above_200dma = bool(close.iloc[-1] > ma200) if ma200 else None

            # RSI (Wilder's smoothed method)
            try:
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=13, min_periods=14).mean()
                avg_loss = loss.ewm(com=13, min_periods=14).mean()
                rsi_series = 100 - (100 / (1 + avg_gain / avg_loss))
                if not rsi_series.dropna().empty:
                    result.rsi_14 = round(float(rsi_series.dropna().iloc[-1]), 1)
            except Exception:
                pass

        # Piotroski
        result.piotroski_score, result.piotroski_detail = compute_piotroski(stock)

        # Check data quality
        filled = sum([
            result.pe_ratio is not None,
            result.revenue_growth_yoy is not None,
            result.rsi_14 is not None,
            result.momentum_6m is not None,
        ])
        result.data_quality = "full" if filled >= 3 else "partial"

        # Compute composite scores
        result = compute_scores(result)

        # Derive final rating from composite score + analyst upside + Piotroski.
        # Raised composite floor to 56 to prevent razor-thin boundary flips (e.g.
        # two stocks at composite 55 getting different labels from a 0.2% upside
        # difference). Piotroski ≥5 ensures financial health gates the top label.
        up = result.upside_to_target if result.upside_to_target is not None else 0.0
        if result.composite_score >= 56 and up >= 30 and result.piotroski_score >= 5:
            result.analyst_rating = "Strong Buy"
        elif result.composite_score >= 35 and up >= 5:
            result.analyst_rating = "Buy"
        elif result.composite_score >= 20:
            result.analyst_rating = "Hold"
        else:
            result.analyst_rating = "Avoid"

        # Contrarian flag: algo says Strong Buy but Street consensus is Hold/Neutral.
        # Surfaces the divergence so users know they'd be swimming against the Street.
        street = result.yahoo_consensus.lower()
        if result.analyst_rating == "Strong Buy" and any(
            x in street for x in ("hold", "neutral", "underperform", "sell")
        ):
            result.analyst_rating = "Contrarian Strong Buy"

    except Exception as e:
        result.error = str(e)
        result.data_quality = "failed"

    return result
