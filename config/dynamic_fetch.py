"""
Dynamic ticker fetching via Yahoo Finance screener (yf.screen + EquityQuery).

Sectors that map cleanly to Yahoo's taxonomy are fetched live, sorted by
market cap, and filtered to US-listed stocks above $500M.

Sectors with no clean Yahoo mapping (ev, fintech) always use the static
curated list.  Any screener failure also falls back to the static list.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.sectors import SECTOR_TICKERS

log = logging.getLogger(__name__)

try:
    import yfinance as yf
    from yfinance import EquityQuery
    _SCREENER_OK = True
except (ImportError, AttributeError):
    _SCREENER_OK = False

# ── Minimum market cap filter ($500M — removes micro/nano-cap noise) ──────────
MIN_MARKET_CAP = 500_000_000

# ── Yahoo Finance query spec per sector key ───────────────────────────────────
# Each entry is a dict with optional "sector" and/or "industry" keys.
# Values may be a single string or a list (list → OR clause).
# None means "no clean mapping — always use static fallback".
_YAHOO_QUERY_SPEC: dict[str, Optional[dict]] = {
    "tech": {
        "sector": "Technology",
    },
    "energy": {
        "sector": "Energy",
    },
    "reit": {
        "sector": "Real Estate",
    },
    "consumer": {
        "sector": ["Consumer Cyclical", "Consumer Defensive"],
    },
    "defense": {
        "industry": "Aerospace & Defense",
    },
    "semiconductor": {
        "industry": "Semiconductors",
    },
    "biotech": {
        "industry": "Biotechnology",
    },
    "pharma": {
        "sector": "Healthcare",
        "industry": [
            "Drug Manufacturers\u2014General",
            "Drug Manufacturers\u2014Specialty & Generic",
            "Health Care Plans",
        ],
    },
    "mining": {
        "sector": "Basic Materials",
        "industry": [
            "Gold",
            "Silver",
            "Copper",
            "Steel",
            "Aluminum",
            "Uranium",
            "Other Industrial Metals & Mining",
        ],
    },
    "chemical": {
        "sector": "Basic Materials",
        "industry": [
            "Specialty Chemicals",
            "Agricultural Inputs",
            "Chemicals",
        ],
    },
    # No clean Yahoo taxonomy for these — always use curated static list
    "ev":     None,
    "fintech": None,
}


def fetch_sector_tickers(
    sector_key: str,
    size: int = 20,
    min_market_cap: int = MIN_MARKET_CAP,
) -> tuple[list[str], str]:
    """
    Return (tickers, source) where source is "live" or "curated".

    Attempts to fetch the top ``size`` tickers for the sector from Yahoo
    Finance's screener, sorted descending by market cap.  Falls back to
    the static curated list if:
      - The sector has no Yahoo Finance mapping (ev, fintech)
      - The EquityQuery / yf.screen API is not available
      - The screener returns fewer than 5 results
      - Any network / parsing error occurs
    """
    static = SECTOR_TICKERS.get(sector_key, [])

    spec = _YAHOO_QUERY_SPEC.get(sector_key)
    if spec is None:
        log.debug("'%s' has no Yahoo mapping — using curated list", sector_key)
        return static, "curated"

    if not _SCREENER_OK:
        log.warning("yfinance EquityQuery not available — using curated list")
        return static, "curated"

    try:
        query = _build_query(spec, min_market_cap)
        result = yf.screen(
            query,
            sortField="intradaymarketcap",
            sortAsc=False,
            size=size,
        )
        quotes = result.get("quotes", [])
        tickers = [q["symbol"] for q in quotes if "symbol" in q]

        if len(tickers) < 5:
            log.warning(
                "Screener returned only %d ticker(s) for '%s' — falling back",
                len(tickers),
                sector_key,
            )
            return static, "curated"

        log.info("Live fetch: %d tickers for '%s'", len(tickers), sector_key)
        return tickers, "live"

    except Exception as exc:
        log.warning(
            "Screener error for '%s': %s — using curated list", sector_key, exc
        )
        return static, "curated"


def _build_query(spec: dict, min_market_cap: int) -> "EquityQuery":
    """Translate a spec dict into a nested EquityQuery."""
    clauses: list[EquityQuery] = [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("gt", ["intradaymarketcap", min_market_cap]),
    ]

    if "sector" in spec:
        val = spec["sector"]
        if isinstance(val, list):
            clauses.append(
                EquityQuery("or", [EquityQuery("eq", ["sector", s]) for s in val])
            )
        else:
            clauses.append(EquityQuery("eq", ["sector", val]))

    if "industry" in spec:
        val = spec["industry"]
        if isinstance(val, list):
            clauses.append(
                EquityQuery("or", [EquityQuery("eq", ["industry", i]) for i in val])
            )
        else:
            clauses.append(EquityQuery("eq", ["industry", val]))

    return EquityQuery("and", clauses)
