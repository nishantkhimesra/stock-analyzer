# sectors.py — NYSE sector → ticker mapping
# Add or remove tickers as needed

SECTOR_TICKERS = {
    "chemical": [
        "DOW", "LYB", "WLK", "OLN", "FMC", "CF", "MOS", "NTR",
        "ASIX", "LXU", "CBT", "HUN", "EMN", "CE", "RPM", "APD",
        "PPG", "SHW", "ALB", "CC", "TROX", "MEOH", "RYAM"
    ],
    "mining": [
        "B", "NEM", "AEM", "KGC", "AU", "GOLD", "HL", "CDE",
        "EXK", "AG", "FCX", "SCCO", "MP", "CCJ", "BHP", "RIO",
        "NEXA", "HYMC", "CLF", "X", "AA"
    ],
    "semiconductor": [
        "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU",
        "AMAT", "LRCX", "KLAC", "MRVL", "ON", "WOLF", "SWKS",
        "MPWR", "ENTG", "ONTO", "ACLS", "CAMT"
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "HAL", "BKR", "OXY",
        "DVN", "FANG", "PXD", "MPC", "VLO", "PSX", "HES",
        "EOG", "APA", "WMB", "KMI", "OKE"
    ],
    "pharma": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN",
        "GILD", "BIIB", "REGN", "VRTX", "ZTS", "CI", "HUM",
        "UNH", "CVS", "MCK", "CAH", "ABC"
    ],
    "fintech": [
        "V", "MA", "PYPL", "SQ", "AFRM", "UPST", "LC",
        "SOFI", "NU", "HOOD", "COIN", "MARA", "RIOT",
        "FIS", "FISV", "GPN", "WEX", "DLO"
    ],
    "ev": [
        "TSLA", "RIVN", "LCID", "NIO", "LI", "XPEV", "FSR",
        "CHPT", "BLNK", "EVGO", "QS", "MVST", "GOEV",
        "PTRA", "REE", "NKLA"
    ],
    "defense": [
        "LMT", "RTX", "NOC", "GD", "BA", "L3T", "HII",
        "LDOS", "SAIC", "CACI", "KTOS", "BWXT", "DRS",
        "VEC", "PLTR", "AI"
    ],
    "reit": [
        "PLD", "AMT", "EQIX", "CCI", "SPG", "O", "WELL",
        "DLR", "PSA", "EQR", "AVB", "VTR", "BXP", "KIM",
        "REG", "COLD", "IIPR", "GLPI"
    ],
    "consumer": [
        "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "MCD",
        "SBUX", "NKE", "LULU", "TJX", "ROST", "DG", "DLTR",
        "YUM", "CMG", "DPZ", "EL", "PG", "CL"
    ],
    "tech": [
        "AAPL", "MSFT", "GOOGL", "META", "AMZN", "CRM",
        "ORCL", "IBM", "ADBE", "NOW", "SNOW", "DDOG",
        "NET", "ZS", "PANW", "CRWD", "S", "MDB", "GTLB"
    ],
    "biotech": [
        "MRNA", "BNTX", "NVAX", "SRPT", "BLUE", "FATE",
        "BEAM", "EDIT", "NTLA", "CRSP", "ALLO", "ALNY",
        "IONS", "EXEL", "INCY", "RARE", "ACAD", "LGND"
    ],
}

# Friendly display names
SECTOR_DISPLAY = {
    "chemical":     "Chemical Manufacturing",
    "mining":       "Mining & Metals",
    "semiconductor":"Semiconductors",
    "energy":       "Oil, Gas & Energy",
    "pharma":       "Pharmaceuticals & Healthcare",
    "fintech":      "Fintech & Payments",
    "ev":           "Electric Vehicles & Clean Transport",
    "defense":      "Defense & Aerospace",
    "reit":         "Real Estate Investment Trusts",
    "consumer":     "Consumer Retail & Brands",
    "tech":         "Technology & Software",
    "biotech":      "Biotechnology",
}
