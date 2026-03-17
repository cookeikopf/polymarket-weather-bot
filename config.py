"""
Polymarket Weather Prediction Bot - Configuration
===================================================
All tunable parameters in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# POLYMARKET API
# ═══════════════════════════════════════════════════════════════════
POLYMARKET_HOST = "https://clob.polymarket.com"
GAMMA_API_HOST = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_GEO_TOKEN = os.getenv("POLYMARKET_GEO_TOKEN", "")

# Polymarket wallet/funder address (the Polymarket proxy wallet, NOT your MetaMask)
# Found in your Polymarket profile URL: https://polymarket.com/profile/<THIS_ADDRESS>
FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")

# Signature type: 0=EOA, 1=POLY_PROXY (Magic/email login), 2=GNOSIS_SAFE (browser wallet)
SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

# ─── Builder API (gasless trading + order attribution + rewards) ───
# Get your keys at: polymarket.com/settings?tab=builder
BUILDER_API_KEY = os.getenv("POLY_BUILDER_API_KEY", "")
BUILDER_SECRET = os.getenv("POLY_BUILDER_SECRET", "")
BUILDER_PASSPHRASE = os.getenv("POLY_BUILDER_PASSPHRASE", "")

# Order execution strategy:
# "taker"    = aggressive (FOK/FAK market orders, instant fill)
# "maker"    = passive (post-only limit orders, earns liquidity rewards)
# "adaptive" = uses maker for small edges, taker for large edges
ORDER_STRATEGY = os.getenv("ORDER_STRATEGY", "adaptive")

# Adaptive thresholds
TAKER_EDGE_THRESHOLD = 0.15  # Use taker (aggressive) if edge > 15%
MAKER_PRICE_OFFSET = 0.001   # Place maker orders 0.1 cent inside spread

# ═══════════════════════════════════════════════════════════════════
# WEATHER DATA SOURCES (Open-Meteo - Free, no API key needed)
# ═══════════════════════════════════════════════════════════════════
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Station coordinates for ALL 20 Polymarket weather cities
# Coordinates point to the exact weather station used for resolution
STATIONS = {
    # ─── USA (°F) ───
    "NYC": {
        "name": "LaGuardia Airport, New York",
        "lat": 40.7769, "lon": -73.8740,
        "unit": "fahrenheit", "tz": "America/New_York",
        "slug_name": "nyc",
    },
    "Miami": {
        "name": "Miami International Airport",
        "lat": 25.7959, "lon": -80.2870,
        "unit": "fahrenheit", "tz": "America/New_York",
        "slug_name": "miami",
    },
    "Atlanta": {
        "name": "Hartsfield-Jackson Airport, Atlanta",
        "lat": 33.6407, "lon": -84.4277,
        "unit": "fahrenheit", "tz": "America/New_York",
        "slug_name": "atlanta",
    },
    "Chicago": {
        "name": "O'Hare International Airport, Chicago",
        "lat": 41.9742, "lon": -87.9073,
        "unit": "fahrenheit", "tz": "America/Chicago",
        "slug_name": "chicago",
    },
    "Dallas": {
        "name": "DFW International Airport, Dallas",
        "lat": 32.8998, "lon": -97.0403,
        "unit": "fahrenheit", "tz": "America/Chicago",
        "slug_name": "dallas",
    },
    "Seattle": {
        "name": "Seattle-Tacoma International Airport",
        "lat": 47.4502, "lon": -122.3088,
        "unit": "fahrenheit", "tz": "America/Los_Angeles",
        "slug_name": "seattle",
    },
    # ─── Europe (°C) ───
    "London": {
        "name": "Heathrow Airport, London",
        "lat": 51.4700, "lon": -0.4543,
        "unit": "celsius", "tz": "Europe/London",
        "slug_name": "london",
    },
    "Paris": {
        "name": "Charles de Gaulle Airport, Paris",
        "lat": 49.0097, "lon": 2.5479,
        "unit": "celsius", "tz": "Europe/Paris",
        "slug_name": "paris",
    },
    "Munich": {
        "name": "Munich Airport",
        "lat": 48.3537, "lon": 11.7750,
        "unit": "celsius", "tz": "Europe/Berlin",
        "slug_name": "munich",
    },
    "Ankara": {
        "name": "Esenboga Airport, Ankara",
        "lat": 40.1281, "lon": 32.9951,
        "unit": "celsius", "tz": "Europe/Istanbul",
        "slug_name": "ankara",
    },
    "Tel Aviv": {
        "name": "Ben Gurion Airport, Tel Aviv",
        "lat": 32.0055, "lon": 34.8854,
        "unit": "celsius", "tz": "Asia/Jerusalem",
        "slug_name": "tel-aviv",
    },
    # ─── Americas (°C) ───
    "Toronto": {
        "name": "Pearson International Airport, Toronto",
        "lat": 43.6777, "lon": -79.6248,
        "unit": "celsius", "tz": "America/Toronto",
        "slug_name": "toronto",
    },
    "Buenos Aires": {
        "name": "Ezeiza Airport, Buenos Aires",
        "lat": -34.8222, "lon": -58.5358,
        "unit": "celsius", "tz": "America/Argentina/Buenos_Aires",
        "slug_name": "buenos-aires",
    },
    "Sao Paulo": {
        "name": "Guarulhos Airport, Sao Paulo",
        "lat": -23.4356, "lon": -46.4731,
        "unit": "celsius", "tz": "America/Sao_Paulo",
        "slug_name": "sao-paulo",
    },
    # ─── Asia (°C) ───
    "Seoul": {
        "name": "Incheon International Airport, Seoul",
        "lat": 37.4602, "lon": 126.4407,
        "unit": "celsius", "tz": "Asia/Seoul",
        "slug_name": "seoul",
    },
    "Shanghai": {
        "name": "Pudong International Airport, Shanghai",
        "lat": 31.1443, "lon": 121.8083,
        "unit": "celsius", "tz": "Asia/Shanghai",
        "slug_name": "shanghai",
    },
    "Tokyo": {
        "name": "Haneda Airport, Tokyo",
        "lat": 35.5494, "lon": 139.7798,
        "unit": "celsius", "tz": "Asia/Tokyo",
        "slug_name": "tokyo",
    },
    "Singapore": {
        "name": "Changi Airport, Singapore",
        "lat": 1.3644, "lon": 103.9915,
        "unit": "celsius", "tz": "Asia/Singapore",
        "slug_name": "singapore",
    },
    "Lucknow": {
        "name": "Chaudhary Charan Singh Airport, Lucknow",
        "lat": 26.7606, "lon": 80.8893,
        "unit": "celsius", "tz": "Asia/Kolkata",
        "slug_name": "lucknow",
    },
    # ─── Oceania (°C) ───
    "Wellington": {
        "name": "Wellington Airport",
        "lat": -41.3272, "lon": 174.8053,
        "unit": "celsius", "tz": "Pacific/Auckland",
        "slug_name": "wellington",
    },
}

# Weather models to ensemble (Open-Meteo provides all of these for free)
WEATHER_MODELS = [
    "best_match",          # Auto-selects best local model
    "ecmwf_ifs025",        # ECMWF IFS 0.25° (European flagship)
    "gfs_seamless",        # GFS (US flagship, NOAA)
    "icon_seamless",       # ICON (German DWD)
    "gem_seamless",        # GEM (Canadian)
    "meteofrance_seamless",# Meteo-France ARPEGE/AROME
    "jma_seamless",        # JMA (Japan)
    "ukmo_seamless",       # UK Met Office
]

# Model weights (learned from historical accuracy, will be optimized)
MODEL_WEIGHTS = {
    "best_match": 0.20,
    "ecmwf_ifs025": 0.25,  # Consistently best
    "gfs_seamless": 0.15,
    "icon_seamless": 0.12,
    "gem_seamless": 0.08,
    "meteofrance_seamless": 0.08,
    "jma_seamless": 0.06,
    "ukmo_seamless": 0.06,
}

# ═══════════════════════════════════════════════════════════════════
# ML & PROBABILITY CALIBRATION
# ═══════════════════════════════════════════════════════════════════
# Historical data window for calibration (years)
CALIBRATION_YEARS = 3

# Kernel Density Estimation bandwidth (auto-tuned via cross-validation)
KDE_BANDWIDTH = "silverman"  # or float, or "scott"

# Platt scaling / isotonic regression for calibration
CALIBRATION_METHOD = "isotonic"  # "platt" or "isotonic"

# Number of Monte Carlo samples for probability estimation
MC_SAMPLES = 10000

# Temperature bucket size (Polymarket uses 2°F buckets)
TEMP_BUCKET_SIZE_F = 2

# ═══════════════════════════════════════════════════════════════════
# EDGE DETECTION & TRADING SIGNALS
# ═══════════════════════════════════════════════════════════════════
# Minimum edge (our prob - market prob) to enter a trade
MIN_EDGE_PCT = 0.04  # 4% minimum edge (weather: 0 fees = lower threshold profitable)

# Minimum absolute probability for a bucket to be tradeable
MIN_PROBABILITY = 0.03  # Don't trade on < 3% outcomes

# Maximum price we'll pay for a share (avoid illiquid extremes)
MAX_ENTRY_PRICE = 0.92

# Confidence threshold (0-1) from ensemble agreement
MIN_ENSEMBLE_AGREEMENT = 0.50  # 50% agreement (relaxed: 20 cities provide diversification)

# ═══════════════════════════════════════════════════════════════════
# KELLY CRITERION & POSITION SIZING
# ═══════════════════════════════════════════════════════════════════
# Fractional Kelly (full Kelly is too aggressive)
KELLY_FRACTION = 0.20  # Fifth Kelly — conservative, smoother equity curve

# Maximum position size as fraction of bankroll
MAX_POSITION_PCT = 0.10  # Max 10% per trade

# Maximum number of concurrent positions
MAX_CONCURRENT_POSITIONS = 10  # Increased for 20-city coverage

# Maximum total exposure (sum of all positions / bankroll)
MAX_TOTAL_EXPOSURE = 0.50  # Max 50% (diversified across 20 cities)

# Minimum trade size in USDC
# Note: Polymarket orderMinSize is typically 5 shares, which at price 0.30 = $1.50
MIN_TRADE_SIZE_USDC = 2.0

# Maximum trade size in USDC
MAX_TRADE_SIZE_USDC = 15.0  # Conservative for $50 bankroll; increase proportionally

# ═══════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
# Stop trading if drawdown exceeds this
MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown halt

# Take profit if edge narrows below this
TAKE_PROFIT_EDGE_PCT = 0.02  # Exit if edge < 2%

# Time-based exit: exit position X hours before market resolution
EXIT_HOURS_BEFORE_RESOLUTION = 2

# Trailing stop on unrealized P&L
TRAILING_STOP_PCT = 0.30  # 30% of peak unrealized profit

# ═══════════════════════════════════════════════════════════════════
# BACKTESTING
# ═══════════════════════════════════════════════════════════════════
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2026-03-15"
BACKTEST_INITIAL_BANKROLL = 50.0  # Starting capital (adjustable via .env)
LIVE_BANKROLL = float(os.getenv("POLYMARKET_BANKROLL", "50.0"))  # Live trading bankroll

# Simulated market parameters
SIM_SPREAD = 0.06        # 6% bid-ask spread (realistic for weather markets)
SIM_SLIPPAGE = 0.02      # 2% slippage
SIM_MARKET_NOISE = 0.015 # Market prices are fairly efficient

# ═══════════════════════════════════════════════════════════════════
# OPERATIONAL
# ═══════════════════════════════════════════════════════════════════
SCAN_INTERVAL_SECONDS = 300  # Check markets every 5 min
LOG_LEVEL = "INFO"
PAPER_TRADING = True  # Start in paper mode
DATA_DIR = "data"
RESULTS_DIR = "results"

# Recommended scan times (UTC hours) — aligned with NWP model update cycles
# Models update at 00Z, 06Z, 12Z, 18Z; data available ~2-4h after
SCAN_TIMES_UTC = [6, 12, 18, 0]

# ═══════════════════════════════════════════════════════════════════
# CLIMATE ZONES (for position correlation management)
# ═══════════════════════════════════════════════════════════════════
CLIMATE_ZONES = {
    "US_East": ["NYC", "Miami", "Atlanta"],
    "US_Central": ["Chicago", "Dallas"],
    "US_West": ["Seattle"],
    "Europe": ["London", "Paris", "Munich"],
    "Middle_East": ["Ankara", "Tel Aviv"],
    "Americas_South": ["Buenos Aires", "Sao Paulo"],
    "Americas_North": ["Toronto"],
    "East_Asia": ["Seoul", "Shanghai", "Tokyo"],
    "South_Asia": ["Singapore", "Lucknow"],
    "Oceania": ["Wellington"],
}
MAX_POSITIONS_PER_ZONE = 3

# Slug-based market discovery: number of days ahead to scan
MARKET_SCAN_DAYS_AHEAD = 7

# ═══════════════════════════════════════════════════════════════════
# WEATHER UNDERGROUND (Resolution Data Source)
# ═══════════════════════════════════════════════════════════════════
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
WU_API_BASE = "https://api.weather.com/v1/location"

# WU station IDs (ICAO:9:COUNTRY) mapped to our station IDs
WU_STATIONS = {
    "NYC":          {"wu": "KLGA:9:US", "units": "e"},
    "Miami":        {"wu": "KMIA:9:US", "units": "e"},
    "Atlanta":      {"wu": "KATL:9:US", "units": "e"},
    "Chicago":      {"wu": "KORD:9:US", "units": "e"},
    "Dallas":       {"wu": "KDFW:9:US", "units": "e"},
    "Seattle":      {"wu": "KSEA:9:US", "units": "e"},
    "London":       {"wu": "EGLL:9:GB", "units": "m"},
    "Paris":        {"wu": "LFPG:9:FR", "units": "m"},
    "Munich":       {"wu": "EDDM:9:DE", "units": "m"},
    "Ankara":       {"wu": "LTAC:9:TR", "units": "m"},
    "Tel Aviv":     {"wu": "LLBG:9:IL", "units": "m"},
    "Toronto":      {"wu": "CYYZ:9:CA", "units": "m"},
    "Buenos Aires": {"wu": "SAEZ:9:AR", "units": "m"},
    "Sao Paulo":    {"wu": "SBGR:9:BR", "units": "m"},
    "Seoul":        {"wu": "RKSI:9:KR", "units": "m"},
    "Shanghai":     {"wu": "ZSPD:9:CN", "units": "m"},
    "Tokyo":        {"wu": "RJTT:9:JP", "units": "m"},
    "Singapore":    {"wu": "WSSS:9:SG", "units": "m"},
    "Lucknow":      {"wu": "VILK:9:IN", "units": "m"},
    "Wellington":   {"wu": "NZWN:9:NZ", "units": "m"},
}

# ML Model
ML_MODEL_PATH = "data/ml_model.pkl"
ML_FEATURES_PATH = "data/ml_features.csv"
WU_DATA_DIR = "data/wu_historical"
WU_CACHE_DIR = "data/wu_cache"
