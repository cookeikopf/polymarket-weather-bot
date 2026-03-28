"""
Polymarket Weather Bot V6.2 — Configuration
=============================================
All tunable parameters in one place.

TRIPLE STRATEGY:
  1. LADDER: BUY YES in 2 buckets around ensemble median at low prices (<$0.25)
  2. CONSERVATIVE NO: BUY NO on unlikely outcomes at high entry (>=0.65)
  3. LATE SNIPER: BUY YES/NO using late-market prices when forecast confidence is high

Weather markets on Polymarket have ZERO taker fees.

V6.2 Real Backtest (178 markets, 10 cities, 18 days):
  1105 trades | 58.4% WR | PF 2.38 | +3009% ROI
  Ladder: 323 trades +$548 | Cons NO: 451 trades +$819 | Sniper: 331 trades +$1643
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
FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

# Builder API (gasless trading)
BUILDER_API_KEY = os.getenv("POLY_BUILDER_API_KEY", "")
BUILDER_SECRET = os.getenv("POLY_BUILDER_SECRET", "")
BUILDER_PASSPHRASE = os.getenv("POLY_BUILDER_PASSPHRASE", "")

# Order strategy: "taker" | "maker" | "adaptive"
ORDER_STRATEGY = os.getenv("ORDER_STRATEGY", "adaptive")
TAKER_EDGE_THRESHOLD = 0.15
MAKER_PRICE_OFFSET = 0.01

# ═══════════════════════════════════════════════════════════════════
# OPEN-METEO WEATHER API
# ═══════════════════════════════════════════════════════════════════
OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "")

# Auto-select commercial endpoints when API key is set
_PREFIX = "customer-" if OPEN_METEO_API_KEY else ""
FORECAST_URL = f"https://{_PREFIX}api.open-meteo.com/v1/forecast"
ARCHIVE_URL = f"https://{_PREFIX}archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = f"https://{_PREFIX}historical-forecast-api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = f"https://{_PREFIX}ensemble-api.open-meteo.com/v1/ensemble"
PREVIOUS_RUNS_URL = f"https://{_PREFIX}previous-runs-api.open-meteo.com/v1/forecast"

# Weather models for deterministic forecasts
WEATHER_MODELS = [
    "best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless",
    "gem_seamless", "meteofrance_seamless", "jma_seamless", "ukmo_seamless",
]

MODEL_WEIGHTS = {
    "best_match": 0.18, "ecmwf_ifs025": 0.22, "gfs_seamless": 0.15,
    "icon_seamless": 0.12, "gem_seamless": 0.08, "meteofrance_seamless": 0.10,
    "jma_seamless": 0.08, "ukmo_seamless": 0.07,
}

# Ensemble models (probabilistic, member-based)
ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs025", "icon_seamless"]

# ═══════════════════════════════════════════════════════════════════
# STATIONS — All 20 Polymarket weather cities
# ═══════════════════════════════════════════════════════════════════
STATIONS = {
    "NYC":          {"name": "LaGuardia Airport, New York",          "lat": 40.7769, "lon": -73.8740, "unit": "fahrenheit", "tz": "America/New_York",                 "slug": "nyc"},
    "Miami":        {"name": "Miami International Airport",          "lat": 25.7959, "lon": -80.2870, "unit": "fahrenheit", "tz": "America/New_York",                 "slug": "miami"},
    "Atlanta":      {"name": "Hartsfield-Jackson Airport, Atlanta",  "lat": 33.6407, "lon": -84.4277, "unit": "fahrenheit", "tz": "America/New_York",                 "slug": "atlanta"},
    "Chicago":      {"name": "O'Hare International Airport, Chicago","lat": 41.9742, "lon": -87.9073, "unit": "fahrenheit", "tz": "America/Chicago",                  "slug": "chicago"},
    "Dallas":       {"name": "DFW International Airport, Dallas",    "lat": 32.8998, "lon": -97.0403, "unit": "fahrenheit", "tz": "America/Chicago",                  "slug": "dallas"},
    "Seattle":      {"name": "Seattle-Tacoma International Airport", "lat": 47.4502, "lon": -122.3088,"unit": "fahrenheit", "tz": "America/Los_Angeles",              "slug": "seattle"},
    "London":       {"name": "Heathrow Airport, London",             "lat": 51.4700, "lon": -0.4543,  "unit": "celsius",    "tz": "Europe/London",                    "slug": "london"},
    "Paris":        {"name": "Charles de Gaulle Airport, Paris",     "lat": 49.0097, "lon": 2.5479,   "unit": "celsius",    "tz": "Europe/Paris",                     "slug": "paris"},
    "Munich":       {"name": "Munich Airport",                       "lat": 48.3537, "lon": 11.7750,  "unit": "celsius",    "tz": "Europe/Berlin",                    "slug": "munich"},
    "Ankara":       {"name": "Esenboga Airport, Ankara",             "lat": 40.1281, "lon": 32.9951,  "unit": "celsius",    "tz": "Europe/Istanbul",                  "slug": "ankara"},
    "Tel Aviv":     {"name": "Ben Gurion Airport, Tel Aviv",         "lat": 32.0055, "lon": 34.8854,  "unit": "celsius",    "tz": "Asia/Jerusalem",                   "slug": "tel-aviv"},
    "Toronto":      {"name": "Pearson International Airport",        "lat": 43.6777, "lon": -79.6248, "unit": "celsius",    "tz": "America/Toronto",                  "slug": "toronto"},
    "Buenos Aires": {"name": "Ezeiza Airport, Buenos Aires",         "lat": -34.8222,"lon": -58.5358, "unit": "celsius",    "tz": "America/Argentina/Buenos_Aires",   "slug": "buenos-aires"},
    "Sao Paulo":    {"name": "Guarulhos Airport, Sao Paulo",         "lat": -23.4356,"lon": -46.4731, "unit": "celsius",    "tz": "America/Sao_Paulo",                "slug": "sao-paulo"},
    "Seoul":        {"name": "Incheon International Airport, Seoul", "lat": 37.4602, "lon": 126.4407, "unit": "celsius",    "tz": "Asia/Seoul",                       "slug": "seoul"},
    "Shanghai":     {"name": "Pudong International Airport",         "lat": 31.1443, "lon": 121.8083, "unit": "celsius",    "tz": "Asia/Shanghai",                    "slug": "shanghai"},
    "Tokyo":        {"name": "Haneda Airport, Tokyo",                "lat": 35.5494, "lon": 139.7798, "unit": "celsius",    "tz": "Asia/Tokyo",                       "slug": "tokyo"},
    "Singapore":    {"name": "Changi Airport, Singapore",            "lat": 1.3644,  "lon": 103.9915, "unit": "celsius",    "tz": "Asia/Singapore",                   "slug": "singapore"},
    "Lucknow":      {"name": "Chaudhary Charan Singh Airport",       "lat": 26.7606, "lon": 80.8893,  "unit": "celsius",    "tz": "Asia/Kolkata",                     "slug": "lucknow"},
    "Wellington":   {"name": "Wellington Airport",                   "lat": -41.3272,"lon": 174.8053, "unit": "celsius",    "tz": "Pacific/Auckland",                 "slug": "wellington"},
}

CLIMATE_ZONES = {
    "US_East":        ["NYC", "Miami", "Atlanta"],
    "US_Central":     ["Chicago", "Dallas"],
    "US_West":        ["Seattle"],
    "Europe":         ["London", "Paris", "Munich"],
    "Middle_East":    ["Ankara", "Tel Aviv"],
    "Americas_South": ["Buenos Aires", "Sao Paulo"],
    "Americas_North": ["Toronto"],
    "East_Asia":      ["Seoul", "Shanghai", "Tokyo"],
    "South_Asia":     ["Singapore", "Lucknow"],
    "Oceania":        ["Wellington"],
}

# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: LADDER (BUY YES around ensemble median)
# ═══════════════════════════════════════════════════════════════════
LADDER_ENABLED = True
LADDER_MAX_ENTRY_PRICE = 0.25   # Buy below 25 cents (optimized from 0.20)
LADDER_BUCKETS = 2              # 2 buckets near median (optimized from 3)
LADDER_BET_PER_BUCKET = 1.50    # $1.50 per bucket (scaled for small bankroll)
LADDER_MAX_SETS_PER_CYCLE = 1   # 1 set per cycle

# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: CONSERVATIVE BUY NO
# ═══════════════════════════════════════════════════════════════════
ALLOW_BUY_NO = True
CONSERVATIVE_NO_MIN_ENTRY = 0.65  # NO price must be >= 65 cents (optimized from 0.55)
CONSERVATIVE_NO_MAX_ENTRY = 0.90  # NO price must be <= 90 cents (optimized from 0.85)
MIN_EDGE_PCT = 0.12              # Minimum 12% edge

# ═══════════════════════════════════════════════════════════════════
# POSITION SIZING & KELLY
# ═══════════════════════════════════════════════════════════════════
KELLY_FRACTION = 0.20           # Optimized from 0.15
MAX_POSITION_PCT = 0.15
MAX_CONCURRENT_POSITIONS = 25
MAX_TOTAL_EXPOSURE = 0.60        # 60% max exposure
MIN_TRADE_SIZE_USDC = 1.0
MAX_TRADE_SIZE_USDC = 5.0

# ═══════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
BANKROLL = float(os.getenv("POLYMARKET_BANKROLL", "100.0"))
MAX_DAILY_LOSS_USDC = 20.0
MAX_DRAWDOWN_PCT = 0.20
TAKE_PROFIT_EDGE_PCT = 0.03
EXIT_HOURS_BEFORE_RESOLUTION = 3
MAX_POSITIONS_PER_ZONE = 3

# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE CONFIDENCE SCALING
# ═══════════════════════════════════════════════════════════════════
ENSEMBLE_SPREAD_LOW_STD = 2.5
ENSEMBLE_SPREAD_HIGH_STD = 5.0
ENSEMBLE_HIGH_CONF_MULT = 1.5
ENSEMBLE_LOW_CONF_MULT = 0.5

MODEL_DISAGREEMENT_THRESH_F = 6.0
MODEL_AGREEMENT_THRESH_F = 3.0
DISAGREEMENT_SIZING_MULT = 0.7
AGREEMENT_SIZING_MULT = 1.8

# Time decay
TIME_DECAY_FULL_DAYS = 2
TIME_DECAY_MED_DAYS = 4
TIME_DECAY_MED_MULT = 0.8
TIME_DECAY_FAR_MULT = 0.5
TIME_DECAY_FAR_MIN_EDGE = 0.20

# Market efficiency
MARKET_EXPECTED_SUM = 1.05
MARKET_SHARP_THRESHOLD = 0.03
MARKET_SOFT_THRESHOLD = 0.10
MARKET_SHARP_EDGE_MULT = 1.5

# ═══════════════════════════════════════════════════════════════════
# OPERATIONAL
# ═══════════════════════════════════════════════════════════════════
SCAN_INTERVAL_SECONDS = 900      # 15 min
SCAN_DAYS_AHEAD = 2              # 1-2 days (highest accuracy)
RESULTS_DIR = "results"
DATA_DIR = "data"

# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: LATE SNIPER
# ═══════════════════════════════════════════════════════════════════
# Uses tighter probability distribution (~1° MAE) near resolution.
# Trades at late prices (80% through market lifetime ≈ 6h before resolution)
# when CLOB prices haven't caught up to improved forecast accuracy.
LATE_SNIPER_ENABLED = True
SNIPER_MIN_EDGE = 0.13             # Minimum 13% edge for sniper
SNIPER_MAX_YES_ENTRY = 0.35        # Max price to buy YES (sniper)
SNIPER_MIN_NO_ENTRY = 0.55         # Min NO entry price for sniper
SNIPER_MAX_NO_ENTRY = 0.88         # Max NO entry price for sniper
SNIPER_BET_PCT = 0.03              # 3% of bankroll per sniper trade
SNIPER_CONFIDENCE_MULT = 1.5       # Size multiplier (higher confidence late)
SNIPER_MAX_BETS = 4                # Max sniper trades per market
SNIPER_STD_F = 1.2                 # Tighter std for F cities (vs ~3° early)
SNIPER_STD_C = 0.7                 # Tighter std for C cities (vs ~1.5° early)
SNIPER_MIN_HOURS = 4               # Only snipe within last N hours
SNIPER_PRICE_TIMING = 0.80         # Use price from 80% through history

# Backtesting defaults
BACKTEST_INITIAL_BANKROLL = 100.0
SIM_SPREAD = 0.06
SIM_SLIPPAGE = 0.02

# Optimized entry timing: use early prices (better fills before market sharpens)
# V6.2 Real backtest: 1105 trades, 58.4% WR, PF 2.38, +3009% ROI
ENTRY_TIMING = "early"
