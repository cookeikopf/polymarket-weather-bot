#!/usr/bin/env python3
"""
VPS Diagnostic Script
======================
Run this on the VPS to find out exactly why the bot finds 0 edges.
Tests each component independently.

Usage: python diagnose.py
"""
import sys
import os
import json
import time
import traceback

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_section(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  ✅ {msg}")

def warn(msg):
    print(f"  ⚠️  {msg}")

def fail(msg):
    print(f"  ❌ {msg}")

issues = []

# ═══════════════════════════════════════════════════════
# TEST 1: Python dependencies
# ═══════════════════════════════════════════════════════
test_section("Python Dependencies")
try:
    import numpy as np
    ok(f"numpy {np.__version__}")
except ImportError:
    fail("numpy not installed!")
    issues.append("CRITICAL: numpy not installed")

try:
    import pandas as pd
    ok(f"pandas {pd.__version__}")
except ImportError:
    fail("pandas not installed!")
    issues.append("CRITICAL: pandas not installed")

try:
    from scipy import stats
    import scipy
    ok(f"scipy {scipy.__version__}")
    # Test t-distribution sampling
    samples = stats.t.rvs(df=5, loc=0, scale=3.5, size=100)
    ok(f"scipy.stats.t.rvs works (mean={samples.mean():.1f}, std={samples.std():.1f})")
except ImportError:
    fail("scipy not installed!")
    issues.append("CRITICAL: scipy not installed")

try:
    from sklearn.isotonic import IsotonicRegression
    import sklearn
    ok(f"scikit-learn {sklearn.__version__}")
except ImportError:
    warn("scikit-learn not installed (calibration will use defaults)")

try:
    import requests
    ok(f"requests {requests.__version__}")
except ImportError:
    fail("requests not installed!")
    issues.append("CRITICAL: requests not installed")

try:
    import lightgbm
    ok(f"lightgbm {lightgbm.__version__}")
except ImportError:
    warn("lightgbm not installed (ML model won't load)")

# ═══════════════════════════════════════════════════════
# TEST 2: Config
# ═══════════════════════════════════════════════════════
test_section("Configuration")
try:
    import config
    ok(f"config.py loaded")
    ok(f"MIN_EDGE_PCT = {config.MIN_EDGE_PCT}")
    ok(f"MIN_ENSEMBLE_AGREEMENT = {config.MIN_ENSEMBLE_AGREEMENT}")
    ok(f"STATIONS = {len(config.STATIONS)} cities")
    ok(f"WEATHER_MODELS = {len(config.WEATHER_MODELS)} models")
    ok(f"LIVE_BANKROLL = ${config.LIVE_BANKROLL}")
    ok(f"BACKTEST_INITIAL_BANKROLL = ${config.BACKTEST_INITIAL_BANKROLL}")
    ok(f"SCAN_INTERVAL = {config.SCAN_INTERVAL_SECONDS}s")
    ok(f"MARKET_SCAN_DAYS_AHEAD = {config.MARKET_SCAN_DAYS_AHEAD}")
except Exception as e:
    fail(f"config.py failed: {e}")
    issues.append(f"CRITICAL: config.py error: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════
# TEST 3: ML Model
# ═══════════════════════════════════════════════════════
test_section("ML Model")
ml_path = config.ML_MODEL_PATH
if os.path.exists(ml_path):
    ok(f"ML model file exists: {ml_path} ({os.path.getsize(ml_path)} bytes)")
    try:
        from ml_model import WeatherMLModel
        ml = WeatherMLModel()
        ml.load(ml_path)
        if ml.is_trained:
            ok("ML model loaded and is_trained=True")
        else:
            warn("ML model loaded but is_trained=False")
    except Exception as e:
        warn(f"ML model file exists but failed to load: {e}")
else:
    warn(f"ML model file NOT found at {ml_path}")
    warn("Bot will use NWP ensemble only (still should find edges)")

# ═══════════════════════════════════════════════════════
# TEST 4: Gamma API (Market Discovery)
# ═══════════════════════════════════════════════════════
test_section("Gamma API (Market Discovery)")
import requests
from datetime import datetime, timedelta

today = datetime.now()
target = today + timedelta(days=1)
month_name = target.strftime("%B").lower()
day = target.day
year = target.year
test_slug = f"highest-temperature-in-nyc-on-{month_name}-{day}-{year}"

try:
    resp = requests.get(f"{config.GAMMA_API_HOST}/events", 
                       params={"slug": test_slug}, timeout=15)
    if resp.status_code == 200:
        events = resp.json()
        if events:
            event = events[0] if isinstance(events, list) else events
            markets = event.get("markets", [])
            ok(f"Gamma API working! Found event: {event.get('title', '?')}")
            ok(f"  {len(markets)} sub-markets")
            
            # Parse first outcome
            if markets:
                sm = markets[0]
                outcome_prices = json.loads(sm.get("outcomePrices", "[]"))
                ok(f"  First outcome price: {outcome_prices}")
        else:
            warn(f"Gamma API returned empty for slug: {test_slug}")
            warn(f"  This could mean no market exists for tomorrow. Try a different day.")
    else:
        fail(f"Gamma API returned status {resp.status_code}")
        issues.append(f"Gamma API error: HTTP {resp.status_code}")
except Exception as e:
    fail(f"Gamma API failed: {e}")
    issues.append(f"Gamma API unreachable: {e}")

# ═══════════════════════════════════════════════════════
# TEST 5: Open-Meteo API (Weather Forecasts)
# ═══════════════════════════════════════════════════════
test_section("Open-Meteo API (Weather Forecasts)")
target_date = target.strftime("%Y-%m-%d")
forecast_count = 0
forecast_failures = 0

for model in config.WEATHER_MODELS:
    try:
        params = {
            "latitude": 40.7769, "longitude": -73.8740,
            "daily": "temperature_2m_max",
            "timezone": "America/New_York",
            "start_date": target_date, "end_date": target_date,
            "temperature_unit": "fahrenheit",
        }
        if model != "best_match":
            params["models"] = model
        
        resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "daily" in data and "temperature_2m_max" in data["daily"]:
                vals = data["daily"]["temperature_2m_max"]
                if vals and vals[0] is not None:
                    forecast_count += 1
                    if model in ["best_match", "ecmwf_ifs025", "gfs_seamless"]:
                        ok(f"{model}: {vals[0]}°F")
                else:
                    warn(f"{model}: returned null value")
                    forecast_failures += 1
            else:
                warn(f"{model}: missing daily data in response")
                forecast_failures += 1
        elif resp.status_code == 429:
            fail(f"{model}: RATE LIMITED (HTTP 429)")
            forecast_failures += 1
            issues.append("Open-Meteo rate limited!")
        else:
            warn(f"{model}: HTTP {resp.status_code}")
            forecast_failures += 1
        time.sleep(0.3)
    except Exception as e:
        fail(f"{model}: {e}")
        forecast_failures += 1

print(f"\n  Forecasts: {forecast_count}/{len(config.WEATHER_MODELS)} succeeded, {forecast_failures} failed")
if forecast_count == 0:
    issues.append("CRITICAL: ALL Open-Meteo forecasts failed!")
elif forecast_count < 3:
    issues.append(f"WARNING: Only {forecast_count} forecasts succeeded (need good ensemble)")

# ═══════════════════════════════════════════════════════
# TEST 6: CLOB API (Orderbook Prices)
# ═══════════════════════════════════════════════════════
test_section("CLOB API (Orderbook Prices)")
try:
    # Get a real token ID from the market we found
    if events and markets:
        sm = markets[0]
        token_ids = json.loads(sm.get("clobTokenIds", "[]"))
        if token_ids:
            token_id = token_ids[0]
            
            # Test /price endpoint
            buy_resp = requests.get(f"{config.POLYMARKET_HOST}/price",
                                   params={"token_id": token_id, "side": "BUY"}, timeout=5)
            sell_resp = requests.get(f"{config.POLYMARKET_HOST}/price",
                                    params={"token_id": token_id, "side": "SELL"}, timeout=5)
            
            if buy_resp.status_code == 200 and sell_resp.status_code == 200:
                buy_price = buy_resp.json().get("price", "?")
                sell_price = sell_resp.json().get("price", "?")
                ok(f"CLOB /price endpoint working! BUY={buy_price}, SELL={sell_price}")
            else:
                warn(f"CLOB /price returned: BUY={buy_resp.status_code}, SELL={sell_resp.status_code}")
            
            # Test /book endpoint
            book_resp = requests.get(f"{config.POLYMARKET_HOST}/book",
                                    params={"token_id": token_id}, timeout=5)
            if book_resp.status_code == 200:
                book = book_resp.json()
                ok(f"CLOB /book endpoint working! {len(book.get('bids',[]))} bids, {len(book.get('asks',[]))} asks")
            else:
                warn(f"CLOB /book returned: {book_resp.status_code}")
    else:
        warn("Skipping CLOB test (no market found)")
except Exception as e:
    fail(f"CLOB API failed: {e}")
    issues.append(f"CLOB API error: {e}")

# ═══════════════════════════════════════════════════════
# TEST 7: Full Probability + Edge Pipeline
# ═══════════════════════════════════════════════════════
test_section("Full Edge Detection Pipeline (NYC)")
try:
    from weather_engine import WeatherEngine
    from edge_detector import EdgeDetector
    from market_scanner import MarketScanner
    
    print("  Initializing NYC weather engine...")
    engine = WeatherEngine("NYC")
    engine.calibrate()
    
    print("  Fetching market...")
    scanner = MarketScanner()
    market = scanner._fetch_event_by_slug(test_slug, "NYC", target_date, "temperature_max")
    
    if not market:
        fail("Could not fetch test market!")
        issues.append("Market fetch failed in pipeline test")
    else:
        ok(f"Market: {market.question} ({len(market.outcomes)} outcomes)")
        
        print("  Fetching forecasts...")
        forecasts = engine.fetch_multi_model_forecasts(target_date)
        ok(f"Got {len(forecasts)} model forecasts")
        
        if not forecasts:
            fail("No forecasts returned!")
            issues.append("CRITICAL: No forecasts in pipeline test")
        else:
            ensemble = engine.compute_ensemble_stats(forecasts)
            ok(f"Ensemble: mean={ensemble['mean']:.1f}°F, agreement={ensemble['agreement']:.1%}")
            
            # Extract bucket edges manually (same as bot)
            import re
            range_lows = []
            for outcome in market.outcomes:
                name = outcome.name
                if re.search(r'or\s*(below|higher)', name, re.I):
                    continue
                mf = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', name)
                if mf:
                    range_lows.append(int(mf.group(1)))
                    continue
                mc = re.search(r'(-?\d+)\s*°\s*C', name)
                if mc:
                    range_lows.append(int(mc.group(1)))
            
            range_lows.sort()
            step = 2  # NYC is °F
            edges = []
            for low in range_lows:
                if low not in edges: edges.append(low)
                top = low + step
                if top not in edges: edges.append(top)
            edges.sort()
            ok(f"Bucket edges: {edges}")
            
            our_probs = engine.compute_probability_distribution(forecasts, edges)
            ok(f"Computed {len(our_probs)} bucket probabilities")
            
            # Show top probabilities
            sorted_probs = sorted(our_probs.items(), key=lambda x: -x[1])
            for label, prob in sorted_probs[:5]:
                print(f"    {label}: {prob:.3f} ({prob*100:.1f}%)")
            
            # Check edge matching
            edge_detector = EdgeDetector()
            relaxed = max(0.02, config.MIN_EDGE_PCT - 0.02)
            matched = 0
            unmatched = 0
            candidates = 0
            best_edge = 0
            
            for outcome in market.outcomes:
                our_prob = edge_detector._match_probability(outcome.name, our_probs)
                if our_prob is None:
                    unmatched += 1
                    warn(f"  NO MATCH: '{outcome.name}' vs our keys: {list(our_probs.keys())[:3]}...")
                    continue
                matched += 1
                gamma_price = outcome.price
                if gamma_price <= 0 or gamma_price >= 1:
                    continue
                edge = abs(our_prob - gamma_price)
                best_edge = max(best_edge, edge)
                if edge >= relaxed:
                    candidates += 1
            
            print(f"\n  Matching: {matched} matched, {unmatched} unmatched out of {len(market.outcomes)} outcomes")
            print(f"  Best edge: {best_edge:.3f} ({best_edge*100:.1f}%)")
            print(f"  Candidates (edge >= {relaxed*100:.0f}%): {candidates}")
            
            if candidates > 0:
                ok(f"PIPELINE WORKS! Found {candidates} candidate edges")
            elif matched == 0:
                fail("NO OUTCOMES MATCHED probability labels!")
                issues.append("CRITICAL: Probability label mismatch — outcome names don't match our bucket labels")
            elif best_edge < relaxed:
                fail(f"All edges below threshold (best: {best_edge*100:.1f}%)")
                issues.append(f"All edges too small — best was {best_edge*100:.1f}%")
            
except Exception as e:
    fail(f"Pipeline test failed: {e}")
    traceback.print_exc()
    issues.append(f"Pipeline crash: {e}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print(f"  DIAGNOSTIC SUMMARY")
print(f"{'='*60}")

if not issues:
    ok("ALL TESTS PASSED!")
    print("\n  If the bot still finds 0 edges, check the VPS logs for")
    print("  the new detailed skip_reasons output after this update.")
else:
    print(f"\n  Found {len(issues)} issues:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print(f"\n{'='*60}")
