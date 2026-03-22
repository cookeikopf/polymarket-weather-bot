# Polymarket Weather Bot V6

Automated weather temperature trading bot for Polymarket prediction markets.

## Strategy

**Dual-strategy approach** exploiting temperature forecast edges:

1. **Ladder (BUY YES):** Buys YES on 3 temperature buckets near the ensemble forecast median at prices below $0.20. One winning bucket pays 5-50x, covering all misses.

2. **Conservative NO (BUY NO):** Buys NO on unlikely temperature outcomes (far from forecast) at entry prices of $0.55–$0.85. High win rate, lower payout per trade.

## Key Features

- Multi-model ensemble forecasting (ECMWF, GFS, ICON, GEM, Meteo-France, JMA, UKMO)
- KDE-based bucket probability computation with ensemble member weighting
- Live CLOB orderbook integration with effective price discovery
- Adaptive order placement (taker for strong edges, maker for weak)
- Full risk management: Kelly sizing, daily loss limits, max drawdown, zone capacity
- Paper + live trading modes
- Historical backtester using Open-Meteo archive + historical forecast APIs
- Gasless trading via Builder API
- Geoblocking detection + automatic fallback

## Architecture

```
config.py     — All configuration, 20 weather stations, strategy params
utils.py      — Logging helpers
weather.py    — Open-Meteo API (forecast, ensemble, historical, archive)
markets.py    — Gamma API + CLOB (market discovery, orderbook)
strategy.py   — Edge detection (ladder + conservative NO)
trader.py     — Execution, positions, P&L, state persistence
backtest.py   — Historical data backtester
main.py       — CLI entry point
```

## Setup

```bash
# Clone
git clone https://github.com/cookeikopf/polymarket-weather-bot.git
cd polymarket-weather-bot

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your credentials
```

## Usage

```bash
# Paper trading (24h continuous)
python main.py paper

# Single scan cycle
python main.py scan

# Quick backtest (14 days, 5 stations)
python main.py backtest

# Full backtest (60 days, all stations)
python main.py backtest full 60

# Live trading (real USDC!)
python main.py live

# Check status
python main.py status
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `POLYMARKET_PRIVATE_KEY` | Live only | Ethereum private key |
| `POLYMARKET_FUNDER_ADDRESS` | Live only | Polygon funder address |
| `POLYMARKET_SIGNATURE_TYPE` | No | Default: 2 |
| `POLYMARKET_BANKROLL` | No | Starting bankroll (default: 100) |
| `POLY_BUILDER_API_KEY` | No | Builder API for gasless trading |
| `POLY_BUILDER_SECRET` | No | Builder API secret |
| `POLY_BUILDER_PASSPHRASE` | No | Builder API passphrase |
| `OPEN_METEO_API_KEY` | Recommended | Commercial API key (€120/mo) |
| `ORDER_STRATEGY` | No | `taker` / `maker` / `adaptive` (default) |

## Covered Markets

20 cities across 10 climate zones: NYC, Miami, Atlanta, Chicago, Dallas, Seattle, London, Paris, Munich, Ankara, Tel Aviv, Toronto, Buenos Aires, São Paulo, Seoul, Shanghai, Tokyo, Singapore, Lucknow, Wellington.

## Risk Parameters

- Max daily loss: $20
- Max drawdown: 20%
- Max exposure: 60% of bankroll
- Max concurrent positions: 25
- Kelly fraction: 0.15
- Ladder: $2 per bucket, max entry $0.20
- Conservative NO: min entry $0.55, min edge 12%
