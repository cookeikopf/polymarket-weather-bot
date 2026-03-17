# Polymarket Weather Prediction Bot 🌡️📈

Ein hochprofitabler Weather Prediction Trading Bot für Polymarket, basierend auf einem revolutionären Multi-Model Ensemble Approach mit 20-City Coverage.

## Revolutionärer Ansatz

### 1. Multi-Model Ensemble Forecasting (8 globale NWP-Modelle)
- **ECMWF IFS** (European Centre) - bestes globales Modell
- **GFS** (NOAA/USA) - US-Flaggschiff
- **ICON** (DWD/Deutschland) - exzellent für Mitteleuropa
- **GEM** (Kanada), **Meteo-France ARPEGE**, **JMA** (Japan), **UKMO** (UK)
- Automatische Gewichtung nach historischer Genauigkeit (inverse RMSE)

### 2. Bayesian Probability Calibration
- 3 Jahre historische Forecast-vs-Actual Daten
- Bias-Korrektur pro Modell (z.B. ECMWF: +1.08°F Bias in NYC)
- Student's t-Verteilung für realistische Tail-Risiken
- Monte Carlo Simulation (10.000 Samples) für Wahrscheinlichkeitsverteilung

### 3. Edge Detection & Kelly Sizing
- Vergleich unserer ML-Wahrscheinlichkeiten vs. Marktpreise
- Minimum 7% Edge bevor ein Trade eingegangen wird
- Fractional Kelly Criterion (0.20x) für konservatives Position Sizing
- Max 8% pro Position, 40% Gesamtexposure

### 4. Multi-City Expansion (20 Städte weltweit)
- Alle 20 Polymarket Weather Cities integriert
- Pro Stadt: individuelle Kalibrierung, Units (°F/°C), Zeitzonen
- °C-Städte generieren ~28x mehr Trades als °F-Städte (schmalere Buckets)
- Diversifikation über 6 Kontinente = geringeres Risiko, höheres Volumen

## Multi-City Backtest (Jan 2025 - März 2026)

### Gesamtergebnisse (20 Städte)

| Metrik | Wert |
|--------|------|
| Total Trades | 684 |
| Win Rate | 76.3% |
| Avg Edge per Trade | 17.6% |
| Trades pro Kalendertag | 1.6 |
| Trades pro Trading Day | 3.4 |
| Trades pro Woche | 10.9 |
| Trades pro Monat | 46.8 |
| Trading-Tage | 46% aller Tage |
| Total P&L ($1000/Stadt) | $118,234 |
| Avg P&L pro Trade | $172.86 |

### Top 15 Performing Cities

| Stadt | Trades | Return | Win% | Profit Factor | Edge |
|-------|--------|--------|------|---------------|------|
| Toronto | 73 | +1450% | 70% | 7.6 | 16.8% |
| Tokyo | 80 | +1385% | 70% | 7.3 | 17.8% |
| Shanghai | 48 | +1051% | 75% | 10.9 | 19.5% |
| Ankara | 36 | +1013% | 72% | 13.6 | 17.6% |
| Buenos Aires | 55 | +980% | 78% | 9.6 | 17.9% |
| München | 62 | +973% | 77% | 8.3 | 16.5% |
| São Paulo | 46 | +840% | 72% | 8.7 | 16.5% |
| Paris | 38 | +832% | 79% | 12.6 | 22.1% |
| Seoul | 25 | +630% | 68% | 11.3 | 15.4% |
| Singapore | 43 | +592% | 74% | 7.2 | 19.3% |
| Lucknow | 42 | +504% | 83% | 9.9 | 14.9% |
| London | 38 | +473% | 92% | 18.7 | 19.3% |
| Wellington | 35 | +448% | 86% | 10.0 | 21.5% |
| Tel Aviv | 39 | +419% | 77% | 7.2 | 16.0% |
| NYC | 24 | +235% | 83% | 11.9 | 11.1% |

**Note:** US °F-Städte (Miami, Atlanta, Chicago, Dallas, Seattle) generieren 0 Trades weil 2°F Buckets zu breit sind. Nur NYC schafft einige Trades.

### Wichtige Erkenntnis: °C vs °F

- **°C Städte**: Durchschnitt 45 Trades/Stadt → 1°C Buckets = engere Ranges = mehr Edges
- **°F Städte (ohne NYC)**: 0 Trades → 2°F ≈ 1.1°C Buckets sind zu breit
- **NYC (°F)**: 24 Trades → einzige °F Stadt mit Trades dank höherer Variabilität

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Private Key eintragen in .env (nur für Live-Trading)
```

## Verwendung

```bash
# Status anzeigen
python main.py status

# Kalibrierung ausführen
python main.py calibrate

# Backtest ausführen (einzelne Stadt)
python main.py backtest

# Multi-City Backtest (alle 20 Städte)
python multi_city_backtest.py

# Parameter-Optimierung
python main.py optimize

# Aktive Weather-Märkte scannen (Paper Mode)
python main.py scan

# Paper Trading (X Minuten)
python main.py paper 360

# Live Trading (ECHTES GELD!)
python main.py live 360
```

## Architektur

```
config.py                - Alle 20 Städte + Parameter zentral konfigurierbar
weather_engine.py        - Multi-Model Ensemble + Kalibrierung (per-city)
market_scanner.py        - Polymarket Market Discovery (Gamma API)
edge_detector.py         - Edge Detection + Signal Generation
backtester.py            - Backtesting Engine (multi-city fähig)
optimizer.py             - Parameter-Optimierung (5 Profile)
live_trader.py           - Paper/Live Trading Execution
main.py                  - CLI Entry Point
multi_city_backtest.py   - 20-City Backtest + Charts
final_analysis.py        - Performance Charts + Report
```

## Risikomanagement

- **Max Drawdown**: 15-20% → Bot stoppt automatisch
- **Max Position**: 8% des Bankrolls
- **Max Exposure**: 40% total
- **Trailing Stop**: 30% des Peak-Profits
- **Exit**: 2h vor Market Resolution
- **Kein Fee**: Weather-Märkte haben keine Trading-Fees auf Polymarket

## Hinweise für echtes Trading

1. **Paper Trading zuerst**: Mindestens 2-4 Wochen mit echten Marktdaten
2. **Klein anfangen**: $50-100 initial
3. **Fokus auf °C Städte**: Toronto, Tokyo, München, Buenos Aires, Shanghai — die bringen das Volumen
4. **Liquidität beachten**: Weather-Märkte haben $2.5K-$15K Volumen
5. **Re-Kalibrierung**: Monatlich Model Weights aktualisieren
6. **Multi-City ist Key**: 15 aktive Städte = ~11 Trades/Woche statt ~0.6 mit nur NYC
7. **Geo-Token**: Ggf. VPN nötig je nach Standort
