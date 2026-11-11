# TradePulse: Intraday AI Directional Trading Indicator

> **Institution:** IIT Patna, IMPACT Lab — End-to-End Machine Learning Guidelines  
> **Asset:** META Platforms Inc. (NASDAQ)  
> **Timeframe:** 5-Minute Intraday Bars  
> **Horizon:** 15 Minutes (3 consecutive bars)  
> **Feed Source:** Alpaca Market Data v2 (`feed=iex`)  

TradePulse is an end-to-end, scientifically validated machine learning indicator designed to predict short-term directional movement of META stock over a 15-minute horizon. It features a complete financial charting terminal powered by **TradingView Lightweight Charts v4.2.1** and a unified **FastAPI + React 18** architecture with zero repainting and verifiable cryptographic audit logs.

---

## 1. Quick Start (One-Command Launch)

TradePulse runs as a unified full-stack application on a single port (`http://localhost:8000`). Both the FastAPI REST/SSE backend and the compiled React production application are served concurrently without CORS friction.

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ and npm (only needed if modifying frontend code; production bundle is pre-built in `frontend/dist/`)

### Setup & Launch
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment credentials
cp .env.example .env
# Edit .env with your Alpaca API credentials (or run in historical simulation mode)

# 4. Launch the application
python run_app.py
```

Open your browser to **`http://localhost:8000`** (the launcher will automatically open your default browser).

---

## 2. Architecture & Data Flow

```text
TradePulse/
├── artifacts/
│   ├── models/                <- model_final.joblib, scaler_final.joblib, calibrator_final.joblib
│   └── metadata/              <- ui_summary_final.json, signal_rule_final.json, feature_schema_final.json
├── backend/app/
│   ├── main_final.py          <- FastAPI application entrypoint (mounts frontend/dist)
│   ├── routes/api_v1.py       <- REST and SSE streaming router
│   ├── schemas/api_schemas.py <- Pydantic request/response schemas
│   └── services/              <- live_inference_final.py, background_worker_final.py, mock_service.py
├── frontend/
│   ├── src/                   <- React 18 + Vite 6 + TypeScript 5 components
│   └── dist/                  <- Compiled production web assets
├── data/
│   ├── processed/             <- META_5min_clean.csv (validated historical dataset)
│   ├── live_signals_final.jsonl <- Append-only live audit log
│   └── live_outcomes_final.jsonl<- Evaluated 15-min trade outcomes
├── ml/                        <- Single shared pipeline: predict_final.py, features.py
├── reports/                   <- ui_log.md, trader_user_guide.md, ui_screenshots/
├── run_app.py                 <- Unified local launcher
└── requirements.txt
```

---

## 3. Configuration & `.env` Settings

TradePulse loads settings from `.env`. An example template is provided in `.env.example`:

```ini
# Alpaca Market Data v2 Credentials
APCA_API_KEY_ID="your_alpaca_key_id_here"
APCA_API_SECRET_KEY="your_alpaca_secret_key_here"
APCA_API_BASE_URL="https://paper-api.alpaca.markets"

# Data Feed Configuration: 'alpaca' (live IEX feed) or 'historical' (offline replay)
TRADEPULSE_FEED_SOURCE="alpaca"

# Live Testing Configuration
TRADEPULSE_ASSET="META"
TRADEPULSE_TIMEFRAME="5Min"
TRADEPULSE_HORIZON_BARS=3
```

*Note: If Alpaca API keys are left empty, TradePulse automatically falls back to historical simulation mode, enabling full demonstration outside US market hours.*

---

## 4. Key Trader Features & Operational Modes

### 4.1 Live Mode
- **Clock-Aligned Execution:** Background daemon wakes 3 seconds past every 5-minute candle close (`:00`, `:05`, `:10`, ..., `:55` + 3s buffer) during regular market hours (9:30 AM – 4:00 PM ET).
- **Auto Mode (`A` key):** Connects to Server-Sent Events (`/api/live/stream`); chart and prediction cards update instantaneously upon candle close.
- **Manual Test Now (`T` key):** Fetches the latest closed bar, computes all 29 causal features, and displays the calibrated probability.

### 4.2 Replay Mode (Instructor Demo Outside Market Hours)
- **Live / Replay Toggle (`R` key):** Switches between live stream and historical session playback.
- **Unmissable Amber Banner:** Permanent warning banner distinguishes replay data from real market feeds.
- **Speed Multipliers:** Playback at `1x`, `5x`, or `20x` speed with Play, Pause, and Step (`+5m`) controls.
- **Separate Replay Audit Log:** Replay predictions write strictly to `data/replay_signals.jsonl`, ensuring official live testing logs are never polluted.

### 4.3 Scoreboard & Model Card
- **Real-Time Scoreboard:** Tracks emitted signals vs expected daily frequency (37.5 signals/day) and computes realized win rate vs the Always-Up baseline.
- **Collapsible Model Card:** Displays full provenance, feature counts, training dates, and held-out test metrics with 95% day-block bootstrap confidence intervals.

---

## 5. Audit Log & Zero-Repainting Guarantee

Signals are logged to `data/live_signals_final.jsonl` in append-only JSON Lines format. Each line contains a SHA-256 tamper-evident hash:

```json
{
  "timestamp": "2026-11-06 15:55:00-05:00",
  "close_price": 612.45,
  "direction": "Up",
  "signal_code": 1,
  "confidence_pct": 68.4,
  "conviction": 0.3541,
  "threshold": 0.2635,
  "is_gated": true,
  "probabilities": {"down": 0.2215, "neutral": 0.4244, "up": 0.3541},
  "model_version": "TradePulse-v1.0-final",
  "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

- **Idempotence:** Querying the same closed bar returns the stored record; it is never recomputed or altered.
- **Immutability:** Restarting the server preserves earlier log lines byte-for-byte.

---

## 6. Verification & Automated Test Suites

TradePulse contains three comprehensive automated verification suites:

```bash
# 1. Parity & Resilience Suite (Feature parity, zero repainting, gaps, restarts)
.venv/Scripts/python -m unittest tests/test_parity_final.py -v

# 2. API Contract & Build Suite (Pydantic schemas, OpenAPI contracts, UI bundle)
.venv/Scripts/python -m unittest tests/test_ui_api_contract_final.py -v

# 3. Phase 3 Replay & Metadata Suite (Replay isolation, model card disclosures)
.venv/Scripts/python -m unittest tests/test_phase3_features_final.py -v
```

### Measured Benchmark Parity:
- **Feature Parity:** Max feature difference across all 29 causal indicators: **$5.45 \times 10^{-6}$** (target strictly $< 1 \times 10^{-5}$).
- **Signal Parity:** **0 mismatches / 780 bars** across 10 complete trading days (100.0% signal match).
- **Restart Audit Invariance:** 100% byte-for-byte identical output for pre-crash records.
- **Test Now Latency:** $< 45$ ms execution time.

---

## 7. Honest Academic Disclosure & Limitations

In accordance with IIT Patna IMPACT Lab guidelines:

> [!WARNING]
> **No Profitable Edge After Costs Was Demonstrated.**  
> While TradePulse demonstrates statistically significant directional predictive capability (achieving **54.11% binary sign accuracy** on held-out data with a 95% bootstrap CI of `[50.82%, 57.22%]`), the average gross return of **+1.29 bps per signal** is completely absorbed by institutional round-trip execution costs of **8.00 bps**. Net expected return after transaction costs is **-6.71 bps per signal** (`edge_demonstrated = false`).

### Operational Limitations:
1. **Paper Trading Only:** No order routing or brokerage trade placement functionality is implemented.
2. **Technical Indicators Only:** The model does not read news headlines, earnings reports, or regulatory filings.
3. **Session Open/Close Regimes:** Signals generated during the first 15 minutes (09:30–09:45 ET) and last 15 minutes (15:45–16:00 ET) are subject to extreme auction spreads and volatility distortions.
