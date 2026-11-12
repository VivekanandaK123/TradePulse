# TradePulse: Trader Interface UI & API Execution Log

**Project:** TradePulse Intraday Directional Indicator (META, NASDAQ, 5-Min Bars)  
**Laboratory:** IIT Patna, IMPACT Lab  
**Log Start Date:** 2026-11-11 23:20:00 IST  

---

### Phase 0: Artifact Inventory & API Contract Definition (2026-11-11 23:35:00 IST)

#### 1. Artifacts Selected & Verified:
All selected model and metadata artifacts strictly adhere to the non-negotiable `_final` provenance:
- [`artifacts/models/model_final.joblib`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/models/model_final.joblib): Refitted production ensemble (50% RF + 50% LR) on all 253 complete sessions ($N=18,868$ valid bars). SHA-256: `07ef98a8...`
- [`artifacts/models/scaler_final.joblib`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/models/scaler_final.joblib): Fitted `RobustScaler` on all 18,868 rows. SHA-256: `0f178107...`
- [`artifacts/models/calibrator_final.joblib`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/models/calibrator_final.joblib): Platt Sigmoid OVR calibrator fitted on 11,317 OOF bars. SHA-256: `77ae883c...`
- [`artifacts/metadata/feature_schema_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/feature_schema_final.json): 29 causal feature schema with warm-up lookback requirements. SHA-256: `39e684a7...`
- [`artifacts/metadata/signal_rule_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/signal_rule_final.json): Gating conviction threshold $\tau_{\text{final}} = 0.2635$ targeting 50.0% signal coverage (~37.5 signals/day). SHA-256: `43630557...`
- [`artifacts/metadata/final_config_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/final_config_final.json): Authoritative frozen pipeline architecture. SHA-256: `a324e521...`
- [`artifacts/metadata/training_manifest_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/training_manifest_final.json): Complete training provenance manifest. SHA-256: `b57ee06e...`
- [`artifacts/metadata/phase2_test_results.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/phase2_test_results.json): One-time held-out test evaluation results ($N=3,525$ bars).
- [`artifacts/metadata/split_config.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/split_config.json): Chronological split boundaries (Train, Embargo 1, Validation, Embargo 2, Test).

#### 2. Reusability Analysis:
- **Backend Reusability:**
  - [`ml/predict_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/predict_final.py): Directly leveraged by backend services for closed-candle feature computation and calibrated probability inference.
  - [`backend/app/services/live_inference_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/live_inference_final.py): Reused for buffer maintenance, missing-bar gap synthesis (`is_filled=1`), immutable append-only JSONL logging, and 15-minute outcome resolution.
  - [`backend/app/routes/signals_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/routes/signals_final.py): Preserved intact under `/api/v1/`.
  - [`data/processed/META_5min_clean.csv`](file:///e:/Projects/Trading%20Project/TradePulse/data/processed/META_5min_clean.csv): Serves ~19.6k historical 5-minute bars for the historical training chart.
  - [`data/live_signals_final.jsonl`](file:///e:/Projects/Trading%20Project/TradePulse/data/live_signals_final.jsonl) & [`data/live_outcomes_final.jsonl`](file:///e:/Projects/Trading%20Project/TradePulse/data/live_outcomes_final.jsonl): Reused as authoritative audit logs.
- **Frontend State:**
  - `frontend/` contains only `.gitkeep`. Clean slate ready for React + Vite + TypeScript setup in Phase 1.

#### 3. Created Files:
- [`artifacts/metadata/ui_summary_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/ui_summary_final.json): Single authoritative machine-readable source of truth for all UI metrics and metadata. Zero metrics hardcoded in UI. Contains `edge_demonstrated = false` with honest disclosure.
- [`.env.example`](file:///e:/Projects/Trading%20Project/TradePulse/.env.example): Documents Alpaca API keys and feed configuration.
- [`backend/app/schemas/api_schemas.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/schemas/api_schemas.py): Complete Pydantic schemas enforcing UTC epoch seconds (`int`) for all timestamps.
- [`backend/app/services/market_calendar.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/market_calendar.py): Real-time NASDAQ market hours, early close, and open/close countdown calculator using `pandas-market-calendars`.
- [`backend/app/services/mock_service.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/mock_service.py): High-fidelity mock backend implementation serving realistic historical data, split period bands, live prediction payloads, and replay states.
- [`backend/app/routes/api_v1.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/routes/api_v1.py): REST and SSE streaming router mounted under `/api/`.
- [`backend/app/main_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/main_final.py): Updated FastAPI entrypoint mounting all core routes with CORS support.
- [`backend/openapi.json`](file:///e:/Projects/Trading%20Project/TradePulse/backend/openapi.json): Standardized OpenAPI 3.1.0 specification with 18 endpoints.

---

### Phase 1: Frontend Implementation & Mock Integration (2026-11-12 00:33:00 IST)

#### 1. Setup & Tooling:
- **Framework:** React 18 + Vite 6 + TypeScript 5 + Tailwind CSS 3.
- **Financial Charting:** `lightweight-charts` v4.2.1.
- **Icons:** `lucide-react`.
- **FastAPI Integration:** Mounted `frontend/dist` on root path (`/`) in `main_final.py` so the application runs as a unified full-stack system on `http://localhost:8000/`. Configured Vite dev proxy `/api` -> `http://localhost:8000`.

#### 2. Components Created:
- [`frontend/src/components/Header.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/Header.tsx): Top navigation with META NASDAQ asset badge, real-time market open/close countdown chip, data feed chip, data freshness, model version, timezone toggle (ET/Local), dark/light theme toggle, and guide trigger.
- [`frontend/src/components/HistoricalChart.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/HistoricalChart.tsx): Lightweight Charts candlestick + volume display with 1D/5D/1M/3M/All range selector, period bands (Train, Embargo 1, Val, Embargo 2, Test), VWAP overlay, OOF past signals toggle, dimmed gap-fill bars (`is_filled`), and OHLCV crosshair tooltip.
- [`frontend/src/components/LiveChart.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/LiveChart.tsx): Main testing screen with 5-minute candles, forming candle in muted style, signal markers (Up, Down, Neutral), horizon band overlay, outcome badge overlays, Test Now button (`T`), Auto mode toggle (`A`), and candle close countdown.
- [`frontend/src/components/PredictionCard.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/PredictionCard.tsx): Dominant one-second read featuring large direction badge (color-blind safe), large calibrated confidence %, probability breakdown bars (Down/Neutral/Up), conviction gauge vs threshold $\tau = 0.2635$, lock icon indicating immutable signal, and advisory chips.
- [`frontend/src/components/Scoreboard.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/Scoreboard.tsx): Running daily signals vs expected 37.5, realized hit rate vs Always-Up baseline, and small-sample warnings.
- [`frontend/src/components/SignalLog.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/SignalLog.tsx): Searchable, filterable audit log table with realized returns and read-only CSV export.
- [`frontend/src/components/ModelCard.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/ModelCard.tsx): Collapsible model card showing all metadata from `ui_summary_final.json`, headline validation/test metrics with 95% CIs, and the explicit non-negotiable notice: *"No profitable edge after costs was demonstrated. Treat signals as experimental."*
- [`frontend/src/components/GuideDrawer.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/GuideDrawer.tsx): Scannable drawer explaining stale data, session boundaries, news catalysts, and calibrated probabilities.
- [`frontend/src/components/ReplayControls.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/ReplayControls.tsx): Unmissable amber "REPLAY, NOT LIVE" banner, playback controls (Play, Pause, Step, Stop, Speed 1x/5x/20x).
- [`frontend/src/components/Footer.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/Footer.tsx): Academic disclaimer & TradingView Lightweight Charts license attribution.
- [`frontend/src/App.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/App.tsx): Master layout supporting desktop stacked or side-by-side charts, mobile single-column, global hotkeys (`T`, `A`, `R`, `?`), SSE auto-streaming, and persistent local preferences.

#### 3. Verification Suite ([`tests/test_ui_api_contract_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_ui_api_contract_final.py)):
- Ran 9 automated contract and UI build verification tests.
- Status: **ALL 9 TESTS PASSED (100%)** in 0.432 seconds.
- Verified TypeScript compilation and production Vite build: `dist/index.html` (0.96 kB), `dist/assets/index.js` (375 kB), `dist/assets/index.css` (22 kB).

---

### Phase 2: Real Backend, Background Worker & Parity/No-Repaint Verification (2026-11-12 00:36:00 IST)

#### 1. Architecture & Background Worker Implementation:
- **Streaming & Polling Harness:** Built [`backend/app/services/background_worker_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/background_worker_final.py) running as an async daemon task inside the FastAPI lifespan context.
- **Clock Alignment:** Wakes 3 seconds past every 5-minute candle close (`:00`, `:05`, `:10`, ..., `:55` + 3s buffer) during regular market hours (9:30 AM - 4:00 PM ET) as determined by `pandas-market-calendars` NASDAQ schedule.
- **Alpaca Client & Fallback:** Handles Alpaca Market Data v2 IEX 5-minute bars via `alpaca-py` with transparent fallback to historical simulation when Alpaca keys are absent or outside market hours.
- **Single Shared ML Pipeline:** Ingests live candles into [`LiveInferenceServiceFinal`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/live_inference_final.py), preserving a 200-bar rolling memory, running feature generation through [`ml/predict_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/predict_final.py) with zero code duplication.
- **Immutable Append-Only Audit Logs:** Every signal written to `data/live_signals_final.jsonl` contains a SHA-256 line hash including the prior line's hash, forming an unbroken cryptographic chain. Repeated calls on the same bar return the cached stored signal (zero repainting).
- **15-Minute Outcome Resolution:** Exactly 3 bars after each prediction, calculates gross realized return and net return after 8 bps round-trip transaction costs, logging to `data/live_outcomes_final.jsonl`.

#### 2. Verification Test Results ([`tests/test_parity_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_parity_final.py)):
Ran complete 7-test suite in 66.712 seconds:
1. `test_01_warmup_and_schema_validation`: **PASS** (Rejects < 35 bars and missing columns with clear `ValueError`).
2. `test_02_historical_replay_feature_parity_last_10_days`: **PASS** (Maximum feature difference across all 29 features between live sliding buffer and batch offline evaluation = $5.45 \times 10^{-6}$, strictly $< 1 \times 10^{-5}$).
3. `test_03_historical_replay_signal_parity_last_10_days`: **PASS** (Zero signal mismatches out of 780 bars across 10 complete trading days: 100.0% signal parity).
4. `test_04_no_repainting_and_restart_byte_identity`: **PASS** (Mid-session server restart produced 100% byte-for-byte identical output for earlier log records).
5. `test_05_missing_bar_gap_handling`: **PASS** (Synthesized missing 5-minute interval bars with `is_filled=1` and `volume=0.0`, incrementing gap counters).
6. `test_06_corrupt_packet_resilience`: **PASS** (Trapped NaN close price, status updated to `DEGRADED`, logged error without crashing application).
7. `test_07_realized_outcome_resolution_after_3_bars`: **PASS** (Evaluated 15-minute gross return and net return @ 8 bps accurately upon 3rd subsequent candle close).

---

### Phase 3: Replay Mode, Scoreboard, Model Card & Reliability Flags (2026-11-12 00:54:00 IST)

#### 1. Replay Mode Architecture:
- **Session Isolation:** Replay mode steps through historical trading sessions bar-by-bar using [`LiveInferenceServiceFinal`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/live_inference_final.py) seeded with the 200 bars preceding the selected session.
- **Log Isolation:** Replay predictions write strictly to `data/replay_signals.jsonl` and `data/replay_outcomes.jsonl`, keeping the official live log `data/live_signals_final.jsonl` completely untouched.
- **Banner & Accents:** Prominent Amber `"REPLAY MODE â€” NOT LIVE DATA"` banner with animated indicator and top-border amber accent.
- **Interactive Controls:** Implemented Play, Pause, Step (+5m), and playback speed multipliers (`1x`, `5x`, `20x`) with automatic interval timers and session selection dropdown.
- **Pipeline Demo Disclaimer:** Clearly discloses that replay demonstrates pipeline mechanics on historical bars, not out-of-sample edge.

#### 2. Scoreboard & Model Card Disclosures:
- **Scoreboard:** Tracks daily signals vs expected 37.5/day target, calculates realized win rate, and renders small sample warning banner when $n < 10$.
- **Model Card:** Displays complete provenance, 29 causal features, 50% RF + 50% LR ensemble architecture, uniform recency weighting, and held-out test metrics with 95% day-block bootstrap confidence intervals.
- **Mandatory Edge Disclosure:** Explicitly presents `edge_demonstrated = false` with non-negotiable notice: *"No profitable edge after costs was demonstrated. Treat signals as experimental."*

#### 3. User Guide & Reliability Anchors:
- Built [`GuideDrawer.tsx`](file:///e:/Projects/Trading%20Project/TradePulse/frontend/src/components/GuideDrawer.tsx) with smooth auto-scroll to anchor tags (`#stale-data`, `#session-boundaries`, `#news-earnings`, `#thin-liquidity`, `#conviction-gating`) when clicking advisory chips on the Prediction Card.

#### 4. Verification Suite ([`tests/test_phase3_features_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_phase3_features_final.py)):
- `test_01_replay_sessions_availability`: **PASS** (Returns available historical sessions).
- `test_02_replay_start_and_isolated_logging`: **PASS** (Zero impact on `live_signals_final.jsonl`, initializes `replay_signals.jsonl`).
- `test_03_replay_step_progression`: **PASS** (Steps through 5 candles, updates bar index and signal).
- `test_04_replay_pause_and_stop_lifecycle`: **PASS** (Handles pause and stop cleanly).
- `test_05_model_card_and_edge_disclosure_integrity`: **PASS** (Verifies negative net return at 8 bps and `edge_demonstrated = false`).
- `test_06_reliability_advisory_flags`: **PASS** (Validates guide anchors and condition levels).

---

### Phase 4: Polish, Accessibility, Documentation & Verification (2026-11-12 01:03:00 IST)

#### 1. Application Launcher & Setup:
- Built [`run_app.py`](file:///e:/Projects/Trading%20Project/TradePulse/run_app.py) for one-command local startup on `http://localhost:8000/`.
- Updated [`README.md`](file:///e:/Projects/Trading%20Project/TradePulse/README.md) with comprehensive installation instructions, `.env` settings, architectural breakdown, replay usage, and academic honest disclosures.
- Authored [`reports/trader_user_guide.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/trader_user_guide.md) documenting one-second reads, operational protocols, regime volatility, and limitations.

#### 2. Visual Screenshots Captured:
Captured full terminal layouts using headless Chrome into [`reports/ui_screenshots/`](file:///e:/Projects/Trading%20Project/TradePulse/reports/ui_screenshots/):
- `desktop_dark.png` (1920x1080): Complete dark theme dashboard with stacked charts and prediction card.
- `desktop_replay.png` (1920x1080): Replay mode active with prominent amber banner and session progress.
- `mobile_dark.png` (390x844): Mobile single-column responsive layout.

#### 3. Accessibility & Design Verification:
- Color-blind safe directional semantic colors (emerald green, rose red, slate neutral).
- Full keyboard hotkey support: `T` (Test now), `A` (Auto mode), `R` (Replay toggle), `?` (Guide).
- Persisted user preferences in `localStorage` (`tp_theme`, `tp_timezone`, `tp_chart_layout`).
- All 3 automated test suites passed (100% test pass rate across 22 tests).






## 2026-11-12 — Chart Timestamp Fix

**Bug:** Historical chart X-axis showed \18:19\ and \21 Jan '70\ (Unix epoch ˜ 0).

**Root cause:** \_get_clean_df()\ in \mock_service.py\ computed \df['time'] = ts.astype('int64') // 10**9\. Pandas \Timestamp.astype('int64')\ returns **nanoseconds** (e.g. \1762180200_000_000\), so dividing by 10^9 gave \1762180\ seconds ˜ Jan 21, 1970. Correct divisor is 10^6 (yielding \1762180200\ seconds = 2025-11-03 09:30 ET).

**Fix:** Changed divisor to \// 10**6\ in \mock_service.py\ line 62.

**Range selector:** Also replaced fixed tail-row counts (78/390/1638/4914) with session-based slicing (\df[df['session_date'].isin(keep_sessions)]\) for correctness when any session has fewer bars (early close, etc.).

**Tests:** 15/15 pass. Frontend rebuilt successfully.
