# TradePulse: Final Model and Live Inference Execution Log

**Project:** TradePulse Intraday Directional Indicator (META, NASDAQ, 5-Min Bars)  
**Laboratory:** IIT Patna, IMPACT Lab  
**Log Start Date:** 2026-11-11 20:45:00 IST  

---

### Phase 0: Configuration Freeze (2026-11-11 20:45:00 IST)
- **Source Artifacts Loaded:**
  1. [`artifacts/metadata/chosen_label_label_expansion.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/chosen_label_label_expansion.json): Selected label `label_fixed_15bp` ($\theta = \pm 0.15\%$, 15-minute / 3-bar forward intraday return).
  2. [`artifacts/metadata/production_signal_rule_label_expansion.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/production_signal_rule_label_expansion.json): Ensemble architecture (50% RF + 50% LR), Platt Sigmoid OVR calibrator, conviction rule $\max(P(\text{Down}), P(\text{Up})) \ge \tau$, target frequency ~5 signals/day (~6.4% coverage).
  3. [`artifacts/metadata/feature_metadata.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/feature_metadata.json): 29 causal features, `max_lookback_bars = 35`.
  4. Hyperparameters:
     - Random Forest: `n_estimators=150`, `max_depth=3`, `min_samples_leaf=50`, `max_features="sqrt"`, `class_weight="balanced"`, `random_state=42`.
     - Logistic Regression: `C=1.0`, `class_weight="balanced"`, `max_iter=1000`, `solver="lbfgs"`, `random_state=42`, internal `RobustScaler`.
- **Action:**
  - Written authoritative configuration to [`artifacts/metadata/final_config_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/final_config_final.json).
  - All hyperparameters, label definitions, feature lists, and signal formulas are permanently frozen. Only the recency weighting half-life will be chosen in Phase 1 via purged walk-forward cross-validation on Train + Validation data.
  - The Test Set (`data/processed/test_features.csv`, Sep 2 – Nov 6, 2026) remains completely sealed and untouched during Phase 1.

---

### Phase 1: Recency Weighting Grid & Selection (2026-11-11 20:53:00 IST)
- **Dataset Partition:** Train + Validation combined ($N = 15,343$ valid bars, 206 trading sessions, Nov 3, 2025 to Aug 31, 2026). Test set strictly untouched.
- **Protocol:** 5-fold `PurgedWalkForwardCV` with 3-bar embargo gap.
- **Predefined Half-Life Grid:**
  - `Uniform (no decay)`: ESS = 15,343.0 (100.0%), UpDown AUC = 0.4990, Sign Acc = 49.43%, Top 6.4% Gross = +4.64 bps, Net @ 8bp = -3.36 bps, $\tau = 0.4505$.
  - `180 trading days`: ESS = 14,592.6 (95.1%), UpDown AUC = 0.4994, Sign Acc = 49.58%, Top 6.4% Gross = +4.46 bps, Net @ 8bp = -3.54 bps.
    - Paired Day-Block Bootstrap (1,000 resamples): $\Delta\text{Gross} = -0.11$ bps, 95% CI: $[-2.24, +2.10]$ bps. Excludes zero: **False**.
  - `90 trading days`: ESS = 12,791.9 (83.4%), UpDown AUC = 0.4992, Sign Acc = 49.73%, Top 6.4% Gross = +3.92 bps, Net @ 8bp = -4.08 bps.
    - Paired Day-Block Bootstrap (1,000 resamples): $\Delta\text{Gross} = -0.73$ bps, 95% CI: $[-4.40, +3.57]$ bps. Excludes zero: **False**.
  - `45 trading days`: ESS = 8,933.4 (58.2%), UpDown AUC = 0.5041, Sign Acc = 50.17%, Top 6.4% Gross = +4.11 bps, Net @ 8bp = -3.89 bps.
    - Paired Day-Block Bootstrap (1,000 resamples): $\Delta\text{Gross} = -0.46$ bps, 95% CI: $[-3.45, +2.48]$ bps. Excludes zero: **False**.
- **Pre-Registered Decision Rule:**
  - Pick a decaying half-life **only if** it beats uniform weighting on those metrics with a paired day-block bootstrap interval that excludes zero.
  - **Verdict:** Decaying half-lives fail to beat uniform weighting with statistical significance. Uniform weighting is retained per pre-registered protocol, preserving 100% of effective sample size.
- **Outputs Created:**
  - [`artifacts/metadata/phase1_recency_weighting_results.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/phase1_recency_weighting_results.json)
  - [`reports/figures/recency_weighting_cv_final.png`](file:///e:/Projects/Trading%20Project/TradePulse/reports/figures/recency_weighting_cv_final.png)
  - Updated [`artifacts/metadata/final_config_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/final_config_final.json).

---

### Phase 3: Final Refit on All Available Data (2026-11-11 21:35:00 IST)
- **Dataset Scope:** Full chronological dataset across 253 complete sessions (Nov 3, 2025 – Nov 6, 2026).
  - Total Raw Bars: 19,627 bars.
  - Valid Training Rows: 18,868 rows (after dropping 759 final 3-bar session boundary entries).
- **Refit Strategy:**
  - 5-fold `PurgedWalkForwardCV` executed on all available data to generate out-of-fold predictions on 11,317 bars.
  - `MulticlassCalibrator(method="sigmoid")` fit strictly on out-of-fold ensemble probabilities to ensure un-biased calibration.
  - Derived final production threshold: $\tau_{\text{final}} = \mathbf{0.2635}$ (targeting 50.0% operational signal coverage, 37.5 signals/day, 0.0% zero-signal days).
  - Refit final production `RandomForestClassifier` (150 trees, depth 3, leaf 50, sqrt features) and `LogisticRegression` (C=1.0, balanced weights, lbfgs) on all 18,868 bars with uniform sample weights.
  - Fitted final `RobustScaler` on all 18,868 bars.
- **Model Manifest & Cryptographic Hashes (SHA-256):**
  - `artifacts/models/model_final.joblib`: `07ef98a8f08f007a967aed474c69d4b68184e81142e48f4e27aabb8cf365a0cd`
  - `artifacts/models/scaler_final.joblib`: `0f178107e4d3804e5e705b44d32fef6338e5268744f28df7721d45a522e1460e`
  - `artifacts/models/calibrator_final.joblib`: `77ae883c9942d494c170f41f02822ab5933f7f5b35f213388b0c977e6b61f7c6`
  - `artifacts/metadata/feature_schema_final.json`: `39e684a757cb85a0b8185eff28d072114a81721553c8a7cce83c778ddba19ada`
  - `artifacts/metadata/signal_rule_final.json`: `43630557dcaa7b98b3ea7269b57fdbae7b96cebca1860542862602eaffdedae9`
  - `artifacts/metadata/final_config_final.json`: `a324e521c046678173c30c2341304dbd9cc104edcde316216d71168f8f98cd65`
  - Git Commit Hash: `b57ee06eaf83b0854d697dfbfe81f92da3d2ec8d`
- **Held-Out Evaluation Notice:**
  - This final model has no held-out data. All 253 sessions were used for the final refit. Expected operational performance (46.81% accuracy, 42.56% balanced accuracy, 0.4266 macro-F1, 54.11% binary sign accuracy, +1.29 bps gross return, -6.71 bps net return @ 8bp, 37.7 signals/day, 0% zero-signal days) is inferred strictly from the Phase 2 held-out test evaluation. Zero in-sample evaluation is reported.

---

### Phase 4: Live Inference Engine Implementation (2026-11-11 21:48:00 IST)
- **Component 1: Inference Pipeline ([`ml/predict_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/predict_final.py)):**
  - Validates minimum candle lookback buffer ($\ge 35$ bars warm-up; recommends 200 bars).
  - Computes exact 29 causal features using the shared pipeline in `ml.features`.
  - Loads production artifacts (`model_final.joblib`, `scaler_final.joblib`, `calibrator_final.joblib`, `signal_rule_final.json`).
  - Evaluates RF (unscaled) and LR (scaled via RobustScaler), computes 50/50 raw ensemble probabilities, and applies Platt Sigmoid calibrator.
  - Applies conviction threshold $\tau = 0.2635$: emits Up (1), Down (-1), or "no clear signal" (0).
  - Outputs standardized unmodifiable dictionary payload: `{timestamp, close_price, direction, signal_code, confidence_pct, conviction, threshold, is_gated, probabilities, model_version}`.
- **Component 2: Live Service ([`backend/app/services/live_inference_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/live_inference_final.py)):**
  - Manages thread-safe 200-bar rolling buffer of closed 5-minute candles.
  - Strictly enforces candle-close evaluation (never evaluates on forming candles).
  - Implements missing-bar synthesis: detects intraday gaps $> 5$ minutes, logs gap events, creates synthetic zero-volume bars (`is_filled=1`), and increments monitoring counters.
  - Appends every emitted prediction immediately upon candle close to [`data/live_signals_final.jsonl`](file:///e:/Projects/Trading%20Project/TradePulse/data/live_signals_final.jsonl) with explicit flush/sync (immutable audit log proving no repainting).
  - Implements 3-bar (15-min) realized outcome resolution: tracks pending signals and appends realized gross and net returns (@ 8bp) and direction correctness to [`data/live_outcomes_final.jsonl`](file:///e:/Projects/Trading%20Project/TradePulse/data/live_outcomes_final.jsonl).
- **Component 3: REST Endpoints ([`backend/app/routes/signals_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/routes/signals_final.py) & [`backend/app/main_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/main_final.py)):**
  - `GET /api/v1/signals/latest`: Latest candle signal, confidence, timestamp.
  - `GET /api/v1/signals/history?limit=100`: Recent signals for financial chart rendering.
  - `GET /api/v1/monitoring/status`: Health status, uptime, buffer size, error count, and running signal count today vs expected (37.5).

---

### Phase 5: Verification Suite & Dry-Run Execution (2026-11-11 21:58:00 IST)
- **Unit & Integration Suite ([`tests/test_parity_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_parity_final.py)):**
  - Executed 7 comprehensive verification tests in 58.8 seconds. Status: **ALL 7 PASSED (100%)**.
  - `test_01_warmup_and_schema_validation`: Passed. Verified short buffer (< 35 bars) and missing column errors.
  - `test_02_historical_replay_feature_parity_last_10_days`: **Passed**. Max feature difference across all 29 features over the last 10 trading days (780 bars) = **$5.45 \times 10^{-6} < 10^{-5}$**.
  - `test_03_historical_replay_signal_parity_last_10_days`: **Passed**. Verified sliding window predictions against batch predictions across all 780 bars: **0 signal mismatches out of 780 bars (100.0% parity)**.
  - `test_04_no_repainting_and_restart_byte_identity`: **Passed**. Simulated mid-session server restart; proved earlier lines in `live_signals_final.jsonl` are byte-identical and timestamps never change after emission.
  - `test_05_missing_bar_gap_handling`: **Passed**. Simulated a 15-minute gap; confirmed synthetic zero-volume bars (`is_filled=1`) were properly inserted and logged.
  - `test_06_corrupt_packet_resilience`: **Passed**. Injected corrupt NaN bar; verified service caught exception, updated status to DEGRADED, and continued without crashing.
  - `test_07_realized_outcome_resolution_after_3_bars`: **Passed**. Verified exact 3-bar resolution and logging of gross and net returns in `live_outcomes_final.jsonl`.
- **Full Session Dry-Run Simulation ([`ml/dry_run_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/dry_run_final.py)):**
  - Simulated complete trading session (2026-11-06, 78 5-minute bars from 09:30 to 16:00 NY time).
  - Execution time: 3.68 seconds, 0 crashes.
  - Emitted: 12 Up, 23 Down, 43 Neutral (35 active directional signals, 44.9% daily coverage, closely matching 50% target).
  - Resolved: 75 3-bar outcomes in `live_outcomes_final.jsonl` (final 3 bars remain pending as expected).
  - Audit log integrity verified: 78 JSONL entries byte-flushed and sequentially indexed.
- **Stopping Point:**
  - In strict compliance with user instructions ("Stop again after Phase 5 for my confirmation before the live test" and "dont make the frontend part just yet"), system is paused for user confirmation before the live test trading day.


