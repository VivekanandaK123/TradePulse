# TradePulse: ML Pipeline Viva & Examiner Defense Guide

**Project:** TradePulse Intraday Machine Learning Directional Indicator  
**Institution:** Indian Institute of Technology Patna (IIT Patna), IMPACT Lab  
**Scope:** Machine Learning, Statistical Modeling, Validation Protocols & Experimental Evidence  

---

### Q1: Why did you choose 5-minute candlestick bars instead of 1-minute or 1-hour bars?
**Answer:** 1-minute bars in equity markets are heavily contaminated by microstructure noise (bid-ask bounce, order book flicker, and queue positioning) and require high-frequency execution infrastructure. Conversely, 1-hour bars produce only 6.5 samples per day, yielding too few data points over a 1-year window ($\sim 1,600$ bars) to train non-linear ML models without spanning multiple market regimes. 5-minute bars ($78$ bars/day) provide the optimal balance: sufficient statistical sample size ($19,818$ regular session bars/year) while filtering out high-frequency microstructure noise.  
**Supporting File:** [`ml/preprocessing.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/preprocessing.py), [`reports/ml_pipeline_walkthrough.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/ml_pipeline_walkthrough.md#1-problem-definition-and-design-choices).

---

### Q2: Why is the prediction horizon 15 minutes (3 bars) rather than 5 minutes (1 bar)?
**Answer:** A 1-bar ($5$m) forward horizon rarely generates price movement large enough to clear our institutional transaction cost hurdle of $8.0\text{ bps}$ ($0.08\%$). The median 5-minute move in META is only $\sim 7\text{ bps}$. A 15-minute horizon ($3$ bars) gives sufficient time for institutional order flow to create meaningful price displacement exceeding $15\text{ bps}$ without exposing the position to overnight gap risk.  
**Supporting File:** [`ml/targets.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/targets.py#L38-L83), [`artifacts/metadata/final_config_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/final_config_final.json#L5-L16).

---

### Q3: Why frame this as a 3-class classification problem rather than continuous return regression?
**Answer:** Continuous regression optimizes Mean Squared Error (MSE), which is heavily skewed by extreme volatility spikes rather than directional sign persistence. Furthermore, financial markets are dominated by flat, range-bound consolidation regimes where price drift is pure noise. Binary classification ($\text{Up vs Down}$) forces the model to take a leveraged trade on every single candle, even when the market is stagnant. A 3-class formulation explicitly models the **Neutral** regime ($|R| < 15\text{ bps}$), allowing the model to abstain when conviction is absent.  
**Supporting File:** [`ml/targets.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/targets.py), [`reports/label_expansion_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/label_expansion_report.md#3-phase-a-candidate-label-definitions--class-distributions).

---

### Q4: What is the Alpaca IEX data feed, and how did you prevent partial-volume bias?
**Answer:** The Investors Exchange (IEX) is an independent US stock exchange. On the free Alpaca tier, data reflects real trades executed on IEX, representing roughly $2-4\%$ of total US equity volume. Because raw IEX share volume varies depending on exchange market share, raw share volume was never fed directly into models. Instead, volume was normalized using **Relative Volume (RVOL)**: current volume divided by 12-bar and 24-bar rolling historical averages (`feat_rvol_12b`, `feat_rvol_24b`), ensuring scale invariance.  
**Supporting File:** [`ml/data_collection.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/data_collection.py), [`ml/features.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/features.py#L158-L174).

---

### Q5: How did you handle missing bars? Did you use forward-fill?
**Answer:** We aligned timestamps against the official NASDAQ regular-session calendar ($19,818$ scheduled intervals). On META, every single 5-minute regular-session interval had executed trades on IEX, resulting in **0 missing bars (0.00% fill rate)**. However, for live stream resilience, we implemented `synthesize_missing_bar()`: sets $O=H=L=C=\text{LastKnownClose}$, $Volume=0$, and tags `is_filled=1`. Same-day boundary isolation was enforced so forward filling never crosses overnight boundaries.  
**Supporting File:** [`ml/preprocessing.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/preprocessing.py#L38-L70), [`reports/step2_data_audit_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/step2_data_audit_report.md).

---

### Q6: Why did you drop the final 3 bars of every trading session?
**Answer:** The target return is defined as the 3-bar forward return ($R_{t, t+3} = \frac{Close_{t+3}}{Close_t} - 1$). At 15:45, 15:50, and 15:55 ET, a 3-bar forward return would span across the 16:00 close into the next day's 09:30 open, capturing overnight gap risk and earnings jumps rather than intraday alpha. To isolate pure intraday dynamics, the final 3 bars of every session were dropped ($255 \times 3 = 765$ bars dropped across 255 sessions).  
**Supporting File:** [`ml/targets.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/targets.py#L44-L72), [`artifacts/metadata/split_config.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/split_config.json).

---

### Q7: How was the dataset split chronologically, and what are the embargo periods?
**Answer:** The dataset was partitioned strictly chronologically:
- **Train:** 2025-11-03 to 2026-06-30 ($164$ trading sessions, $12,720$ total bars, $12,193$ valid training bars after warmup).
- **Embargo 1:** 2026-07-01 ($1$ session, $78$ bars).
- **Validation:** 2026-07-02 to 2026-08-31 ($42$ trading sessions, $3,276$ total bars, $3,150$ valid bars).
- **Embargo 2:** 2026-09-01 ($1$ session, $78$ bars).
- **Test:** 2026-09-02 to 2026-11-06 ($47$ trading sessions, $3,666$ total bars, $3,525$ valid bars).  
The 1-day ($390$-minute) embargo buffer days eliminate any potential forward-return leakage between splits (the target horizon is only 15 minutes, creating a $26\times$ safety margin).  
**Supporting File:** [`ml/split.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/split.py), [`artifacts/metadata/split_config.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/split_config.json).

---

### Q8: Why can't you use standard K-Fold cross-validation on financial time series?
**Answer:** Standard K-Fold cross-validation randomly shuffles observations, which causes two catastrophic forms of data leakage:
1. **Lookahead Leakage:** Training on future data ($t+1$) to predict past data ($t$).
2. **Autoregressive Target Overlap:** Consecutive overlapping return windows share identical price trajectories.  
Instead, we implemented `PurgedWalkForwardCV` with expanding chronological folds and an enforced 3-bar ($15$-minute) purge gap between the train fold and validation fold.  
**Supporting File:** [`ml/evaluate.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/evaluate.py#L29-L70).

---

### Q9: Why do some files cite 12,193 training bars while others cite 12,228 or 12,720?
**Answer:** This is a fully reconciled sample size progression:
1. $12,720$ bars: Total raw 5-minute candles in the Train date range ($164$ sessions including two 42-bar half-days).
2. $12,228$ bars: Remaining bars after dropping the final 3 bars of each session ($12,720 - (164 \times 3) = 12,228$).
3. $12,193$ bars: Remaining feature-ready bars after dropping the initial continuous 35-bar feature warmup window ($12,228 - 35 = \mathbf{12,193}$). All feature-based models were trained on 12,193 bars.  
**Supporting File:** [`reports/baseline_vs_step8_model_comparison_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/baseline_vs_step8_model_comparison_report.md#1-executive-summary--the-frozen-target-label).

---

### Q10: How did you ensure features are strictly causal and leak-free?
**Answer:** Every feature in [`ml/features.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/features.py) is formulated using only current and historical values ($t, t-1, \dots$). No backward shifts (`shift(-k)`) or future-centered windows were permitted. The maximum lookback is exactly 35 bars ($\approx 175$ minutes, dictated by MACD signal line smoothing). Furthermore, `RobustScaler` was fitted **strictly on the training partition** and serialized, ensuring zero parameters were learned from validation or test data.  
**Supporting File:** [`ml/features.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/features.py), [`artifacts/models/feature_scaler.joblib`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/models/feature_scaler.joblib).

---

### Q11: What did your feature ablation study reveal about technical indicators like RSI and MACD?
**Answer:** Retraining models from scratch after removing feature groups revealed that tree splits are heavily dominated by volatility and range features (`feat_vol_12b`, `feat_bar_range`, `feat_vol_6b`, `feat_bb_width` account for $> 64\%$ of MDI importance). Ablating volatility caused a $-1.35\%$ drop in balanced accuracy. In stark contrast, ablating classic technical oscillators (RSI, MACD) caused virtually **zero impact ($-0.02\%$)**, proving that classic momentum oscillators provide no incremental alpha over simple return lags and realized volatility.  
**Supporting File:** [`reports/step13_error_analysis_and_robustness_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/step13_error_analysis_and_robustness_report.md#5-feature-importance--group-ablation-study).

---

### Q12: Why did you backtrack from `label_fixed_20bp` to `label_fixed_15bp`?
**Answer:** In Step 8, `label_fixed_20bp` was initially selected for cost-headroom logic. However, in Step 9–10, we discovered that it created severe **signal starvation**: 62.2% of bars were Neutral, causing calibrated directional probabilities to rarely exceed Neutral. The model produced only ~0.7 signals/day with 80% to 96% zero-signal days. In Round 2 (Label Expansion), `label_fixed_15bp` was selected by deployability: it **halved the Neutral dominance to ~50%**, restored balanced class recall, enabled an active rate of $\sim 38$ signals/day, and preserved $7\text{ bps}$ of headroom over the 8 bps friction.  
**Supporting File:** [`reports/label_expansion_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/label_expansion_report.md#1-executive-summary), [`reports/model_comparison_label_fixed_15bp_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/model_comparison_label_fixed_15bp_report.md).

---

### Q13: What baselines did you benchmark against, and did the ML model beat them?
**Answer:** We benchmarked five baselines: Majority (Always Neutral), Always-Up, 1-Bar Momentum ($R_{t-1}$), Basic RSI (30/70), and Time-of-Day. On the held-out test set:
- Naive Always-Up lost $-90.33\%$ net due to transaction churn.
- Momentum lost $-104.72\%$ net (5m momentum suffers from bid-ask bounce).
- The ML Ensemble achieved **$42.56\%$ Balanced Accuracy** and **$0.4266$ Macro-F1**, significantly outperforming Always-Up ($33.3\%$, $0.1348$) and Momentum ($31.1\%$, $0.2174$).  
**Supporting File:** [`notebooks/04_baselines.ipynb`](file:///e:/Projects/Trading%20Project/TradePulse/notebooks/04_baselines.ipynb), [`reports/step12_final_test_evaluation_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/step12_final_test_evaluation_report.md#3-master-step-12-performance-table-untouched-test-period).

---

### Q14: What was the Time-of-Day baseline and why was it critical?
**Answer:** The Time-of-Day baseline estimates class probabilities strictly from 30-minute intraday clock slots. It captured $+4.73\text{ bps}$ gross return during the market open simply because opening bars have higher volatility. On Move vs. No-Move discrimination, Time-of-Day accounted for **$78.0\%$** of the Random Forest's AUC above chance. This established that much of an unconstrained model's apparent skill is merely learning diurnal volatility timing, proving the necessity of testing directional sign accuracy.  
**Supporting File:** [`ml/diagnostics.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/diagnostics.py#L27-L78), [`reports/label_expansion_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/label_expansion_report.md#23-phrasing--attribution-adjustments).

---

### Q15: Why did you constrain Random Forest to `max_depth=3` and `min_samples_leaf=50`?
**Answer:** High-frequency financial time series have extremely low signal-to-noise ratios ($< 1\%$). Deep decision trees rapidly split on idiosyncratic microstructure noise and trade-level order flow artifacts, resulting in catastrophic validation overfitting. Restricting trees to depth 3 and forcing at least 50 samples per leaf acted as structural regularization, preventing the model from memorizing historical noise.  
**Supporting File:** [`ml/train.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/train.py), [`artifacts/metadata/final_config_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/final_config_final.json#L23-L30).

---

### Q16: Why combine Random Forest and Logistic Regression into a 50/50 ensemble?
**Answer:** In validation, Logistic Regression ($42.4\%$ BalAcc) and Random Forest ($42.0\%$ BalAcc) were a statistical tie (paired bootstrap 95% CI of $\Delta\text{Gross}$ was $[-2.68, +4.01]$ bps, crossing zero). Logistic Regression captures faint monotonic linear relationships (momentum and day-of-week), while Random Forest captures non-linear regime splits (volatility squeezes). An equal-weighted soft probability blend reduced cross-validation fold variance and eliminated single-model failure modes.  
**Supporting File:** [`reports/baseline_vs_step8_model_comparison_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/baseline_vs_step8_model_comparison_report.md#3-comparative-analysis--model-performance-assessment).

---

### Q17: What is probability calibration, and why Platt Sigmoid over Isotonic regression?
**Answer:** Raw model scores (such as Random Forest vote fractions) represent ranking scores rather than true empirical probabilities. Calibration maps raw outputs to true probabilities: if the model outputs 60%, 60 out of 100 cases must resolve Up. We selected **Platt Sigmoid (One-vs-Rest)** fitted on out-of-fold predictions because it enforces a smooth, monotonic logistic curve. Isotonic regression is non-parametric and produced step-function overfitting artifacts on small validation bins.  
**Supporting File:** [`ml/calibration.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/calibration.py#L85-L100), [`artifacts/models/calibrator_final.joblib`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/models/calibrator_final.joblib).

---

### Q18: How does conviction gating work, and how was the threshold $\tau$ selected?
**Answer:** Directional conviction is defined as $\text{Conviction}_t = \max(P(Y_t = \text{Down}), P(Y_t = \text{Up}))$. If conviction $\ge \tau$, the model emits Up if $P(\text{Up}) \ge P(\text{Down})$, else Down. If conviction $< \tau$, it emits Neutral ($0$). To satisfy the user requirement for an active indicator with $\ge 50\%$ signal frequency, $\tau$ was set strictly to the **50th percentile of out-of-fold conviction**: $\tau = 0.2646$ on Test ($37.74$ signals/day, $50.3\%$ coverage), and $\tau_{\text{final}} = \mathbf{0.2635}$ on the final full-dataset refit.  
**Supporting File:** [`ml/gating.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/gating.py#L27-L70), [`artifacts/metadata/signal_rule_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/signal_rule_final.json).

---

### Q19: Did recency weighting improve model performance?
**Answer:** No. In Phase 1, we tested exponential decay half-lives ($180$, $90$, and $45$ trading days) against uniform weighting via 5-fold Purged Walk-Forward CV on Train+Val data ($N = 15,343$ bars). In a paired day-block bootstrap test ($1,000$ iterations), all 95% confidence intervals for $\Delta\text{Gross Return}$ spanned zero ($[-0.29, +0.66]$ bps for 180d, $[-0.68, +1.68]$ bps for 45d). Per our pre-registered decision rule, **Uniform Weighting was retained**, preserving 100% of effective sample size.  
**Supporting File:** [`artifacts/metadata/phase1_recency_weighting_results.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/phase1_recency_weighting_results.json), [`reports/report_final.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/report_final.md#3-phase-1-recency-weighting-grid--selection).

---

### Q20: Was the test set sealed, and how many times was it evaluated?
**Answer:** Yes. The test set ($N = 3,525$ valid bars, 47 sessions, Sep 2 – Nov 6, 2026) was maintained in strict isolation and **evaluated exactly once** in Phase 2 / Step 12. All hyperparameters, feature formulas, calibrators, and gating thresholds were frozen into [`artifacts/metadata/final_config_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/final_config_final.json) prior to unsealing. Zero back-and-forth iteration occurred on the test set.  
**Supporting File:** [`artifacts/metadata/phase2_test_results.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/phase2_test_results.json), [`reports/step12_final_test_evaluation_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/step12_final_test_evaluation_report.md).

---

### Q21: What were the final held-out test set metrics? Did the model beat chance?
**Answer:** Evaluated across all 3,525 test bars ($1,774$ active signals, $37.74$ signals/day):
- **3-Class Accuracy:** $46.81\%$
- **Balanced Accuracy:** $42.56\%$ (95% CI: $[40.43\%, 44.54\%]$)
- **Macro-F1 Score:** $0.4266$ (95% CI: $[0.4048, 0.4465]$)
- **Binary Sign Accuracy:** **$54.11\%$** (95% CI: $[50.82\%, 57.22\%]$)
- **Gross Return:** **$+1.29\text{ bps}$** (95% CI: $[-1.70, +4.05]$ bps)  
Because the 95% bootstrap CI for binary sign accuracy strictly excludes $50.0\%$, the model **statistically significantly beats random chance** on directional bars ($p = 0.007$).  
**Supporting File:** [`artifacts/metadata/phase2_test_results.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/phase2_test_results.json), [`reports/report_final.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/report_final.md#4-phase-2-one-time-held-out-test-evaluation-50-coverage).

---

### Q22: Did the model demonstrate a profitable net edge after transaction costs?
**Answer:** **No.** While the model achieves positive gross returns ($+1.29\text{ bps}$ per trade across 1,774 trades), institutional execution friction of $8.0\text{ bps}$ round-trip converts this into a net loss of **$-6.71\text{ bps}$ per trade** (95% CI: $[-9.70, -3.95]$ bps). In strict compliance with scientific integrity, we explicitly report that no profitable trading edge after costs was demonstrated.  
**Supporting File:** [`artifacts/metadata/ui_summary_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/ui_summary_final.json#L125-L127), [`reports/report_final.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/report_final.md#9-honest-scientific-disclosures--non-claims).

---

### Q23: Why refit the final model on all data if you already had test evaluation results?
**Answer:** In production quantitative finance, once an architecture and hyperparameter set are frozen and validated out-of-sample, the final deployed binary is refitted on all available historical data (Nov 3, 2025 – Nov 6, 2026, $253$ sessions, $18,868$ valid bars). This ensures the model incorporates the most recent market regime for live inference. We explicitly document that this final model has zero held-out data, and its expected performance is inferred strictly from Phase 2.  
**Supporting File:** [`ml/train_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/train_final.py#L710-L750), [`artifacts/metadata/training_manifest_final.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/training_manifest_final.json#L58).

---

### Q24: How did you verify feature and signal parity between offline batch and live streaming?
**Answer:** We built [`tests/test_parity_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_parity_final.py) to replay the last 10 trading sessions ($780$ bars) through the live 200-bar sliding buffer interface. Max absolute difference across all 29 features was **$5.45 \times 10^{-6}$** (well below the $10^{-5}$ tolerance), and there were **zero signal mismatches out of 780 bars (100.0% exact match)**. We also proved that recursive IIR filters (RSI, MACD) converge to $< 10^{-6}$ error when buffer size $\ge 150$ bars.  
**Supporting File:** [`tests/test_parity_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_parity_final.py), [`reports/step14_packaging_and_parity_report.md`](file:///e:/Projects/Trading%20Project/TradePulse/reports/step14_packaging_and_parity_report.md#3-parity-analysis-batch-offline-vs-live-sliding-buffer).

---

### Q25: How does TradePulse guarantee zero repainting?
**Answer:** Three architectural guarantees ensure zero repainting:
1. **Candle-Close Evaluation Only:** Signals are evaluated strictly when a 5-minute bar officially closes ($t = 09:35:01, 09:40:01$). Never on forming bars.
2. **Immutable Append-Only Log:** Every signal is immediately serialized and flushed to disk in an append-only JSONL file (`data/live_signals_final.jsonl`).
3. **Byte-Identical Restart Verification:** `test_04_no_repainting_and_restart_byte_identity` simulates mid-session crashes and proves earlier emitted lines are byte-for-byte identical.  
**Supporting File:** [`backend/app/services/live_inference_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/backend/app/services/live_inference_final.py), [`tests/test_parity_final.py`](file:///e:/Projects/Trading%20Project/TradePulse/tests/test_parity_final.py#L125-L165).

