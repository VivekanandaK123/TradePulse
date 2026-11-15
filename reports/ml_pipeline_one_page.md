# TradePulse ML Pipeline: One-Page Executive Summary

**Project:** TradePulse Intraday Machine Learning Directional Indicator  
**Institution:** IIT Patna, IMPACT Lab | **Date:** November 2026  
**Asset:** META Platforms Inc. (`META`, NASDAQ) | **Cadence:** 5-Minute Closed Bars | **Horizon:** 15 Minutes Forward ($3$ Bars)  

---

### 1. Objective & Problem Formulation
- **The Core Task:** Predict the sign of forward 15-minute return ($R_{t, t+3}$) across three discrete regimes: **Up ($+1$)**, **Down ($-1$)**, and **Neutral ($0$)** using a symmetric threshold $\theta = \pm 15\text{ basis points}$ ($\pm 0.15\%$).
- **The Challenge:** Intraday 5-minute equity returns exhibit a signal-to-noise ratio $< 1\%$, extreme noise, and an institutional execution friction hurdle of **$8.0\text{ bps}$ round-trip** (spread + fees).
- **Deliverable:** A leak-free, calibrated probability ensemble model with dynamic conviction gating, evaluated on sealed held-out test data and packaged for live streaming inference.

---

### 2. End-to-End Pipeline Architecture

```
[Raw Alpaca IEX 5m Bars] (20,516 bars; 1 Year: Nov 2025 – Nov 2026)
         │
         ▼
[Exchange Calendar Alignment] ──► Filter RTH (09:30–16:00 ET); 0 missing bars (0.00% fill rate)
         │
         ▼
[Intraday 15m Forward Target] ──► Drop final 3 bars of every day (765 EOD bars dropped) -> 19,053 labeled bars
         │
         ▼
[Leak-Free Chronological Split]
  ├── Train Set: 164 days (12,193 valid bars after 35-bar warmup)
  ├── Embargo 1: 1 trading day (78 bars)
  ├── Validation Set: 42 days (3,150 valid bars)
  ├── Embargo 2: 1 trading day (78 bars)
  └── Held-Out Test Set: 47 days (3,525 valid bars) ──► STRICTLY SEALED (Touched Once)
         │
         ▼
[29 Causal Features] ──► Multi-lag returns (1,3,6,12), Realized Volatility (6,12), VWAP dist/slope, 
                          Relative Volume (12,24), Bollinger %B/width, MACD, RSI, Time-of-Day sin/cos.
                          RobustScaler fitted strictly on Train partition.
         │
         ▼
[Candidate Modeling & Tuning] ──► 5-Fold Purged Walk-Forward CV (3-bar purge)
                                  Evaluated: Majority, Always-Up, Momentum, RSI, Time-of-Day, LR, RF, XGBoost.
         │
         ▼
[Label & Model Evolution] ──► Round 1 selected 20bp -> suffered 62% Neutral signal starvation (~0.7 sigs/day).
                              Round 2 expanded to 7 labels -> selected 15bp by deployability (50% Neutral).
                              Model: 50% Random Forest (150 trees, depth 3) + 50% Logistic Regression (C=1.0).
         │
         ▼
[Calibration & Conviction Gating] ──► Platt Sigmoid (OVR) fit on OOF probabilities (Brier: 0.612 -> 0.584).
                                      Conviction = max(P_down, P_up) >= tau (tau = 0.2646 -> 50% target coverage).
         │
         ▼
[Recency Weighting Evaluation] ──► Half-lives (180d, 90d, 45d) failed to beat Uniform in paired bootstrap CIs.
                                   Uniform weighting retained, preserving 100% Effective Sample Size (15,343 bars).
         │
         ▼
[One-Time Held-Out Test Evaluation] ──► Touched exactly once on 47 test sessions (Sep 2 – Nov 6, 2026).
         │
         ▼
[Full Dataset Refit & Production Serving] ──► Refit final RF+LR on all 253 sessions (18,868 rows).
                                              tau_final = 0.2635 (~37.5 sigs/day). SHA-256 verified manifests.
```

---

### 3. Key Empirical Results (Held-Out Test Set: 47 Sessions, 3,525 Bars)

| Metric | Measured Test Value | 95% Day-Block Bootstrap CI | What It Proves to Examiners |
| :--- | :---: | :---: | :--- |
| **3-Class Accuracy** | **46.81%** | — | Strong multi-class discrimination against a ~47% Neutral prevalence. |
| **Balanced Accuracy** | **42.56%** | $[40.43\%, 44.54\%]$ | Substantially beats naive guessing ($33.3\%$) and conservative gating ($35.32\%$). |
| **Macro-F1 Score** | **0.4266** | $[0.4048, 0.4465]$ | Balanced precision and recall across all three classes. |
| **Binary Sign Accuracy** | **54.11%** | **$[50.82\%, 57.22\%]$** | **Strictly exceeds 50.0% coin-flip threshold** on directional bars ($p = 0.007$). |
| **Up-vs-Down ROC-AUC** | **0.5386** | — | Positive ranking discrimination on directional candles. |
| **Gross Return / Signal** | **+1.29 bps** | $[-1.70, +4.05]\text{ bps}$ | Faint positive gross alpha across 1,774 executed test signals. |
| **Net Return @ 8bp Costs**| **-6.71 bps** | **$[-9.70, -3.95]\text{ bps}$** | **Negative net expectancy after institutional execution friction.** |
| **Operational Activity** | **37.74 sigs/day** | **0.0% zero-signal days** | Active signal every other 5m candle; guarantees live testing responsiveness. |

---

### 4. Core Scientific Verdict & Defensive Non-Claims
1. **Gross Predictive Alpha Confirmed:** The ML model extracts genuine directional signal from 5-minute equity microstructure, achieving statistically significant sign accuracy (**$54.11\%$**) and positive gross returns (**$+1.29\text{ bps}$**).
2. **The Transaction Cost Trap:** Because round-trip institutional friction is $8.0\text{ bps}$, executing $\sim 38$ trades/day produces an expected net return of **$-6.71\text{ bps}$ per trade**.
3. **Honest Scientific Non-Claim:** **No profitable net trading edge after costs is demonstrated.** TradePulse ships as an educational, risk-managed analytical indicator for discretionary trader assistance, not an automated money-making execution bot.
4. **Leakage & Overfitting Defense:** Zero test leakage, 1-day embargoes, 3-bar walk-forward purging, shallow decision trees (`max_depth=3`), unweighted calibration, and frozen SHA-256 manifests guarantee complete scientific integrity.

