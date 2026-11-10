# TradePulse - Corrected Model Comparison Report: Baselines vs. Step 8 Models
**Author:** TradePulse Team (IIT Patna, IMPACT Lab)  
**Asset:** META (NASDAQ) | **Timeframe:** 5-minute candles | **Prediction Horizon:** 15-minute (3 bars ahead)  
**Evaluation Dates:** Training (Nov 3, 2025 – Jun 30, 2026, $N = 12,193$ valid bars) | Validation (Jul 2, 2026 – Aug 31, 2026, $N = 3,150$ valid bars)  
**Strict Protocol:** Chronological splits with 1-day embargoes; 3-bar purging between CV folds; Test Set ($N = 3,525$ bars) 100% UNTOUCHED.

---

## 1. Executive Summary & The Frozen Target Label

### Sample Size Reconciliation ($N = 12,193$ vs. $12,228$)
- In raw split accounting, the Training partition contains 12,720 total candles across 164 trading sessions.
- Dropping the final 3 bars of each session (which cannot have a 3-bar forward target within the same day) leaves **12,228** bars with valid targets.
- To compute causal rolling technical indicators (with a maximum feature lookback of 35 bars / 175 minutes), the first continuous 35 bars of the training period are reserved as warm-up bars and dropped so feature rows contain zero NaNs.
- Therefore, all feature-based model evaluations in both Cross-Validation and final training utilize strictly **$N = 12,193$ valid training bars**. All tables in this report strictly reflect this reconciled sample size.

### The Chosen Target Label: `label_fixed_20bp`
Following Step 8 Phase 1, the candidate label **`label_fixed_20bp`** was frozen into `artifacts/metadata/chosen_label.json`:
$$\theta = \pm 0.20\% \quad (\pm 20\text{ basis points})$$
$$Y_t = \begin{cases} +1 \quad (\text{Up}) & \text{if } R_{t, t+3} \ge +0.0020 \\ -1 \quad (\text{Down}) & \text{if } R_{t, t+3} \le -0.0020 \\ 0 \quad (\text{Neutral}) & \text{if } -0.0020 < R_{t, t+3} < +0.0020 \end{cases}$$

### Class Distribution on `label_fixed_20bp`
| Partition | Total Valid Labeled Bars | Down (-1) Count & % | Neutral (0) Count & % | Up (+1) Count & % | Directional Coverage |
|---|---|---|---|---|---|
| **Training Set (Feature-Ready)** | 12,193 bars | 2,308 (18.9%) | 7,587 (62.2%) | 2,298 (18.8%) | 37.8% |
| **Validation Set** | 3,150 bars | 675 (21.4%) | 1,813 (57.6%) | 662 (21.0%) | 42.4% |

### Justification of `label_fixed_20bp`
The choice of `label_fixed_20bp` is justified primarily by **economic viability and cross-label realized returns**:
1. **Economic Viability Over Transaction Costs:** An assumed round-trip transaction friction of $0.08\%$ (8 bps spread + slippage) consumes $40\%$ of a $20$ bps move. In contrast, narrower thresholds like `label_quantile_40` ($11.3$ bps) require $71\%$ of the target move simply to pay for friction, causing large negative net returns ($-9.97$ bps on XGBoost).
2. **Label-Independent Realized Return Comparison:** On the Validation Set, `label_fixed_20bp` produced the lowest transaction drag across both Logistic Regression ($-7.38$ bps) and Random Forest ($-7.05$ bps), outperforming `label_quantile_50` ($-7.66$ to $-9.17$ bps) and `label_quantile_40` ($-7.82$ to $-9.97$ bps).
3. **Note on Classification Metrics:** Balanced accuracy and Macro-F1 are not directly comparable across candidate labels because they depend on varying base rates (e.g. 62% Neutral for fixed 20bp vs 40% Neutral for quantile 40). Label selection is therefore grounded in actual price moves and cost-adjusted returns.

---

## 2. Reconciled Comparison Tables

### Definition of "Top 10% Cov Prec"
Bars are ranked in descending order of model conviction:
$$\text{Conviction}_t = \max\big(P(Y_t = \text{Down}), P(Y_t = \text{Up})\big)$$
The top $10\%$ highest-conviction bars are selected. Precision is defined as:
$$\text{Precision}_{\text{Top 10\%}} = \frac{\sum_{t \in \text{Top 10\%}} \mathbb{I}\Big(\big(\hat{Y}_t = +1 \land R_{t, t+3} \ge +0.0008\big) \lor \big(\hat{Y}_t = -1 \land R_{t, t+3} \le -0.0008\big)\Big)}{\sum_{t \in \text{Top 10\%}} \mathbb{I}(\hat{Y}_t \neq 0)}$$
For constant predictors (Majority, Always-Up), all bars share identical constant probability with zero variation in conviction. Because no meaningful ranking exists, Top 10% Conviction Precision is reported as **N/A**.

---

### Table A: Out-of-Sample Performance on Validation Set ($N = 3,150$ bars)
*Evaluated on the unseen Validation partition (July – August 2026). All numbers generated from saved metadata artifacts:*

| Category | Model Name | Accuracy | Balanced Acc | Macro-F1 | Coverage % | **Gross Ret/Trade** | **Net Ret @ 8bp** | 8bp Win Rate | **Top 10% Cov Prec** |
|---|---|---|---|---|---|---|---|---|---|
| **Simple Baseline** | Majority (Always Neutral) | 57.6% | 33.3% | 0.2435 | 0.0% | +0.00 bps | +0.00 bps | 0.0% | N/A |
| **Simple Baseline** | Always-Up (+1) | 21.0% | 33.3% | 0.1158 | 100.0% | +0.47 bps | -7.53 bps | 35.5% | N/A |
| **Simple Baseline** | Momentum (Sign $R_{t-1}$) | 22.0% | 34.3% | 0.2062 | 99.8% | +0.54 bps | -7.46 bps | 36.1% | 37.8% |
| **Simple Baseline** | Basic RSI (30/70) | 55.1% | 35.7% | 0.3280 | 13.8% | -1.21 bps | -9.21 bps | 39.1% | 41.0% |
| **Linear Model** | Logistic Regression (L2) | 49.7% | 42.4% | 0.4194 | 48.2% | +0.62 bps | -7.38 bps | 40.1% | 42.9% |
| **Non-Linear Tree** | Random Forest (Default Phase 1)| 47.1% | 41.8% | 0.3997 | 53.6% | +0.95 bps | -7.05 bps | 40.9% | 41.3% |
| **Non-Linear Tree** | Random Forest (Tuned Phase 2)  | 47.9% | 42.0% | 0.3920 | 51.7% | +0.91 bps | -7.09 bps | 40.6% | 42.9% |
| **Gradient Boosting**| XGBoost (Default Phase 1)     | 44.5% | 40.1% | 0.3908 | 57.6% | -1.18 bps | -9.18 bps | 38.3% | 41.0% |
| **Gradient Boosting**| XGBoost (Tuned Phase 2)       | 43.1% | 39.9% | 0.3861 | 60.1% | -0.52 bps | -8.52 bps | 38.2% | 39.4% |

---

### Table B: Cross-Validation Performance on Training Set (5-Fold Purged Walk-Forward CV, $N = 12,193$ bars)
*Evaluated with an expanding chronological window and an enforced 3-bar purge gap:*

| Category | Model Name | Accuracy | Balanced Acc | Macro-F1 | Coverage % | **Gross Ret/Trade** | **Net Ret @ 8bp** | 8bp Win Rate | **Top 10% Cov Prec** |
|---|---|---|---|---|---|---|---|---|---|
| **Simple Baseline** | Majority (Always Neutral) | $60.57\% \pm 3.17\%$ | $33.33\% \pm 0.00\%$ | $0.2513 \pm 0.0082$ | 0.0% | $+0.00$ bps | $+0.00$ bps | 0.0% | N/A |
| **Simple Baseline** | Always-Up (+1) | $19.20\% \pm 0.87\%$ | $33.33\% \pm 0.00\%$ | $0.1074 \pm 0.0041$ | 100.0% | $+0.03$ bps | $-7.97$ bps | 35.3% | N/A |
| **Simple Baseline** | Momentum (Sign $R_{t-1}$) | $19.41\% \pm 1.58\%$ | $32.50\% \pm 1.13\%$ | $0.1856 \pm 0.0110$ | 99.6% | $-0.43$ bps | $-8.43$ bps | 35.5% | $37.7\%$ |
| **Simple Baseline** | Basic RSI (30/70) | $58.51\% \pm 3.88\%$ | $36.56\% \pm 1.42\%$ | $0.3419 \pm 0.0203$ | 13.2% | $+0.97$ bps | $-7.03$ bps | 42.3% | $43.0\%$ |
| **Linear Model** | Logistic Regression | $51.29\% \pm 2.40\%$ | $41.78\% \pm 1.37\%$ | $0.4125 \pm 0.0119$ | 44.8% | $+1.00$ bps | $-7.00$ bps | 40.3% | $44.2\%$ |
| **Non-Linear Tree** | Random Forest (Tuned) | $47.88\% \pm 2.15\%$ | $42.10\% \pm 0.80\%$ | $0.3897 \pm 0.0169$ | 51.5% | $+1.48$ bps | $-6.52 \pm 2.52\text{ bps}$| 41.2% | $43.5\%$ |
| **Gradient Boosting**| XGBoost (Tuned) | $45.10\% \pm 2.20\%$ | $41.76\% \pm 1.90\%$ | $0.4032 \pm 0.0167$ | 57.8% | $+0.78$ bps | $-7.22 \pm 1.56\text{ bps}$| 39.8% | $41.8\%$ |

---

## 3. Comparative Analysis & Model Performance Assessment

### Model Comparison: Logistic Regression and Random Forest are a Statistical Tie
When comparing Logistic Regression and Random Forest, the empirical metrics show no clear superior model:
- **Validation Balanced Accuracy:** Logistic Regression ($42.4\%$) vs. Random Forest ($42.0\%$).
- **Validation Macro-F1:** Logistic Regression ($0.4194$) vs. Random Forest ($0.3920$).
- **Validation Net Return @ 8bp:** Random Forest ($-7.09$ bps) vs. Logistic Regression ($-7.38$ bps) — a marginal difference of $0.29$ bps.
- **Top 10% Precision:** Exactly tied at **$42.9\%$**.
- **Paired Day-Level Block Bootstrap Test:** The difference in gross return between Random Forest and Logistic Regression is $+0.37$ bps, with a 95% confidence interval of $[-2.68, +4.01]$ bps. Because the interval straddles zero, **the performance difference between Random Forest and Logistic Regression is statistically indistinguishable from zero**.

### Performance Hypotheses Regarding Model Differences
The following are working hypotheses regarding why the models behave differently, rather than proven facts:
1. **Low Signal-to-Noise Ratio Hypothesis:** If 5-minute stock returns are dominated by noise rather than signal, sequential boosting (XGBoost) may iteratively adjust weights to fit microstructure noise, leading to slightly lower validation performance ($39.9\%$ BalAcc). In contrast, bagging (Random Forest) averages independent trees, which may dampen variance.
2. **Linear Robustness Hypothesis:** L2-regularized Logistic Regression achieves comparable results to tree ensembles, suggesting that the faint linear predictive relationship (momentum and time of day) is the primary driver of signal, while non-linear interaction terms offer minimal incremental predictive value.

---

## 4. Summary of the Friction Reality
Across all models, the unconstrained gross return per signal hovers between $+0.5$ and $+1.5$ bps, which is within noise of zero. Because the round-trip transaction friction is 8 bps, taking unconstrained trades on $\sim 50\%$ of bars leads to negative net expectancy. Steps 9 and 10 investigate whether confidence gating can isolate a subset of signals where gross return exceeds 8 bps.
