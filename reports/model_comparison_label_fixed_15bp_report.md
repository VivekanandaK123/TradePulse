# TradePulse - Model Comparison Report on Selected Target (`label_fixed_15bp`)
**Comprehensive Benchmark: Simple Baselines, Logistic Regression, Tuned Random Forest, Tuned XGBoost & Ensemble**  
**Author:** TradePulse Team (IIT Patna, IMPACT Lab)  
**Asset:** META (NASDAQ) | **Timeframe:** 5-minute candles | **Prediction Horizon:** 15-minute (3 bars ahead)  
**Evaluation Partitions:**  
- **Training Set:** Nov 3, 2025 – Jun 30, 2026 ($N = 12,193$ feature-ready bars, 163 sessions)  
- **Validation Set:** Jul 2, 2026 – Aug 31, 2026 ($N = 3,150$ valid bars, 42 sessions)  
- **Test Set:** Sep 2, 2026 – Nov 6, 2026 ($N = 3,525$ valid bars, 47 sessions, **100% SEALED & UNTOUCHED**)  
**Strict Protocol:** Chronological splits with 1-day embargoes; 3-bar purging between CV folds; 26 causal features.

---

## 1. Executive Summary & The Selected Target Label

### Sample Size Reconciliation ($N = 12,193$ vs. $12,228$)
- In raw split accounting, the Training partition contains 12,720 total candles across 164 trading sessions.
- Dropping the final 3 bars of each session (which cannot form a 3-bar forward target within the same session) leaves **12,228** bars with valid forward returns.
- To compute causal rolling technical indicators (with a maximum feature lookback of 35 bars / 175 minutes), the first continuous 35 bars of the training period are reserved as warm-up bars and dropped so feature rows contain zero NaNs.
- Therefore, all feature-based model evaluations in both Cross-Validation and final training utilize strictly **$N = 12,193$ valid training bars**. All tables in this report reflect this reconciled sample size.

### The Selected Target Label: `label_fixed_15bp`
Following the Phase C pre-registered label selection protocol, **`label_fixed_15bp`** was selected by deployability into [`artifacts/metadata/chosen_label_label_expansion.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/chosen_label_label_expansion.json):
$$\theta = \pm 0.15\% \quad (\pm 15\text{ basis points})$$
$$Y_t = \begin{cases} +1 \quad (\text{Up}) & \text{if } R_{t, t+3} \ge +0.0015 \\ -1 \quad (\text{Down}) & \text{if } R_{t, t+3} \le -0.0015 \\ 0 \quad (\text{Neutral}) & \text{if } -0.0015 < R_{t, t+3} < +0.0015 \end{cases}$$

### Class Distribution on `label_fixed_15bp`
| Partition | Total Valid Labeled Bars | Down (-1) Count & % | Neutral (0) Count & % | Up (+1) Count & % | Directional Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Set (Feature-Ready)** | 12,193 bars | 3,026 (24.8%) | 6,192 (50.8%) | 2,975 (24.4%) | 49.2% |
| **Validation Set** | 3,150 bars | 847 (26.9%) | 1,481 (47.0%) | 822 (26.1%) | 53.0% |

### Why `label_fixed_15bp` Replaced `label_fixed_20bp`
1. **Resolution of the Signal Starvation Problem:** Under `label_fixed_20bp`, the Neutral class dominated at 62.2% in training and 57.6% in validation. Consequently, calibrated probability models rarely allowed directional conviction to surpass Neutral, resulting in an unviable live frequency of ~0.7 signals/day with 80% to 96% zero-signal days. Under `label_fixed_15bp`, the Neutral class is reduced to ~50% (50.8% in Train, 47.0% in Val), allowing directional signals to fire at viable operational rates (~5 signals/day).
2. **Preservation of Economic Headroom:** While narrower labels like 5 bp and 10 bp have even smaller neutral classes (19% and 36%), they lack economic headroom over our institutional 8 bps round-trip friction ($0.08\%$ = 5 bps spread + 3 bps slippage). A 5 bps move is unprofitable by design ($5 - 8 = -3$ bps), and a 10 bps move leaves only 2 bps headroom ($10 - 8 = +2$ bps). In contrast, `label_fixed_15bp` preserves $+7$ bps of theoretical headroom ($15 - 8 = +7$ bps).

---

## 2. Reconciled Comparison Tables

### Metric Definitions
- **Accuracy:** Overall multi-class fraction of exact matches across all three classes ($\{-1, 0, +1\}$).
- **Balanced Accuracy:** Macro-averaged recall across all 3 classes: $\frac{1}{3}(\text{Recall}_{-1} + \text{Recall}_0 + \text{Recall}_{+1})$. A naive constant predictor scores $33.3\%$.
- **Macro-F1:** Unweighted mean of F1 scores across the three classes.
- **Coverage %:** Share of bars where the model emits an active directional trade ($\hat{Y}_t \neq 0$).
- **Gross Ret / Trade:** Average realized return $R_{t, t+3} \times \hat{Y}_t$ across all active directional signals, in basis points ($1\text{ bp} = 0.01\%$).
- **Net Ret @ 8bp:** Average realized return per active trade after subtracting institutional transaction friction ($8\text{ bps} = 0.0008$).
- **8bp Win Rate:** Percentage of active trades whose realized return exceeds 8 bps ($(\hat{Y}_t \times R_{t, t+3}) > 0.0008$).
- **Top 10% Cov Prec:** Precision among the top 10% highest-conviction bars (ranked by $\max(P(\text{Down}), P(\text{Up}))$), defined as the percentage of signals where the actual price moved by at least $+8$ bps in the predicted direction. For constant baselines without probability rankings, reported as **N/A**.
- **Binary Sign Accuracy:** Forced directional choice among truly directional bars ($Y_t \in \{-1, +1\}$), matching $\text{sign}(P(\text{Up}) - P(\text{Down})) == Y_t$. An uninformative coin flip evaluates to $50.0\%$.

---

### Table A: Out-of-Sample Performance on Validation Set ($N = 3,150$ bars)
*Evaluated on the unseen Validation partition (Jul 2, 2026 – Aug 31, 2026, 42 trading sessions). All models trained on the full $N = 12,193$ training set:*

| Category | Model Architecture | Accuracy | Balanced Acc | Macro-F1 | Coverage % | Gross Ret / Trade | Net Ret @ 8bp | 8bp Win Rate | Top 10% Cov Prec | Binary Sign Acc (95% CI) | Up-Down ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Baseline** | Majority (Always Neutral) | **47.0%** | 33.3% | 0.2132 | 0.0% | +0.00 bps | +0.00 bps | 0.0% | N/A | 50.0% (50.0, 50.0) | N/A |
| **Simple Baseline** | Always-Up (+1) | 26.1% | 33.3% | 0.1380 | 100.0% | +0.47 bps | -7.53 bps | 35.3% | N/A | 50.0% (50.0, 50.0) | N/A |
| **Simple Baseline** | Momentum (Sign $R_{t-1}$) | 27.1% | 34.0% | 0.2372 | 99.8% | +0.54 bps | -7.46 bps | 36.2% | N/A | 50.0% (50.0, 50.0) | N/A |
| **Simple Baseline** | Basic RSI (30/70) | 46.1% | 35.0% | 0.2946 | 13.8% | -1.21 bps | -9.21 bps | 39.2% | N/A | 50.0% (50.0, 50.0) | N/A |
| **Simple Baseline** | Time-of-Day Baseline | 47.8% | 35.5% | 0.2721 | 8.0% | +4.73 bps | -3.27 bps | 41.3% | 39.3% | 47.9% (45.5, 50.3) | 0.4734 |
| **Linear Model** | Logistic Regression (L2, tuned) | 44.3% | **40.5%** | **0.4036** | 51.8% | +0.76 bps | -7.24 bps | **39.8%** | **42.2%** | **51.0% (48.6, 53.4)** | 0.4953 |
| **Non-Linear Tree** | Random Forest (Tuned) | 43.8% | 40.3% | 0.3809 | 53.4% | +0.28 bps | -7.72 bps | 38.9% | 41.9% | 50.2% (47.8, 52.6) | **0.5085** |
| **Gradient Boosting**| XGBoost (Tuned) | 39.5% | 37.6% | 0.3747 | 62.5% | -1.76 bps | -9.76 bps | 36.7% | 35.2% | 48.7% (46.3, 51.1) | 0.4839 |
| **Model Ensemble** | **Ensemble (50% RF + 50% LR)** | 43.9% | 40.1% | 0.3967 | 52.0% | **+0.92 bps** | **-7.08 bps** | 38.8% | 41.6% | 50.9% (48.5, 53.3) | 0.4910 |

---

### Table B: Cross-Validation Performance on Training Set (5-Fold Purged Walk-Forward CV, $N = 12,193$ bars)
*Evaluated with an expanding chronological window and an enforced 3-bar purge gap (mean $\pm$ std across 5 folds):*

| Category | Model Architecture | Accuracy | Balanced Acc | Macro-F1 | Coverage % | Gross Ret / Trade | Net Ret @ 8bp | 8bp Win Rate | Top 10% Cov Prec | Binary Sign Acc | Up-Down ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Baseline** | Majority (Always Neutral) | $48.6\% \pm 3.0\%$ | $33.3\% \pm 0.0\%$ | $0.2180 \pm 0.0089$ | 0.0% | $+0.00$ bps | $+0.00$ bps | 0.0% | N/A | 50.0% | N/A |
| **Simple Baseline** | Always-Up (+1) | $24.8\% \pm 0.9\%$ | $33.3\% \pm 0.0\%$ | $0.1323 \pm 0.0037$ | 100.0% | $+0.03$ bps | $-7.97$ bps | 35.3% | N/A | 50.0% | N/A |
| **Simple Baseline** | Momentum (Sign $R_{t-1}$) | $25.0\% \pm 1.6\%$ | $32.2\% \pm 1.2\%$ | $0.2204 \pm 0.0097$ | 99.6% | $-0.43$ bps | $-8.43$ bps | 35.6% | N/A | 50.0% | N/A |
| **Simple Baseline** | Basic RSI (30/70) | $48.4\% \pm 3.4\%$ | $35.9\% \pm 1.4\%$ | $0.3056 \pm 0.0203$ | 13.2% | $+0.97$ bps | $-7.03$ bps | 42.3% | N/A | 50.0% | N/A |
| **Simple Baseline** | Time-of-Day Baseline | $49.4\% \pm 3.0\%$ | $35.9\% \pm 0.1\%$ | $0.2821 \pm 0.0082$ | 9.6% | $-5.65$ bps | $-13.65$ bps | 40.2% | 40.3% | 50.2% | 0.5076 |
| **Linear Model** | Logistic Regression (L2) | $44.6\% \pm 2.1\%$ | $39.6\% \pm 1.0\%$ | $0.3923 \pm 0.0094$ | 48.5% | $+0.67$ bps | $-7.33$ bps | 39.2% | $45.1\%$ | 49.8% | 0.5100 |
| **Non-Linear Tree** | Random Forest (Tuned) | $43.8\% \pm 3.5\%$ | $\mathbf{40.0\% \pm 1.2\%}$ | $0.3729 \pm 0.0158$ | 52.1% | $+0.58$ bps | $-7.42$ bps | 38.8% | $\mathbf{45.4\%}$ | 50.9% | $\mathbf{0.5202}$ |
| **Gradient Boosting**| XGBoost (Tuned) | $42.4\% \pm 2.4\%$ | $39.6\% \pm 1.8\%$ | $\mathbf{0.3922 \pm 0.0184}$ | 57.5% | $+0.54$ bps | $-7.46$ bps | 38.9% | 38.3% | $\mathbf{51.1\%}$ | 0.5176 |
| **Model Ensemble** | **Ensemble (50% RF + 50% LR)** | $\mathbf{44.6\% \pm 2.4\%}$ | $39.8\% \pm 1.3\%$ | $0.3905 \pm 0.0124$ | 49.6% | $+0.42$ bps | $-7.58$ bps | 38.9% | $45.0\%$ | 50.2% | 0.5114 |

---

### Table C: Per-Class Breakdown on Validation Set ($N = 3,150$ bars)
*Class-level precision, recall, and F1 scores under `label_fixed_15bp`:*

| Model Architecture | Class | Precision | Recall | F1-Score | Confusion Matrix $[N_{\text{Down}}, N_{\text{Neutral}}, N_{\text{Up}}]$ |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **Down (-1)** | 32.8% | 27.3% | 0.2977 | Pred: [231 Down, 335 Neut, 281 Up] (True Down $N=847$) |
| | **Neutral (0)** | 57.8% | 59.2% | 0.5849 | Pred: [245 Down, 877 Neut, 359 Up] (True Neut $N=1,481$) |
| | **Up (+1)** | 31.0% | 34.9% | 0.3282 | Pred: [229 Down, 306 Neut, 287 Up] (True Up $N=822$) |
| **Random Forest (Tuned)** | **Down (-1)** | 26.5% | 12.0% | 0.1656 | Pred: [102 Down, 319 Neut, 426 Up] (True Down $N=847$) |
| | **Neutral (0)** | 58.2% | 57.7% | 0.5797 | Pred: [177 Down, 855 Neut, 449 Up] (True Neut $N=1,481$) |
| | **Up (+1)** | 32.5% | 51.2% | 0.3975 | Pred: [106 Down, 295 Neut, 421 Up] (True Up $N=822$) |
| **XGBoost (Tuned)** | **Down (-1)** | 27.8% | 29.2% | 0.2847 | Pred: [247 Down, 248 Neut, 352 Up] (True Down $N=847$) |
| | **Neutral (0)** | 58.6% | 46.7% | 0.5199 | Pred: [364 Down, 692 Neut, 425 Up] (True Neut $N=1,481$) |
| | **Up (+1)** | 28.1% | 37.0% | 0.3195 | Pred: [277 Down, 241 Neut, 304 Up] (True Up $N=822$) |
| **Ensemble (50% RF + 50% LR)** | **Down (-1)** | 31.6% | 22.5% | 0.2631 | Pred: [191 Down, 332 Neut, 324 Up] (True Down $N=847$) |
| | **Neutral (0)** | 57.8% | 59.0% | 0.5840 | Pred: [216 Down, 874 Neut, 391 Up] (True Neut $N=1,481$) |
| | **Up (+1)** | 30.8% | 38.7% | 0.3429 | Pred: [198 Down, 306 Neut, 318 Up] (True Up $N=822$) |

---

## 3. Deep-Dive Comparative Analysis

### 3.1 Model Comparison: Statistical Parity Between LR, RF, and Ensemble; XGBoost Lags
1. **Balanced Accuracy and Macro-F1:**
   - On Validation Balanced Accuracy, **Logistic Regression ($40.5\%$)**, **Random Forest ($40.3\%$)**, and the **Ensemble ($40.1\%$)** are within $0.4\%$ of each other. In 5-Fold Cross-Validation, RF ($40.0\% \pm 1.2\%$), LR ($39.6\% \pm 1.0\%$), and Ensemble ($39.8\% \pm 1.3\%$) are statistically indistinguishable.
   - **XGBoost lags notably on Validation**: its balanced accuracy drops to **$37.6\%$** (compared to $40.5\%$ for LR), and its Macro-F1 is **$0.3747$** (vs. $0.4036$ for LR).
2. **Net Realized Return at 8 bps Friction:**
   - The unconstrained net return per trade is virtually identical across the three top models: **Ensemble ($-7.08$ bps)**, **Logistic Regression ($-7.24$ bps)**, and **Random Forest ($-7.72$ bps)**.
   - In contrast, **XGBoost produces substantially higher drag: $-9.76$ bps per trade** (a $-2.68$ bps worse penalty than the Ensemble).
3. **High-Conviction Subset (Top 10% Conviction):**
   - When filtering for the top 10% most confident bars, Logistic Regression achieves the highest precision (**$42.2\%$**, gross return $+4.83$ bps, net $-3.17$ bps).
   - Random Forest achieves **$41.9\%$** precision (gross $+2.59$ bps, net $-5.41$ bps).
   - The Ensemble achieves **$41.6\%$** precision (gross $+3.15$ bps, net $-4.85$ bps).
   - XGBoost drops to **$35.2\%$** precision, generating a negative gross return ($-1.36$ bps) and a severe net return of **$-9.36$ bps**.
4. **Binary Direction-Only Sign Accuracy:**
   - Forced choice between Up and Down on directional bars ($Y \in \{-1, +1\}$):
     - Logistic Regression: **$51.0\%$** (95% CI: $[48.6\%, 53.4\%]$)
     - Ensemble: **$50.9\%$** (95% CI: $[48.5\%, 53.3\%]$)
     - Random Forest: **$50.2\%$** (95% CI: $[47.8\%, 52.6\%]$)
     - XGBoost: **$48.7\%$** (95% CI: $[46.3\%, 51.1\%]$)
   - Every confidence interval comfortably spans $50.0\%$, confirming that binary sign predictions are statistically indistinguishable from a random coin toss.

---

### 3.2 Performance Hypotheses: Why Do These Differences Arise?
The observed empirical differences suggest several working hypotheses:
1. **Noise Overfitting in Sequential Boosting (XGBoost):** In ultra-low signal-to-noise intraday regimes (5-minute equity returns), gradient boosting iteratively reweights misclassified samples. In doing so, it appears to fit microstructure noise and transient bid-ask bounces, degrading out-of-sample generalization.
2. **Bagging Variance Reduction (Random Forest):** Random Forest fits independent trees on bootstrap samples and averages predictions across the forest. This bagging architecture dampens variance and provides a natural regularizer against intraday noise spikes.
3. **Linear Parsimony (Logistic Regression):** L2-regularized Logistic Regression matches or exceeds tree models on Macro-F1 and balanced accuracy. This indicates that whatever faint predictability exists in the features operates linearly (e.g., short-term momentum and time-of-day decay), and complex non-linear feature interactions do not contribute reliable incremental edge.
4. **Ensemble Smoothing:** The 50/50 Ensemble combines the probability estimates of Random Forest and Logistic Regression, dampening idiosyncratic probability miscalibrations and producing the lowest net transaction drag ($-7.08$ bps) among all multi-class models.

---

### 3.3 Analysis of Simple Baselines
1. **Majority Class (Always Neutral):**
   - Shows the highest raw accuracy ($47.0\%$ in Val, $48.6\%$ in CV) solely by predicting Neutral on every bar.
   - However, its Balanced Accuracy is strictly $33.3\%$ and its coverage is $0\%$, yielding zero operational utility as a trading indicator.
2. **Always-Up (+1):**
   - Emits an Up signal on 100% of bars. It captures the overall market upward drift ($+0.47$ bps gross return), but suffers catastrophic friction drag: after 8 bps costs, it loses **$-7.53$ bps per bar**.
3. **Momentum (Sign $R_{t-1}$):**
   - Following the previous 5-minute bar's direction achieves $27.1\%$ accuracy and $34.0\%$ balanced accuracy.
   - It captures $+0.54$ bps gross return, which is wiped out by costs to **$-7.46$ bps net return**, demonstrating that 5-minute trend-following on single stocks is destroyed by transaction friction.
4. **Basic RSI (30/70):**
   - Emits signals on $13.8\%$ of bars. Gross return is negative ($-1.21$ bps), yielding **$-9.21$ bps net return**, indicating that simple 14-period RSI extremes do not predict 15-minute reversals on META.
5. **Time-of-Day Baseline:**
   - Predicts class based solely on the 30-minute intraday time slot.
   - While it achieves $+4.73$ bps gross return on Validation due to opening-range volatility clustering, its binary sign accuracy is **$47.9\%$** (below 50%) and its Up-Down AUC is $0.4734$. It learns when the market moves, not which way it moves.

---

## 4. Probability Calibration Assessment on `label_fixed_15bp`

All models were evaluated with Platt Sigmoid calibration fit strictly on the 5-fold Training Out-of-Fold predictions:

| Model Architecture | Raw Log Loss | Calibrated Log Loss | Brier Score | Reliability Curve Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 1.0958 | **1.0732** | **0.6493** | Monotonic, well-aligned with diagonal across all classes. |
| **Random Forest (Tuned)** | **1.0771** | 1.0874 | 0.6599 | Raw probabilities slightly compressed; Platt calibration expands dynamic range. |
| **XGBoost (Tuned)** | 1.1074 | 1.0781 | 0.6529 | Substantial reduction in log loss after sigmoid scaling, but direction accuracy remains low. |
| **Ensemble (50% RF + 50% LR)**| 1.0790 | 1.0785 | 0.6534 | **Best calibrated balance**: smooth sigmoid curve with empirical probabilities closely matching predictions. |

---

## 5. Summary of the Friction Reality & Verdict

1. **Unconstrained Signal Expectancy is Dominated by Friction:**  
   Across all unconstrained ML models, gross return per signal hovers between $+0.3$ and $+0.9$ bps. Because the institutional round-trip friction is 8 bps, trading unconstrained signals on $\sim 52\%$ of bars produces an average loss of $-7.08$ to $-7.72$ bps per trade.
2. **Conviction Gating Lifts Gross Return but Remains Sub-Hurdle:**  
   Focusing exclusively on the top 10% highest-conviction signals boosts gross return to $+2.6$ to $+4.8$ bps. While this represents a $4\times$ to $5\times$ expansion in gross expectancy, it remains below the 8 bps transaction cost hurdle, resulting in net returns of $-3.17$ to $-5.41$ bps.
3. **Selected Deployment Configuration:**  
   The **50/50 Ensemble (Random Forest + Logistic Regression)** with Platt Sigmoid calibration is selected as the production architecture. Calibrated to an operational frequency of ~5 signals/day (conviction threshold $\tau = 0.2208$), it delivers **4.8 signals/day** on unseen validation data, resolving the signal deprivation of earlier thresholds while maintaining full empirical transparency that no profitable net edge exists after transaction costs.

