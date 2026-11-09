# TradePulse: Feature Engineering Documentation
**Author:** TradePulse Team (IIT Patna, IMPACT Lab)  
**Pipeline Module:** [`ml/features.py`](file:///e:/Projects/Trading%20Project/TradePulse/ml/features.py)  
**Fitted Scaler Artifact:** [`artifacts/models/feature_scaler.joblib`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/models/feature_scaler.joblib)  
**Feature Metadata:** [`artifacts/metadata/feature_metadata.json`](file:///e:/Projects/Trading%20Project/TradePulse/artifacts/metadata/feature_metadata.json)  

---

## 1. Architectural Overview & Causal Design Rules

The feature pipeline transforms 5-minute OHLCV bars into a 29-dimensional stationary feature space. It strictly satisfies IIT Patna IMPACT Lab leakage prevention criteria:

1. **Strictly Causal:** Every feature at bar $t$ relies exclusively on information known at or before bar $t$. Zero future shift or lookahead bias.
2. **Shared Training & Live Pipeline:** The exact same function `compute_features()` is called during training, walk-forward cross-validation, and live inference.
3. **IEX-Relative Normalization:** Because the free Alpaca IEX feed represents a variable fraction of total SIP volume, raw share volume is never fed directly into models. All volume and trade quantities are expressed as ratios relative to rolling historical averages.
4. **Warm-Up Requirement:** The maximum historical lookback is **35 bars** (~175 minutes, or ~3 hours of trading). In live production, the backend requires a rolling buffer of 35 closed bars before generating signals.
5. **Robust Scaling:** Scalers (`RobustScaler`) are fit **strictly on training partition rows** using median and Interquartile Range (IQR) to prevent leptokurtic jump distortion.

---

## 2. Comprehensive Feature Catalog (29 Features)

### A. Trailing Multi-Horizon Returns (4 Features)
Captures short-to-medium term momentum and price velocity across multiple candle lookbacks.

| Feature Name | Definition / Formula | Lookback | Trading & Economic Rationale |
|---|---|---|---|
| `feat_ret_1b` | $\frac{Close_t}{Close_{t-1}} - 1$ | 1 bar (5 min) | Immediate 1-bar price momentum; captures immediate reaction. |
| `feat_ret_3b` | $\frac{Close_t}{Close_{t-3}} - 1$ | 3 bars (15 min) | Matches the prediction horizon; detects medium-frequency momentum. |
| `feat_ret_6b` | $\frac{Close_t}{Close_{t-6}} - 1$ | 6 bars (30 min) | Half-hour trend velocity; identifies persistent directional drift. |
| `feat_ret_12b` | $\frac{Close_t}{Close_{t-12}} - 1$ | 12 bars (60 min) | Hourly momentum; differentiates minor noise from broader session trends. |

---

### B. Volatility & Candle Geometry (5 Features)
Measures intraday price dispersion, buying/selling pressure within the candle, and regime shifts.

| Feature Name | Definition / Formula | Lookback | Trading & Economic Rationale |
|---|---|---|---|
| `feat_vol_6b` | $\text{Std}(R_{t-5 \dots t})$ | 6 bars (30 min) | Short-term realized volatility; alerts model to sudden volatility spikes. |
| `feat_vol_12b` | $\text{Std}(R_{t-11 \dots t})$ | 12 bars (60 min) | Baseline 1-hour realized volatility; normalizes signal conviction. |
| `feat_bar_range` | $\frac{High_t - Low_t}{Close_t}$ | 1 bar (5 min) | Total bar range normalized by price; proxies instantaneous volatility. |
| `feat_body_size` | $\frac{\|Close_t - Open_t\|}{Close_t}$ | 1 bar (5 min) | Candle real-body magnitude; distinguishes decisive bars from indecisive dojis. |
| `feat_close_pos` | $\frac{Close_t - Low_t}{High_t - Low_t + \epsilon}$ | 1 bar (5 min) | Close position in bar range $[0, 1]$. Values $\approx 1$ indicate strong closing buy pressure; $\approx 0$ indicate aggressive selling pressure. |

---

### C. Volume-Weighted Average Price (VWAP) Relationships (2 Features)
VWAP represents institutional benchmark execution price; deviations signal intraday overextension or mean reversion.

| Feature Name | Definition / Formula | Lookback | Trading & Economic Rationale |
|---|---|---|---|
| `feat_vwap_dist` | $\frac{Close_t - VWAP_t}{VWAP_t}$ | Current session | Relative premium or discount to the session VWAP. Classic institutional mean-reversion feature. |
| `feat_vwap_slope_3b` | $\frac{VWAP_t - VWAP_{t-3}}{VWAP_{t-3}}$ | 3 bars (15 min) | Rate of change in the institutional benchmark, indicating whether liquidity is shifting higher or lower. |

---

### D. Relative Volume & Order Flow (5 Features)
Overcomes the partial liquidity limitation of the IEX feed by tracking normalized participation surges.

| Feature Name | Definition / Formula | Lookback | Trading & Economic Rationale |
|---|---|---|---|
| `feat_rvol_12b` | $\frac{Volume_t}{\text{Mean}(Volume_{t-11 \dots t})}$ | 12 bars (1 hr) | 1-Hour Relative Volume (RVOL). Values $> 2.0$ flag institutional volume breakouts. |
| `feat_rvol_24b` | $\frac{Volume_t}{\text{Mean}(Volume_{t-23 \dots t})}$ | 24 bars (2 hr) | 2-Hour RVOL; confirms whether elevated activity is sustained or transient. |
| `feat_rtrades_12b` | $\frac{Trades_t}{\text{Mean}(Trades_{t-11 \dots t})}$ | 12 bars (1 hr) | Normalized trade count; measures participation breadth across market participants. |
| `feat_avg_trade_size_ratio` | $\frac{\text{TradeSize}_t}{\text{Mean}(\text{TradeSize}_{t-11 \dots t})}$ | 12 bars (1 hr) | Average shares per trade relative to average. Spikes indicate block-order institutional activity. |
| `feat_filled_density_12b` | $\sum_{i=0}^{11} is\_filled_{t-i}$ | 12 bars (1 hr) | Count of synthetic bars in last hour; informs the model when trading is thin vs. genuine consolidation. |

---

### E. Classic Technical Oscillators & Moving Averages (8 Features)
Standard technical indicators adapted to 5-minute bars and normalized for stationarity.

| Feature Name | Definition / Formula | Lookback | Trading & Economic Rationale |
|---|---|---|---|
| `feat_rsi_14` | $\text{Wilder RSI}(14) / 100.0$ | 14 bars (70 min) | Normalized to $[0, 1]$. Values $> 0.70$ flag overbought conditions; $< 0.30$ flag oversold conditions. |
| `feat_macd_line` | $\frac{\text{EMA}_{12} - \text{EMA}_{26}}{Close_t}$ | 26 bars (130 min) | Price-normalized MACD line; measures trend acceleration across time horizons. |
| `feat_macd_signal` | $\text{EMA}_9(\text{macd\_line})$ | 35 bars (175 min) | Signal line smoothing for the normalized MACD. |
| `feat_macd_hist` | $\text{macd\_line} - \text{macd\_signal}$ | 35 bars (175 min) | MACD histogram; leading oscillator detecting loss of directional momentum before price turns. |
| `feat_bb_pct_b` | $\frac{Close_t - Lower_{20}}{Upper_{20} - Lower_{20}}$ | 20 bars (100 min) | Bollinger %B $[0, 1]$. Identifies statistical band touches, breakouts ($> 1.0$), and breakdowns ($< 0.0$). |
| `feat_bb_width` | $\frac{Upper_{20} - Lower_{20}}{Middle_{20}}$ | 20 bars (100 min) | Bollinger Bandwidth; identifies low-volatility squeezes that precede explosive directional expansions. |
| `feat_ema_dist_9` | $\frac{Close_t - \text{EMA}_9}{Close_t}$ | 9 bars (45 min) | Fast moving average displacement; measures short-term extension from trend. |
| `feat_ema_dist_21` | $\frac{Close_t - \text{EMA}_{21}}{Close_t}$ | 21 bars (105 min) | Medium moving average displacement; provides dynamic support/resistance reference. |

---

### F. Session Context & Temporal Features (5 Features)
Encodes exchange calendar dynamics and the pronounced intraday volume/volatility U-curve.

| Feature Name | Definition / Formula | Lookback | Trading & Economic Rationale |
|---|---|---|---|
| `feat_time_sin` | $\sin\left(\frac{2\pi \cdot (\text{minute} - 570)}{390}\right)$ | Current bar | Continuous cyclical representation of time of day (starts at 0.0 at 9:30 AM open). |
| `feat_time_cos` | $\cos\left(\frac{2\pi \cdot (\text{minute} - 570)}{390}\right)$ | Current bar | Quadrature companion to `feat_time_sin`; preserves distance between trading phases. |
| `feat_day_of_week` | $\frac{\text{DayOfWeek}}{4.0}$ | Current day | Normalized day index ($0.0 = \text{Monday}, 1.0 = \text{Friday}$). Captures day-of-week liquidity patterns. |
| `feat_ret_since_open` | $\frac{Close_t}{Open_{\text{session}}} - 1$ | Current session | Cumulative intraday session gain/loss. Distinguishes trend days from range-bound chop. |
| `feat_overnight_gap` | $\frac{Open_{\text{session}}}{Close_{\text{prev\_session}}} - 1$ | 1 session (causal) | Gap at 9:30 AM open relative to previous 16:00 close. Anchors opening price discovery regime. |

---

## 3. Dataset Outputs

The pipeline executed and partitioned feature matrices into:
- [`data/processed/train_features.csv`](file:///e:/Projects/Trading%20Project/TradePulse/data/processed/train_features.csv): **12,685 bars** (Train partition after dropping 35 warm-up bars)
- [`data/processed/val_features.csv`](file:///e:/Projects/Trading%20Project/TradePulse/data/processed/val_features.csv): **3,276 bars** (Validation partition)
- [`data/processed/test_features.csv`](file:///e:/Projects/Trading%20Project/TradePulse/data/processed/test_features.csv): **3,666 bars** (Test partition, untouched)

