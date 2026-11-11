# TradePulse: Trader User Guide & Operational Manual

**Project:** AI-Powered Intraday Directional Indicator for META (NASDAQ, 5-Minute Bars)  
**Laboratory:** IIT Patna, IMPACT Lab — End-to-End Machine Learning Guidelines  
**Model Version:** `TradePulse-v1.0-final` (Git commit `b57ee06e`, Model Hash `07ef98a8`)  
**Target Horizon:** 15 Minutes (3 consecutive 5-minute bars)  
**Primary Class Label:** Directional threshold $\theta = \pm 15$ bps ($\pm 0.15\%$)  

---

## 1. Executive Summary & One-Second Trader Read

TradePulse is engineered for active intraday traders who need an instantaneous, unambiguous assessment of short-term price momentum without cognitive overload. When glancing at the interface, the **Prediction Card** delivers the essential decision in one second:

```text
+-------------------------------------------------------------+
| PREDICTED DIRECTION (NEXT 15M)                 [LOCKED]     |
| [  UP  ]   68.4% Confidence                                 |
| Conviction: 0.3541 vs Threshold: 0.2635       [GATED: YES]  |
| Down: 22.1%  |  Neutral: 42.4%  |  Up: 35.4%                |
+-------------------------------------------------------------+
```

### Signal Interpretation:
- **`UP` (Green Badge):** Platt-calibrated probability indicates that META is favored to advance by at least $+15$ bps over the next 15 minutes.
- **`DOWN` (Red Badge):** Calibrated probability indicates META is favored to decline by at least $-15$ bps over the next 15 minutes.
- **`NO CLEAR SIGNAL` (Grey Badge):** Directional conviction $\max(P(\text{Down}), P(\text{Up}))$ failed to breach the gating threshold $\tau = 0.2635$. The model advises against taking directional positions during this consolidation bar.

---

## 2. Core Statistical & Machine Learning Principles

### 2.1 Calibrated Confidence vs. Guarantees
The displayed confidence percentage (e.g., `"Up, 68.4%"`) represents the **empirically calibrated posterior probability** from an out-of-fold Platt Sigmoid calibrator. 
> [!IMPORTANT]
> A 68% confidence rating means that under historically identical feature distributions, META exceeded $+15$ bps in approximately 68 out of 100 historical instances. It is **not** a guarantee of profit, nor does it predict the absolute price extreme reached within the 15-minute window.

### 2.2 Zero-Repainting Guarantee
A fatal flaw in commercial technical indicators is "repainting"—modifying prior signals after subsequent candles close. TradePulse strictly enforces zero repainting:
1. Predictions are computed **strictly upon candle close** (evaluated at 3 seconds past every 5-minute mark).
2. The still-forming candle is displayed in a dimmed, dashed style and **never** displays a prediction marker.
3. Every emitted signal is cryptographically appended to `data/live_signals_final.jsonl` with an SHA-256 line hash chained to the prior signal.
4. Subsequent calls on the same bar timestamp return the existing stored record; signals can never mutate.

### 2.3 Conviction Gating ($\tau = 0.2635$)
To avoid overtrading during choppy, low-signal market conditions, TradePulse applies a conviction gating filter:
$$\text{Conviction} = \max\left(P(\text{Down}), P(\text{Up})\right)$$
If $\text{Conviction} < \tau_{\text{final}} = 0.2635$, the prediction is filtered to `Neutral` (`0`). This gates out ~49.7% of unconvincing market noise, delivering approximately 37.5 high-conviction signals per 78-bar trading day.

---

## 3. Operational Protocols: When NOT to Trust the Indicator

Trading signals must be contextualized by market structure. Do not blindly execute signals under the following regimes:

### 3.1 Feed Latency & Stale Data
- **Indicator:** The top bar freshness chip flashes amber/red if the latest closed candle is $> 5$ minutes old during market hours.
- **Action:** If the Alpaca IEX data feed lags, do not trade on delayed signals. Wait until connectivity recovers and real-time candle alignment resumes.

### 3.2 Opening & Closing Volatility Regimes (Session Boundaries)
- **Opening Regime (09:30 – 09:45 ET):** The opening auction generates heavy institutional order matching, widening bid-ask spreads, and extreme price discovery.
- **Closing Regime (15:45 – 16:00 ET):** The closing cross causes sharp index rebalancing distortions.
- **Training Protocol:** The model was trained with the final 3 bars of every session removed to prevent overnight gap distortion. Exercise heightened skepticism on signals generated in the first and last 15 minutes of the trading day.

### 3.3 Scheduled News & Earnings Announcements
- **Fundamental Blindspot:** TradePulse operates strictly on 29 causal price-volume technical indicators. It does **not** process news sentiment, quarterly earnings reports, FOMC interest rate announcements, or macroeconomic releases.
- **Action:** Stand aside 30 minutes before and after major scheduled releases (CPI, PPI, FOMC rate decisions, and META quarterly earnings).

### 3.4 Low-Liquidity & Early-Close Sessions
- Half-day sessions (such as Black Friday or Christmas Eve) feature atypical participant liquidity and suppressed volume profiles.
- Volume-weighted indicators (VWAP distance, RVOL, volume flow ratios) may exhibit distortion.

---

## 4. Academic Honest Disclosure & Transaction Cost Realities

Under IIT Patna IMPACT Lab guidelines, scientific integrity supercedes optimistic claims:

| Metric | Validation Set ($N=3,150$) | Held-Out Test Set ($N=3,525$) | 95% Day-Block CI |
|---|---|---|---|
| **Binary Sign Accuracy** | $50.20\%$ | **$54.11\%$** | $[50.82\%, 57.22\%]$ |
| **Gross Return per Signal** | $+0.42$ bps | **$+1.29$ bps** | $[-1.70, +4.05]$ bps |
| **Institutional Friction (8 bps)** | $-8.00$ bps | **$-8.00$ bps** | Fixed assumption |
| **Net Return per Signal** | $-7.58$ bps | **$-6.71$ bps** | $[-9.70, -3.95]$ bps |
| **Demonstrated Edge After Costs** | **No** | **No (`edge_demonstrated = false`)** | Statistically Negative |

### Key Takeaway for the Trader:
While TradePulse captures genuine statistical structure in intraday price momentum (achieving 54.11% directional accuracy with a 95% bootstrap CI that excludes 50%), **institutional bid-ask execution frictions of 8 bps round-trip exceed the gross edge of +1.29 bps**. Consequently, net realized performance is negative. This indicator is strictly intended for educational and analytical study, never for unhedged automated execution.

---

## 5. Interface Modes & Key Controls

| Hotkey | Action | Description |
|---|---|---|
| `T` | **Test Now** | Triggers instantaneous evaluation of the latest closed candle. |
| `A` | **Auto Mode** | Connects to Server-Sent Events (SSE) stream; automatically updates upon every 5-minute candle close. |
| `R` | **Replay Toggle** | Switches between Live Alpaca feed and Historical Session Replay mode. |
| `?` | **User Guide** | Toggles the interactive, scannable Trader User Guide drawer. |

### Replay Mode (Demo Outside US Hours):
- Select any historical trading session from the dropdown.
- Adjust playback speed (`1x`, `5x`, `20x`).
- Step candle-by-candle (`+5m`) to observe real-time feature computation, gating decisions, and outcome resolutions.
- Replay writes strictly to `data/replay_signals.jsonl`, preserving the official live testing logs intact.

