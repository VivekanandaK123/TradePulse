# TradePulse: Step 2 Data Audit & Cleaning Report
**Author:** TradePulse Team (IIT Patna, IMPACT Lab)  
**Date:** 2026-11-07 18:18:47  
**Asset:** META (NASDAQ) | **Timeframe:** 5-minute | **Feed:** Alpaca IEX  

---

## 1. Executive Summary
- **Raw Observations:** 20,516 total bars (including pre-market and post-market sessions).
- **Cleaned Regular Session Bars:** **19,818 bars** across **255 trading days**.
- **Exchange Schedule Conformance:** Exactly matches `pandas-market-calendars` NASDAQ regular hours (78 bars per standard day, 42 bars per half-day).
- **Missing Bar Fill Rate:** **0.00%** (0 bars filled out of 19,818).
- **Data Quality Assessment:** PASSED. All geometric OHLC invariants, non-negativity checks, VWAP bounds, and time grid alignments passed with zero violations.

---

## 2. Validation Checks

### Raw Data Integrity Checks
- **Required Columns Present:** Yes (`symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `vwap`)
- **Null Values:** 0
- **Geometric Invariants ($High \ge Max(Open, Close)$, $Low \le Min(Open, Close)$):** 100% Valid (20,516/20,516)
- **Positive Prices & Non-negative Volume:** 100% Valid
- **VWAP Bounds ($Low \le VWAP \le High$):** 100% Valid
- **Raw Validation Status:** PASSED

### Cleaned Regular Session Integrity Checks
- **Grid Completeness:** 19,818 scheduled regular session bars verified against NASDAQ calendar.
- **Overnight Gap / Spikes:** Zero regular-session 5-minute returns exceeding $\pm 3\%$.
- **Cleaned Validation Status:** PASSED

---

## 3. Fill Rate Analysis by Time of Day (America/New_York)

| Hour (ET) | Total Scheduled Bars | Filled Bars | Fill Rate (%) | Avg Volume (Shares) | Avg Trade Count |
|---|---|---|---|---|---|
| 09:00 | 1,530 | 0 | 0.00% | 15,938 | 343.5 |
| 10:00 | 3,060 | 0 | 0.00% | 8,249 | 204.3 |
| 11:00 | 3,060 | 0 | 0.00% | 6,035 | 157.7 |
| 12:00 | 3,060 | 0 | 0.00% | 5,051 | 135.1 |
| 13:00 | 3,036 | 0 | 0.00% | 4,584 | 126.9 |
| 14:00 | 3,036 | 0 | 0.00% | 4,720 | 130.5 |
| 15:00 | 3,036 | 0 | 0.00% | 10,851 | 299.8 |

> [!NOTE]
> **Key Finding on IEX Liquidity:** META exhibits exceptional liquidity on the IEX feed. Every single 5-minute regular session interval in the 1-year window had trades executed on IEX, resulting in **0 missing bars (0.00% fill rate)**. The fill policy (`synthesize_missing_bar` with `is_filled=1`) is fully implemented and tested for future live stream resiliency.

---

## 4. Intraday Return Statistics (5-Minute Regular Session)
- **Mean Return:** 0.0005%
- **Standard Deviation:** 0.189%
- **Minimum 5-Min Return:** -2.91%
- **Maximum 5-Min Return:** 1.70%
- **Kurtosis:** 10.17 (moderate fat tails, typical for high-frequency equity returns)
- **Skewness:** -0.07
- **Spikes > 3%:** 0
- **Spikes > 2%:** 2

---

## 5. Artifacts Produced
- **Cleaned Dataset (CSV):** `data/processed/META_5min_clean.csv`
- **Cleaned Dataset (Parquet):** `data/processed/META_5min_clean.parquet`
- **Data Dictionary:** `reports/data_dictionary.md`
- **Validation Figures:**
  - Hourly volume & trade profile: `reports/figures/hourly_volume_profile.png`
  - Fill rate by hour: `reports/figures/missing_bars_by_hour.png`
  - Intraday return distribution: `reports/figures/intraday_return_distribution.png`
  - 1-Year price trajectory: `reports/figures/price_trajectory.png`
