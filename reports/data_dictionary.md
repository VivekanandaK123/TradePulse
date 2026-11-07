# TradePulse: Data Dictionary

**Dataset:** `META_5min_clean`  
**Description:** Validated and calendar-reindexed 5-minute OHLCV bars for META (NASDAQ) from Alpaca IEX feed.  
**Frequency:** 5 minutes  
**Sessions:** Regular trading sessions (9:30 AM - 4:00 PM ET; 9:30 AM - 1:00 PM ET on half-days)  

## Schema & Specifications

| Column | Type | Unit | Description | Missing / Synthetic Policy |
|---|---|---|---|---|
| `symbol` | `string` | ticker | Equity ticker symbol ('META'). | Non-nullable. |
| `timestamp` | `datetime64[ns, UTC]` | ISO-8601 UTC | Bar start timestamp in UTC. | Reindexed to NASDAQ calendar grid. |
| `timestamp_ny` | `datetime64[ns, America/New_York]` | ISO-8601 ET | Bar start timestamp in local exchange time (America/New_York). | Derived from timestamp. |
| `open` | `float64` | USD | Opening trade price in the 5-minute interval. | Forward-filled from prior close if is_filled=1. |
| `high` | `float64` | USD | Highest trade price in the 5-minute interval. | Forward-filled from prior close if is_filled=1. |
| `low` | `float64` | USD | Lowest trade price in the 5-minute interval. | Forward-filled from prior close if is_filled=1. |
| `close` | `float64` | USD | Closing trade price in the 5-minute interval. | Forward-filled from prior close if is_filled=1. |
| `volume` | `float64` | shares | Total shares executed on IEX in the 5-minute interval. | 0.0 if is_filled=1. |
| `trade_count` | `float64` | trades | Total number of transactions executed on IEX in the interval. | 0.0 if is_filled=1. |
| `vwap` | `float64` | USD | Volume-Weighted Average Price for the interval. | Forward-filled from prior vwap if is_filled=1. |
| `is_filled` | `int64` | binary flag | Data quality flag: 0 = observed market trade bar; 1 = synthesized missing bar. | Non-nullable. |
