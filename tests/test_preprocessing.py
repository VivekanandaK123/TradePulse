"""
TradePulse - Preprocessing Unit Tests (Built-in unittest)
Author: TradePulse Team (IIT Patna, IMPACT Lab)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import (
    synthesize_missing_bar,
    build_daily_5min_grid,
    clean_and_reindex_dataset,
)
from ml.validation import validate_cleaned_dataframe


class TestPreprocessing(unittest.TestCase):

    def test_synthesize_missing_bar(self):
        ts = pd.Timestamp("2026-11-06 14:35:00+00:00")
        synth = synthesize_missing_bar(
            symbol="META",
            timestamp=ts,
            last_known_close=600.5,
            last_known_vwap=600.2,
        )

        self.assertEqual(synth["symbol"], "META")
        self.assertEqual(synth["timestamp"], ts)
        self.assertEqual(synth["open"], 600.5)
        self.assertEqual(synth["high"], 600.5)
        self.assertEqual(synth["low"], 600.5)
        self.assertEqual(synth["close"], 600.5)
        self.assertEqual(synth["volume"], 0.0)
        self.assertEqual(synth["trade_count"], 0.0)
        self.assertEqual(synth["vwap"], 600.2)
        self.assertEqual(synth["is_filled"], 1)

    def test_build_daily_5min_grid_full_day(self):
        open_ts = pd.Timestamp("2026-11-06 14:30:00+00:00")
        close_ts = pd.Timestamp("2026-11-06 21:00:00+00:00")
        grid = build_daily_5min_grid(open_ts, close_ts)

        self.assertEqual(len(grid), 78)
        self.assertEqual(grid[0], open_ts)
        self.assertEqual(grid[-1], pd.Timestamp("2026-11-06 20:55:00+00:00"))

    def test_build_daily_5min_grid_half_day(self):
        open_ts = pd.Timestamp("2025-11-28 14:30:00+00:00")
        close_ts = pd.Timestamp("2025-11-28 18:00:00+00:00")
        grid = build_daily_5min_grid(open_ts, close_ts)

        self.assertEqual(len(grid), 42)
        self.assertEqual(grid[0], open_ts)
        self.assertEqual(grid[-1], pd.Timestamp("2025-11-28 17:55:00+00:00"))

    def test_leading_missing_bar_reindexing(self):
        """
        Verifies that when leading bars of a session are missing,
        the previous regular session close is used causally with is_filled=1.
        """
        # Day 1: full 78 bars
        day1_grid = build_daily_5min_grid(
            pd.Timestamp("2026-01-05 14:30:00+00:00"),
            pd.Timestamp("2026-01-05 21:00:00+00:00"),
        )
        df1 = pd.DataFrame({
            "symbol": "META",
            "timestamp": day1_grid,
            "open": 600.0,
            "high": 601.0,
            "low": 599.0,
            "close": 600.5,
            "volume": 1000.0,
            "trade_count": 50.0,
            "vwap": 600.2,
        })

        # Day 2: missing the first 2 bars (9:30 and 9:35)
        day2_grid = build_daily_5min_grid(
            pd.Timestamp("2026-01-06 14:30:00+00:00"),
            pd.Timestamp("2026-01-06 21:00:00+00:00"),
        )
        df2 = pd.DataFrame({
            "symbol": "META",
            "timestamp": day2_grid[2:],  # drop bar 0 and bar 1
            "open": 605.0,
            "high": 606.0,
            "low": 604.0,
            "close": 605.5,
            "volume": 1200.0,
            "trade_count": 60.0,
            "vwap": 605.2,
        })

        mock_raw = pd.concat([df1, df2], ignore_index=True)
        clean = clean_and_reindex_dataset(mock_raw, symbol="META")

        self.assertEqual(len(clean), 156)

        # Bars 78 and 79 (first two bars of Day 2) should be filled with Day 1 close (600.5)
        bar78 = clean.iloc[78]
        bar79 = clean.iloc[79]

        self.assertEqual(bar78["is_filled"], 1)
        self.assertEqual(bar78["close"], 600.5)
        self.assertEqual(bar78["open"], 600.5)
        self.assertEqual(bar78["volume"], 0.0)

        self.assertEqual(bar79["is_filled"], 1)
        self.assertEqual(bar79["close"], 600.5)
        self.assertEqual(bar79["volume"], 0.0)

        # Bar 80 (9:40 ET) is observed
        bar80 = clean.iloc[80]
        self.assertEqual(bar80["is_filled"], 0)
        self.assertEqual(bar80["close"], 605.5)

    def test_cleaned_validation_passes(self):
        csv_path = PROJECT_ROOT / "data" / "processed" / "META_5min_clean.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            valid, issues = validate_cleaned_dataframe(df)
            self.assertTrue(valid, f"Cleaned validation failed: {issues}")


if __name__ == "__main__":
    unittest.main()
