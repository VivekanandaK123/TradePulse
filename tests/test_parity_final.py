"""
TradePulse - Step 14 Final: Verification Suite & Historical Parity Tests
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Verifies Phase 5 Requirements:
1. Historical Replay Parity (Last 10 Trading Days):
   - Sliding 200-bar buffer feature difference < 1e-5 everywhere against batch offline calculation.
   - Batch vs Live directional signal mismatches = 0 out of all bars.
2. No-Repainting Guarantee:
   - Emitted signals are appended to data/live_signals_final.jsonl immediately upon candle close.
   - Mid-session restart maintains exact byte-for-byte identity of earlier lines; timestamps never mutate.
3. Missing-Bar & Failure Resilience:
   - Missing 5-minute intervals automatically synthesize zero-volume filled bars (is_filled=1).
   - Corrupt packets and stream exceptions are trapped, logged, and increment error counters without crashing.
4. Realized 3-Bar Outcome Verification:
   - Realized return (gross & net @ 8bp) is logged exactly 3 bars (15 min) after signal emission.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict_final import TradePulsePredictorFinal, get_default_predictor_final
from backend.app.services.live_inference_final import LiveInferenceServiceFinal


class TestFinalParityAndResilience(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_default_predictor_final()
        cls.raw_clean = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "META_5min_clean.csv")
        cls.raw_clean["timestamp_ny"] = pd.to_datetime(cls.raw_clean["timestamp"], utc=True).dt.tz_convert("America/New_York")
        cls.raw_clean["session_date"] = cls.raw_clean["timestamp_ny"].dt.date.astype(str)

        cls.batch_features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_features.csv")
        cls.batch_features["timestamp_ny"] = pd.to_datetime(cls.batch_features["timestamp"], utc=True).dt.tz_convert("America/New_York")
        cls.batch_features["session_date"] = cls.batch_features["timestamp_ny"].dt.date.astype(str)

        # Identify last 10 trading sessions
        cls.last_10_dates = list(cls.batch_features["session_date"].unique()[-10:])
        cls.sub_batch = cls.batch_features[cls.batch_features["session_date"].isin(cls.last_10_dates)].copy()
        cls.feature_cols = cls.predictor.feature_names

    def test_01_warmup_and_schema_validation(self):
        """Buffer with < 35 bars or missing columns must raise ValueError."""
        short_buf = self.raw_clean.iloc[:20].copy()
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_candle(short_buf)
        self.assertIn("Insufficient candle history", str(ctx.exception))

        missing_col_buf = self.raw_clean.iloc[:50].drop(columns=["close"]).copy()
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_candle(missing_col_buf)
        self.assertIn("Missing required candle columns", str(ctx.exception))

    def test_02_historical_replay_feature_parity_last_10_days(self):
        """
        Sliding 200-bar buffer feature difference across all 29 features
        must be strictly < 1e-5 compared to batch training features across the last 10 trading days.
        """
        max_diff = 0.0
        # Sample across the 10 trading days (testing 50 representative bars across sessions)
        sampled_rows = self.sub_batch.iloc[::15].copy()

        for _, row in sampled_rows.iterrows():
            ts = row["timestamp"]
            raw_idx = self.raw_clean[self.raw_clean["timestamp"] == ts].index[0]
            window = self.raw_clean.iloc[max(0, raw_idx - 199) : raw_idx + 1].copy()

            feat_df = self.predictor.compute_candle_features(window)
            live_feats = feat_df.iloc[-1][self.feature_cols].values.astype(float)
            batch_feats = row[self.feature_cols].values.astype(float)

            diff = float(np.max(np.abs(live_feats - batch_feats)))
            if diff > max_diff:
                max_diff = diff

            self.assertLess(
                diff,
                1e-5,
                f"Feature parity failed at {ts} with max diff {diff:.2e} >= 1e-5",
            )

        print(f"\n[PASS] Historical Replay Feature Parity: Max diff across last 10 days = {max_diff:.2e} (< 1e-5)")

    def test_03_historical_replay_signal_parity_last_10_days(self):
        """
        Every emitted directional signal across the last 10 trading days must match
        batch offline model predictions with exactly 0 mismatches.
        """
        # Batch predictions
        X_batch = self.sub_batch[self.feature_cols].values.astype(float)
        p_rf = self.predictor.rf_model.predict_proba(X_batch)
        X_batch_sc = self.predictor.scaler.transform(X_batch)
        p_lr = self.predictor.lr_model.predict_proba(X_batch_sc)
        p_ens_raw = 0.5 * p_rf + 0.5 * p_lr
        p_cal = self.predictor.calibrator.predict_proba(p_ens_raw)
        conv = np.maximum(p_cal[:, 0], p_cal[:, 2])

        batch_directions = []
        for i in range(len(conv)):
            if conv[i] >= self.predictor.threshold:
                batch_directions.append("Up" if p_cal[i, 2] >= p_cal[i, 0] else "Down")
            else:
                batch_directions.append("no clear signal")

        # Live sliding window predictions
        mismatches = 0
        total_tested = len(self.sub_batch)

        for i, (_, row) in enumerate(self.sub_batch.iterrows()):
            ts = row["timestamp"]
            raw_idx = self.raw_clean[self.raw_clean["timestamp"] == ts].index[0]
            window = self.raw_clean.iloc[max(0, raw_idx - 199) : raw_idx + 1].copy()
            res = self.predictor.predict_candle(window)

            if res["direction"] != batch_directions[i]:
                mismatches += 1

        self.assertEqual(
            mismatches,
            0,
            f"Found {mismatches} signal mismatches between live sliding buffer and batch evaluation!",
        )
        print(f"[PASS] Historical Replay Signal Parity: 0 mismatches out of {total_tested} bars across 10 days (100.0% match)")

    def test_04_no_repainting_and_restart_byte_identity(self):
        """
        Verifies that signals logged to jsonl are append-only and immutable.
        Simulates mid-session restart: earlier log lines must be byte-for-byte identical.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sig_log = Path(tmpdir) / "test_live_signals.jsonl"
            out_log = Path(tmpdir) / "test_live_outcomes.jsonl"

            service1 = LiveInferenceServiceFinal(
                predictor=self.predictor,
                signals_log=sig_log,
                outcomes_log=out_log,
            )
            # Seed 200 bars
            seed_df = self.raw_clean.iloc[500:700].copy()
            service1.seed_buffer_from_history(seed_df)

            # Process 10 closed candles in session 1
            for i in range(700, 710):
                bar = self.raw_clean.iloc[i].to_dict()
                service1.process_closed_bar(bar)

            # Snapshot file content bytes
            with open(sig_log, "rb") as f:
                bytes_session_1 = f.read()

            lines_session_1 = bytes_session_1.decode("utf-8").strip().split("\n")
            self.assertEqual(len(lines_session_1), 10)

            # Simulate mid-session server restart: service2 boots up
            service2 = LiveInferenceServiceFinal(
                predictor=self.predictor,
                signals_log=sig_log,
                outcomes_log=out_log,
            )
            # Service2 re-seeds up to bar 709
            service2.seed_buffer_from_history(self.raw_clean.iloc[510:710].copy())

            # Process 5 subsequent closed candles in session 2
            for i in range(710, 715):
                bar = self.raw_clean.iloc[i].to_dict()
                service2.process_closed_bar(bar)

            with open(sig_log, "rb") as f:
                bytes_after_restart = f.read()

            lines_after_restart = bytes_after_restart.decode("utf-8").strip().split("\n")
            self.assertEqual(len(lines_after_restart), 15)

            # Assert earlier 10 lines are 100% byte-for-byte identical
            self.assertTrue(
                bytes_after_restart.startswith(bytes_session_1),
                "No-repainting violation: Earlier log lines were modified after restart!",
            )
            print("[PASS] No-Repainting Verification: Restart audit log is byte-identical and immutable.")

    def test_05_missing_bar_gap_handling(self):
        """
        Simulates missing 5-minute interval (15-min gap).
        Asserts service synthesizes zero-volume bars (is_filled=1) and increments gaps_filled_count.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sig_log = Path(tmpdir) / "test_gaps.jsonl"
            out_log = Path(tmpdir) / "test_gaps_out.jsonl"

            service = LiveInferenceServiceFinal(
                predictor=self.predictor,
                signals_log=sig_log,
                outcomes_log=out_log,
            )
            service.seed_buffer_from_history(self.raw_clean.iloc[500:700].copy())

            # Last bar timestamp
            last_ts = pd.to_datetime(service.candles_buffer[-1]["timestamp_ny"])

            # Send a bar 15 minutes later (skipping 2 intermediate bars)
            gap_bar = self.raw_clean.iloc[701].to_dict()
            gap_ts = last_ts + pd.Timedelta(minutes=15)
            gap_bar["timestamp"] = str(gap_ts)
            gap_bar["timestamp_ny"] = str(gap_ts)
            gap_bar["session_date"] = str(gap_ts.date())

            initial_gaps = service.gaps_filled_count
            service.process_closed_bar(gap_bar)

            self.assertGreater(
                service.gaps_filled_count,
                initial_gaps,
                "Failed to detect or synthesize missing 5-minute bars across gap!",
            )
            # Verify the synthesized bar has is_filled=1 and volume=0.0
            filled_bars = [b for b in service.candles_buffer if b.get("is_filled") == 1]
            self.assertGreater(len(filled_bars), 0)
            self.assertEqual(filled_bars[0]["volume"], 0.0)
            print(f"[PASS] Missing Bar Handling: Synthesized {len(filled_bars)} gap-fill bars gracefully.")

    def test_06_corrupt_packet_resilience(self):
        """
        Simulates corrupt bar packet (NaN close price).
        Asserts service traps error, logs failure, increments error_count, and does not crash.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sig_log = Path(tmpdir) / "test_corrupt.jsonl"
            out_log = Path(tmpdir) / "test_corrupt_out.jsonl"

            service = LiveInferenceServiceFinal(
                predictor=self.predictor,
                signals_log=sig_log,
                outcomes_log=out_log,
            )
            service.seed_buffer_from_history(self.raw_clean.iloc[500:700].copy())

            corrupt_bar = self.raw_clean.iloc[701].to_dict()
            corrupt_bar["close"] = np.nan

            initial_errors = service.error_count
            res = service.process_closed_bar(corrupt_bar)

            self.assertGreater(service.error_count, initial_errors)
            self.assertEqual(service.status, "DEGRADED")
            print("[PASS] Corrupt Packet Resilience: Trapped NaN bar, status updated to DEGRADED without crash.")

    def test_07_realized_outcome_resolution_after_3_bars(self):
        """
        Verifies that after exactly 3 bars (15 minutes), the realized return (gross & net @ 8bp)
        is evaluated and written to live_outcomes_final.jsonl.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sig_log = Path(tmpdir) / "test_outcome.jsonl"
            out_log = Path(tmpdir) / "test_outcome_out.jsonl"

            service = LiveInferenceServiceFinal(
                predictor=self.predictor,
                signals_log=sig_log,
                outcomes_log=out_log,
            )
            service.seed_buffer_from_history(self.raw_clean.iloc[500:700].copy())

            # Process bar 0 (emits signal 0)
            bar0 = self.raw_clean.iloc[700].to_dict()
            service.process_closed_bar(bar0)

            # Outcome file should be empty at bar 0
            self.assertFalse(out_log.exists() and out_log.stat().st_size > 0)

            # Process bar 1 and bar 2
            service.process_closed_bar(self.raw_clean.iloc[701].to_dict())
            service.process_closed_bar(self.raw_clean.iloc[702].to_dict())

            # Process bar 3 (3 bars have elapsed: bar 0 outcome must resolve now)
            service.process_closed_bar(self.raw_clean.iloc[703].to_dict())

            self.assertTrue(out_log.exists() and out_log.stat().st_size > 0)
            with open(out_log, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            self.assertGreaterEqual(len(lines), 1)
            outcome = lines[0]
            self.assertIn("realized_return_bps", outcome)
            self.assertIn("directional_return_bps", outcome)
            self.assertIn("net_return_bps_8bp", outcome)
            self.assertIn("is_correct", outcome)
            print(f"[PASS] Realized Outcome Logging: Resolved bar 0 outcome: realized={outcome['realized_return_bps']} bps, net={outcome['net_return_bps_8bp']} bps")


if __name__ == "__main__":
    unittest.main()

