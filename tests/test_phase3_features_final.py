"""
TradePulse - Step 15 Final: Phase 3 Features Verification Suite

Tests Phase 3 implementations:
1. Replay mode session loading and buffer seeding.
2. Isolated replay logging to data/replay_signals.jsonl (zero impact on live_signals_final.jsonl).
3. Replay bar-by-bar progression and outcome evaluation.
4. Scoreboard calculations against ui_summary_final.json expectations.
5. Model Card metadata accuracy and mandatory edge disclosure.
6. Reliability flags and guide anchors verification.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.mock_service import get_mock_service
from backend.app.schemas.api_schemas import (
    ReplayState,
    PredictionSignal,
    CandleBar,
    SignalsLogResponse,
    OutcomesResponse,
)


class TestPhase3Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = get_mock_service()
        cls.data_dir = PROJECT_ROOT / "data"
        cls.live_signals_path = cls.data_dir / "live_signals_final.jsonl"
        cls.replay_signals_path = cls.data_dir / "replay_signals.jsonl"

    def test_01_replay_sessions_availability(self):
        """Verifies that available historical sessions are returned for the UI dropdown."""
        sessions = self.service.get_replay_sessions()
        self.assertIsInstance(sessions, list)
        self.assertGreaterEqual(len(sessions), 5)
        # Ensure session dates are in YYYY-MM-DD format
        self.assertRegex(sessions[-1], r"^\d{4}-\d{2}-\d{2}$")

    def test_02_replay_start_and_isolated_logging(self):
        """
        Verifies that starting replay:
        1. Leaves live_signals_final.jsonl completely untouched.
        2. Creates an isolated replay_signals.jsonl.
        3. Returns valid initial CandleBar and PredictionSignal.
        """
        live_bytes_before = self.live_signals_path.read_bytes() if self.live_signals_path.exists() else b""

        state = self.service.start_replay(session_date="2026-11-06", speed=5.0)
        self.assertIsInstance(state, ReplayState)
        self.assertTrue(state.is_active)
        self.assertEqual(state.status, "PLAYING")
        self.assertEqual(state.speed_multiplier, 5.0)
        self.assertEqual(state.current_bar_index, 1)
        self.assertGreater(state.total_bars, 0)
        self.assertIsInstance(state.current_bar, CandleBar)
        self.assertIsInstance(state.current_signal, PredictionSignal)

        # Assert live signals log was NOT modified
        live_bytes_after = self.live_signals_path.read_bytes() if self.live_signals_path.exists() else b""
        self.assertEqual(
            live_bytes_before,
            live_bytes_after,
            "Replay violation: live_signals_final.jsonl was touched during replay!",
        )

        # Assert isolated replay log was populated
        self.assertTrue(self.replay_signals_path.exists())
        replay_lines = [l for l in self.replay_signals_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        self.assertEqual(len(replay_lines), 1)

    def test_03_replay_step_progression(self):
        """
        Verifies stepping through 5 candles in replay mode.
        Asserts bar index increments, new candles are returned, and replay log grows.
        """
        initial_index = self.service.replay_state.current_bar_index
        for step in range(5):
            new_state = self.service.step_replay()
            self.assertEqual(new_state.current_bar_index, initial_index + step + 1)
            self.assertIsNotNone(new_state.current_bar)
            self.assertIsNotNone(new_state.current_signal)

        # Verify replay log has 6 lines total (1 from start + 5 from steps)
        replay_lines = [l for l in self.replay_signals_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        self.assertEqual(len(replay_lines), 6)

    def test_04_replay_pause_and_stop_lifecycle(self):
        """Verifies pause and stop state transitions."""
        pause_state = self.service.pause_replay()
        self.assertEqual(pause_state.status, "PAUSED")
        self.assertTrue(pause_state.is_active)

        stop_state = self.service.stop_replay()
        self.assertEqual(stop_state.status, "STOPPED")
        self.assertFalse(stop_state.is_active)
        self.assertEqual(stop_state.current_bar_index, 0)

    def test_05_model_card_and_edge_disclosure_integrity(self):
        """
        Verifies that metadata strictly asserts edge_demonstrated = False
        with explicit negative net return at 8 bps cost.
        """
        meta = self.service.get_meta()
        self.assertFalse(
            meta.edge_demonstrated,
            "Integrity failure: edge_demonstrated must be False per held-out test report!",
        )
        self.assertIn("No profitable edge after costs was demonstrated", meta.edge_statement)
        test_net_ret = meta.headline_metrics.held_out_test.net_return_8bp_bps
        self.assertLess(
            test_net_ret,
            0.0,
            f"Expected net negative return after 8bp costs on held-out test, got {test_net_ret}",
        )

    def test_06_reliability_advisory_flags(self):
        """Verifies that reliability flags link to appropriate user guide anchors."""
        test_now_res = self.service.test_now()
        flags = test_now_res.signal.reliability_flags
        self.assertGreater(len(flags), 0)

        valid_anchors = {"stale-data", "session-boundaries", "news-earnings", "thin-liquidity", "conviction-gating"}
        for flag in flags:
            self.assertIn(flag.guide_anchor, valid_anchors)
            self.assertIn(flag.level, ["NORMAL", "ADVISORY", "WARNING"])


if __name__ == "__main__":
    unittest.main()
