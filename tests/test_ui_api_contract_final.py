"""
TradePulse - Step 15: UI API Contract & Mock Verification Tests
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Verifies:
1. ui_summary_final.json existence and schema completeness (edge_demonstrated = false).
2. All endpoints in api_schemas.py return strictly valid Pydantic models.
3. All exposed timestamp fields are integer UTC epoch seconds.
4. Historical chart returns proper period bands (Train, Val, Test).
5. Frontend production distribution (frontend/dist/index.html) is built and present.
"""

import sys
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.api_schemas import (
    HealthResponse,
    MetaResponse,
    HistoryResponse,
    OOFSignalsResponse,
    TestNowResponse,
    SignalsLogResponse,
    OutcomesResponse,
    ReplayState,
)
from backend.app.services.mock_service import get_mock_service
from backend.app.services.market_calendar import get_current_market_status


class TestUIAPIContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = get_mock_service()
        cls.summary_path = PROJECT_ROOT / "artifacts" / "metadata" / "ui_summary_final.json"
        cls.dist_path = PROJECT_ROOT / "frontend" / "dist" / "index.html"

    def test_01_ui_summary_final_integrity(self):
        """Verifies that ui_summary_final.json contains all required schema keys."""
        self.assertTrue(self.summary_path.exists(), "ui_summary_final.json does not exist!")
        with open(self.summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "asset", "exchange", "timeframe", "label_name", "theta_bps", "horizon_minutes",
            "feature_count", "model_architecture", "probability_calibration", "recency_weighting",
            "signal_rule", "dataset_scope", "headline_metrics", "cost_assumption",
            "edge_demonstrated", "edge_statement", "disclaimer"
        ]
        for k in required_keys:
            self.assertIn(k, data, f"Missing key '{k}' in ui_summary_final.json")

        # Non-negotiable honest disclosure: edge_demonstrated must be False
        self.assertFalse(data["edge_demonstrated"], "edge_demonstrated must be false per experiment findings.")
        self.assertIn("No profitable edge after costs", data["edge_statement"])
        self.assertEqual(data["cost_assumption"]["institutional_friction_bps"], 8.0)

    def test_02_health_endpoint_schema(self):
        """Verifies /api/health schema and market status integration."""
        health = self.service.get_health()
        self.assertIsInstance(health, HealthResponse)
        self.assertEqual(health.active_symbol, "META")
        self.assertEqual(health.status, "HEALTHY")
        self.assertIsInstance(health.market_status.current_time_utc, int)
        self.assertIn(health.market_status.session_state, ["OPEN", "CLOSED", "EARLY_CLOSE", "PRE_MARKET", "AFTER_HOURS"])

    def test_03_meta_endpoint_schema(self):
        """Verifies /api/meta schema and headline test metrics."""
        meta = self.service.get_meta()
        self.assertIsInstance(meta, MetaResponse)
        self.assertEqual(meta.asset, "META")
        self.assertEqual(meta.theta_bps, 15.0)
        self.assertEqual(meta.signal_rule.conviction_threshold_tau, 0.2635)
        # Verify headline test metrics from Phase 2
        test_m = meta.headline_metrics.held_out_test
        self.assertAlmostEqual(test_m.accuracy, 0.4681, places=2)
        self.assertAlmostEqual(test_m.balanced_accuracy, 0.4256, places=2)
        self.assertAlmostEqual(test_m.macro_f1, 0.4266, places=2)
        self.assertAlmostEqual(test_m.gross_return_bps, 1.29, places=2)
        self.assertAlmostEqual(test_m.net_return_8bp_bps, -6.71, places=2)

    def test_04_history_endpoint_and_period_bands(self):
        """Verifies /api/history returns valid bars, UTC timestamps, and split period bands."""
        hist_1d = self.service.get_history("1D")
        self.assertIsInstance(hist_1d, HistoryResponse)
        self.assertEqual(hist_1d.range, "1D")
        self.assertEqual(hist_1d.bars_count, 78)
        self.assertIsInstance(hist_1d.bars[0].time, int)

        # Check period bands
        bands = hist_1d.period_bands
        band_names = [b.name for b in bands]
        self.assertIn("train", band_names)
        self.assertIn("validation", band_names)
        self.assertIn("test", band_names)
        for b in bands:
            self.assertIsInstance(b.start_time, int)
            self.assertIsInstance(b.end_time, int)
            self.assertLess(b.start_time, b.end_time)

    def test_05_oof_signals_endpoint(self):
        """Verifies /api/history/oof-signals returns valid historical markers with notice."""
        oof = self.service.get_oof_signals()
        self.assertIsInstance(oof, OOFSignalsResponse)
        self.assertTrue(oof.available)
        self.assertIn("Out-of-fold, historical, not live results", oof.note)
        self.assertGreater(oof.count, 0)
        self.assertIsInstance(oof.signals[0].time, int)

    def test_06_test_now_endpoint_idempotence(self):
        """Verifies /api/live/test-now returns standardized prediction, forming candle, and flags."""
        res1 = self.service.test_now()
        self.assertIsInstance(res1, TestNowResponse)
        self.assertEqual(res1.status, "SUCCESS")
        self.assertIn(res1.signal.direction, ["Up", "Down", "no clear signal"])
        self.assertIsInstance(res1.signal.bar_time, int)
        self.assertIsInstance(res1.latest_closed_bar.time, int)
        self.assertIsNotNone(res1.forming_bar)

        # Idempotence: subsequent call on the same closed bar returns identical payload
        res2 = self.service.test_now()
        self.assertEqual(res1.signal.bar_time, res2.signal.bar_time)
        self.assertEqual(res1.signal.direction, res2.signal.direction)
        self.assertEqual(res1.signal.confidence_pct, res2.signal.confidence_pct)

    def test_07_signals_log_and_outcomes(self):
        """Verifies /api/signals and /api/outcomes return typed audit log entries."""
        sigs = self.service.get_signals_log()
        self.assertIsInstance(sigs, SignalsLogResponse)
        self.assertGreater(sigs.count, 0)
        self.assertIsInstance(sigs.signals[0].bar_time, int)
        self.assertTrue(len(sigs.signals[0].checksum) > 0)

        outs = self.service.get_outcomes()
        self.assertIsInstance(outs, OutcomesResponse)
        self.assertGreater(outs.count, 0)
        self.assertIsInstance(outs.outcomes[0].signal_time, int)

    def test_08_replay_state_machine(self):
        """Verifies /api/replay state transitions: start -> step -> pause -> stop."""
        start_state = self.service.start_replay(session_date="2026-11-06", speed=1.0)
        self.assertIsInstance(start_state, ReplayState)
        self.assertTrue(start_state.is_active)
        self.assertEqual(start_state.status, "PLAYING")
        self.assertEqual(start_state.current_bar_index, 1)

        step_state = self.service.step_replay()
        self.assertEqual(step_state.current_bar_index, 2)

        pause_state = self.service.pause_replay()
        self.assertEqual(pause_state.status, "PAUSED")

        stop_state = self.service.stop_replay()
        self.assertFalse(stop_state.is_active)
        self.assertEqual(stop_state.status, "STOPPED")

    def test_09_frontend_distribution_build(self):
        """Verifies that the React + TypeScript frontend built successfully to frontend/dist/."""
        self.assertTrue(self.dist_path.exists(), "frontend/dist/index.html is missing! Run npm run build.")
        self.assertGreater(self.dist_path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()

