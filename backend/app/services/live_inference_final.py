"""
TradePulse - Step 14 Final: Live Inference Service

Production live inference engine for META 5-minute bars:
1. Connects to Alpaca IEX data feed (paper/live) via .env credentials.
2. Maintains a thread-safe rolling buffer of closed 5-minute candles (200 bars).
3. Evaluates predictions strictly at candle close (no repainting guarantee).
4. Handles missing bar gaps gracefully: logs gaps, synthesizes zero-volume bars (is_filled=1).
5. Appends signals to data/live_signals_final.jsonl immediately upon candle close.
6. Resolves 3-bar (15-min) outcomes and appends to data/live_outcomes_final.jsonl.
7. Exposes thread-safe inspection methods for monitoring and REST API endpoints.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from collections import deque

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict_final import TradePulsePredictorFinal, get_default_predictor_final

DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_LOG_PATH = DATA_DIR / "live_signals_final.jsonl"
OUTCOMES_LOG_PATH = DATA_DIR / "live_outcomes_final.jsonl"


class LiveInferenceServiceFinal:
    """
    Live streaming and inference coordinator for TradePulse.
    Maintains candle buffer, evaluates model at candle close, and tracks realized outcomes.
    """

    def __init__(
        self,
        predictor: Optional[TradePulsePredictorFinal] = None,
        buffer_capacity: int = 250,
        signals_log: Path = SIGNALS_LOG_PATH,
        outcomes_log: Path = OUTCOMES_LOG_PATH,
    ):
        self.symbol = "META"
        self.predictor = predictor or get_default_predictor_final()
        self.buffer_capacity = buffer_capacity
        self.signals_log = Path(signals_log)
        self.outcomes_log = Path(outcomes_log)

        # Thread synchronization
        self.lock = threading.Lock()

        # Candle buffer & pending outcomes queue
        self.candles_buffer: List[Dict[str, Any]] = []
        self.signals_history: List[Dict[str, Any]] = []
        self.pending_outcomes: List[Dict[str, Any]] = []
        self.total_bars_processed = 0

        # Monitoring statistics
        self.start_time = time.time()
        self.status = "INITIALIZING"
        self.last_update_ny: Optional[str] = None
        self.error_count = 0
        self.gaps_filled_count = 0
        self.expected_signals_day = 37.5
        self.active_feed = "IEX"

        # Ensure output directories exist
        self.signals_log.parent.mkdir(parents=True, exist_ok=True)
        self.outcomes_log.parent.mkdir(parents=True, exist_ok=True)

        # Feed configuration
        self.feed_source = os.getenv("TRADEPULSE_FEED_SOURCE", "historical").lower()
        self.active_feed = "IEX" if self.feed_source == "alpaca" else "HISTORICAL_REPLAY"

        # Load Alpaca configuration
        load_dotenv(PROJECT_ROOT / ".env")
        self.alpaca_key = os.getenv("APCA_API_KEY_ID", "")
        self.alpaca_secret = os.getenv("APCA_API_SECRET_KEY", "")
        self.alpaca_base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    def seed_buffer_from_history(self, seed_df: pd.DataFrame, num_bars: int = 200) -> int:
        """
        Seeds the rolling buffer from a historical DataFrame (e.g., during warm-up or replay).
        """
        with self.lock:
            tail = seed_df.tail(num_bars).copy()
            for _, row in tail.iterrows():
                bar = {
                    "timestamp": str(row.get("timestamp", row.get("timestamp_ny"))),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "trade_count": float(row.get("trade_count", 0)),
                    "vwap": float(row.get("vwap", row["close"])),
                    "is_filled": int(row.get("is_filled", 0)),
                }
                if "timestamp_ny" in row:
                    bar["timestamp_ny"] = str(row["timestamp_ny"])
                if "session_date" in row:
                    bar["session_date"] = str(row["session_date"])
                self.candles_buffer.append(bar)

            if len(self.candles_buffer) > self.buffer_capacity:
                self.candles_buffer = self.candles_buffer[-self.buffer_capacity :]

            self.status = "HEALTHY" if len(self.candles_buffer) >= 35 else "INITIALIZING"
            return len(self.candles_buffer)

    def seed_buffer_from_alpaca(self, num_bars: int = 200, force_alpaca: bool = False) -> int:
        """
        Fetches the latest completed 5-minute bars to initialize the warm-up buffer.
        If feed_source is 'historical' (or offline sandbox), seeds directly from processed dataset.
        If feed_source is 'alpaca' (or force_alpaca=True), fetches live IEX bars from Alpaca API.
        """
        clean_path = DATA_DIR / "processed" / "META_5min_clean.csv"

        if not force_alpaca and self.feed_source != "alpaca":
            if clean_path.exists():
                df = pd.read_csv(clean_path)
                return self.seed_buffer_from_history(df, num_bars=num_bars)
            return 0

        if not self.alpaca_key or not self.alpaca_secret:
            if clean_path.exists():
                df = pd.read_csv(clean_path)
                return self.seed_buffer_from_history(df, num_bars=num_bars)
            return 0

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed, Adjustment

            client = StockHistoricalDataClient(self.alpaca_key, self.alpaca_secret)
            now_utc = datetime.now(timezone.utc)
            start_utc = now_utc - timedelta(days=7)

            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_utc,
                end=now_utc - timedelta(minutes=15),
                feed=DataFeed.IEX,
                adjustment=Adjustment.RAW,
            )

            bars = client.get_stock_bars(request_params)
            bar_list = []
            for bar in bars[self.symbol]:
                bar_list.append(
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "trade_count": float(getattr(bar, "trade_count", 0)),
                        "vwap": float(getattr(bar, "vwap", bar.close)),
                        "is_filled": 0,
                    }
                )
            if bar_list:
                df = pd.DataFrame(bar_list)
                return self.seed_buffer_from_history(df, num_bars=num_bars)
        except Exception as e:
            self.error_count += 1
            clean_path = DATA_DIR / "processed" / "META_5min_clean.csv"
            if clean_path.exists():
                df = pd.read_csv(clean_path)
                return self.seed_buffer_from_history(df, num_bars=num_bars)

        return len(self.candles_buffer)

    def _fill_missing_gaps(self, new_bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects missing 5-minute intervals between the latest buffer candle and new_bar.
        Returns a list of synthetic bars with is_filled=1.
        """
        if not self.candles_buffer:
            return []

        last_bar = self.candles_buffer[-1]
        try:
            last_ts = pd.to_datetime(last_bar.get("timestamp_ny", last_bar["timestamp"]), utc=True).tz_convert("America/New_York")
            new_ts = pd.to_datetime(new_bar.get("timestamp_ny", new_bar["timestamp"]), utc=True).tz_convert("America/New_York")
        except Exception:
            return []

        # If on the same trading day and gap > 5 minutes
        gap_minutes = (new_ts - last_ts).total_seconds() / 60.0
        synthetic_bars = []

        if last_ts.date() == new_ts.date() and gap_minutes > 5.0 and gap_minutes <= 120.0:
            # Create synthetic bars for each 5-minute interval
            curr_ts = last_ts + timedelta(minutes=5)
            last_close = float(last_bar["close"])
            while curr_ts < new_ts:
                synth = {
                    "timestamp": str(curr_ts),
                    "timestamp_ny": str(curr_ts),
                    "session_date": str(curr_ts.date()),
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "volume": 0.0,
                    "trade_count": 0.0,
                    "vwap": last_close,
                    "is_filled": 1,
                }
                synthetic_bars.append(synth)
                curr_ts += timedelta(minutes=5)
                self.gaps_filled_count += 1

        return synthetic_bars

    def process_closed_bar(self, raw_bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a newly closed 5-minute candle:
        1. Checks and fills missing candle intervals.
        2. Appends candle to buffer.
        3. Generates prediction via predict_final.py.
        4. Appends signal immediately to data/live_signals_final.jsonl.
        5. Evaluates mature 3-bar pending outcomes and appends to data/live_outcomes_final.jsonl.
        """
        with self.lock:
            # Handle missing gaps
            gap_bars = self._fill_missing_gaps(raw_bar)
            all_incoming = gap_bars + [raw_bar]

            latest_payload = None

            for bar in all_incoming:
                # Add candle to buffer
                self.candles_buffer.append(bar)
                if len(self.candles_buffer) > self.buffer_capacity:
                    self.candles_buffer = self.candles_buffer[-self.buffer_capacity :]

                self.total_bars_processed += 1
                curr_idx = self.total_bars_processed
                bar_close = float(bar["close"])
                bar_ts = str(bar.get("timestamp_ny", bar.get("timestamp")))

                # Check if buffer has sufficient warm-up history
                if len(self.candles_buffer) < 35:
                    self.status = "INITIALIZING"
                    continue

                buffer_df = pd.DataFrame(self.candles_buffer)
                try:
                    payload = self.predictor.predict_candle(buffer_df)
                    payload["bar_index"] = curr_idx
                    self.last_update_ny = payload["timestamp"]
                    self.status = "HEALTHY"

                    # 1. Immediately append signal to live_signals_final.jsonl (Audit Trail)
                    with open(self.signals_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")
                        f.flush()

                    self.signals_history.append(payload)
                    latest_payload = payload

                    # 2. Register for outcome resolution 3 bars ahead (15 min)
                    self.pending_outcomes.append(
                        {
                            "signal_timestamp": payload["timestamp"],
                            "bar_index": curr_idx,
                            "entry_price": bar_close,
                            "direction": payload["direction"],
                            "signal_code": payload["signal_code"],
                            "confidence_pct": payload["confidence_pct"],
                            "conviction": payload["conviction"],
                            "target_bar_index": curr_idx + 3,
                        }
                    )
                except Exception as e:
                    self.error_count += 1
                    self.status = "DEGRADED"
                    print(f"[ERROR] Live inference failed on bar {bar_ts}: {e}", file=sys.stderr)

                # 3. Resolve mature pending outcomes
                still_pending = []
                for item in self.pending_outcomes:
                    if curr_idx >= item["target_bar_index"]:
                        # 3 bars have elapsed: compute realized outcome
                        exit_price = bar_close
                        entry_price = item["entry_price"]
                        ret_bps = ((exit_price - entry_price) / entry_price) * 10000.0
                        code = item["signal_code"]
                        dir_bps = code * ret_bps
                        net_bps = (dir_bps - 8.0) if code != 0 else 0.0

                        is_correct = False
                        if code == 1 and ret_bps >= 15.0:
                            is_correct = True
                        elif code == -1 and ret_bps <= -15.0:
                            is_correct = True
                        elif code == 0 and abs(ret_bps) < 15.0:
                            is_correct = True

                        outcome_record = {
                            "signal_timestamp": item["signal_timestamp"],
                            "resolution_timestamp": bar_ts,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exit_price, 2),
                            "direction": item["direction"],
                            "signal_code": code,
                            "confidence_pct": item["confidence_pct"],
                            "realized_return_bps": round(ret_bps, 2),
                            "directional_return_bps": round(dir_bps, 2),
                            "net_return_bps_8bp": round(net_bps, 2),
                            "is_correct": is_correct,
                        }

                        with open(self.outcomes_log, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps(outcome_record) + "\n")
                            f_out.flush()
                    else:
                        still_pending.append(item)

                self.pending_outcomes = still_pending

            return latest_payload

    def get_latest_signal(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent emitted signal payload."""
        with self.lock:
            if self.signals_history:
                return self.signals_history[-1]
            return None

    def get_signal_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns the recent emitted signals history for charting."""
        with self.lock:
            return self.signals_history[-limit:]

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Returns comprehensive monitoring statistics."""
        with self.lock:
            now = time.time()
            uptime = round(now - self.start_time, 1)
            today_ny = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")

            # Count signals emitted today
            sigs_today = sum(
                1
                for s in self.signals_history
                if s.get("timestamp", "").startswith(today_ny) and s.get("signal_code", 0) != 0
            )

            return {
                "status": self.status,
                "symbol": self.symbol,
                "uptime_seconds": uptime,
                "buffer_bars_count": len(self.candles_buffer),
                "min_required_bars": 35,
                "last_update_ny": self.last_update_ny,
                "total_bars_processed": self.total_bars_processed,
                "signals_today_count": sigs_today,
                "expected_signals_per_day": self.expected_signals_day,
                "gaps_filled_count": self.gaps_filled_count,
                "error_count": self.error_count,
                "model_version": self.predictor.version,
                "active_feed": self.active_feed,
            }


# Singleton service instance
_SERVICE_INSTANCE: Optional[LiveInferenceServiceFinal] = None


def get_live_service_final() -> LiveInferenceServiceFinal:
    """Returns the singleton live inference service instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = LiveInferenceServiceFinal()
    return _SERVICE_INSTANCE
