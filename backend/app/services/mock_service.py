"""
TradePulse - Mock Backend Service (Phase 0 API Contract Delivery)
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Provides high-fidelity mock implementations of all required endpoints:
- Uses real ui_summary_final.json, split_config.json, and META_5min_clean.csv data.
- Guarantees 100% schema compliance with backend.app.schemas.api_schemas.
- Serves as the interactive contract for Phase 1 frontend development and tests.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.api_schemas import (
    MarketStatus,
    HealthResponse,
    MetaResponse,
    CandleBar,
    PeriodBand,
    HistoryResponse,
    OOFSignalItem,
    OOFSignalsResponse,
    ReliabilityFlag,
    ProbabilityDistribution,
    PredictionSignal,
    TestNowResponse,
    SignalLogItem,
    SignalsLogResponse,
    OutcomeItem,
    OutcomesResponse,
    ReplayState,
)
from backend.app.services.market_calendar import get_current_market_status
from backend.app.services.live_inference_final import LiveInferenceServiceFinal

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"

# Cached historical data
_CACHED_CLEAN_DF: Optional[pd.DataFrame] = None


def _get_clean_df() -> pd.DataFrame:
    global _CACHED_CLEAN_DF
    if _CACHED_CLEAN_DF is None:
        clean_path = DATA_DIR / "processed" / "META_5min_clean.csv"
        if clean_path.exists():
            df = pd.read_csv(clean_path)
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df["time"] = ts.astype("int64") // 10**9
            df["timestamp_ny"] = ts.dt.tz_convert("America/New_York")
            df["session_date"] = df["timestamp_ny"].dt.date.astype(str)
            if "vwap" not in df.columns:
                df["vwap"] = df["close"]
            if "is_filled" not in df.columns:
                df["is_filled"] = 0
            _CACHED_CLEAN_DF = df
        else:
            # Fallback synthetic frame if offline
            now = int(time.time())
            synth = []
            price = 600.0
            for i in range(200):
                t = now - (200 - i) * 300
                synth.append({
                    "time": t,
                    "timestamp": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.2,
                    "volume": 12000.0,
                    "vwap": price + 0.1,
                    "is_filled": 0,
                    "session_date": "2026-11-06",
                })
                price += 0.2
            _CACHED_CLEAN_DF = pd.DataFrame(synth)
    return _CACHED_CLEAN_DF


class MockBackendService:
    def __init__(self):
        self.start_time = time.time()
        self.symbol = "META"
        self.feed_name = "Alpaca IEX, 5-min"
        self.model_version = "TradePulse-v1.0-final (b57ee06e)"

        # Replay mock state
        self.replay_state = ReplayState(
            is_active=False,
            status="STOPPED",
            session_date="2026-11-06",
            current_bar_index=0,
            total_bars=78,
            speed_multiplier=1.0,
            current_bar=None,
            current_signal=None,
            signals_emitted_count=0,
        )

    def get_health(self) -> HealthResponse:
        market_status = get_current_market_status()

        # Load artifact hashes from training_manifest_final.json
        manifest_path = METADATA_DIR / "training_manifest_final.json"
        artifact_hashes = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                artifact_hashes = manifest.get("artifact_hashes_sha256", {})

        return HealthResponse(
            status="HEALTHY",
            uptime_seconds=round(time.time() - self.start_time, 1),
            feed_connectivity="CONNECTED",
            feed_name=self.feed_name,
            active_symbol=self.symbol,
            market_status=market_status,
            artifact_hashes=artifact_hashes,
            model_version=self.model_version,
            last_data_freshness_seconds=42,
        )

    def get_meta(self) -> MetaResponse:
        summary_path = METADATA_DIR / "ui_summary_final.json"
        if not summary_path.exists():
            raise FileNotFoundError("artifacts/metadata/ui_summary_final.json missing.")

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["market_status"] = get_current_market_status().model_dump()
        return MetaResponse(**data)

    def get_history(self, range_param: str = "1M") -> HistoryResponse:
        df = _get_clean_df()
        range_upper = range_param.upper()

        if range_upper == "1D":
            sub_df = df.tail(78).copy()
        elif range_upper == "5D":
            sub_df = df.tail(390).copy()
        elif range_upper == "1M":
            sub_df = df.tail(1638).copy()
        elif range_upper == "3M":
            sub_df = df.tail(4914).copy()
        else: # "ALL"
            sub_df = df.copy()

        bars: List[CandleBar] = []
        for _, row in sub_df.iterrows():
            bars.append(
                CandleBar(
                    time=int(row["time"]),
                    open=round(float(row["open"]), 2),
                    high=round(float(row["high"]), 2),
                    low=round(float(row["low"]), 2),
                    close=round(float(row["close"]), 2),
                    volume=round(float(row["volume"]), 0),
                    vwap=round(float(row.get("vwap", row["close"])), 2),
                    is_filled=bool(row.get("is_filled", 0) == 1),
                )
            )

        # Period Bands matching split_config.json
        period_bands = [
            PeriodBand(
                name="train",
                label="Training Set (Nov 2025 – Jun 2026)",
                start_time=int(pd.to_datetime("2025-11-03 09:30:00-05:00").timestamp()),
                end_time=int(pd.to_datetime("2026-06-30 16:00:00-04:00").timestamp()),
                color="rgba(59, 130, 246, 0.08)",
                tooltip="Initial model fitting partition (164 trading days, 12,228 bars).",
            ),
            PeriodBand(
                name="embargo_1",
                label="Embargo 1 (Jul 1, 2026)",
                start_time=int(pd.to_datetime("2026-07-01 09:30:00-04:00").timestamp()),
                end_time=int(pd.to_datetime("2026-07-01 16:00:00-04:00").timestamp()),
                color="rgba(107, 114, 128, 0.12)",
                tooltip="1-day causal purge barrier to prevent lookahead leakage into validation.",
            ),
            PeriodBand(
                name="validation",
                label="Validation Set (Jul 2026 – Aug 2026)",
                start_time=int(pd.to_datetime("2026-07-02 09:30:00-04:00").timestamp()),
                end_time=int(pd.to_datetime("2026-08-31 16:00:00-04:00").timestamp()),
                color="rgba(245, 158, 11, 0.08)",
                tooltip="Hyperparameter tuning & label threshold selection partition (42 days, 3,150 bars).",
            ),
            PeriodBand(
                name="embargo_2",
                label="Embargo 2 (Sep 1, 2026)",
                start_time=int(pd.to_datetime("2026-09-01 09:30:00-04:00").timestamp()),
                end_time=int(pd.to_datetime("2026-09-01 16:00:00-04:00").timestamp()),
                color="rgba(107, 114, 128, 0.12)",
                tooltip="1-day causal purge barrier to prevent lookahead leakage into test set.",
            ),
            PeriodBand(
                name="test",
                label="Held-Out Test Set (Sep 2026 – Nov 2026)",
                start_time=int(pd.to_datetime("2026-09-02 09:30:00-04:00").timestamp()),
                end_time=int(pd.to_datetime("2026-11-06 16:00:00-05:00").timestamp()),
                color="rgba(16, 185, 129, 0.08)",
                tooltip="One-time held-out evaluation partition (47 days, 3,525 bars, 50.3% coverage).",
            ),
        ]

        prev_close = round(float(sub_df.iloc[-2]["close"]), 2) if len(sub_df) > 1 else None
        etag_content = f"{range_upper}_{len(bars)}_{sub_df.iloc[-1]['time']}"
        etag = hashlib.md5(etag_content.encode("utf-8")).hexdigest()

        return HistoryResponse(
            symbol=self.symbol,
            range=range_upper,
            bars_count=len(bars),
            bars=bars,
            period_bands=period_bands,
            prev_close=prev_close,
            etag=etag,
        )

    def get_oof_signals(self) -> OOFSignalsResponse:
        # Check if saved or serve representative sample
        df = _get_clean_df()
        test_df = df.tail(3525) # test portion
        # Sample 100 representative out-of-fold signals
        signals = []
        for i in range(0, min(100, len(test_df)), 5):
            row = test_df.iloc[i]
            t = int(row["time"])
            # Alternating mock directions matching 50% coverage
            code = 1 if i % 10 == 0 else (-1 if i % 10 == 5 else 0)
            direction = "Up" if code == 1 else ("Down" if code == -1 else "no clear signal")
            conf = 68.4 if code == 1 else (64.2 if code == -1 else 46.1)
            conv = 0.3541 if code != 0 else 0.2215
            signals.append(
                OOFSignalItem(
                    time=t,
                    direction=direction,
                    signal_code=code,
                    confidence_pct=conf,
                    conviction=conv,
                    is_gated=(code != 0),
                )
            )

        return OOFSignalsResponse(
            available=True,
            count=len(signals),
            signals=signals,
            note="Out-of-fold, historical, not live results.",
        )

    def test_now(self) -> TestNowResponse:
        df = _get_clean_df()
        market_status = get_current_market_status()

        # Last closed candle
        last_closed_row = df.iloc[-1]
        last_time = int(last_closed_row["time"])
        last_close = round(float(last_closed_row["close"]), 2)

        latest_closed_bar = CandleBar(
            time=last_time,
            open=round(float(last_closed_row["open"]), 2),
            high=round(float(last_closed_row["high"]), 2),
            low=round(float(last_closed_row["low"]), 2),
            close=last_close,
            volume=round(float(last_closed_row["volume"]), 0),
            vwap=round(float(last_closed_row.get("vwap", last_close)), 2),
            is_filled=bool(last_closed_row.get("is_filled", 0) == 1),
        )

        # Muted forming live candle (5 minutes ahead)
        forming_time = last_time + 300
        forming_bar = CandleBar(
            time=forming_time,
            open=last_close,
            high=round(last_close + 0.35, 2),
            low=round(last_close - 0.20, 2),
            close=round(last_close + 0.15, 2),
            volume=4500.0,
            vwap=round(last_close + 0.08, 2),
            is_filled=False,
        )

        # Recent 150 closed candles for live chart view
        recent_df = df.tail(150)
        recent_bars = [
            CandleBar(
                time=int(r["time"]),
                open=round(float(r["open"]), 2),
                high=round(float(r["high"]), 2),
                low=round(float(r["low"]), 2),
                close=round(float(r["close"]), 2),
                volume=round(float(r["volume"]), 0),
                vwap=round(float(r.get("vwap", r["close"])), 2),
                is_filled=bool(r.get("is_filled", 0) == 1),
            )
            for _, r in recent_df.iterrows()
        ]

        # Reliability advisory flags
        reliability_flags = [
            ReliabilityFlag(
                id="SESSION_BOUNDARY",
                name="Mid-Session Activity",
                level="NORMAL",
                message="Candle is outside opening (09:30-09:45) and closing (15:45-16:00) volatility regimes.",
                guide_anchor="session-boundaries",
            ),
            ReliabilityFlag(
                id="DATA_FRESHNESS",
                name="Data Feed Fresh",
                level="NORMAL",
                message="Bar closed at expected interval with zero feed disconnects.",
                guide_anchor="stale-data",
            ),
            ReliabilityFlag(
                id="CONVICTION_GATING",
                name="Conviction Evaluated",
                level="NORMAL",
                message="Conviction exceeds calibrated threshold tau = 0.2635.",
                guide_anchor="conviction-gating",
            ),
        ]

        # Production prediction payload (Up signal example)
        signal = PredictionSignal(
            bar_time=last_time,
            bar_timestamp_ny=str(last_closed_row.get("timestamp_ny", "15:55:00 ET")),
            close_price=last_close,
            direction="Up",
            signal_code=1,
            confidence_pct=68.4,
            conviction=0.3541,
            threshold=0.2635,
            is_gated=True,
            probabilities=ProbabilityDistribution(down=0.2215, neutral=0.4244, up=0.3541),
            locked_at_utc=int(time.time()),
            horizon_end_utc=last_time + 900, # 15 minutes = 3 bars
            model_version=self.model_version,
            reliability_flags=reliability_flags,
            outcome=None,
        )

        return TestNowResponse(
            status="SUCCESS",
            signal=signal,
            latest_closed_bar=latest_closed_bar,
            forming_bar=forming_bar,
            recent_bars=recent_bars,
            market_status=market_status,
        )

    def get_signals_log(self, session: Optional[str] = None) -> SignalsLogResponse:
        # Load real live signals log or isolated replay log
        if session == "replay":
            log_path = DATA_DIR / "replay_signals.jsonl"
        else:
            log_path = DATA_DIR / "live_signals_final.jsonl"
            
        items: List[SignalLogItem] = []

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    bar_ts = rec.get("timestamp", "")
                    if session and session != "replay" and not bar_ts.startswith(session):
                        continue
                    ts_epoch = int(pd.to_datetime(bar_ts, utc=True).timestamp())
                    items.append(
                        SignalLogItem(
                            signal_id=f"sig_{ts_epoch}",
                            bar_time=ts_epoch,
                            bar_timestamp_ny=bar_ts,
                            direction=rec.get("direction", "no clear signal"),
                            signal_code=rec.get("signal_code", 0),
                            confidence_pct=rec.get("confidence_pct", 50.0),
                            conviction=rec.get("conviction", 0.0),
                            entry_price=rec.get("close_price", 0.0),
                            is_gated=rec.get("is_gated", False),
                            checksum=hashlib.sha256(line.encode("utf-8")).hexdigest()[:12],
                        )
                    )

        if not items and session != "replay":
            # Fallback mock items
            now = int(time.time())
            for i in range(10):
                t = now - (10 - i) * 300
                items.append(
                    SignalLogItem(
                        signal_id=f"sig_mock_{t}",
                        bar_time=t,
                        bar_timestamp_ny="2026-11-06 15:00:00 ET",
                        direction="Up" if i % 2 == 0 else "Down",
                        signal_code=1 if i % 2 == 0 else -1,
                        confidence_pct=65.2,
                        conviction=0.3340,
                        entry_price=612.45 + i * 0.3,
                        is_gated=True,
                        checksum="a7f8e32901bb",
                    )
                )

        return SignalsLogResponse(
            session=session or "Latest",
            count=len(items),
            signals=items,
        )

    def get_outcomes(self, mode: Optional[str] = None) -> OutcomesResponse:
        if mode == "replay":
            log_path = DATA_DIR / "replay_outcomes.jsonl"
        else:
            log_path = DATA_DIR / "live_outcomes_final.jsonl"
            
        items: List[OutcomeItem] = []

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    sig_ts = int(pd.to_datetime(rec["signal_timestamp"], utc=True).timestamp())
                    res_ts = int(pd.to_datetime(rec["resolution_timestamp"], utc=True).timestamp())
                    items.append(
                        OutcomeItem(
                            signal_id=f"sig_{sig_ts}",
                            signal_time=sig_ts,
                            resolution_time=res_ts,
                            direction=rec.get("direction", "Up"),
                            signal_code=rec.get("signal_code", 1),
                            confidence_pct=rec.get("confidence_pct", 65.0),
                            entry_price=rec.get("entry_price", 600.0),
                            exit_price=rec.get("exit_price", 601.0),
                            realized_return_bps=rec.get("realized_return_bps", 16.5),
                            directional_return_bps=rec.get("directional_return_bps", 16.5),
                            net_return_bps_8bp=rec.get("net_return_bps_8bp", 8.5),
                            is_hit=rec.get("is_correct", True),
                        )
                    )

        return OutcomesResponse(count=len(items), outcomes=items)

    def get_replay_sessions(self) -> List[str]:
        df = _get_clean_df()
        if "session_date" in df.columns:
            dates = df["session_date"].unique().tolist()
            return dates[-15:]
        return ["2026-11-06", "2026-11-05", "2026-11-04", "2026-11-03", "2026-10-31"]

    def start_replay(self, session_date: str = "2026-11-06", speed: float = 1.0) -> ReplayState:
        df = _get_clean_df()

        session_mask = df["session_date"] == session_date if "session_date" in df.columns else pd.Series(False, index=df.index)
        if not session_mask.any():
            session_date = df["session_date"].iloc[-1] if "session_date" in df.columns else "2026-11-06"
            session_mask = df["session_date"] == session_date if "session_date" in df.columns else pd.Series(True, index=df.index)

        session_df = df[session_mask].copy()
        first_idx = session_df.index[0]
        seed_start = max(0, first_idx - 200)
        seed_df = df.iloc[seed_start:first_idx].copy()

        replay_sig_path = DATA_DIR / "replay_signals.jsonl"
        replay_out_path = DATA_DIR / "replay_outcomes.jsonl"

        if replay_sig_path.exists():
            replay_sig_path.unlink()
        if replay_out_path.exists():
            replay_out_path.unlink()

        self.replay_inference_service = LiveInferenceServiceFinal(
            signals_log=replay_sig_path,
            outcomes_log=replay_out_path,
        )
        self.replay_inference_service.seed_buffer_from_history(seed_df)
        self.replay_session_bars = [row.to_dict() for _, row in session_df.iterrows()]

        # Step bar 1 immediately so initial candle & prediction are populated
        first_bar = self.replay_session_bars[0]
        live_res = self.replay_inference_service.process_closed_bar(first_bar)

        first_ts_epoch = int(pd.to_datetime(first_bar.get("timestamp", first_bar.get("timestamp_ny")), utc=True).timestamp())
        c_bar = CandleBar(
            time=first_ts_epoch,
            open=round(float(first_bar["open"]), 2),
            high=round(float(first_bar["high"]), 2),
            low=round(float(first_bar["low"]), 2),
            close=round(float(first_bar["close"]), 2),
            volume=round(float(first_bar["volume"]), 0),
            vwap=round(float(first_bar.get("vwap", first_bar["close"])), 2),
            is_filled=bool(first_bar.get("is_filled", 0) == 1),
        )

        sig = None
        if live_res:
            sig = PredictionSignal(
                bar_time=first_ts_epoch,
                bar_timestamp_ny=str(first_bar.get("timestamp_ny", first_bar.get("timestamp"))),
                close_price=c_bar.close,
                direction=live_res["direction"],
                signal_code=live_res["signal_code"],
                confidence_pct=live_res["confidence_pct"],
                conviction=live_res["conviction"],
                threshold=live_res["threshold"],
                is_gated=live_res["is_gated"],
                probabilities=ProbabilityDistribution(**live_res["probabilities"]),
                locked_at_utc=int(time.time()),
                horizon_end_utc=first_ts_epoch + 900,
                model_version=live_res.get("model_version", self.model_version),
                reliability_flags=[
                    ReliabilityFlag(
                        id="REPLAY_FLAG",
                        name="Historical Replay",
                        level="NORMAL",
                        message="Bar processed through LiveInferenceServiceFinal in isolated replay mode.",
                        guide_anchor="session-boundaries",
                    )
                ],
            )

        sigs_count = 1 if live_res and live_res["direction"] in ("Up", "Down") else 0

        self.replay_state = ReplayState(
            is_active=True,
            status="PLAYING",
            session_date=session_date,
            current_bar_index=1,
            total_bars=len(self.replay_session_bars),
            speed_multiplier=speed,
            current_bar=c_bar,
            current_signal=sig,
            signals_emitted_count=sigs_count,
        )
        return self.replay_state

    def step_replay(self) -> ReplayState:
        if not self.replay_state.is_active or not hasattr(self, "replay_session_bars") or not hasattr(self, "replay_inference_service"):
            return self.replay_state

        idx = self.replay_state.current_bar_index
        if idx < len(self.replay_session_bars):
            bar_dict = self.replay_session_bars[idx]
            live_res = self.replay_inference_service.process_closed_bar(bar_dict)

            ts_epoch = int(pd.to_datetime(bar_dict.get("timestamp", bar_dict.get("timestamp_ny")), utc=True).timestamp())
            c_bar = CandleBar(
                time=ts_epoch,
                open=round(float(bar_dict["open"]), 2),
                high=round(float(bar_dict["high"]), 2),
                low=round(float(bar_dict["low"]), 2),
                close=round(float(bar_dict["close"]), 2),
                volume=round(float(bar_dict["volume"]), 0),
                vwap=round(float(bar_dict.get("vwap", bar_dict["close"])), 2),
                is_filled=bool(bar_dict.get("is_filled", 0) == 1),
            )

            sig = None
            if live_res:
                sig = PredictionSignal(
                    bar_time=ts_epoch,
                    bar_timestamp_ny=str(bar_dict.get("timestamp_ny", bar_dict.get("timestamp"))),
                    close_price=c_bar.close,
                    direction=live_res["direction"],
                    signal_code=live_res["signal_code"],
                    confidence_pct=live_res["confidence_pct"],
                    conviction=live_res["conviction"],
                    threshold=live_res["threshold"],
                    is_gated=live_res["is_gated"],
                    probabilities=ProbabilityDistribution(**live_res["probabilities"]),
                    locked_at_utc=int(time.time()),
                    horizon_end_utc=ts_epoch + 900,
                    model_version=live_res.get("model_version", self.model_version),
                    reliability_flags=[
                        ReliabilityFlag(
                            id="REPLAY_FLAG",
                            name="Historical Replay",
                            level="NORMAL",
                            message="Bar processed through LiveInferenceServiceFinal in isolated replay mode.",
                            guide_anchor="session-boundaries",
                        )
                    ],
                )
                if live_res["direction"] in ("Up", "Down"):
                    self.replay_state.signals_emitted_count += 1

            self.replay_state.current_bar = c_bar
            self.replay_state.current_signal = sig
            self.replay_state.current_bar_index += 1
            if self.replay_state.current_bar_index >= self.replay_state.total_bars:
                self.replay_state.status = "PAUSED"

        return self.replay_state

    def pause_replay(self) -> ReplayState:
        self.replay_state.status = "PAUSED"
        return self.replay_state

    def stop_replay(self) -> ReplayState:
        self.replay_state.is_active = False
        self.replay_state.status = "STOPPED"
        self.replay_state.current_bar_index = 0
        return self.replay_state

    def get_replay_state(self) -> ReplayState:
        return self.replay_state


# Singleton instance
_MOCK_SERVICE: Optional[MockBackendService] = None


def get_mock_service() -> MockBackendService:
    global _MOCK_SERVICE
    if _MOCK_SERVICE is None:
        _MOCK_SERVICE = MockBackendService()
    return _MOCK_SERVICE
