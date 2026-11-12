"""
TradePulse - Mock Backend Service (Phase 0 API Contract Delivery)

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
from backend.app.services.live_inference_final import LiveInferenceServiceFinal, get_live_service_final

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"

# Cached historical data
_CACHED_CLEAN_DF: Optional[pd.DataFrame] = None
_CACHED_DEMO_DF: Optional[pd.DataFrame] = None


def _get_demo_df() -> Optional[pd.DataFrame]:
    global _CACHED_DEMO_DF
    if _CACHED_DEMO_DF is None:
        demo_path = DATA_DIR / "processed" / "demo_live_testing_data.csv"
        if demo_path.exists():
            df = pd.read_csv(demo_path)
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df["time"] = ts.astype("int64") // 10**6
            df["timestamp_ny"] = ts.dt.tz_convert("America/New_York")
            df["session_date"] = df["timestamp_ny"].dt.date.astype(str)
            if "vwap" not in df.columns:
                df["vwap"] = df["close"]
            if "is_filled" not in df.columns:
                df["is_filled"] = 0
            _CACHED_DEMO_DF = df
    return _CACHED_DEMO_DF


def _get_clean_df() -> pd.DataFrame:
    global _CACHED_CLEAN_DF
    if _CACHED_CLEAN_DF is None:
        clean_path = DATA_DIR / "processed" / "META_5min_clean.csv"
        if clean_path.exists():
            df = pd.read_csv(clean_path)
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df["time"] = ts.astype("int64") // 10**6
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
        self.live_since_epoch: Optional[int] = None

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

        # Use session-based slicing so each range shows complete trading sessions
        # regardless of bars-per-day variation (early closes, data gaps etc.)
        all_sessions = df["session_date"].unique()  # already sorted chronologically
        n_sessions = len(all_sessions)

        if range_upper == "1D":
            keep_sessions = all_sessions[-1:]        # last 1 session
        elif range_upper == "5D":
            keep_sessions = all_sessions[-5:]        # last 5 sessions
        elif range_upper == "1M":
            keep_sessions = all_sessions[-21:]       # ~1 month of trading days
        elif range_upper == "3M":
            keep_sessions = all_sessions[-63:]       # ~3 months of trading days
        else:  # "ALL"
            keep_sessions = all_sessions             # every session

        sub_df = df[df["session_date"].isin(keep_sessions)].copy()

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

    def _build_live_test_now(self, market_status: MarketStatus) -> Optional[TestNowResponse]:
        """
        Fetches latest live Alpaca IEX bars during trading hours, evaluates any
        newly closed 5-minute bar idempotently via LiveInferenceServiceFinal,
        and returns the real live TestNowResponse. Returns None on offline/error.
        """
        live_svc = get_live_service_final()
        if not live_svc.alpaca_key or not live_svc.alpaca_secret:
            return None

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed, Adjustment
            from zoneinfo import ZoneInfo

            # Ensure buffer has >= 35 warm-up bars
            if len(live_svc.candles_buffer) < 35:
                hist_df = _get_clean_df()
                live_svc.seed_buffer_from_history(hist_df.tail(200))

            client = StockHistoricalDataClient(live_svc.alpaca_key, live_svc.alpaca_secret)
            now_utc = datetime.now(timezone.utc)
            current_bucket = now_utc.replace(minute=(now_utc.minute // 5) * 5, second=0, microsecond=0)
            ny_tz = ZoneInfo("America/New_York")

            req = StockBarsRequest(
                symbol_or_symbols=live_svc.symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=now_utc - timedelta(days=5),
                end=now_utc,
                feed=DataFeed.IEX,
                adjustment=Adjustment.RAW,
            )
            bars_resp = client.get_stock_bars(req)
            raw_list = bars_resp[live_svc.symbol] if live_svc.symbol in bars_resp.data else []
            if not raw_list:
                return None

            # Filter to regular NASDAQ trading hours (09:30 - 15:55 ET)
            rth_bars = []
            for b in raw_list:
                ny_t = b.timestamp.astimezone(ny_tz).time()
                if (ny_t.hour > 9 or (ny_t.hour == 9 and ny_t.minute >= 30)) and ny_t.hour < 16:
                    rth_bars.append(b)

            closed_bars = [b for b in rth_bars if b.timestamp < current_bucket]
            forming_candidates = [b for b in rth_bars if b.timestamp >= current_bucket]

            if not closed_bars:
                return None

            today_ny = now_utc.astimezone(ny_tz).date()
            last_buf_ts = (
                pd.to_datetime(live_svc.candles_buffer[-1]["timestamp"], utc=True)
                if live_svc.candles_buffer
                else None
            )

            # Seed prior bars (and any bars closed before live_since_epoch) into buffer without emitting live signals
            warmup_dicts = []
            today_bars = []
            for b in closed_bars:
                if last_buf_ts is not None and b.timestamp <= last_buf_ts:
                    continue
                b_ny_date = b.timestamp.astimezone(ny_tz).date()
                bar_close_epoch = int(b.timestamp.timestamp()) + 300
                is_before_clear = self.live_since_epoch is not None and bar_close_epoch <= self.live_since_epoch
                if b_ny_date < today_ny or is_before_clear:
                    warmup_dicts.append({
                        "timestamp": str(b.timestamp),
                        "timestamp_ny": str(b.timestamp.astimezone(ny_tz)),
                        "session_date": str(b.timestamp.astimezone(ny_tz).date()),
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                        "trade_count": float(getattr(b, "trade_count", 0)),
                        "vwap": float(getattr(b, "vwap", b.close)),
                        "is_filled": 0,
                    })
                else:
                    today_bars.append(b)

            if warmup_dicts:
                with live_svc.lock:
                    live_svc.candles_buffer.extend(warmup_dicts)
                    if len(live_svc.candles_buffer) > live_svc.buffer_capacity:
                        live_svc.candles_buffer = live_svc.candles_buffer[-live_svc.buffer_capacity:]

            # Process today's new closed bars chronologically (idempotent by timestamp)
            for b in today_bars:
                bar_dict = {
                    "timestamp": str(b.timestamp),
                    "timestamp_ny": str(b.timestamp.astimezone(ny_tz)),
                    "session_date": str(b.timestamp.astimezone(ny_tz).date()),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "trade_count": float(getattr(b, "trade_count", 0)),
                    "vwap": float(getattr(b, "vwap", b.close)),
                    "is_filled": 0,
                }
                live_svc.process_closed_bar(bar_dict)

            forming_bar: Optional[CandleBar] = None
            if forming_candidates:
                fb = forming_candidates[-1]
                forming_bar = CandleBar(
                    time=int(fb.timestamp.timestamp()),
                    open=round(float(fb.open), 2),
                    high=round(float(fb.high), 2),
                    low=round(float(fb.low), 2),
                    close=round(float(fb.close), 2),
                    volume=round(float(fb.volume), 0),
                    vwap=round(float(getattr(fb, "vwap", fb.close)), 2),
                    is_filled=False,
                )

            # Build recent_bars from live buffer (filtered by live_since_epoch if cleared)
            buf_tail = live_svc.candles_buffer[-150:]
            recent_bars: List[CandleBar] = []
            for r in buf_tail:
                t_sec = int(pd.to_datetime(r["timestamp"], utc=True).timestamp())
                if self.live_since_epoch is not None and (t_sec + 300) <= self.live_since_epoch:
                    continue
                recent_bars.append(
                    CandleBar(
                        time=t_sec,
                        open=round(float(r["open"]), 2),
                        high=round(float(r["high"]), 2),
                        low=round(float(r["low"]), 2),
                        close=round(float(r["close"]), 2),
                        volume=round(float(r["volume"]), 0),
                        vwap=round(float(r.get("vwap", r["close"])), 2),
                        is_filled=bool(r.get("is_filled", 0) == 1),
                    )
                )

            latest_sig = live_svc.get_latest_signal()
            if self.live_since_epoch is not None:
                if not recent_bars:
                    return TestNowResponse(
                        status="WAITING_NEXT_BAR",
                        signal=None,
                        latest_closed_bar=None,
                        forming_bar=forming_bar,
                        recent_bars=[],
                        market_status=market_status,
                    )
                if latest_sig:
                    sig_t = int(pd.to_datetime(latest_sig["timestamp"], utc=True).timestamp())
                    if (sig_t + 300) <= self.live_since_epoch:
                        latest_sig = None

            if not latest_sig or not recent_bars:
                return None

            latest_closed_bar = recent_bars[-1]

            sig_ts = int(pd.to_datetime(latest_sig["timestamp"], utc=True).timestamp())
            sig_ny = pd.to_datetime(latest_sig["timestamp"], utc=True).tz_convert("America/New_York")
            is_boundary = (sig_ny.hour == 9 and sig_ny.minute < 45) or (sig_ny.hour == 15 and sig_ny.minute >= 45)

            reliability_flags = [
                ReliabilityFlag(
                    id="SESSION_BOUNDARY",
                    name="Opening / Closing Regime" if is_boundary else "Mid-Session Activity",
                    level="ADVISORY" if is_boundary else "NORMAL",
                    message=(
                        "Candle is inside opening (09:30-09:45) or closing (15:45-16:00) high-volatility regime."
                        if is_boundary
                        else "Candle is outside opening (09:30-09:45) and closing (15:45-16:00) volatility regimes."
                    ),
                    guide_anchor="session-boundaries",
                ),
                ReliabilityFlag(
                    id="DATA_FRESHNESS",
                    name="Live Alpaca IEX Feed",
                    level="NORMAL",
                    message="Evaluated on live closed Alpaca IEX 5-minute bar.",
                    guide_anchor="stale-data",
                ),
                ReliabilityFlag(
                    id="CONVICTION_GATING",
                    name="Conviction Evaluated",
                    level="NORMAL" if latest_sig["is_gated"] else "ADVISORY",
                    message=(
                        f"Conviction ({latest_sig['conviction']:.4f}) exceeds threshold τ = {latest_sig['threshold']:.4f}."
                        if latest_sig["is_gated"]
                        else f"Conviction ({latest_sig['conviction']:.4f}) is below threshold τ = {latest_sig['threshold']:.4f}."
                    ),
                    guide_anchor="conviction-gating",
                ),
            ]

            signal = PredictionSignal(
                bar_time=sig_ts,
                bar_timestamp_ny=str(latest_sig["timestamp"]),
                close_price=round(float(latest_sig["close_price"]), 2),
                direction=latest_sig["direction"],
                signal_code=int(latest_sig["signal_code"]),
                confidence_pct=float(latest_sig["confidence_pct"]),
                conviction=float(latest_sig["conviction"]),
                threshold=float(latest_sig["threshold"]),
                is_gated=bool(latest_sig["is_gated"]),
                probabilities=ProbabilityDistribution(**latest_sig["probabilities"]),
                locked_at_utc=int(time.time()),
                horizon_end_utc=sig_ts + 900,
                model_version=latest_sig.get("model_version", self.model_version),
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
        except Exception as e:
            print(f"[LIVE TEST NOW ERROR] Falling back to cached history: {e}", file=sys.stderr)
            return None

    def test_now(self) -> TestNowResponse:
        market_status = get_current_market_status()
        if market_status.is_open:
            live_resp = self._build_live_test_now(market_status)
            if live_resp is not None:
                return live_resp

        df = _get_clean_df()

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

    def clear_live_session(self) -> Dict[str, Any]:
        """
        Resets the live view baseline to the current timestamp so only candles
        closing AFTER this moment are plotted, logged, and scored.
        """
        self.live_since_epoch = int(time.time())
        return {"status": "CLEARED", "live_since_epoch": self.live_since_epoch}

    def get_signals_log(self, session: Optional[str] = None) -> SignalsLogResponse:
        # Load real live signals log or isolated replay log
        if session == "replay":
            log_path = DATA_DIR / "replay_signals.jsonl"
            outcomes_path = DATA_DIR / "replay_outcomes.jsonl"
        else:
            log_path = DATA_DIR / "live_signals_final.jsonl"
            outcomes_path = DATA_DIR / "live_outcomes_final.jsonl"

        # Index resolved 15-min (3-bar) outcomes by signal_timestamp
        outcomes_by_ts: Dict[str, Dict[str, Any]] = {}
        if outcomes_path.exists():
            with open(outcomes_path, "r", encoding="utf-8") as f_out:
                for line in f_out:
                    if not line.strip():
                        continue
                    rec_out = json.loads(line)
                    sig_ts_key = rec_out.get("signal_timestamp", "")
                    ret_bps = float(rec_out.get("realized_return_bps", 0.0))
                    if ret_bps > 15.0:
                        actual_dir = "Up"
                    elif ret_bps < -15.0:
                        actual_dir = "Down"
                    else:
                        actual_dir = "Neutral"
                    outcomes_by_ts[sig_ts_key] = {
                        "exit_price": rec_out.get("exit_price", 0.0),
                        "realized_return_bps": ret_bps,
                        "net_return_bps_8bp": rec_out.get("net_return_bps_8bp", 0.0),
                        "is_hit": bool(rec_out.get("is_correct", False)),
                        "actual_direction": actual_dir,
                    }

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
                    if session != "replay" and self.live_since_epoch is not None and (ts_epoch + 300) <= self.live_since_epoch:
                        continue
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
                            outcome=outcomes_by_ts.get(bar_ts),
                        )
                    )

        # Show newest predictions first
        items.reverse()

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
                    if mode != "replay" and self.live_since_epoch is not None and (sig_ts + 300) <= self.live_since_epoch:
                        continue
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
        demo_df = _get_demo_df()
        if demo_df is not None and "session_date" in demo_df.columns:
            dates = demo_df["session_date"].unique().tolist()
            if dates:
                return dates
        df = _get_clean_df()
        if "session_date" in df.columns:
            dates = df["session_date"].unique().tolist()
            return dates[-15:]
        return ["2026-11-09", "2026-11-10", "2026-11-11"]

    def start_replay(self, session_date: str = "2026-11-09", speed: float = 1.0) -> ReplayState:
        hist_df = _get_clean_df()
        demo_df = _get_demo_df()

        # Prefer unseen demo dataset (Nov 7-11, 2026) when session_date matches
        if demo_df is not None and "session_date" in demo_df.columns and (demo_df["session_date"] == session_date).any():
            # Combine tail of historical data + demo_df so rolling warm-up is 100% causal
            base_cols = [c for c in hist_df.columns if c in demo_df.columns]
            combined_df = pd.concat([hist_df[base_cols].tail(200), demo_df[base_cols]], ignore_index=True)
            session_mask = combined_df["session_date"] == session_date
            session_df = combined_df[session_mask].copy()
            first_idx = session_df.index[0]
            seed_start = max(0, first_idx - 200)
            seed_df = combined_df.iloc[seed_start:first_idx].copy()
        else:
            df = hist_df
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

    def resume_replay(self) -> ReplayState:
        if self.replay_state.is_active and self.replay_state.current_bar_index < self.replay_state.total_bars:
            self.replay_state.status = "PLAYING"
        return self.replay_state

    def set_replay_speed(self, speed: float) -> ReplayState:
        self.replay_state.speed_multiplier = float(speed)
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
