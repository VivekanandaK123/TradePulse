"""
TradePulse - Production API Pydantic Schemas (FastAPI & React Trader Interface)

Strict Contract Rules:
- All timestamps exposed to the frontend are UTC epoch seconds (int).
- Complete typing for all OpenAPI endpoints.
- Standardized schemas for health, metadata, historical charts, live signals, and replay.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ==============================================================================
# Market Status & Diagnostics
# ==============================================================================

class MarketStatus(BaseModel):
    is_open: bool = Field(..., description="True if US equity markets are currently in regular trading hours (09:30-16:00 ET)")
    session_state: str = Field(..., description="'OPEN', 'CLOSED', 'EARLY_CLOSE', 'PRE_MARKET', 'AFTER_HOURS'")
    current_time_utc: int = Field(..., description="Current server time in UTC epoch seconds")
    next_open_utc: int = Field(..., description="Next market open timestamp in UTC epoch seconds")
    next_close_utc: int = Field(..., description="Next market close timestamp in UTC epoch seconds")
    countdown_seconds: int = Field(..., description="Seconds remaining until next market open or close")
    description: str = Field(..., description="Human-readable market status description")


class HealthResponse(BaseModel):
    status: str = Field(..., description="'HEALTHY', 'DEGRADED', or 'OFFLINE'")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    feed_connectivity: str = Field(..., description="'CONNECTED', 'REPLAY_MODE', or 'DISCONNECTED'")
    feed_name: str = Field(..., description="Data feed identifier, e.g. 'Alpaca IEX, 5-min'")
    active_symbol: str = Field(..., description="Asset symbol, 'META'")
    market_status: MarketStatus
    artifact_hashes: Dict[str, str] = Field(..., description="SHA-256 hashes of frozen model artifacts")
    model_version: str = Field(..., description="Model version tag and git commit hash")
    last_data_freshness_seconds: Optional[int] = Field(None, description="Seconds since the last closed 5-minute candle")


# ==============================================================================
# Model Metadata & Summary (from ui_summary_final.json)
# ==============================================================================

class SplitBoundary(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date: Optional[str] = None
    sessions: int
    valid_bars: Optional[int] = None


class ExperimentSplits(BaseModel):
    train: SplitBoundary
    embargo_1: SplitBoundary
    validation: SplitBoundary
    embargo_2: SplitBoundary
    test: SplitBoundary
    split_note: str


class TrainingDateRange(BaseModel):
    start_date: str
    end_date: str
    total_sessions: int
    total_raw_bars: int
    valid_training_rows: int
    dropped_eod_bars: int


class DatasetScope(BaseModel):
    training_date_range: TrainingDateRange
    experiment_splits: ExperimentSplits


class RecencyWeightingMeta(BaseModel):
    half_life: str
    decay_rate_alpha: float
    rationale: str


class SignalRuleMeta(BaseModel):
    conviction_threshold_tau: float
    formula: str
    expected_signals_per_day: float
    expected_coverage_pct: float
    expected_zero_signal_days_pct: float
    direction_logic: str


class ValidationMetrics(BaseModel):
    dataset: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    binary_sign_accuracy: float
    updown_auc: float
    gross_return_bps: float
    net_return_8bp_bps: float


class TestMetrics(BaseModel):
    dataset: str
    signals_emitted: int
    signals_per_day: float
    actual_coverage_pct: float
    zero_signal_days_pct: float
    accuracy: float
    balanced_accuracy: float
    balanced_accuracy_95ci: List[float]
    macro_f1: float
    macro_f1_95ci: List[float]
    binary_sign_accuracy: float
    binary_sign_accuracy_95ci: List[float]
    updown_auc: float
    gross_return_bps: float
    gross_return_95ci: List[float]
    net_return_5bp_bps: float
    net_return_8bp_bps: float
    net_return_8bp_95ci: List[float]
    net_return_12bp_bps: float


class HeadlineMetrics(BaseModel):
    validation: ValidationMetrics
    held_out_test: TestMetrics


class CostAssumption(BaseModel):
    institutional_friction_bps: float
    conservative_friction_bps: float
    low_friction_bps: float
    convention: str


class MetaResponse(BaseModel):
    asset: str
    exchange: str
    timeframe: str
    label_name: str
    theta_bps: float
    theta_pct: float
    horizon_bars: int
    horizon_minutes: int
    classes: Dict[str, int]
    feature_count: int
    features_summary: str
    model_architecture: str
    probability_calibration: str
    recency_weighting: RecencyWeightingMeta
    signal_rule: SignalRuleMeta
    dataset_scope: DatasetScope
    headline_metrics: HeadlineMetrics
    cost_assumption: CostAssumption
    edge_demonstrated: bool
    edge_statement: str
    disclaimer: str
    market_status: MarketStatus


# ==============================================================================
# Historical Chart & Candles
# ==============================================================================

class CandleBar(BaseModel):
    time: int = Field(..., description="UTC epoch seconds (candlestick open or close anchor for Lightweight Charts)")
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    is_filled: bool = Field(False, description="True if bar was synthetically filled due to missing feed interval")


class PeriodBand(BaseModel):
    name: str = Field(..., description="'train', 'validation', 'test', or 'embargo'")
    label: str = Field(..., description="Display label, e.g. 'Training Set'")
    start_time: int = Field(..., description="Start timestamp in UTC epoch seconds")
    end_time: int = Field(..., description="End timestamp in UTC epoch seconds")
    color: str = Field(..., description="RGBA color code for chart background tint")
    tooltip: str = Field(..., description="Informational tooltip explaining the split purpose")


class HistoryResponse(BaseModel):
    symbol: str
    range: str = Field(..., description="'1D', '5D', '1M', '3M', or 'All'")
    bars_count: int
    bars: List[CandleBar]
    period_bands: List[PeriodBand]
    prev_close: Optional[float] = None
    etag: str


class OOFSignalItem(BaseModel):
    time: int = Field(..., description="UTC epoch seconds of closed candle")
    direction: str = Field(..., description="'Up', 'Down', or 'no clear signal'")
    signal_code: int = Field(..., description="1, -1, or 0")
    confidence_pct: float
    conviction: float
    is_gated: bool


class OOFSignalsResponse(BaseModel):
    available: bool
    count: int
    signals: List[OOFSignalItem]
    note: str = Field(..., description="'Out-of-fold, historical, not live results.'")


# ==============================================================================
# Live Testing & Signals
# ==============================================================================

class ReliabilityFlag(BaseModel):
    id: str = Field(..., description="Machine identifier, e.g. 'DATA_STALE', 'BAR_FILLED', 'SESSION_BOUNDARY'")
    name: str = Field(..., description="User-friendly chip title")
    level: str = Field(..., description="'NORMAL', 'ADVISORY', or 'WARNING'")
    message: str = Field(..., description="Plain-language explanation of reliability impact")
    guide_anchor: str = Field(..., description="Anchor link to user guide section")


class ProbabilityDistribution(BaseModel):
    down: float
    neutral: float
    up: float


class PredictionSignal(BaseModel):
    bar_time: int = Field(..., description="UTC epoch seconds of closed candle")
    bar_timestamp_ny: str = Field(..., description="Candle close in New York time, e.g. '14:35:00 ET'")
    close_price: float
    direction: str = Field(..., description="'Up', 'Down', or 'no clear signal'")
    signal_code: int = Field(..., description="1, -1, or 0")
    confidence_pct: float = Field(..., description="Calibrated probability of signaled class in percent")
    conviction: float = Field(..., description="Directional conviction: max(P(Down), P(Up))")
    threshold: float = Field(..., description="Conviction threshold (0.2635)")
    is_gated: bool = Field(..., description="True if conviction >= threshold")
    probabilities: ProbabilityDistribution
    locked_at_utc: int = Field(..., description="Timestamp when prediction was locked into append-only log")
    horizon_end_utc: int = Field(..., description="Timestamp when the 15-minute prediction horizon ends")
    model_version: str
    reliability_flags: List[ReliabilityFlag]
    outcome: Optional[Dict[str, Any]] = None


class TestNowResponse(BaseModel):
    status: str = Field(..., description="'SUCCESS', 'ALREADY_STORED', 'WAITING_NEXT_BAR', or 'MARKET_CLOSED'")
    signal: Optional[PredictionSignal] = None
    latest_closed_bar: Optional[CandleBar] = None
    forming_bar: Optional[CandleBar] = Field(None, description="Muted still-forming live bar (no signal attached)")
    recent_bars: List[CandleBar] = Field(..., description="Closed candles for live chart view")
    market_status: MarketStatus


class SignalLogItem(BaseModel):
    signal_id: str
    bar_time: int
    bar_timestamp_ny: str
    direction: str
    signal_code: int
    confidence_pct: float
    conviction: float
    entry_price: float
    is_gated: bool
    checksum: str
    outcome: Optional[Dict[str, Any]] = None


class SignalsLogResponse(BaseModel):
    session: str
    count: int
    signals: List[SignalLogItem]


class OutcomeItem(BaseModel):
    signal_id: str
    signal_time: int
    resolution_time: int
    direction: str
    signal_code: int
    confidence_pct: float
    entry_price: float
    exit_price: float
    realized_return_bps: float
    directional_return_bps: float
    net_return_bps_8bp: float
    is_hit: bool


class OutcomesResponse(BaseModel):
    count: int
    outcomes: List[OutcomeItem]


# ==============================================================================
# Replay Mode Schemas
# ==============================================================================

class ReplayStartRequest(BaseModel):
    session_date: str = Field("2026-11-06", description="YYYY-MM-DD trading session date to replay")
    speed_multiplier: float = Field(1.0, description="Replay playback speed multiplier: 1.0, 5.0, or 20.0")


class ReplayState(BaseModel):
    is_active: bool
    status: str = Field(..., description="'STOPPED', 'PLAYING', or 'PAUSED'")
    session_date: str
    current_bar_index: int
    total_bars: int
    speed_multiplier: float
    current_bar: Optional[CandleBar] = None
    current_signal: Optional[PredictionSignal] = None
    signals_emitted_count: int = 0

