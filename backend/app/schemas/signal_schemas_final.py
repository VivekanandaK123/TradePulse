"""
TradePulse - Step 14 Final: Pydantic Schemas for Live Signal API
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ProbabilityDistribution(BaseModel):
    down: float = Field(..., description="Calibrated probability of Down move (<= -15 bps)")
    neutral: float = Field(..., description="Calibrated probability of Neutral move (< 15 bps)")
    up: float = Field(..., description="Calibrated probability of Up move (>= +15 bps)")


class SignalPayload(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the closed 5-minute candle in New York time")
    close_price: float = Field(..., description="Closing price of the candle in USD")
    direction: str = Field(..., description="'Up', 'Down', or 'no clear signal'")
    signal_code: int = Field(..., description="Direction code: 1 (Up), -1 (Down), 0 (Neutral / Gated out)")
    confidence_pct: float = Field(..., description="Calibrated confidence percentage (e.g. 68.2)")
    conviction: float = Field(..., description="Directional conviction: max(P(Down), P(Up))")
    threshold: float = Field(..., description="Conviction threshold required for directional signal (0.2635)")
    is_gated: bool = Field(..., description="True if conviction >= threshold and signal is active")
    probabilities: ProbabilityDistribution
    model_version: str = Field(..., description="Production model version tag")


class SignalHistoryResponse(BaseModel):
    count: int
    signals: List[SignalPayload]


class MonitoringStatusResponse(BaseModel):
    status: str = Field(..., description="'HEALTHY', 'DEGRADED', 'WARNING', or 'INITIALIZING'")
    symbol: str
    uptime_seconds: float
    buffer_bars_count: int
    min_required_bars: int
    last_update_ny: Optional[str]
    total_bars_processed: int
    signals_today_count: int
    expected_signals_per_day: float
    gaps_filled_count: int
    error_count: int
    model_version: str
    active_feed: str

