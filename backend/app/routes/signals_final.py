"""
TradePulse - Step 14 Final: Live Signal REST Endpoints

Exposes FastAPI routes:
- GET /api/v1/signals/latest: latest candle's signal, confidence, timestamp.
- GET /api/v1/signals/history: historical signals for chart rendering.
- GET /api/v1/monitoring/status: service health, uptime, buffer size, error count, and daily signal counter.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.signal_schemas_final import (
    SignalPayload,
    SignalHistoryResponse,
    MonitoringStatusResponse,
)
from backend.app.services.live_inference_final import get_live_service_final

router = APIRouter(prefix="/api/v1", tags=["Signals Final"])


@router.get("/signals/latest", response_model=SignalPayload)
def get_latest_signal():
    """
    Returns the latest emitted prediction for the most recent closed 5-minute candle.
    """
    service = get_live_service_final()
    latest = service.get_latest_signal()
    if not latest:
        raise HTTPException(
            status_code=404,
            detail="No signals generated yet. Buffer may still be warming up (< 35 bars).",
        )
    return latest


@router.get("/signals/history", response_model=SignalHistoryResponse)
def get_signal_history(limit: int = Query(100, ge=1, le=500)):
    """
    Returns recent closed candle signals for lightweight financial chart rendering.
    """
    service = get_live_service_final()
    history = service.get_signal_history(limit=limit)
    return {"count": len(history), "signals": history}


@router.get("/monitoring/status", response_model=MonitoringStatusResponse)
def get_monitoring_status():
    """
    Returns system health, buffer size, uptime, and running signal frequency vs target (37.5/day).
    """
    service = get_live_service_final()
    return service.get_monitoring_status()


@router.post("/signals/feed_bar", response_model=Optional[SignalPayload])
def feed_candle_bar(bar_dict: Dict[str, Any]):
    """
    Internal / Testing endpoint: feeds a single closed candle into the inference service.
    """
    service = get_live_service_final()
    payload = service.process_closed_bar(bar_dict)
    if payload is None:
        raise HTTPException(
            status_code=202,
            detail="Bar accepted, but insufficient warm-up buffer for signal generation.",
        )
    return payload

