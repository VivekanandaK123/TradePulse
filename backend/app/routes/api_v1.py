"""
TradePulse - REST & Streaming API Router (Phase 0 API Contract)
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Strict adherence to Product Definition (Sections 3-9):
- All timestamps in UTC epoch seconds.
- Idempotent signal testing and append-only log inspection.
- Streaming via Server-Sent Events (SSE) for auto mode.
- Interactive OpenAPI documentation generated automatically.
"""

import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Query, Response, Header
from fastapi.responses import StreamingResponse

from backend.app.schemas.api_schemas import (
    HealthResponse,
    MetaResponse,
    HistoryResponse,
    OOFSignalsResponse,
    TestNowResponse,
    SignalsLogResponse,
    OutcomesResponse,
    ReplayStartRequest,
    ReplayState,
)
from backend.app.services.mock_service import get_mock_service

router = APIRouter(prefix="/api", tags=["TradePulse Core API"])


@router.get("/health", response_model=HealthResponse, summary="System Health & Diagnostic Status")
def get_health():
    """
    Returns service health status, artifact cryptographic hashes,
    data feed connectivity ('Alpaca IEX, 5-min'), and real-time NASDAQ market calendar status.
    """
    service = get_mock_service()
    return service.get_health()


@router.get("/meta", response_model=MetaResponse, summary="Model Metadata & Performance Metrics")
def get_meta():
    """
    Returns complete model metadata loaded from ui_summary_final.json:
    - Target label definition (theta = 15 bps, 15-minute horizon)
    - 29 causal feature count & summaries
    - Headline validation and one-time held-out test metrics with 95% CIs
    - Cost assumptions (8 bp institutional convention)
    - edge_demonstrated boolean flag (false)
    - Real-time NASDAQ market hours & countdown
    """
    service = get_mock_service()
    return service.get_meta()


@router.get("/history", response_model=HistoryResponse, summary="Historical Candlesticks & Split Bands")
def get_history(
    range: str = Query(default="1M", pattern="^(1D|5D|1M|3M|All|all)$", description="Historical range: 1D, 5D, 1M, 3M, or All"),
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
    response: Response = Response(),
):
    """
    Returns lightweight financial candlesticks for the full training history (~19.6k bars)
    or zoomed ranges, along with labeled period bands for Train, Validation, and Test splits.
    Supports ETag HTTP caching for smooth client-side zooming and panning.
    """
    service = get_mock_service()
    range_str = "1M" if not isinstance(range, str) else range
    res = service.get_history(range_param=range_str)

    # Set ETag caching
    if response is not None and hasattr(response, "headers"):
        response.headers["ETag"] = f'"{res.etag}"'
        response.headers["Cache-Control"] = "public, max-age=300"

    return res


@router.get("/history/oof-signals", response_model=OOFSignalsResponse, summary="Out-of-Fold Historical Signals")
def get_oof_signals():
    """
    Returns out-of-fold historical predictions from the training cross-validation pipeline.
    Labeled with strict disclosure: 'Out-of-fold, historical, not live results.'
    """
    service = get_mock_service()
    return service.get_oof_signals()


@router.post("/live/test-now", response_model=TestNowResponse, summary="Evaluate Closed Candle Signal ('Test Now')")
def test_now():
    """
    Fetches the latest live closed 5-minute candle, evaluates directional conviction,
    applies the frozen conviction gating rule (tau = 0.2635), and returns the prediction payload.
    
    Idempotent: If a prediction for the current bar is already logged, returns the stored signal
    without recomputing or modifying the append-only log.
    """
    service = get_mock_service()
    return service.test_now()


@router.get("/live/stream", summary="Server-Sent Events (SSE) Live Bar & Signal Stream")
async def live_stream():
    """
    Server-Sent Events (SSE) streaming endpoint for 'Auto Mode'.
    Pushes newly closed 5-minute bar events and predictions automatically at candle close.
    """
    service = get_mock_service()

    async def event_generator():
        while True:
            # Emit a heartbeat/closed bar update every 15 seconds
            res = service.test_now()
            payload = {
                "event": "candle_close",
                "data": res.model_dump(),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(15)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/signals", response_model=SignalsLogResponse, summary="Immutable Append-Only Signal Log")
def get_signals_log(session: Optional[str] = Query(default=None, description="Filter by session date, e.g. 2026-11-06")):
    """
    Returns the immutable audit log of emitted directional signals from live_signals_final.jsonl.
    Includes tamper-evident checksums for verifiable audit compliance.
    """
    service = get_mock_service()
    session_str = session if isinstance(session, str) else None
    return service.get_signals_log(session=session_str)


@router.get("/outcomes", response_model=OutcomesResponse, summary="Realized 3-Bar Outcomes")
def get_outcomes(mode: Optional[str] = Query(default=None, description="Optional mode, e.g. 'replay'")):
    """
    Returns realized 15-minute (3-bar) trade outcomes from live_outcomes_final.jsonl (or replay_outcomes.jsonl):
    gross return, directional return, net return (@ 8bp cost convention), and hit/miss status.
    """
    service = get_mock_service()
    mode_str = mode if isinstance(mode, str) else None
    return service.get_outcomes(mode=mode_str)


# ==============================================================================
# Replay Mode Endpoints
# ==============================================================================

@router.post("/replay/start", response_model=ReplayState, summary="Start Historical Session Replay")
def replay_start(req: ReplayStartRequest):
    """
    Starts stepping through a chosen historical trading session candle-by-candle.
    Replay writes strictly to an isolated replay log, never modifying live logs.
    """
    service = get_mock_service()
    return service.start_replay(session_date=req.session_date, speed=req.speed_multiplier)


@router.post("/replay/step", response_model=ReplayState, summary="Step Forward One Replay Candle")
def replay_step():
    """
    Advances replay by one 5-minute candle.
    """
    service = get_mock_service()
    return service.step_replay()


@router.post("/replay/pause", response_model=ReplayState, summary="Pause Replay Playback")
def replay_pause():
    """
    Pauses active session replay.
    """
    service = get_mock_service()
    return service.pause_replay()


@router.post("/replay/resume", response_model=ReplayState, summary="Resume Replay Playback")
def replay_resume():
    """
    Resumes paused session replay from current bar index without restarting.
    """
    service = get_mock_service()
    return service.resume_replay()


@router.post("/replay/speed", response_model=ReplayState, summary="Set Replay Speed Multiplier")
def replay_speed(speed_multiplier: float = Query(..., ge=0.5, le=50.0, description="Playback speed multiplier")):
    """
    Updates playback speed in place without restarting the replay session.
    """
    service = get_mock_service()
    return service.set_replay_speed(speed=speed_multiplier)


@router.post("/replay/stop", response_model=ReplayState, summary="Stop Replay Playback")
def replay_stop():
    """
    Stops replay and returns state to STOPPED.
    """
    service = get_mock_service()
    return service.stop_replay()


@router.get("/replay/state", response_model=ReplayState, summary="Get Current Replay Playback State")
def replay_state():
    """
    Returns the current replay playback position, active candle, and emitted signals count.
    """
    service = get_mock_service()
    return service.get_replay_state()


@router.get("/replay/sessions", summary="Get Available Replay Trading Sessions")
def replay_sessions():
    """
    Returns list of recent available trading sessions that can be selected for historical replay.
    """
    service = get_mock_service()
    return {"sessions": service.get_replay_sessions()}

