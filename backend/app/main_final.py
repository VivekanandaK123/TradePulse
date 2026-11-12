"""
TradePulse - Step 14 Final: FastAPI Application Server

Main backend entrypoint for live inference server.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.routes.signals_final import router as signals_router
from backend.app.routes.api_v1 import router as api_v1_router
from backend.app.services.live_inference_final import get_live_service_final
from backend.app.services.background_worker_final import get_background_worker_final


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize service, seed buffer, and launch background worker
    service = get_live_service_final()
    seeded = service.seed_buffer_from_alpaca(num_bars=200)
    print(f"[STARTUP] LiveInferenceServiceFinal initialized with {seeded} warm-up bars.")

    worker = get_background_worker_final()
    worker.start()

    yield
    # Shutdown
    worker.stop()
    print("[SHUTDOWN] LiveInferenceServiceFinal gracefully stopped.")


app = FastAPI(
    title="TradePulse Live Indicator API (Final)",
    description="Intraday AI-powered directional trading indicator for META (NASDAQ) on 5-minute bars.",
    version="1.0.0-final",
    lifespan=lifespan,
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(signals_router)
app.include_router(api_v1_router)


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {
            "app": "TradePulse Live Indicator",
            "version": "1.0.0-final",
            "symbol": "META",
            "timeframe": "5Min",
            "status": "online",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main_final:app", host="0.0.0.0", port=8000, reload=False)

