"""
TradePulse - One-Command Application Launcher

Starts the unified FastAPI + React web application on http://localhost:8000.
Serves both the backend REST & SSE APIs and the compiled TradingView React interface.
"""

import os
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

def main():
    print("=" * 70)
    print("  TradePulse: AI Intraday Directional Indicator (META, NASDAQ 5-Min)")
    print("  IIT Patna - IMPACT Lab End-to-End Machine Learning Project")
    print("=" * 70)
    print("  Local Trader Interface : http://localhost:8000")
    print("  Interactive OpenAPI Docs: http://localhost:8000/docs")
    print("  Health Status Endpoint : http://localhost:8000/api/health")
    print("=" * 70)
    print("  Press CTRL+C to stop the server cleanly.")
    print("=" * 70)

    # Launch browser automatically if not running in CI/headless mode
    if os.environ.get("HEADLESS", "0") != "1":
        try:
            webbrowser.open("http://localhost:8000")
        except Exception:
            pass

    uvicorn.run(
        "backend.app.main_final:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()

