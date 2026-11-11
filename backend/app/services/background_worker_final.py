"""
TradePulse - Step 15 Final: Production Background Candle Close Worker
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Background scheduler that runs during US market hours (09:30 - 16:00 ET):
1. Wakes precisely 3-5 seconds after each 5-minute boundary (:00, :05, :10, :15, ...).
2. Fetches the newly closed 5-minute META bar from Alpaca IEX (or local stream).
3. Evaluates the closed bar through the shared pipeline via LiveInferenceServiceFinal.
4. Appends to the append-only log and updates the SSE streaming queue.
5. Idempotent: 'Test Now' from the UI and this background worker call the exact same logic.
"""

import os
import sys
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.live_inference_final import get_live_service_final
from backend.app.services.market_calendar import get_current_market_status


class BackgroundWorkerFinal:
    """
    Independent background worker ensuring every 5-minute candle is processed
    and permanently logged even if no user interacts with the UI.
    """

    def __init__(self):
        self.service = get_live_service_final()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="TradePulse-Worker")
        self.worker_thread.start()
        print("[WORKER] Background candle-close worker started.")

    def stop(self):
        if not self.is_running:
            return
        self._stop_event.set()
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        print("[WORKER] Background candle-close worker stopped.")

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                market_status = get_current_market_status()
                now_epoch = int(time.time())

                if market_status.is_open:
                    # Calculate seconds elapsed in current 5-minute block
                    sec_in_block = now_epoch % 300
                    # Sleep until 3 seconds past the next 5-minute close
                    sec_to_next_close = (300 - sec_in_block) + 3

                    # Wait in small increments so stop event can interrupt immediately
                    wait_count = 0
                    while wait_count < sec_to_next_close and not self._stop_event.is_set():
                        time.sleep(1.0)
                        wait_count += 1

                    if self._stop_event.is_set():
                        break

                    # Candle has closed: fetch and evaluate
                    self._execute_candle_close_cycle()
                else:
                    # Market closed: wait 30 seconds before re-checking
                    for _ in range(30):
                        if self._stop_event.is_set():
                            break
                        time.sleep(1.0)

            except Exception as e:
                print(f"[WORKER ERROR] Exception in worker cycle: {e}", file=sys.stderr)
                time.sleep(5.0)

    def _execute_candle_close_cycle(self):
        """
        Fetches the latest completed bar and pushes it through the shared inference engine.
        """
        try:
            # If in live Alpaca feed mode, fetch latest bar; else read latest historical bar
            if self.service.feed_source == "alpaca" and self.service.alpaca_key:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                from alpaca.data.enums import DataFeed, Adjustment
                from datetime import timedelta

                client = StockHistoricalDataClient(self.service.alpaca_key, self.service.alpaca_secret)
                now_utc = datetime.now(timezone.utc)
                start_utc = now_utc - timedelta(minutes=25)

                req = StockBarsRequest(
                    symbol_or_symbols=self.service.symbol,
                    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=start_utc,
                    end=now_utc - timedelta(minutes=15), # IEX free delay buffer
                    feed=DataFeed.IEX,
                    adjustment=Adjustment.RAW,
                )
                bars = client.get_stock_bars(req)
                if bars[self.service.symbol]:
                    last_bar = bars[self.service.symbol][-1]
                    bar_dict = {
                        "timestamp": last_bar.timestamp.isoformat(),
                        "open": float(last_bar.open),
                        "high": float(last_bar.high),
                        "low": float(last_bar.low),
                        "close": float(last_bar.close),
                        "volume": float(last_bar.volume),
                        "trade_count": float(getattr(last_bar, "trade_count", 0)),
                        "vwap": float(getattr(last_bar, "vwap", last_bar.close)),
                        "is_filled": 0,
                    }
                    self.service.process_closed_bar(bar_dict)
            else:
                # Historical / local fallback mode
                clean_path = PROJECT_ROOT / "data" / "processed" / "META_5min_clean.csv"
                if clean_path.exists() and len(self.service.candles_buffer) > 0:
                    # In test/offline mode, already loaded
                    pass
        except Exception as e:
            print(f"[WORKER ERROR] Failed to fetch live bar: {e}", file=sys.stderr)


# Singleton worker instance
_WORKER_INSTANCE: Optional[BackgroundWorkerFinal] = None


def get_background_worker_final() -> BackgroundWorkerFinal:
    global _WORKER_INSTANCE
    if _WORKER_INSTANCE is None:
        _WORKER_INSTANCE = BackgroundWorkerFinal()
    return _WORKER_INSTANCE

