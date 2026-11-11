"""
TradePulse - Market Calendar & Session Status Service
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Uses pandas-market-calendars (NASDAQ) to determine:
- Current regular trading status (09:30 to 16:00 ET).
- Half-day session handling (early close at 13:00 ET).
- Accurate countdown to next market open or close in UTC epoch seconds.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd
import pandas_market_calendars as mcal

from backend.app.schemas.api_schemas import MarketStatus

NASDAQ_CALENDAR = mcal.get_calendar("NASDAQ")


def get_current_market_status(reference_time: Optional[datetime] = None) -> MarketStatus:
    """
    Computes accurate NASDAQ market status, next open/close timestamps, and countdown.
    """
    if reference_time is None:
        now_utc = datetime.now(timezone.utc)
    else:
        now_utc = reference_time.astimezone(timezone.utc)

    now_ts = int(now_utc.timestamp())

    # Query schedule for a window of 14 days around now
    start_date = (now_utc - timedelta(days=2)).strftime("%Y-%m-%d")
    end_date = (now_utc + timedelta(days=12)).strftime("%Y-%m-%d")

    schedule = NASDAQ_CALENDAR.schedule(start_date=start_date, end_date=end_date)
    if schedule.empty:
        # Fallback if empty schedule
        return MarketStatus(
            is_open=False,
            session_state="CLOSED",
            current_time_utc=now_ts,
            next_open_utc=now_ts + 3600,
            next_close_utc=now_ts + 7200,
            countdown_seconds=3600,
            description="Market closed (Weekend / Holiday)",
        )

    # Check if current time falls within any scheduled session
    is_open = False
    session_state = "CLOSED"
    next_open_utc = now_ts
    next_close_utc = now_ts
    countdown = 0
    desc = "Market closed"

    for _, row in schedule.iterrows():
        open_time = row["market_open"].to_pydatetime()
        close_time = row["market_close"].to_pydatetime()

        open_epoch = int(open_time.timestamp())
        close_epoch = int(close_time.timestamp())

        # If currently within market hours
        if open_epoch <= now_ts < close_epoch:
            is_open = True
            is_half_day = (close_time - open_time).total_seconds() < 21600 # < 6 hours
            session_state = "EARLY_CLOSE" if is_half_day else "OPEN"
            next_close_utc = close_epoch
            countdown = close_epoch - now_ts

            # Find next open after this session
            future_opens = [
                int(r["market_open"].to_pydatetime().timestamp())
                for _, r in schedule.iterrows()
                if int(r["market_open"].to_pydatetime().timestamp()) > now_ts
            ]
            next_open_utc = future_opens[0] if future_opens else close_epoch + 17.5 * 3600

            desc = f"Market open ({session_state}), closes in {timedelta(seconds=countdown)}"
            break

    if not is_open:
        # Find next scheduled market open
        future_opens = [
            int(r["market_open"].to_pydatetime().timestamp())
            for _, r in schedule.iterrows()
            if int(r["market_open"].to_pydatetime().timestamp()) > now_ts
        ]
        if future_opens:
            next_open_utc = future_opens[0]
            countdown = next_open_utc - now_ts
            desc = f"Market closed, opens in {timedelta(seconds=countdown)}"
        else:
            next_open_utc = now_ts + 14400
            countdown = 14400
            desc = "Market closed"

        # Check pre-market (04:00 to 09:30 ET) or after-hours
        # For simplicity, session_state = "CLOSED"
        session_state = "CLOSED"

    return MarketStatus(
        is_open=is_open,
        session_state=session_state,
        current_time_utc=now_ts,
        next_open_utc=next_open_utc,
        next_close_utc=next_close_utc,
        countdown_seconds=countdown,
        description=desc,
    )
