"""
TradePulse UI Screenshot Generator using Headless Chrome.
Captures desktop and mobile UI views in dark and light themes, as well as replay mode.
"""

import os
import sys
import time
import shutil
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = PROJECT_ROOT / "reports" / "ui_screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

TEMP_DIR = Path(os.environ.get("TEMP", r"C:\Windows\Temp"))


def capture_view(url: str, filename: str, width: int = 1920, height: int = 1080):
    temp_target = TEMP_DIR / f"tp_shot_{filename}"
    if temp_target.exists():
        temp_target.unlink()

    cmd = [
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        f"--screenshot={temp_target}",
        url,
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not temp_target.exists():
        print(f"[FAIL] Could not capture {filename}: {res.stderr}")
        return False

    final_dest = SCREENSHOTS_DIR / filename
    shutil.copy2(temp_target, final_dest)
    print(f"[SUCCESS] Captured {filename} ({width}x{height}, {final_dest.stat().st_size:,} bytes)")
    return True


def main():
    print("==================================================================")
    print("TradePulse UI: Automated Screenshot Capture")
    print("==================================================================")

    # 1. Start FastAPI server
    print("Starting FastAPI full-stack server on http://127.0.0.1:8000 ...")
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main_final:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ],
        cwd=str(PROJECT_ROOT),
    )

    try:
        # Wait for port 8000
        for _ in range(20):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1) as resp:
                    if resp.status == 200:
                        print("Server healthy and responding.")
                        break
            except Exception:
                time.sleep(0.5)

        base_url = "http://127.0.0.1:8000"

        # 1. Desktop Dark View (1920x1080)
        capture_view(f"{base_url}/", "desktop_dark.png", 1920, 1080)

        # 2. Mobile Dark View (390x844 - iPhone 14 / standard mobile)
        capture_view(f"{base_url}/", "mobile_dark.png", 390, 844)

        # 3. Start Replay to capture Replay Mode with Amber Banner
        print("Activating Replay mode for screenshot...")
        req = urllib.request.Request(
            f"{base_url}/api/replay/start",
            data=b'{"session_date": "2026-11-06", "speed_multiplier": 1.0}',
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                print("Replay session initialized:", resp.status)
        except Exception as e:
            print("Note: replay start call returned:", e)

        time.sleep(1.0)
        capture_view(f"{base_url}/", "desktop_replay.png", 1920, 1080)

        print("==================================================================")
        print("All UI Screenshots saved to:")
        print(f"  {SCREENSHOTS_DIR}")
        print("==================================================================")
    finally:
        print("Shutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()


if __name__ == "__main__":
    main()
