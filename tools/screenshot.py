"""Capture the stroke casebook with system Chrome."""

from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
PORT = 8771
BASE = f"http://127.0.0.1:{PORT}"
CHROME = "/usr/bin/google-chrome-stable"
PAGES = [
    ("file_and_age.png", f"{BASE}/?shot=open", "1440,900"),
    ("model_comparison.png", f"{BASE}/?shot=models", "1440,980"),
    ("desk_threshold.png", f"{BASE}/?shot=point&t=0.10", "1440,900"),
    ("calibration_odds.png", f"{BASE}/?shot=burden", "1440,980"),
]


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(BASE, timeout=1)
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("server did not start")
        for name, url, size in PAGES:
            dest = SHOTS / name
            subprocess.run(
                [
                    CHROME,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=1",
                    f"--window-size={size}",
                    "--user-data-dir=/tmp/stroke-chrome",
                    "--disk-cache-dir=/tmp/stroke-chrome-cache",
                    "--virtual-time-budget=8000",
                    f"--screenshot={dest}",
                    url,
                ],
                check=True,
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(dest, dest.stat().st_size)
    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    main()
