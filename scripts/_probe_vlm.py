"""Tiny probe: invoke call_vlm() once per candidate model with a real frame.

Prints OK + short snippet or FAIL + error. Used as a cheap pre-benchmark check
so we don't burn 6 minutes on a benchmark where one model is invalid.

Usage:
    python scripts/_probe_vlm.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import cv2  # noqa: E402

from src.run import call_vlm  # noqa: E402

MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-image-preview",
]

VIDEO = REPO_ROOT / "data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4"
PROMPT = (
    "Respond with one short sentence describing what is visible in this frame. "
    "Plain text only, no JSON."
)


def main():
    # Match run.py's order: MY_OPENROUTER_API takes precedence over any
    # stale PROD_OPENROUTER_API that may still live in .env after rotation.
    key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("MY_OPENROUTER_API")
        or os.getenv("PROD_OPENROUTER_API")
    )
    if not key:
        print("ERROR: no OpenRouter key found in env", file=sys.stderr)
        sys.exit(1)

    if not VIDEO.exists():
        print(f"ERROR: missing video: {VIDEO}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("ERROR: could not read frame 1500", file=sys.stderr)
        sys.exit(1)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode()

    for model in MODELS:
        print(f"\n=== {model} ===")
        try:
            resp = call_vlm(key, b64, PROMPT, model=model)
            snippet = (resp or "").replace("\n", " ")[:240]
            print(f"OK: {snippet}")
        except Exception as e:
            print(f"FAIL: {str(e)[:300]}")


if __name__ == "__main__":
    main()
