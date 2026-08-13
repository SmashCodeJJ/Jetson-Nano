#!/usr/bin/env python3
"""
Color tracking demo using OpenCV HSV detection.
Mac/PC compatible version — uses a sample image instead of Jetson GStreamer camera.
Original Jetson version: color_detection.py (CSI camera + OpenCV)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = ROOT / "color_detection_pho.png"
OUTPUT_DIR = ROOT / "results"


def detect_color(frame: np.ndarray, lower_hsv, upper_hsv) -> tuple[np.ndarray, str, tuple[int, int] | None]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, "no target", None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    center = (int(x + w / 2), int(y + h / 2))

    cv2.drawContours(frame, [largest], -1, (255, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

    screen_center = frame.shape[1] / 2
    offset = 50
    if center[0] < screen_center - offset:
        direction = "turn left"
    elif center[0] > screen_center + offset:
        direction = "turn right"
    else:
        direction = "keep (centered)"

    return mask, direction, center


def main() -> int:
    if not SAMPLE_IMAGE.exists():
        print(f"Sample image not found: {SAMPLE_IMAGE}")
        return 1

    frame = cv2.imread(str(SAMPLE_IMAGE))
    if frame is None:
        print("Failed to read sample image")
        return 1

    # Green target HSV range (matches object_tracking.py)
    lower = np.array([36, 25, 25])
    upper = np.array([70, 255, 255])

    mask, direction, center = detect_color(frame.copy(), lower, upper)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_frame = OUTPUT_DIR / "color_detection_output.png"
    out_mask = OUTPUT_DIR / "color_detection_mask.png"
    cv2.imwrite(str(out_frame), frame)
    cv2.imwrite(str(out_mask), mask)

    print(f"Sample image: {SAMPLE_IMAGE.name}")
    print(f"Target center: {center}")
    print(f"Robot command: {direction}")
    print(f"Saved annotated output to {out_frame}")
    print(f"Saved mask to {out_mask}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
