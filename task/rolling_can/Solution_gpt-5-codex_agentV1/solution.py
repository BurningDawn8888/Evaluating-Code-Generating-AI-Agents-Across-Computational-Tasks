"""Circle tracking for rollcan video.

Generates solution_circle_coordinates.csv and an overlay video.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

VIDEO_PATH = Path("rollcan.mp4")
CSV_OUTPUT = Path("solution_circle_coordinates.csv")
VIDEO_OUTPUT = Path("rollcan_overlay.mp4")

# Detection hyperparameters tuned for the provided footage.
MIN_RADIUS = 110
MAX_RADIUS = 180
ROI_SCALE = 1.6
SMOOTHING_ALPHA = 0.2


def _clip_circle(circle: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Clamp circle coordinates to frame bounds."""
    h, w = frame_shape
    x = float(np.clip(circle[0], 0, w - 1))
    y = float(np.clip(circle[1], 0, h - 1))
    r = float(np.clip(circle[2], MIN_RADIUS, MAX_RADIUS))
    return np.array([x, y, r], dtype=np.float32)


def _select_best_circle(circles: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
    """Pick the circle closest to the reference (if available)."""
    circles = circles[0]
    if reference is None:
        # When nothing to compare against, prioritise the largest radius to avoid false positives.
        idx = np.argmax(circles[:, 2])
        return circles[idx]
    distances = np.linalg.norm(circles[:, :2] - reference[:2], axis=1)
    idx = int(np.argmin(distances))
    return circles[idx]


def _hough_detect(image: np.ndarray, min_radius: int, max_radius: int, reference: Optional[np.ndarray]) -> Optional[np.ndarray]:
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=180,
        param1=140,
        param2=38,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    return _select_best_circle(circles, reference)


def detect_circle(gray_frame: np.ndarray, previous: Optional[np.ndarray]) -> Optional[np.ndarray]:
    blurred = cv2.GaussianBlur(gray_frame, (7, 7), 1.5)
    h, w = gray_frame.shape
    candidate: Optional[np.ndarray] = None

    if previous is not None:
        x, y, r = previous
        pad = int(r * ROI_SCALE)
        x1 = max(int(x - pad), 0)
        y1 = max(int(y - pad), 0)
        x2 = min(int(x + pad), w)
        y2 = min(int(y + pad), h)
        roi = blurred[y1:y2, x1:x2]
        if roi.size > 0:
            roi_min = int(max(r * 0.75, MIN_RADIUS))
            roi_max = int(min(r * 1.25, MAX_RADIUS))
            roi_circles = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(roi.shape) / 2,
                param1=140,
                param2=30,
                minRadius=roi_min,
                maxRadius=roi_max,
            )
            if roi_circles is not None:
                circle = _select_best_circle(roi_circles, previous - np.array([x1, y1, 0]))
                candidate = np.array([circle[0] + x1, circle[1] + y1, circle[2]])

    if candidate is None:
        candidate = _hough_detect(blurred, MIN_RADIUS, MAX_RADIUS, previous)

    if candidate is None:
        return None

    return _clip_circle(candidate, (h, w))


def rotate_cw_coordinates(point: np.ndarray, frame_shape: tuple[int, int]) -> tuple[float, float, float]:
    """Rotate coordinates 90 degrees  clockwise to match evaluation reference."""
    h, _ = frame_shape
    x, y, r = point
    rotated_x = float(h - 1 - y)
    rotated_y = float(x)
    return rotated_x, rotated_y, float(r)


def process_video() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open video stream")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rotated_size = (frame_h, frame_w)  # width, height after clockwise rotation
    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        rotated_size,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open video writer")

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame_Index", "X", "Y", "Radius"])

        prev_raw: Optional[np.ndarray] = None
        prev_smoothed: Optional[np.ndarray] = None
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = detect_circle(gray, prev_raw)

            if detected is None:
                if prev_raw is not None:
                    detected = prev_raw.copy()
                else:
                    frame_index += 1
                    writer.write(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
                    continue

            prev_raw = detected.copy()

            if prev_smoothed is None:
                smoothed = detected
            else:
                smoothed = SMOOTHING_ALPHA * detected + (1.0 - SMOOTHING_ALPHA) * prev_smoothed
            prev_smoothed = smoothed

            rot_x, rot_y, rot_r = rotate_cw_coordinates(smoothed, (frame_h, frame_w))
            csv_writer.writerow([
                frame_index,
                f"{rot_x:.3f}",
                f"{rot_y:.3f}",
                f"{rot_r:.3f}",
            ])

            display = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            center = (int(round(rot_x)), int(round(rot_y)))
            radius = int(round(rot_r))
            cv2.circle(display, center, radius, (0, 255, 0), 3)
            cv2.circle(display, center, 5, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"Frame {frame_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"x={rot_x:.1f}, y={rot_y:.1f}, r={rot_r:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(display)
            frame_index += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    process_video()
