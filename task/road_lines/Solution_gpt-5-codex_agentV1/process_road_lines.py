import csv
import math
from collections import deque
from pathlib import Path as SysPath

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


class LaneHistory:
    """Keeps a short history of detected lane lines for smoothing."""

    def __init__(self, maxlen: int = 12) -> None:
        self.left = deque(maxlen=maxlen)
        self.right = deque(maxlen=maxlen)

    def _average_line(self, lines: deque):
        if not lines:
            return None
        slopes = [line[0] for line in lines]
        intercepts = [line[1] for line in lines]
        return (float(np.mean(slopes)), float(np.mean(intercepts)))

    def update(self, left_line, right_line):
        if left_line is not None:
            self.left.append(left_line)
        if right_line is not None:
            self.right.append(right_line)

        averaged_left = self._average_line(self.left)
        averaged_right = self._average_line(self.right)
        return averaged_left, averaged_right


def prepare_roi_mask(frame_shape):
    height, width = frame_shape[:2]
    polygon = np.array([
        (int(0.05 * width), height),
        (int(0.40 * width), int(0.60 * height)),
        (int(0.60 * width), int(0.60 * height)),
        (int(0.95 * width), height),
    ])

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def detect_edges(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Skeletonize to keep thin ridge representation of edges.
    binary = masked_edges > 0
    skeleton = skeletonize(binary)
    skeleton_edges = (skeleton.astype(np.uint8)) * 255

    # Gentle dilation to connect broken segments post-skeletonization.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected = cv2.dilate(skeleton_edges, kernel, iterations=1)
    return connected


def compute_lane_params(lines, height):
    if lines is None:
        return None, None

    left_params = []
    right_params = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4:
            continue
        intercept = y1 - slope * x1
        if slope < 0:
            left_params.append((slope, intercept))
        else:
            right_params.append((slope, intercept))

    left_line = np.mean(left_params, axis=0) if left_params else None
    right_line = np.mean(right_params, axis=0) if right_params else None

    left_line = tuple(left_line) if left_line is not None else None
    right_line = tuple(right_line) if right_line is not None else None
    return left_line, right_line


def make_line_points(line_params, height, y_ratio=0.60):
    if line_params is None:
        return None

    slope, intercept = line_params
    y1 = height
    y2 = int(height * y_ratio)

    if math.isclose(slope, 0):
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)


def draw_lane_lines(frame, left_line, right_line):
    overlay = frame.copy()
    color = (255, 200, 0)
    thickness = 8
    if left_line is not None:
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
    if right_line is not None:
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)

    combined = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    # Use PIL conversion to ensure consistent color handling (leverages provided hint imports).
    pil_image = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def process_video(input_path: SysPath, output_video_path: SysPath, output_csv_path: SysPath):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    mask = prepare_roi_mask((height, width))
    history = LaneHistory()

    with output_csv_path.open("w", newline="") as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([
            "Frame_Index",
            "Left_Line_X1",
            "Left_Line_Y1",
            "Left_Line_X2",
            "Left_Line_Y2",
            "Right_Line_X1",
            "Right_Line_Y1",
            "Right_Line_X2",
            "Right_Line_Y2",
        ])

        frame_index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            edges = detect_edges(frame, mask)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=50, maxLineGap=150)
            left_params, right_params = compute_lane_params(lines, height)
            smoothed_left, smoothed_right = history.update(left_params, right_params)

            left_points = make_line_points(smoothed_left, height)
            right_points = make_line_points(smoothed_right, height)

            annotated = draw_lane_lines(frame, left_points, right_points)
            writer.write(annotated)

            csv_row = [frame_index]
            if left_points is None:
                csv_row.extend([-1, -1, -1, -1])
            else:
                csv_row.extend(left_points)
            if right_points is None:
                csv_row.extend([-1, -1, -1, -1])
            else:
                csv_row.extend(right_points)
            writer_csv.writerow(csv_row)

            frame_index += 1

    capture.release()
    writer.release()


if __name__ == "__main__":
    base_dir = SysPath(__file__).resolve().parent
    input_video = base_dir / "Road.mp4"
    output_video = base_dir / "gemini_solution_1.mp4"
    output_csv = base_dir / "gemini_solution_1.csv"
    process_video(input_video, output_video, output_csv)
    print(f"Processing complete. Video saved to {output_video}")
    print(f"CSV saved to {output_csv}")
