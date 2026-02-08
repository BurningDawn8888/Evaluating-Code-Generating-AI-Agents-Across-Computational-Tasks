import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

@dataclass
class FaceMatch:
    rect: Tuple[int, int, int, int]
    score: float


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def detect_faces(image: np.ndarray, cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [tuple(map(int, face)) for face in faces]


def expand_rect(rect: Tuple[int, int, int, int], image_shape: Tuple[int, int, int], scale: float = 1.15) -> Tuple[int, int, int, int]:
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w, new_h = w * scale, h * scale
    new_x = int(max(0, cx - new_w / 2.0))
    new_y = int(max(0, cy - new_h / 2.0))
    new_w = int(min(image_shape[1] - new_x, new_w))
    new_h = int(min(image_shape[0] - new_y, new_h))
    return new_x, new_y, new_w, new_h


def preprocess_face(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = expand_rect(rect, image.shape, scale=1.2)
    face = image[y : y + h, x : x + w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (140, 140), interpolation=cv2.INTER_CUBIC)
    return face


def score_match(target_face: np.ndarray, candidate_face: np.ndarray) -> float:
    result = cv2.matchTemplate(target_face, candidate_face, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def select_best_match(target_img: np.ndarray, crowd_img: np.ndarray, cascade: cv2.CascadeClassifier) -> Tuple[Optional[FaceMatch], List[Tuple[int, int, int, int]]]:
    target_faces = detect_faces(target_img, cascade)
    if not target_faces:
        raise RuntimeError("No face detected in target image")
    target_rect = max(target_faces, key=lambda rect: rect[2] * rect[3])
    target_face = preprocess_face(target_img, target_rect)

    crowd_faces = detect_faces(crowd_img, cascade)
    best: Optional[FaceMatch] = None
    for rect in crowd_faces:
        candidate_face = preprocess_face(crowd_img, rect)
        score = score_match(target_face, candidate_face)
        if best is None or score > best.score:
            best = FaceMatch(rect=rect, score=score)
    return best, crowd_faces


def annotate_image(image: np.ndarray, faces: List[Tuple[int, int, int, int]], target_match: Optional[FaceMatch]) -> np.ndarray:
    annotated = image.copy()
    for rect in faces:
        if target_match and rect == target_match.rect:
            continue
        x, y, w, h = rect
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (160, 160, 160), 1)
    if target_match:
        x, y, w, h = target_match.rect
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "Target",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def write_csv(output_path: Path, image_name: str, match: Optional[FaceMatch]) -> None:
    headers = [
        "Image_Name",
        "Target_Found",
        "Bounding_Box_X",
        "Bounding_Box_Y",
        "Bounding_Box_Width",
        "Bounding_Box_Height",
        "Confidence",
    ]
    if match:
        x, y, w, h = match.rect
        row = [image_name, True, x, y, w, h, round(match.score, 4)]
    else:
        row = [image_name, False, "", "", "", "", 0.0]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(row)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    target_path = project_root / "target.png"
    crowd_path = project_root / "crowd.jpg"
    output_image_path = base_dir / "gemini_solution_1_output.png"
    output_csv_path = base_dir / "gemini_solution_1_results.csv"

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Unable to load cascade data from {cascade_path}")

    target_img = load_image(target_path)
    crowd_img = load_image(crowd_path)

    match, faces = select_best_match(target_img, crowd_img, cascade)
    confidence_threshold = 0.25
    if match and match.score < confidence_threshold:
        match = None

    annotated = annotate_image(crowd_img, faces, match)
    cv2.imwrite(str(output_image_path), annotated)
    write_csv(output_csv_path, crowd_path.name, match)


if __name__ == "__main__":
    main()
