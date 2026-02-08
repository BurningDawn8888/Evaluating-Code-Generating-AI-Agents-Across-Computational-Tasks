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
    ratio: float
    good_matches: int


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
        scaleFactor=1.03,
        minNeighbors=3,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [tuple(map(int, face)) for face in faces]


def expand_rect(rect: Tuple[int, int, int, int], image_shape: Tuple[int, int, int], scale: float = 1.25) -> Tuple[int, int, int, int]:
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


def create_sift():
    try:
        return cv2.SIFT_create()
    except AttributeError as exc:  # pragma: no cover - SIFT should exist once contrib installed.
        raise RuntimeError("SIFT feature extractor unavailable") from exc


def score_match(target_face: np.ndarray, candidate_face: np.ndarray, sift: cv2.SIFT, matcher: cv2.BFMatcher) -> Tuple[float, float, int]:
    kp_t, des_t = sift.detectAndCompute(target_face, None)
    kp_c, des_c = sift.detectAndCompute(candidate_face, None)
    if des_t is None or des_c is None or len(des_t) == 0 or len(des_c) == 0:
        return 0.0, 0.0, 0
    matches = matcher.knnMatch(des_t, des_c, k=2)
    good_matches = [m for m, n in matches if n is not None and m.distance < 0.72 * n.distance]
    ratio = len(good_matches) / (len(matches) + 1e-6)
    ncc = cv2.matchTemplate(target_face, candidate_face, cv2.TM_CCOEFF_NORMED)[0][0]
    combined = 0.7 * ratio + 0.3 * ((ncc + 1.0) / 2.0)
    return combined, ratio, len(good_matches)


def select_best_match(target_img: np.ndarray, crowd_img: np.ndarray, cascade: cv2.CascadeClassifier) -> Tuple[Optional[FaceMatch], List[Tuple[int, int, int, int]]]:
    target_faces = detect_faces(target_img, cascade)
    if target_faces:
        target_rect = max(target_faces, key=lambda rect: rect[2] * rect[3])
    else:
        h, w = target_img.shape[:2]
        size = min(h, w)
        offset_x = (w - size) // 2
        offset_y = (h - size) // 2
        target_rect = (offset_x, offset_y, size, size)
    target_face = preprocess_face(target_img, target_rect)

    crowd_faces = detect_faces(crowd_img, cascade)
    if not crowd_faces:
        return None, []

    sift = create_sift()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best: Optional[FaceMatch] = None
    for rect in crowd_faces:
        candidate_face = preprocess_face(crowd_img, rect)
        score, ratio, good = score_match(target_face, candidate_face, sift, matcher)
        if best is None or score > best.score:
            best = FaceMatch(rect=rect, score=score, ratio=ratio, good_matches=good)
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
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        if match:
            x, y, w, h = match.rect
            writer.writerow([
                image_name,
                True,
                x,
                y,
                w,
                h,
                round(match.score, 4),
            ])
        else:
            writer.writerow([image_name, False, "", "", "", "", 0.0])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    target_path = project_root / "target.png"
    crowd_path = project_root / "crowd.jpg"
    output_image_path = base_dir / "gemini_solution_2_output.png"
    output_csv_path = base_dir / "gemini_solution_2_results.csv"

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Unable to load cascade data from {cascade_path}")

    target_img = load_image(target_path)
    crowd_img = load_image(crowd_path)

    match, faces = select_best_match(target_img, crowd_img, cascade)

    if match:
        if match.ratio < 0.12 or match.good_matches < 15:
            match = None

    annotated = annotate_image(crowd_img, faces, match)
    cv2.imwrite(str(output_image_path), annotated)
    write_csv(output_csv_path, crowd_path.name, match)


if __name__ == "__main__":
    main()
