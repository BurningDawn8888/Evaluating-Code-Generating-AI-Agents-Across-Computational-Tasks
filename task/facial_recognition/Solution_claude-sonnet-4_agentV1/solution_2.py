import cv2
import numpy as np
import csv
from pathlib import Path

def detect_and_match_faces(target_path, crowd_path, output_image_path, output_csv_path):
    """
    Detect faces in crowd image and match with target face using advanced facial recognition.
    Uses multiple detection passes and feature-based matching.
    """

    # Load images
    target_img = cv2.imread(target_path)
    crowd_img = cv2.imread(crowd_path)

    if target_img is None or crowd_img is None:
        print("Error: Could not load images")
        return

    # Convert to grayscale for face detection
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    crowd_gray = cv2.cvtColor(crowd_img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect face in target image
    target_faces = face_cascade.detectMultiScale(target_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(target_faces) == 0:
        print("Error: No face detected in target image")
        return

    # Get the largest face in target
    target_face = max(target_faces, key=lambda rect: rect[2] * rect[3])
    tx, ty, tw, th = target_face
    target_face_roi = target_gray[ty:ty+th, tx:tx+tw]
    target_face_color = target_img[ty:ty+th, tx:tx+tw]

    print(f"Target face size: {tw}x{th}")

    # Detect faces in crowd image with multiple scale factors for better detection
    all_faces = []

    # Multiple detection passes with different parameters
    for scale in [1.03, 1.05, 1.08]:
        for neighbors in [3, 4, 5]:
            faces = face_cascade.detectMultiScale(
                crowd_gray,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(40, 40),
                maxSize=(120, 120)
            )
            for face in faces:
                all_faces.append(face)

    # Remove duplicate detections using Non-Maximum Suppression
    if len(all_faces) > 0:
        crowd_faces = non_max_suppression(np.array(all_faces), 0.3)
    else:
        crowd_faces = []

    print(f"Detected {len(crowd_faces)} unique faces in crowd image after NMS")

    # Match target face with crowd faces
    best_match = None
    best_score = float('inf')

    # Prepare target for matching
    target_face_resized = cv2.resize(target_face_roi, (100, 100))
    target_face_eq = cv2.equalizeHist(target_face_resized)

    # Extract features from target
    orb = cv2.ORB_create()
    kp_target, des_target = orb.detectAndCompute(target_face_eq, None)

    for i, (x, y, w, h) in enumerate(crowd_faces):
        # Extract face ROI from crowd
        face_roi = crowd_gray[y:y+h, x:x+w]
        face_roi_color = crowd_img[y:y+h, x:x+w]

        # Resize to same size as target
        face_roi_resized = cv2.resize(face_roi, (100, 100))
        face_roi_eq = cv2.equalizeHist(face_roi_resized)

        # Method 1: Template matching with normalized correlation
        result = cv2.matchTemplate(face_roi_eq, target_face_eq, cv2.TM_SQDIFF_NORMED)
        template_score = result[0][0]

        # Method 2: Histogram comparison (multiple color spaces)
        # Grayscale histogram
        target_hist = cv2.calcHist([target_face_resized], [0], None, [256], [0, 256])
        face_hist = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])
        cv2.normalize(target_hist, target_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(face_hist, face_hist, 0, 1, cv2.NORM_MINMAX)
        hist_score = cv2.compareHist(target_hist, face_hist, cv2.HISTCMP_CHISQR)

        # Color histogram comparison (HSV)
        target_hsv = cv2.cvtColor(cv2.resize(target_face_color, (100, 100)), cv2.COLOR_BGR2HSV)
        face_hsv = cv2.cvtColor(cv2.resize(face_roi_color, (100, 100)), cv2.COLOR_BGR2HSV)

        target_hist_h = cv2.calcHist([target_hsv], [0], None, [180], [0, 180])
        face_hist_h = cv2.calcHist([face_hsv], [0], None, [180], [0, 180])
        cv2.normalize(target_hist_h, target_hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(face_hist_h, face_hist_h, 0, 1, cv2.NORM_MINMAX)
        hist_score_color = cv2.compareHist(target_hist_h, face_hist_h, cv2.HISTCMP_CHISQR)

        # Method 3: Feature matching using ORB
        kp_face, des_face = orb.detectAndCompute(face_roi_eq, None)
        feature_score = 1.0  # default high score if no match

        if des_target is not None and des_face is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            try:
                matches = bf.match(des_target, des_face)
                if len(matches) > 0:
                    # Lower distance is better
                    feature_score = sum([m.distance for m in matches]) / (len(matches) + 1)
                    feature_score = feature_score / 100.0  # normalize
            except:
                pass

        # Size similarity (faces should be roughly similar size)
        size_diff = abs(w - tw) / tw + abs(h - th) / th

        # Combine scores (weighted average, lower is better)
        combined_score = (
            template_score * 0.35 +
            hist_score * 0.0002 +
            hist_score_color * 0.0002 +
            feature_score * 0.25 +
            size_diff * 0.05
        )

        print(f"Face {i}: Pos ({x:3d}, {y:3d}), Size ({w:2d}x{h:2d}), "
              f"Template: {template_score:.4f}, Hist: {hist_score:.2f}, "
              f"Feature: {feature_score:.4f}, Combined: {combined_score:.4f}")

        if combined_score < best_score:
            best_score = combined_score
            best_match = (x, y, w, h)

    # Create output image with bounding box on best match
    output_img = crowd_img.copy()

    target_found = False
    bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, 0, 0

    if best_match is not None:
        x, y, w, h = best_match
        # Draw bounding box around the matched face
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(output_img, 'TARGET', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        target_found = True
        bbox_x, bbox_y, bbox_w, bbox_h = x, y, w, h

        print(f"\n{'='*60}")
        print(f"Best match found at: ({x}, {y}) with size ({w}x{h})")
        print(f"Match score: {best_score:.4f}")
        print(f"{'='*60}")

    # Save output image
    cv2.imwrite(output_image_path, output_img)
    print(f"\nOutput image saved to: {output_image_path}")

    # Save CSV results
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image_Name', 'Target_Found', 'Bounding_Box_X', 'Bounding_Box_Y',
                        'Bounding_Box_Width', 'Bounding_Box_Height'])
        writer.writerow(['crowd.jpg', target_found, bbox_x, bbox_y, bbox_w, bbox_h])

    print(f"CSV results saved to: {output_csv_path}")

    return output_img, target_found, (bbox_x, bbox_y, bbox_w, bbox_h)

def non_max_suppression(boxes, overlap_thresh):
    """
    Apply non-maximum suppression to remove overlapping bounding boxes.
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")

if __name__ == "__main__":
    # File paths
    target_path = "target.png"
    crowd_path = "crowd.jpg"
    output_image_path = "solution_2_output.jpg"
    output_csv_path = "solution_2_results.csv"

    # Run facial recognition
    print("Starting improved facial recognition...")
    print("=" * 60)

    detect_and_match_faces(target_path, crowd_path, output_image_path, output_csv_path)

    print("=" * 60)
    print("Facial recognition complete!")
