import cv2
import numpy as np
import csv
from pathlib import Path

def detect_and_match_faces(target_path, crowd_path, output_image_path, output_csv_path):
    """
    Detect faces in crowd image and match with target face using facial recognition.

    Args:
        target_path: Path to target face image
        crowd_path: Path to crowd image
        output_image_path: Path to save output image with bounding boxes
        output_csv_path: Path to save CSV results
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

    # Get the largest face in target (assuming it's the main face)
    target_face = max(target_faces, key=lambda rect: rect[2] * rect[3])
    tx, ty, tw, th = target_face
    target_face_roi = target_gray[ty:ty+th, tx:tx+tw]

    # Detect faces in crowd image
    crowd_faces = face_cascade.detectMultiScale(
        crowd_gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(20, 20)
    )

    print(f"Detected {len(crowd_faces)} faces in crowd image")

    # Match target face with crowd faces using template matching and histogram comparison
    best_match = None
    best_score = float('inf')

    # Resize target face for comparison
    target_face_resized = cv2.resize(target_face_roi, (100, 100))

    for i, (x, y, w, h) in enumerate(crowd_faces):
        # Extract face ROI from crowd
        face_roi = crowd_gray[y:y+h, x:x+w]

        # Resize to same size as target for comparison
        face_roi_resized = cv2.resize(face_roi, (100, 100))

        # Method 1: Template matching using correlation coefficient
        result = cv2.matchTemplate(face_roi_resized, target_face_resized, cv2.TM_SQDIFF_NORMED)
        template_score = result[0][0]

        # Method 2: Histogram comparison
        target_hist = cv2.calcHist([target_face_resized], [0], None, [256], [0, 256])
        face_hist = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])

        cv2.normalize(target_hist, target_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(face_hist, face_hist, 0, 1, cv2.NORM_MINMAX)

        hist_score = cv2.compareHist(target_hist, face_hist, cv2.HISTCMP_CHISQR)

        # Combine scores (lower is better for both)
        combined_score = template_score * 0.6 + hist_score * 0.0001

        print(f"Face {i}: Position ({x}, {y}), Size ({w}x{h}), Score: {combined_score:.4f}")

        if combined_score < best_score:
            best_score = combined_score
            best_match = (x, y, w, h)

    # Create output image with bounding box on best match
    output_img = crowd_img.copy()

    target_found = False
    bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, 0, 0

    if best_match is not None:
        x, y, w, h = best_match
        # Draw bounding box around the matched face (green for target)
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(output_img, 'TARGET', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        target_found = True
        bbox_x, bbox_y, bbox_w, bbox_h = x, y, w, h

        print(f"\nBest match found at: ({x}, {y}) with size ({w}x{h})")
        print(f"Match score: {best_score:.4f}")

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

if __name__ == "__main__":
    # File paths
    target_path = "target.png"
    crowd_path = "crowd.jpg"
    output_image_path = "solution_output.jpg"
    output_csv_path = "solution_results.csv"

    # Run facial recognition
    print("Starting facial recognition...")
    print("=" * 60)

    detect_and_match_faces(target_path, crowd_path, output_image_path, output_csv_path)

    print("=" * 60)
    print("Facial recognition complete!")
