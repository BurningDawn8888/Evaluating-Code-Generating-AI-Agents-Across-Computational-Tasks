import csv
import numpy as np
import math

def load_csv(filename):
    """Load CSV file and return as list of dictionaries"""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'frame_index': int(row['Frame_Index']),
                'left_x1': int(row['Left_Line_X1']),
                'left_y1': int(row['Left_Line_Y1']),
                'left_x2': int(row['Left_Line_X2']),
                'left_y2': int(row['Left_Line_Y2']),
                'right_x1': int(row['Right_Line_X1']),
                'right_y1': int(row['Right_Line_Y1']),
                'right_x2': int(row['Right_Line_X2']),
                'right_y2': int(row['Right_Line_Y2'])
            })
    return data

def filter_non_zero_annotations(data):
    """Remove all rows where all coordinates are 0"""
    filtered = []
    for row in data:
        if not all([
            row['left_x1'] == 0, row['left_y1'] == 0,
            row['left_x2'] == 0, row['left_y2'] == 0,
            row['right_x1'] == 0, row['right_y1'] == 0,
            row['right_x2'] == 0, row['right_y2'] == 0
        ]):
            filtered.append(row)
    return filtered

def calculate_endpoint_distance(x1, y1, x2, y2, x1_ref, y1_ref, x2_ref, y2_ref):
    """Calculate average Euclidean distance between corresponding endpoints"""
    dist1 = math.sqrt((x1 - x1_ref)**2 + (y1 - y1_ref)**2)
    dist2 = math.sqrt((x2 - x2_ref)**2 + (y2 - y2_ref)**2)
    return (dist1 + dist2) / 2

def calculate_angular_error(x1, y1, x2, y2, x1_ref, y1_ref, x2_ref, y2_ref):
    """Calculate angular difference between two lines in degrees"""
    # Calculate angles
    if x2 - x1 == 0:
        angle1 = 90.0
    else:
        angle1 = math.degrees(math.atan((y2 - y1) / (x2 - x1)))

    if x2_ref - x1_ref == 0:
        angle2 = 90.0
    else:
        angle2 = math.degrees(math.atan((y2_ref - y1_ref) / (x2_ref - x1_ref)))

    # Return absolute angular difference
    return abs(angle1 - angle2)

def calculate_line_similarity(x1, y1, x2, y2, x1_ref, y1_ref, x2_ref, y2_ref):
    """Calculate similarity score between two lines (0-100, 100 = identical)"""
    # Maximum reasonable distance in a 1280x720 frame
    max_distance = math.sqrt(1280**2 + 720**2)

    # Calculate endpoint distance error
    endpoint_dist = calculate_endpoint_distance(x1, y1, x2, y2, x1_ref, y1_ref, x2_ref, y2_ref)

    # Calculate angular error
    angular_error = calculate_angular_error(x1, y1, x2, y2, x1_ref, y1_ref, x2_ref, y2_ref)

    # Normalize scores (0-1, where 1 is perfect)
    distance_score = max(0, 1 - (endpoint_dist / max_distance))
    angular_score = max(0, 1 - (angular_error / 90))  # 90 degrees max reasonable error

    # Combined score (weighted: 70% distance, 30% angle)
    combined_score = (distance_score * 0.7) + (angular_score * 0.3)

    return combined_score * 100, endpoint_dist, angular_error

def find_closest_frame(frame_index, reference_data, tolerance=15):
    """Find the closest reference frame within tolerance (about 0.5 seconds at 29fps)"""
    closest = None
    min_diff = float('inf')

    for ref_row in reference_data:
        diff = abs(ref_row['frame_index'] - frame_index)
        if diff < min_diff and diff <= tolerance:
            min_diff = diff
            closest = ref_row

    return closest, min_diff

def compare_csv_files(solution_file, reference_file):
    """Compare solution CSV against reference (manual annotations)"""
    print("="*80)
    print("ROAD LINE TRACKING - ACCURACY ASSESSMENT")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    solution_data = load_csv(solution_file)
    reference_data = load_csv(reference_file)

    print(f"Solution data: {len(solution_data)} frames")
    print(f"Reference data (raw): {len(reference_data)} frames")

    # Filter out zero annotations from reference
    reference_data = filter_non_zero_annotations(reference_data)
    print(f"Reference data (non-zero): {len(reference_data)} frames")
    print()

    # Comparison metrics
    left_line_scores = []
    right_line_scores = []
    left_endpoint_errors = []
    right_endpoint_errors = []
    left_angular_errors = []
    right_angular_errors = []

    frames_compared = 0
    frames_with_detection = 0
    frames_without_detection = 0

    print("Comparing frames...")
    print()

    # For each reference annotation, find corresponding solution frame
    for ref_row in reference_data:
        ref_frame = ref_row['frame_index']

        # Find closest solution frame
        solution_row, frame_diff = find_closest_frame(ref_frame, solution_data)

        if solution_row is None:
            continue

        frames_compared += 1

        # Check if solution has detection (not all zeros)
        has_detection = not all([
            solution_row['left_x1'] == 0, solution_row['left_y1'] == 0,
            solution_row['left_x2'] == 0, solution_row['left_y2'] == 0,
            solution_row['right_x1'] == 0, solution_row['right_y1'] == 0,
            solution_row['right_x2'] == 0, solution_row['right_y2'] == 0
        ])

        if has_detection:
            frames_with_detection += 1

            # Compare left line
            left_score, left_dist, left_angle = calculate_line_similarity(
                solution_row['left_x1'], solution_row['left_y1'],
                solution_row['left_x2'], solution_row['left_y2'],
                ref_row['left_x1'], ref_row['left_y1'],
                ref_row['left_x2'], ref_row['left_y2']
            )
            left_line_scores.append(left_score)
            left_endpoint_errors.append(left_dist)
            left_angular_errors.append(left_angle)

            # Compare right line
            right_score, right_dist, right_angle = calculate_line_similarity(
                solution_row['right_x1'], solution_row['right_y1'],
                solution_row['right_x2'], solution_row['right_y2'],
                ref_row['right_x1'], ref_row['right_y1'],
                ref_row['right_x2'], ref_row['right_y2']
            )
            right_line_scores.append(right_score)
            right_endpoint_errors.append(right_dist)
            right_angular_errors.append(right_angle)

            # Print detailed comparison for first few and some sample frames
            if frames_compared <= 3 or frames_compared % 3 == 0:
                print(f"Frame {ref_frame} (matched with solution frame {solution_row['frame_index']}, diff={frame_diff}):")
                print(f"  Left Line  - Score: {left_score:.2f}%, Endpoint Error: {left_dist:.2f}px, Angular Error: {left_angle:.2f}°")
                print(f"  Right Line - Score: {right_score:.2f}%, Endpoint Error: {right_dist:.2f}px, Angular Error: {right_angle:.2f}°")
                print()
        else:
            frames_without_detection += 1

    # Calculate overall statistics
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()

    print(f"Frames Compared: {frames_compared}")
    print(f"  - With Detection: {frames_with_detection}")
    print(f"  - Without Detection: {frames_without_detection}")
    print()

    if len(left_line_scores) > 0:
        # Detection rate
        detection_rate = (frames_with_detection / frames_compared) * 100

        # Average scores
        avg_left_score = np.mean(left_line_scores)
        avg_right_score = np.mean(right_line_scores)
        overall_score = (avg_left_score + avg_right_score) / 2

        # Endpoint errors
        avg_left_endpoint = np.mean(left_endpoint_errors)
        avg_right_endpoint = np.mean(right_endpoint_errors)

        # Angular errors
        avg_left_angular = np.mean(left_angular_errors)
        avg_right_angular = np.mean(right_angular_errors)

        print("LEFT LINE METRICS:")
        print(f"  Average Accuracy Score: {avg_left_score:.2f}%")
        print(f"  Average Endpoint Error: {avg_left_endpoint:.2f} pixels")
        print(f"  Average Angular Error: {avg_left_angular:.2f} degrees")
        print(f"  Std Dev Endpoint Error: {np.std(left_endpoint_errors):.2f} pixels")
        print()

        print("RIGHT LINE METRICS:")
        print(f"  Average Accuracy Score: {avg_right_score:.2f}%")
        print(f"  Average Endpoint Error: {avg_right_endpoint:.2f} pixels")
        print(f"  Average Angular Error: {avg_right_angular:.2f} degrees")
        print(f"  Std Dev Endpoint Error: {np.std(right_endpoint_errors):.2f} pixels")
        print()

        print("="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        print()
        print(f"Detection Rate: {detection_rate:.2f}%")
        print(f"Overall Accuracy Score: {overall_score:.2f}%")
        print(f"Overall Margin of Error: {100 - overall_score:.2f}%")
        print()

        # Quality grading based on task.md requirements
        print("QUALITY GRADING:")
        print()

        # Line accuracy (endpoint error should be < 50 pixels for good quality)
        if avg_left_endpoint < 30 and avg_right_endpoint < 30:
            accuracy_grade = "EXCELLENT"
        elif avg_left_endpoint < 50 and avg_right_endpoint < 50:
            accuracy_grade = "GOOD"
        elif avg_left_endpoint < 100 and avg_right_endpoint < 100:
            accuracy_grade = "FAIR"
        else:
            accuracy_grade = "POOR"

        print(f"1. Line Accuracy to Road Edges: {accuracy_grade}")
        print(f"   (Avg endpoint error: {(avg_left_endpoint + avg_right_endpoint)/2:.2f} pixels)")
        print()

        # Detection completeness
        if detection_rate > 90:
            detection_grade = "EXCELLENT"
        elif detection_rate > 70:
            detection_grade = "GOOD"
        elif detection_rate > 50:
            detection_grade = "FAIR"
        else:
            detection_grade = "POOR"

        print(f"2. Road Edge Detection: {detection_grade}")
        print(f"   (Detection rate: {detection_rate:.2f}%)")
        print()

        # Angular consistency
        if avg_left_angular < 5 and avg_right_angular < 5:
            angle_grade = "EXCELLENT"
        elif avg_left_angular < 10 and avg_right_angular < 10:
            angle_grade = "GOOD"
        elif avg_left_angular < 15 and avg_right_angular < 15:
            angle_grade = "FAIR"
        else:
            angle_grade = "POOR"

        print(f"3. Line Angle Consistency: {angle_grade}")
        print(f"   (Avg angular error: {(avg_left_angular + avg_right_angular)/2:.2f}°)")
        print()

        # Overall grade
        grades = {'EXCELLENT': 4, 'GOOD': 3, 'FAIR': 2, 'POOR': 1}
        avg_grade_score = (grades[accuracy_grade] + grades[detection_grade] + grades[angle_grade]) / 3

        if avg_grade_score >= 3.5:
            overall_grade = "EXCELLENT"
        elif avg_grade_score >= 2.5:
            overall_grade = "GOOD"
        elif avg_grade_score >= 1.5:
            overall_grade = "FAIR"
        else:
            overall_grade = "POOR"

        print("="*80)
        print(f"OVERALL GRADE: {overall_grade}")
        print("="*80)
        print()

        # Save detailed results
        with open('comparison_results.txt', 'w') as f:
            f.write("ROAD LINE TRACKING - DETAILED COMPARISON RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Detection Rate: {detection_rate:.2f}%\n")
            f.write(f"Overall Accuracy Score: {overall_score:.2f}%\n")
            f.write(f"Overall Margin of Error: {100 - overall_score:.2f}%\n\n")
            f.write(f"Left Line Accuracy: {avg_left_score:.2f}%\n")
            f.write(f"Right Line Accuracy: {avg_right_score:.2f}%\n\n")
            f.write(f"Average Endpoint Error (Left): {avg_left_endpoint:.2f} pixels\n")
            f.write(f"Average Endpoint Error (Right): {avg_right_endpoint:.2f} pixels\n\n")
            f.write(f"Average Angular Error (Left): {avg_left_angular:.2f} degrees\n")
            f.write(f"Average Angular Error (Right): {avg_right_angular:.2f} degrees\n\n")
            f.write(f"Overall Grade: {overall_grade}\n")

        print("Detailed results saved to: comparison_results.txt")

    else:
        print("ERROR: No valid comparisons could be made!")
        print("This could mean:")
        print("  - Solution has no detections for any reference frames")
        print("  - Frame indices don't align between files")

if __name__ == "__main__":
    compare_csv_files('solution_coordinates.csv', 'manual_annotations_coordinates.csv')
