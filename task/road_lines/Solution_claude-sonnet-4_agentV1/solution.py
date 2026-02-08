import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque
from PIL import Image
import csv

class RoadLineTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize line smoothing with moving average
        self.left_line_history = deque(maxlen=10)
        self.right_line_history = deque(maxlen=10)

    def define_roi(self, frame):
        """Define region of interest (ROI) to focus on road area"""
        height, width = frame.shape[:2]

        # Define ROI trapezoid (middle of screen, avoiding car hood)
        roi_vertices = np.array([[
            (width * 0.1, height),  # Bottom left
            (width * 0.4, height * 0.6),  # Top left
            (width * 0.6, height * 0.6),  # Top right
            (width * 0.9, height)  # Bottom right
        ]], dtype=np.int32)

        return roi_vertices

    def apply_roi_mask(self, image, vertices):
        """Apply ROI mask to image"""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def detect_edges(self, frame):
        """Detect edges in frame using Canny edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        return edges

    def detect_lines(self, edges):
        """Detect lines using Hough transform"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )
        return lines

    def separate_lines(self, lines, frame_shape):
        """Separate lines into left and right based on slope"""
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        height, width = frame_shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter out horizontal lines and separate by slope
            if abs(slope) < 0.5:  # Too horizontal
                continue

            if slope < 0:  # Negative slope = left line
                left_lines.append(line[0])
            else:  # Positive slope = right line
                right_lines.append(line[0])

        return left_lines, right_lines

    def average_lines(self, lines, frame_shape):
        """Average multiple lines into a single line"""
        if len(lines) == 0:
            return None

        height, width = frame_shape[:2]

        # Calculate average slope and intercept
        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        # Fit a line using polyfit
        if len(x_coords) > 0:
            poly = np.polyfit(y_coords, x_coords, 1)

            # Define y coordinates (from 60% height to bottom)
            y1 = height
            y2 = int(height * 0.6)

            # Calculate x coordinates
            x1 = int(poly[0] * y1 + poly[1])
            x2 = int(poly[0] * y2 + poly[1])

            return [x1, y1, x2, y2]

        return None

    def smooth_line(self, line, history):
        """Smooth line using moving average"""
        if line is None:
            if len(history) > 0:
                return history[-1]
            return None

        history.append(line)

        if len(history) == 0:
            return line

        # Calculate average of recent lines
        avg_line = np.mean(history, axis=0).astype(int)
        return avg_line.tolist()

    def process_frame(self, frame):
        """Process a single frame and detect road lines"""
        # Define ROI
        roi_vertices = self.define_roi(frame)

        # Detect edges
        edges = self.detect_edges(frame)

        # Apply ROI mask
        masked_edges = self.apply_roi_mask(edges, roi_vertices)

        # Detect lines
        lines = self.detect_lines(masked_edges)

        # Separate left and right lines
        left_lines, right_lines = self.separate_lines(lines, frame.shape)

        # Average lines
        left_line = self.average_lines(left_lines, frame.shape)
        right_line = self.average_lines(right_lines, frame.shape)

        # Smooth lines
        left_line = self.smooth_line(left_line, self.left_line_history)
        right_line = self.smooth_line(right_line, self.right_line_history)

        return left_line, right_line

    def draw_lines(self, frame, left_line, right_line):
        """Draw lines on frame"""
        line_image = np.zeros_like(frame)

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        # Combine with original frame
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        return result

    def process_video(self, output_video_path, output_csv_path):
        """Process entire video and save results"""
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))

        # Initialize CSV file
        csv_data = []

        frame_index = 0

        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.frame_count}")
        print(f"FPS: {self.fps}")
        print(f"Resolution: {self.width}x{self.height}")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Process frame
            left_line, right_line = self.process_frame(frame)

            # Draw lines on frame
            result_frame = self.draw_lines(frame, left_line, right_line)

            # Write frame to output video
            out.write(result_frame)

            # Save line coordinates to CSV
            if left_line is not None and right_line is not None:
                csv_data.append([
                    frame_index,
                    left_line[0], left_line[1], left_line[2], left_line[3],
                    right_line[0], right_line[1], right_line[2], right_line[3]
                ])
            else:
                # Use previous values or zeros if no lines detected
                if len(csv_data) > 0:
                    prev_row = csv_data[-1]
                    csv_data.append([frame_index] + prev_row[1:])
                else:
                    csv_data.append([frame_index, 0, 0, 0, 0, 0, 0, 0, 0])

            frame_index += 1

            if frame_index % 30 == 0:
                print(f"Processed {frame_index}/{self.frame_count} frames")

        # Release video objects
        self.cap.release()
        out.release()

        # Write CSV file
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame_Index', 'Left_Line_X1', 'Left_Line_Y1', 'Left_Line_X2', 'Left_Line_Y2',
                           'Right_Line_X1', 'Right_Line_Y1', 'Right_Line_X2', 'Right_Line_Y2'])
            writer.writerows(csv_data)

        print(f"\nProcessing complete!")
        print(f"Output video: {output_video_path}")
        print(f"Output CSV: {output_csv_path}")

def main():
    # Input video
    video_path = "Road.mp4"

    # Output files
    output_video = "output.mp4"
    output_csv = "solution_coordinates.csv"

    # Create tracker and process video
    tracker = RoadLineTracker(video_path)
    tracker.process_video(output_video, output_csv)

if __name__ == "__main__":
    main()
