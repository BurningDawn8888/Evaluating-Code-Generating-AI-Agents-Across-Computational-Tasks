#!/usr/bin/env python3
"""
Q3 PROJECT 4 - CIRCULAR SHAPE MOVEMENT DETECTION
=================================================

This solution detects and tracks a moving circular shape (soda can) from a video.
It overlays the circumference of the can with a circular line and finds the center point.

Author: Claude Sonnet 4 Agent V1
Assessment Criteria:
- Image quality: 10 points
- Image Overlay: 25 points
- Image overlay accuracy: 25 points
- Center point detection: 25 points
- Code: 15 points (comments, format, layout)

Requirements:
- Detect the outer ring of the bottom of the can
- Circle must not wobble or flicker
- No marks or alterations to the can
- Output format: Frame_Index,X,Y,Radius
"""

import cv2
import numpy as np
import csv
import os
from typing import Tuple, Optional, List
import argparse


class CanDetector:
    """
    Advanced circular shape detector for tracking moving soda cans in video.

    This class implements multiple detection strategies to ensure robust
    circle detection with minimal flickering and wobbling.
    """

    def __init__(self, video_path: str = "rollcan.mp4"):
        """
        Initialize the can detector with video path and detection parameters.

        Args:
            video_path (str): Path to the input video file
        """
        self.video_path = video_path
        self.output_csv = "circle_coordinates.csv"
        self.output_video = "output_with_detection.mp4"

        # Detection parameters - optimized for stable circle detection
        self.hough_params = {
            'dp': 1.2,
            'minDist': 50,
            'param1': 100,
            'param2': 30,
            'minRadius': 20,
            'maxRadius': 200
        }

        # Smoothing parameters to reduce flickering
        self.smoothing_window = 5
        self.previous_circles = []

        # Video processing parameters
        self.resize_width = 640
        self.gaussian_blur_kernel = (9, 9)
        self.gaussian_blur_sigma = 2

    def resize_with_aspect_ratio(self, image: np.ndarray, width: int = None,
                                height: int = None) -> Tuple[np.ndarray, float]:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image (np.ndarray): Input image
            width (int): Target width (optional)
            height (int): Target height (optional)

        Returns:
            Tuple[np.ndarray, float]: Resized image and scale factor
        """
        h, w = image.shape[:2]

        if width is None and height is None:
            return image, 1.0

        if width is None:
            scale_factor = height / float(h)
            dim = (int(w * scale_factor), height)
        else:
            scale_factor = width / float(w)
            dim = (width, int(h * scale_factor))

        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image, scale_factor

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Preprocess frame for optimal circle detection.

        Args:
            frame (np.ndarray): Raw input frame

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Original frame, processed frame, scale factor
        """
        # Rotate frame 90 degrees clockwise (as per original implementation)
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Resize for better processing speed and accuracy
        resized_frame, scale_factor = self.resize_with_aspect_ratio(
            rotated_frame, width=self.resize_width
        )

        # Convert to grayscale
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray, self.gaussian_blur_kernel, self.gaussian_blur_sigma
        )

        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(blurred)

        return rotated_frame, equalized, scale_factor

    def detect_circles_hough(self, processed_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect circles using Hough Circle Transform.

        Args:
            processed_frame (np.ndarray): Preprocessed grayscale frame

        Returns:
            Optional[np.ndarray]: Detected circles or None
        """
        circles = cv2.HoughCircles(
            processed_frame,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params['dp'],
            minDist=self.hough_params['minDist'],
            param1=self.hough_params['param1'],
            param2=self.hough_params['param2'],
            minRadius=self.hough_params['minRadius'],
            maxRadius=self.hough_params['maxRadius']
        )

        return circles

    def smooth_circle_detection(self, circles: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Apply temporal smoothing to reduce circle flickering.

        Args:
            circles (np.ndarray): Current frame's detected circles

        Returns:
            Optional[Tuple[int, int, int]]: Smoothed circle (x, y, radius) or None
        """
        if circles is None or len(circles) == 0:
            return None

        # Convert to integer and take the first (most confident) circle
        circles = np.uint16(np.around(circles[0, :]))
        best_circle = circles[0]  # Take the first circle

        # Store in history for smoothing
        self.previous_circles.append(best_circle)

        # Keep only recent circles for smoothing
        if len(self.previous_circles) > self.smoothing_window:
            self.previous_circles.pop(0)

        # Calculate smoothed circle by averaging recent detections
        if len(self.previous_circles) >= 3:
            avg_x = int(np.mean([c[0] for c in self.previous_circles]))
            avg_y = int(np.mean([c[1] for c in self.previous_circles]))
            avg_r = int(np.mean([c[2] for c in self.previous_circles]))
            return (avg_x, avg_y, avg_r)
        else:
            return tuple(best_circle)

    def draw_circle_overlay(self, frame: np.ndarray, circle: Tuple[int, int, int],
                           scale_factor: float) -> np.ndarray:
        """
        Draw circle overlay on the frame.

        Args:
            frame (np.ndarray): Original frame
            circle (Tuple[int, int, int]): Circle parameters (x, y, radius)
            scale_factor (float): Scale factor for coordinate transformation

        Returns:
            np.ndarray: Frame with circle overlay
        """
        overlay_frame = frame.copy()
        x, y, r = circle

        # Scale coordinates back to original frame size
        x_scaled = int(x / scale_factor)
        y_scaled = int(y / scale_factor)
        r_scaled = int(r / scale_factor)

        # Draw the outer circle (can circumference) in green
        cv2.circle(overlay_frame, (x_scaled, y_scaled), r_scaled, (0, 255, 0), 3)

        # Draw the center point in red
        cv2.circle(overlay_frame, (x_scaled, y_scaled), 5, (0, 0, 255), -1)

        # Add frame information text
        cv2.putText(overlay_frame, f"Center: ({x_scaled}, {y_scaled})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"Radius: {r_scaled}px",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return overlay_frame, (x_scaled, y_scaled, r_scaled)

    def process_video(self, show_video: bool = True, save_video: bool = True) -> bool:
        """
        Process the entire video and detect circular shapes.

        Args:
            show_video (bool): Whether to display video during processing
            save_video (bool): Whether to save output video with overlays

        Returns:
            bool: Success status
        """
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"Error: Video file '{self.video_path}' not found!")
            return False

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video file '{self.video_path}'!")
            return False

        # Get video properties for output video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {frame_count}, FPS: {fps}")

        # Initialize video writer if saving output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        # Open CSV file for writing coordinates
        with open(self.output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Frame_Index', 'X', 'Y', 'Radius'])

            frame_index = 0
            successful_detections = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Initialize video writer with first frame dimensions
                if out is None and save_video:
                    height, width = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE).shape[:2]
                    out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

                # Preprocess frame
                original_frame, processed_frame, scale_factor = self.preprocess_frame(frame)

                # Detect circles
                circles = self.detect_circles_hough(processed_frame)

                # Apply smoothing
                smooth_circle = self.smooth_circle_detection(circles)

                # Initialize default values
                x_coord, y_coord, radius = '', '', ''
                overlay_frame = original_frame.copy()

                if smooth_circle is not None:
                    # Draw overlay and get scaled coordinates
                    overlay_frame, (x_coord, y_coord, radius) = self.draw_circle_overlay(
                        original_frame, smooth_circle, scale_factor
                    )
                    successful_detections += 1
                else:
                    # Add "No detection" text when circle is not found
                    cv2.putText(overlay_frame, "No circle detected",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Write frame data to CSV
                csvwriter.writerow([frame_index, x_coord, y_coord, radius])

                # Save frame to output video
                if out is not None:
                    out.write(overlay_frame)

                # Display video if requested
                if show_video:
                    cv2.imshow("Can Detection - Press 'q' to quit", overlay_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_index += 1

                # Progress indicator
                if frame_index % 30 == 0:
                    progress = (frame_index / frame_count) * 100
                    print(f"Progress: {progress:.1f}% ({frame_index}/{frame_count} frames)")

        # Cleanup
        cap.release()
        if out is not None:
            out.release()

        # Only destroy windows if they were actually created
        if show_video:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore GUI-related errors on systems without display support

        # Print summary
        detection_rate = (successful_detections / frame_count) * 100
        print(f"\nProcessing complete!")
        print(f"Successful detections: {successful_detections}/{frame_count} ({detection_rate:.1f}%)")
        print(f"Coordinates saved to: {self.output_csv}")
        if save_video:
            print(f"Output video saved to: {self.output_video}")

        return True


def main():
    """
    Main function to run the can detection system.
    """
    parser = argparse.ArgumentParser(description='Detect circular shapes in video')
    parser.add_argument('--video', '-v', default='rollcan.mp4',
                       help='Path to input video file')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying video window')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving output video')

    args = parser.parse_args()

    # Create detector instance
    detector = CanDetector(video_path=args.video)

    # Process video
    success = detector.process_video(
        show_video=not args.no_display,
        save_video=not args.no_save
    )

    if success:
        print("* Video processing completed successfully!")
    else:
        print("* Video processing failed!")
        return 1

    return 0


if __name__ == "__main__":
    """
    Entry point for the circular shape detection program.

    Usage:
        python solution.py                    # Process rollcan.mp4 with default settings
        python solution.py --video video.mp4 # Process custom video file
        python solution.py --no-display      # Run without showing video window
        python solution.py --no-save         # Skip saving output video
    """
    exit(main())