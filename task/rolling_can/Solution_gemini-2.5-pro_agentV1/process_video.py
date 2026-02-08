
import cv2
import numpy as np
import csv

# Input and output file paths
video_path = 'rollcan.mp4'
output_video_path = 'output.mp4'
csv_path = 'output.csv'

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Open CSV file for writing
with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame_Index', 'X', 'Y', 'Radius'])

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=70,
            minRadius=80,
            maxRadius=150
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw the circle and center
                # cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                # cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                
                # Write data to CSV
                csv_writer.writerow([frame_index, x, y, r])

        # Write the frame to the output video
        out.write(frame)

        frame_index += 1

# Release everything when job is finished
cap.release()
out.release()


print(f"Processing complete. Output video saved to {output_video_path} and data to {csv_path}")
