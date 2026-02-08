
import cv2
import numpy as np
import csv

# Input and output file paths
video_path = 'rollcan.mp4'
output_video_path = 'gemini_solution_1.mp4'
csv_path = 'gemini_solution_1.csv'

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
    # Store the last known circle
    last_circle = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=80,
            param2=60,
            minRadius=80,
            maxRadius=150
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Assume the first detected circle is the one we want
            x, y, r = circles[0]
            last_circle = (x, y, r)
            # Draw the circle and center
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            
            # Write data to CSV
            csv_writer.writerow([frame_index, x, y, r])
        else:
            # If no circle is detected, use the last known circle if available
            if last_circle is not None:
                x, y, r = last_circle
                # Draw the circle and center
                cv2.circle(frame, (x, y), r, (0, 0, 255), 4) # Draw in red to indicate it's a tracked circle
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                csv_writer.writerow([frame_index, x, y, r])
            else:
                # If no circle has been detected yet, write N/A
                csv_writer.writerow([frame_index, 'N/A', 'N/A', 'N/A'])


        # Write the frame to the output video
        out.write(frame)

        frame_index += 1

# Release everything when job is finished
cap.release()
out.release()

print(f"Processing complete. Output video saved to {output_video_path} and data to {csv_path}")
