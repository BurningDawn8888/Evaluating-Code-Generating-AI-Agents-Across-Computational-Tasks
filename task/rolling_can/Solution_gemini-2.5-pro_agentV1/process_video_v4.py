
import cv2
import numpy as np
import csv

# Input and output file paths
video_path = 'rollcan.mp4'
output_video_path = 'gemini_solution_3.mp4'
csv_path = 'gemini_solution_3.csv'

# Kalman Filter setup
class KalmanFilter:
    def __init__(self, dt=1, u=0, std_acc=1, std_meas=1):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.std_meas = std_meas

        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[self.dt**2 / 2], [self.dt]])
        self.H = np.array([[1, 0]]) # Corrected H matrix

        self.Q = np.array([[(self.dt**4) / 4, (self.dt**3) / 2], [(self.dt**3) / 2, self.dt**2]]) * self.std_acc**2
        self.R = self.std_meas**2
        self.P = np.eye(self.A.shape[1])
        self.x = np.array([[0], [0]])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

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

# Initialize Kalman Filters for x, y, and radius
kf_x = KalmanFilter(dt=1, std_acc=1, std_meas=10)
kf_y = KalmanFilter(dt=1, std_acc=1, std_meas=10)
kf_r = KalmanFilter(dt=1, std_acc=1, std_meas=10)

# Open CSV file for writing
with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame_Index', 'X', 'Y', 'Radius'])

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Predict the next state
        x_pred = kf_x.predict()[0,0]
        y_pred = kf_y.predict()[0,0]
        r_pred = kf_r.predict()[0,0]

        # Detect circles
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
            x, y, r = circles[0]
            
            # Update Kalman filters with measurement
            kf_x.update(x)
            kf_y.update(y)
            kf_r.update(r)

            # Get the smoothed state
            x_smooth = int(kf_x.x[0,0])
            y_smooth = int(kf_y.x[0,0])
            r_smooth = int(kf_r.x[0,0])

            # Draw the circle and center
            cv2.circle(frame, (x_smooth, y_smooth), r_smooth, (0, 255, 0), 4)
            cv2.rectangle(frame, (x_smooth - 5, y_smooth - 5), (x_smooth + 5, y_smooth + 5), (0, 128, 255), -1)
            csv_writer.writerow([frame_index, x_smooth, y_smooth, r_smooth])
        else:
            # Use the predicted state if no circle is detected
            x_smooth = int(x_pred)
            y_smooth = int(y_pred)
            r_smooth = int(r_pred)
            
            # Only draw and write if the predicted values are reasonable
            if r_smooth > 0:
                cv2.circle(frame, (x_smooth, y_smooth), r_smooth, (0, 0, 255), 4)
                cv2.rectangle(frame, (x_smooth - 5, y_smooth - 5), (x_smooth + 5, y_smooth + 5), (0, 128, 255), -1)
                csv_writer.writerow([frame_index, x_smooth, y_smooth, r_smooth])
            else:
                 csv_writer.writerow([frame_index, 'N/A', 'N/A', 'N/A'])

        # Write the frame to the output video
        out.write(frame)

        frame_index += 1

# Release everything when job is finished
cap.release()
out.release()

print(f"Processing complete. Output video saved to {output_video_path} and data to {csv_path}")
