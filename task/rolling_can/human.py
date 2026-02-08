import cv2
import numpy as np
import csv


# Resizes the camera
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
   dim = None
   (h, w) = image.shape[:2]

   if width is None and height is None:
       return image, 1.0

   if width is None:
       r = height / float(h)
       dim = (int(w * r), height)
   else:
       r = width / float(w)
       dim = (width, int(h * r))

   resized_image = cv2.resize(image, dim, interpolation=inter)
   return resized_image, r


# Function to detect the can
def detect_cans_in_video():
   cap = cv2.VideoCapture("rollcan.mp4")
   
   # Open CSV file for writing
   with open('human_circle_coordinates.csv', 'w', newline='') as csvfile:
       csvwriter = csv.writer(csvfile)
       csvwriter.writerow(['Frame_Index', 'X', 'Y', 'Radius'])
       
       frame_index = 0
       
       while True:
           ret, frame = cap.read()
           # cap = turn_vid(cap)
           if not ret:
               break
           # rotate the frame 90 degrees
           frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
           # Call resize function to enhance circle detection accuracy
           resized_frame, scale_factor = resize_with_aspect_ratio(frame, width=300)

           # Gray and blur photos for better image processing
           gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
           blurred = cv2.GaussianBlur(gray, (9, 9), 2)


           # Detect circles with HoughCircles
           circles = cv2.HoughCircles(
               blurred,
               cv2.HOUGH_GRADIENT,
               dp=1.2,
               minDist=13,
               param1=60,
               param2=80,
               minRadius=0,
               maxRadius=0
           )

           overlay_frame = frame.copy()
           
           # Initialize coordinates as empty
           x_coord, y_coord, radius = '', '', ''

           if circles is not None:
               circles = np.uint16(np.around(circles[0, :]))

               for circle in circles:
                   x, y, r = circle
                   # Scale the size of the drawn circle back to the original
                   x_scaled = int(x * (1 / scale_factor))
                   y_scaled = int(y * (1 / scale_factor))
                   r_scaled = int(r * (1 / scale_factor))
                   
                   # Store coordinates for CSV
                   x_coord, y_coord, radius = x_scaled, y_scaled, r_scaled

                   # Draw the outer circle on the original frame
                   cv2.circle(overlay_frame, (x_scaled, y_scaled), r_scaled, (0, 255, 0), 3)
                   # Draw the center of the circle on the original frame
                   cv2.circle(overlay_frame, (x_scaled, y_scaled), 2, (0, 0, 255), 3)

           # Write frame data to CSV
           csvwriter.writerow([frame_index, x_coord, y_coord, radius])

           cv2.imshow("Original Video with Circle Detection", overlay_frame)

           # Exit on pressing 'q'
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
               
           frame_index += 1

   cap.release()
   cv2.destroyAllWindows()


# Start the program
detect_cans_in_video()
