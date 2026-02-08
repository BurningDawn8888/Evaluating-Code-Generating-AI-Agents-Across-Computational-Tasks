
import face_recognition
import cv2
import csv
import numpy as np

# Load the images
target_image = face_recognition.load_image_file("target.png")
crowd_image = face_recognition.load_image_file("crowd.jpg")

# Find face locations and encodings
try:
    target_face_encoding = face_recognition.face_encodings(target_image)[0]
except IndexError:
    print("No face found in target.png")
    exit()

crowd_face_locations = face_recognition.face_locations(crowd_image)
crowd_face_encodings = face_recognition.face_encodings(crowd_image, crowd_face_locations)

# Default values for CSV
target_found = False
bounding_box = (0, 0, 0, 0)
best_match_index = -1

# Find the target in the crowd
if len(crowd_face_encodings) > 0:
    face_distances = face_recognition.face_distance(crowd_face_encodings, target_face_encoding)
    best_match_index = np.argmin(face_distances)
    
    # Set a tolerance for the match
    if face_distances[best_match_index] < 0.6:
        target_found = True
        top, right, bottom, left = crowd_face_locations[best_match_index]
        bounding_box = (left, top, right - left, bottom - top)

        # Draw a box around the face
        cv2.rectangle(crowd_image, (left, top), (right, bottom), (0, 0, 255), 3) # Thicker box
        # Draw a label with a name below the face
        cv2.rectangle(crowd_image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(crowd_image, "Target", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


# Save the output image
cv2.imwrite("gemini_solution_3.png", cv2.cvtColor(crowd_image, cv2.COLOR_RGB2BGR))

# Write to CSV
with open("gemini_solution_3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_Name", "Target_Found", "Bounding_Box_X", "Bounding_Box_Y", "Bounding_Box_Width", "Bounding_Box_Height"])
    writer.writerow(["crowd.jpg", target_found, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]])

print("Processing complete. Output saved to gemini_solution_3.png and gemini_solution_3.csv")
