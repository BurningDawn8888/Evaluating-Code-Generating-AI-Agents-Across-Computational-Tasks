
import face_recognition
import cv2
import csv

# Load the images
target_image = face_recognition.load_image_file("target.png")
crowd_image = face_recognition.load_image_file("crowd.jpg")

# Find face locations and encodings
try:
    target_face_encoding = face_recognition.face_encodings(target_image)[0]
except IndexError:
    print("No face found in target.png")
    exit()

crowd_face_locations = face_recognition.face_locations(crowd_image, number_of_times_to_upsample=5)
crowd_face_encodings = face_recognition.face_encodings(crowd_image, crowd_face_locations)

# Default values for CSV
target_found = False
bounding_box = (0, 0, 0, 0)

# Find the target in the crowd
for i, face_encoding in enumerate(crowd_face_encodings):
    matches = face_recognition.compare_faces([target_face_encoding], face_encoding)

    if True in matches:
        target_found = True
        top, right, bottom, left = crowd_face_locations[i]
        bounding_box = (left, top, right - left, bottom - top)

        # Draw a box around the face
        cv2.rectangle(crowd_image, (left, top), (right, bottom), (0, 0, 255), 2)

# Save the output image
cv2.imwrite("gemini_solution_1.png", cv2.cvtColor(crowd_image, cv2.COLOR_RGB2BGR))

# Write to CSV
with open("gemini_solution_1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_Name", "Target_Found", "Bounding_Box_X", "Bounding_Box_Y", "Bounding_Box_Width", "Bounding_Box_Height"])
    writer.writerow(["crowd.jpg", target_found, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]])

print("Processing complete. Output saved to gemini_solution_1.png and gemini_solution_1.csv")
