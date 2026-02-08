[task.md](https://github.com/user-attachments/files/25170794/task.md)

### **PROJECT – FACIAL RECOGNITION**

**Task:**
Create a program capable of performing facial recognition in an entire picture given the picture of the target's face. The system should detect and identify the target person's face within a larger image containing multiple faces or a crowd.

**Rules:**

* Input: A target face image (target.jpg) and a larger image (crowd.jpg) to search within.
* The program must accurately identify the target person's face in the larger image.
* Handle various lighting conditions, angles, and facial expressions.
* Process images with multiple faces and correctly identify the target.
* No modifications to the original image content.
* There should only be one identification box you generated and it should be on the target's face

**Output:**

1. CSV: Image_Name,Target_Found,Bounding_Box_X,Bounding_Box_Y,Bounding_Box_Width,Bounding_Box_Height
2. Output image with bounding boxes around detected faces, highlighting the target face.

**Assessment:**

* Facial recognition accuracy
* Performance with multiple faces in the image
* Bounding box accuracy

**Gemini Instructions:**

* Create a facial recognition program per task.md.
* Import the provided target.png and crowd.jpg and generate:

  * Output image with face detection and target highlighting
  * CSV of detection results with confidence scores
  * Feedback on your generated **IMAGE** results
  * Improved solutions, images and csv in new files: first solution would be gemini_solution_1, second solution would be gemini_solution_2, etc.
* Never delete files from the directory.
* Only use target.png, crowd.jpg, task.md, and files you generate.
* Use .venv for the python environment
* Stop when you feel like the requirements has been met

**Hint:**
har cascade
