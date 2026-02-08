[task.md](https://github.com/user-attachments/files/25170601/task.md)

### **Q3 PROJECT 5 – ROAD LINE TRACKING**

**Task:**
Track the sides of a road in a dashboard camera video and apply straight lines onto the left and right edges of the road. The system should detect road boundaries and overlay stable, non-flickering lines that accurately follow the road edges.

**Rules:**

* Video must be from a dashboard camera view of a car driving on a road.
* No modifications to the original video content.
* Lines must be straight and follow the road edges accurately within the ROI.
* No flicker or wobble allowed in the line overlay.
* Lines should be clearly visible and distinguishable.

**Output:**

1. CSV: Frame_Index,Left_Line_X1,Left_Line_Y1,Left_Line_X2,Left_Line_Y2,Right_Line_X1,Right_Line_Y1,Right_Line_X2,Right_Line_Y2 (OpenCV coordinates).
2. Output video with line overlays.

**Assessment:**

* Video processing quality
* Road edge detection
* Line overlay stability
* Line accuracy to road edges

**Gemini Instructions:**

* Create an image-processing program per task.md.
* Import the provided Road.mp4 and generate:

  * Output video with line overlays
  * CSV of line coordinates per frame
  * Feedback on your generated **VIDEO** results (you **CAN** read videos (mp4) with read_file)
  * Improved solutions, video and csv in new files: gemini_solution_1, gemini_solution_2, etc.
* Never delete files from the directory.
* Only use Road.mp4, task.md, and files you generate.
* Use .venv for the python environment
* Stop when you feel like the requirements has been met

**Hint:**

* ROI should be on the middle of the screen, therefore it doesn't detect the front of the car
* You would need to following libraries in your code:
import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque
from PIL import Image
