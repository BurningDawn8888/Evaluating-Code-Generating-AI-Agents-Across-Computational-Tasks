[task.md](https://github.com/user-attachments/files/25170607/task.md)

### **Q3 PROJECT 4 – CIRCULAR SHAPE MOVEMENT**

**Task:**
Detect a moving circular shape (soda can base) in a self-taken video. Overlay a non-wobbling, stable circular line on the can’s bottom edge, track its center point, and output results.

**Rules:**

* Video must be of a soda can or similar shape.
* No markings or appearance changes to the can.
* The circle must match the outer rim of the base.
* No flicker or wobble allowed.

**Output:**

1. CSV: Frame_Index,X,Y,Radius (OpenCV coordinates).
2. Output video with overlay.

**Assessment (100 pts):**

* Image quality – 10
* Image overlay – 25
* Overlay accuracy – 25
* Center detection – 25
* Code quality – 15 (comments, formatting, layout, pseudocode).

**Gemini Instructions:**

* Create an image-processing program per task.md.
* Import the provided rollcan.mp4 and generate:

  * Output video
  * CSV of X, Y, Radius per frame
  * Feedback on results(look at the video you generated)
  * Improved solutions, video and csv in new files: gemini_solution_1, gemini_solution_2, etc.
* Never delete files from v2.
* Only use rollcan.mp4, task.md, and files you generate.
* Use .venv for the python environment
* Stop when you feel like the requirements has been met
