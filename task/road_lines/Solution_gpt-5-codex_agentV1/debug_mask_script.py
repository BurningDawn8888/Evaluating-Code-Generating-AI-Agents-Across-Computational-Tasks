import cv2
import numpy as np

frame = cv2.imread("frame0.png")
hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

lower_road = np.array([0, 120, 0])
upper_road = np.array([180, 210, 120])
road_mask = cv2.inRange(hls, lower_road, upper_road)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
road_clean = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imwrite("debug_road_mask.png", road_mask)
cv2.imwrite("debug_road_clean.png", road_clean)
