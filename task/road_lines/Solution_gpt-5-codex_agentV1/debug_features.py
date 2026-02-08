import cv2
import numpy as np
from process_road_lines_v2 import prepare_roi_mask, extract_lane_features

cap = cv2.VideoCapture('Road.mp4')
ret, frame = cap.read()
cap.release()
height, width = frame.shape[:2]
roi_mask = prepare_roi_mask((height, width))
features = extract_lane_features(frame, roi_mask)
cv2.imwrite('debug_features.png', features)
lines = cv2.HoughLinesP(features, 1, np.pi / 180, threshold=25, minLineLength=40, maxLineGap=120)
print('lines', None if lines is None else len(lines))
if lines is not None:
    center_x = width / 2
    neg = []
    pos = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.35:
            neg.append(((x1, y1, x2, y2), slope))
        elif slope > 0.35:
            pos.append(((x1, y1, x2, y2), slope))
    print('neg', len(neg))
    print('pos', len(pos))
    print('sample neg', neg[:5])
    print('sample pos', pos[:5])
