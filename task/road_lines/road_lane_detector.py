
import cv2
import numpy as np
import csv

def make_points(image, line):
    slope, intercept = line
    if abs(slope) < 1e-6:
        return None
    y1 = int(image.shape[0])
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_points(image, left_fit_average)
        right_line = make_points(image, right_fit_average)
        if left_line is not None and right_line is not None:
            return [left_line, right_line]
    return None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(int(width * 0.1), height), (int(width * 0.9), height), (int(width * 0.55), int(height*0.6)), (int(width * 0.45), int(height*0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture("Road.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

csv_file = open('lines.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame_Index', 'Left_Line_X1', 'Left_Line_Y1', 'Left_Line_X2', 'Left_Line_Y2', 'Right_Line_X1', 'Right_Line_Y1', 'Right_Line_X2', 'Right_Line_Y2'])

frame_index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    
    line_image = np.zeros_like(frame)
    left_line_coords = [0,0,0,0]
    right_line_coords = [0,0,0,0]

    if averaged_lines is not None and len(averaged_lines) == 2:
        left_line = averaged_lines[0]
        right_line = averaged_lines[1]

        lx1, ly1, lx2, ly2 = left_line
        cv2.line(line_image, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 0), 10)
        left_line_coords = [lx1, ly1, lx2, ly2]

        rx1, ry1, rx2, ry2 = right_line
        cv2.line(line_image, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 0, 255), 10)
        right_line_coords = [rx1, ry1, rx2, ry2]

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    out.write(combo_image)
    
    csv_writer.writerow([frame_index, left_line_coords[0], left_line_coords[1], left_line_coords[2], left_line_coords[3], right_line_coords[0], right_line_coords[1], right_line_coords[2], right_line_coords[3]])
    
    frame_index += 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
print("Processing complete. Output video saved as output.mp4 and line coordinates saved as lines.csv")
