import csv
from statistics import mean

left_slopes = []
right_slopes = []
left_missing = 0
right_missing = 0
with open("gemini_solution_3.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lx1 = int(row["Left_Line_X1"])
        if lx1 == -1:
            left_missing += 1
        else:
            x1 = int(row["Left_Line_X1"])
            y1 = int(row["Left_Line_Y1"])
            x2 = int(row["Left_Line_X2"])
            y2 = int(row["Left_Line_Y2"])
            if x2 != x1:
                left_slopes.append((y2 - y1) / (x2 - x1))

        rx1 = int(row["Right_Line_X1"])
        if rx1 == -1:
            right_missing += 1
        else:
            x1 = int(row["Right_Line_X1"])
            y1 = int(row["Right_Line_Y1"])
            x2 = int(row["Right_Line_X2"])
            y2 = int(row["Right_Line_Y2"])
            if x2 != x1:
                right_slopes.append((y2 - y1) / (x2 - x1))

print("left count", len(left_slopes), "missing", left_missing)
print("right count", len(right_slopes), "missing", right_missing)
if left_slopes:
    print("left slope mean", mean(left_slopes))
if right_slopes:
    print("right slope mean", mean(right_slopes))
