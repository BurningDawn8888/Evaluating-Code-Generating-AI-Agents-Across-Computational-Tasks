import json
import csv

# Read the JSON file
with open('csvpoints.json', 'r') as f:
    data = json.load(f)

# Extract metadata entries
metadata = data['metadata']

# Create a dictionary to store frame data
# Key: frame_index, Value: {'left': [x1, y1, x2, y2], 'right': [x1, y1, x2, y2]}
frame_data = {}

# Process each metadata entry
for entry_id, entry in metadata.items():
    # Get the timestamp (z array has one value which is the timestamp in seconds)
    timestamp = entry['z'][0]

    # Convert timestamp to frame index (assuming 44 fps based on typical video)
    # We'll need to determine the actual FPS from the video
    # For now, let's use the timestamp directly and we'll adjust

    # xy array format: [shape_type, x1, y1, x2, y2]
    # shape_type = 5 means polyline
    xy_data = entry['xy']
    shape_type = xy_data[0]

    if shape_type == 5:  # Polyline
        x1, y1, x2, y2 = xy_data[1], xy_data[2], xy_data[3], xy_data[4]

        # Determine which line this is (left or right)
        # Left lines typically have smaller x coordinates
        # Right lines typically have larger x coordinates

        # For each timestamp, we should have two annotations (left and right)
        if timestamp not in frame_data:
            frame_data[timestamp] = {}

        # Determine if this is left or right line based on x1 coordinate
        # Left line: x1 < 700 (approximate midpoint)
        # Right line: x1 > 700
        if x1 < 700:
            frame_data[timestamp]['left'] = [x1, y1, x2, y2]
        else:
            frame_data[timestamp]['right'] = [x1, y1, x2, y2]

# Now we need to determine the FPS to convert timestamps to frame indices
# Let's check what timestamps we have
timestamps = sorted(frame_data.keys())
print(f"Found {len(timestamps)} annotated timestamps")
print(f"First timestamp: {timestamps[0]}")
print(f"Last timestamp: {timestamps[-1]}")

# Based on verify_output.py, the video is at 29 fps
# Let's convert timestamps to frame indices
fps = 29

# Create the output CSV
output_rows = []

# We need to determine the total number of frames
# Based on verify_output.py, the video has 5161 frames
total_frames = 5161

print(f"Total frames estimated: {total_frames}")

# Create rows for each frame
for frame_idx in range(total_frames):
    # Calculate the timestamp for this frame
    frame_timestamp = frame_idx / fps

    # Find the closest annotated timestamp
    closest_timestamp = None
    min_diff = float('inf')

    for ts in timestamps:
        diff = abs(ts - frame_timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_timestamp = ts

    # If the closest timestamp is within 0.5 seconds, use those coordinates
    if closest_timestamp and min_diff < 0.5:
        frame_annotations = frame_data[closest_timestamp]

        # Get left and right line data
        left_line = frame_annotations.get('left', [0, 0, 0, 0])
        right_line = frame_annotations.get('right', [0, 0, 0, 0])

        row = [
            frame_idx,
            int(left_line[0]), int(left_line[1]), int(left_line[2]), int(left_line[3]),
            int(right_line[0]), int(right_line[1]), int(right_line[2]), int(right_line[3])
        ]
    else:
        # No annotation for this frame
        row = [frame_idx, 0, 0, 0, 0, 0, 0, 0, 0]

    output_rows.append(row)

# Write to CSV - using a different filename to not overwrite existing file
output_filename = 'manual_annotations_coordinates.csv'
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow([
        'Frame_Index',
        'Left_Line_X1', 'Left_Line_Y1', 'Left_Line_X2', 'Left_Line_Y2',
        'Right_Line_X1', 'Right_Line_Y1', 'Right_Line_X2', 'Right_Line_Y2'
    ])

    # Write all rows
    writer.writerows(output_rows)

print(f"Wrote {len(output_rows)} rows to {output_filename}")
