import os
import numpy as np
import cv2
from PIL import Image
import csv

# Load coordinates from CSV file
coordinates = []
with open('ver.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        coordinates.append({
            'folder_name': row['folder_name'],
            'box': [float(row['box_x1']), float(row['box_y1']), float(row['box_x1']) + float(row['width']), float(row['box_y1']) + float(row['height'])],
            'frame': int(row['frame']),
            'point_x': float(row['point_x']) if row['point_x'] else None,
            'point_y': float(row['point_y']) if row['point_y'] else None
        })

for coord in coordinates:
    folder_name = coord['folder_name']
    box = np.array(coord['box'], dtype=np.float32)
    frame_index = coord['frame']
    point_x = coord['point_x']
    point_y = coord['point_y']

    frame_dir = f"./HOIST/valid/JPEGImages/{folder_name}"
    output_dir = f"./TEST/ver2"
    os.makedirs(output_dir, exist_ok=True)

    frame_names = sorted([p for p in os.listdir(frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]], key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    if frame_names and frame_index < len(frame_names):
        frame_path = os.path.join(frame_dir, frame_names[frame_index])
        frame = cv2.imread(frame_path)

        # Draw the box on the specified frame
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the center point on the specified frame
        if point_x is not None and point_y is not None:
            cv2.circle(frame, (int(point_x), int(point_y)), 5, (0, 0, 255), -1)

        # Save the image with the box and point
        output_frame_path = os.path.join(output_dir, f"{folder_name}_frame_{frame_index}_with_box_and_point.jpg")
        cv2.imwrite(output_frame_path, frame)

        print(f"Processed folder {folder_name}, frame {frame_index}")

print("Verification complete.")