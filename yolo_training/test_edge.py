import numpy as np
import cv2

# Define an extremely skewed rectangular box, e.g., tilted 60 degrees.
# Original: Width=10, Height=3
points = np.array([
    [0.0, 5.0],      # Far Left
    [5.0, 0.0],      # Top
    [8.0, 3.0],      # Far Right
    [3.0, 8.0]       # Bottom
], dtype=np.float32)

rect = cv2.minAreaRect(points)
box = cv2.boxPoints(rect)

def edge_center(idxA, idxB):
    return ((box[idxA][0] + box[idxB][0]) / 2.0, (box[idxA][1] + box[idxB][1]) / 2.0)

edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

leftmost_edge = min(edge_indices, key=lambda e: edge_center(e[0], e[1])[0])
bottommost_edge = max(edge_indices, key=lambda e: edge_center(e[0], e[1])[1])

bl_idx = None
for i in leftmost_edge:
    if i in bottommost_edge:
        bl_idx = i
        break

if bl_idx is None:
    bl_idx = 1

tl_idx = (bl_idx + 1) % 4
tr_idx = (bl_idx + 2) % 4
br_idx = (bl_idx + 3) % 4

tl = box[tl_idx].tolist()
tr = box[tr_idx].tolist()
br = box[br_idx].tolist()
bl = box[bl_idx].tolist()

print("Original Points:", points.tolist())
print(f"Top-Left (Mapped Left-most): {tl}")
print(f"Top-Right: {tr}")
print(f"Bottom-Right: {br}")
print(f"Bottom-Left (Left/Bottom Edge Intersection): {bl}")

