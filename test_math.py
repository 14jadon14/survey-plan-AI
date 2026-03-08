import math

def rotate_point(px, py, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    # PIL rotate is counter-clockwise around center, but image Y is down.
    # To rotate counter-clockwise in an image (where +Y is down), 
    # it corresponds to clockwise rotation in standard math (+Y up).
    # Wait, PIL's `rotate(angle)`:
    # "Rotates the image counter-clockwise"
    # A counter-clockwise physical rotation mapping (x,y) to (X,Y) around (cx,cy):
    # X = cx + (px - cx) * cos(theta) + (py - cy) * sin(theta)
    # Y = cy - (px - cx) * sin(theta) + (py - cy) * cos(theta)
    pass

# OBB corners in original image
obb_raw = [
    (2219.2, 1678.3), # TL
    (2271.8, 1708.8), # TR
    (2178.8, 1869.4), # BR
    (2126.2, 1838.9)  # BL
]

# AABB of the OBB
# min/max x and y of OBB
min_x = min(p[0] for p in obb_raw) # 2126.2
max_x = max(p[0] for p in obb_raw) # 2271.8
min_y = min(p[1] for p in obb_raw) # 1678.3
max_y = max(p[1] for p in obb_raw) # 1869.4

print(f"Computed AABB: [{min_x}, {min_y}, {max_x}, {max_y}]")
print(f"User AABB: [2146, 1692, 2259, 1861]")

# Notice how the Computed AABB doesn't perfectly match User AABB.
# Why? Because the `[DEBUG OBB]` prints the RAW mask bounding box,
# while the user's JSON bbox is `bbox` from `prediction.bbox`.
# Prediction bbox might be slightly different.
# BUT, let's look at the center:
center_obb_x = sum(p[0] for p in obb_raw) / 4.0
center_obb_y = sum(p[1] for p in obb_raw) / 4.0

center_aabb_x = (min_x + max_x) / 2.0
center_aabb_y = (min_y + max_y) / 2.0

print(f"Center OBB: ({center_obb_x:.1f}, {center_obb_y:.1f})")
print(f"Center AABB: ({center_aabb_x:.1f}, {center_aabb_y:.1f})")
