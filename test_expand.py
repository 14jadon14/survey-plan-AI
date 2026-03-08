import math

# We have an image of size (W, H)
# AABB of the user crop:
min_x, min_y, max_x, max_y = 2146, 1692, 2259, 1861
aabb_w = max_x - min_x # 113
aabb_h = max_y - min_y # 169

cx_img = aabb_w / 2.0  # 56.5
cy_img = aabb_h / 2.0  # 84.5

angle_deg = -59.9
angle_rad = math.radians(angle_deg)
cos_a = math.cos(angle_rad)
sin_a = math.sin(angle_rad)

# In PIL rotate(expand=True), the new image size is calculated by rotating the 4 corners of the AABB
# and finding the new bounding box.
corners = [
    (-cx_img, -cy_img),
    (cx_img, -cy_img),
    (cx_img, cy_img),
    (-cx_img, cy_img)
]

rotated_corners = []
for x, y in corners:
    # PIL counter-clockwise matrix but Y is down.
    # It turns out PIL rotate matrix for angle theta degrees:
    # x' = x * cos(theta) + y * sin(theta)
    # y' = -x * sin(theta) + y * cos(theta)
    # Let's use standard rotation of corners
    new_x = x * cos_a + y * sin_a
    new_y = -x * sin_a + y * cos_a
    rotated_corners.append((new_x, new_y))

new_w = max(p[0] for p in rotated_corners) - min(p[0] for p in rotated_corners)
new_h = max(p[1] for p in rotated_corners) - min(p[1] for p in rotated_corners)

print(f"PIL expanded image size: {new_w:.1f} x {new_h:.1f}")

# Tight crop is center
rect_w, rect_h = 185.5, 60.8
print(f"Tight crop size: {rect_w} x {rect_h}")
print(f"Does tight crop fit in expanded image? w_fits={rect_w <= new_w}, h_fits={rect_h <= new_h}")
