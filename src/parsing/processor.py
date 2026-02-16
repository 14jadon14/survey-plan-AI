
import cv2
import numpy as np
from PIL import Image

def crop_region(image, box):
    """
    Crops a region from the image based on the bounding box.
    args:
        image: PIL Image or numpy array
        box: Tuple (x1, y1, x2, y2) or List [x1, y1, x2, y2]
    returns:
        Cropped PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Ensure box is integer
    box = [int(c) for c in box]
    
    # Crop
    # PIL crop expects (left, upper, right, lower)
    cropped_img = image.crop(box)
    
    return cropped_img

def rectify_and_crop(image, obb):
    """
    Crops and rectifies (rotates) a region based on OBB.
    args:
        image: PIL Image or numpy array
        obb: Tuple (cx, cy, w, h, angle_deg)
             angle_deg is expected to be in degrees.
    returns:
        Rectified (straightened) PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    cx, cy, w, h, angle = obb
    
    # OpenCv expects (cx, cy), (w, h), angle
    rect = ((cx, cy), (w, h), angle)
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get rotation matrix for the center of the OBB
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # Rotate the entire image? This is expensive for large images.
    # Optimization: Crop a slightly larger bounding square, then rotate that crop.
    
    # 1. Get bounding rect of the rotated box
    W = int(w)
    H = int(h)
    
    # We can rotate the *whole* image if it's small, but for 10k x 10k it's bad.
    # Let's perform the "crop rotated rect" standard approach:
    # Get the bounding box of the OBB
    x_min = max(0, int(np.min(box[:, 0])))
    y_min = max(0, int(np.min(box[:, 1])))
    x_max = min(image.shape[1], int(np.max(box[:, 0])))
    y_max = min(image.shape[0], int(np.max(box[:, 1])))
    
    # Crop the bounding box
    crop = image[y_min:y_max, x_min:x_max]
    
    # Adjust center to the crop's coordinate system
    # New center
    cx_new = cx - x_min
    cy_new = cy - y_min
    
    # Rotate the crop around the new center
    M_crop = cv2.getRotationMatrix2D((cx_new, cy_new), angle, 1.0)
    
    # Determine new width/height to avoid clipping after rotation
    # Actually, we just need to warp it. 
    # But usually we want the final output to be exactly W x H (upright).
    
    rotated_crop = cv2.warpAffine(crop, M_crop, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR)
    
    # Now extract the center WxH region from the rotated crop
    # The center of the upright box in 'rotated_crop' is (cx_new, cy_new)
    
    get_w, get_h = int(w), int(h)
    
    start_x = int(cx_new - get_w / 2)
    start_y = int(cy_new - get_h / 2)
    end_x = start_x + get_w
    end_y = start_y + get_h
    
    # Pad if out of bounds (black padding)
    if start_x < 0 or start_y < 0 or end_x > rotated_crop.shape[1] or end_y > rotated_crop.shape[0]:
        # Padding logic could be added here, but for simplicity let's just clamp or pad
        # For now, safe cropping check:
        final_crop = rotated_crop[max(0, start_y):end_y, max(0, start_x):end_x]
    else:
        final_crop = rotated_crop[start_y:end_y, start_x:end_x]
        
    return Image.fromarray(final_crop)

def prepare_inputs(processor, image, device="cpu", task_prompt="<s_sroie-donut>"):
    """
    Prepares inputs for the Donut model.
    """
    # Donut expects RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    return pixel_values.to(device)
