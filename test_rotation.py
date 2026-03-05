import cv2
import numpy as np
from PIL import Image

def get_rotated_crop(image_pil, bbox, angle):
    """
    Given a PIL Image, bounding box [xmin, ymin, xmax, ymax], and an angle
    from cv2.minAreaRect, return the deskewed and properly cropped image.
    """
    # 1. minAreaRect returns angle in range [-90, 0)
    # The angle is the one between the lowest point and the horizontal axis.
    
    # Let's use OpenCV for the rotation as it's easier to reason about cv2 angles with cv2 rotations.
    img = np.array(image_pil)
    
    # Calculate box center
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # Get rotation matrix around the center of the bounding box
    # NOTE: minAreaRect angle is negative, cv2.warpAffine expects positive for counter-clockwise.
    # But Donut json might have different conventions, let's just rotate the *original* image
    # around the box center by `angle` (we'll try positive first to counteract cv2's negative angle)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # Perform rotation on the full image (can be slow, but mathematically safest for crops)
    # To speed up, we should only rotate a padded region around the bbox.
    
    # Padding size
    pad = int(max(w, h) * 1.5)
    
    pxmin = max(0, int(cx - pad))
    pxmax = min(img.shape[1], int(cx + pad))
    pymin = max(0, int(cy - pad))
    pymax = min(img.shape[0], int(cy + pad))
    
    # Crop padded region
    padded_crop = img[pymin:pymax, pxmin:pxmax]
    
    # New center in padded region
    ncx = cx - pxmin
    ncy = cy - pymin
    
    # Rotate padded region
    M_padded = cv2.getRotationMatrix2D((ncx, ncy), angle, 1.0)
    rotated_padded = cv2.warpAffine(padded_crop, M_padded, (padded_crop.shape[1], padded_crop.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # Now crop out the actual bounding box from the rotated padded region.
    # Since we rotated exactly around the center, the new top-left is just (ncx - w/2, ncy - h/2)
    # NOTE: minAreaRect width and height might be swapped depending on the angle.
    # If angle < -45, w and h are swapped in cv2.minAreaRect. Let's assume the provided bbox 
    # [xmin, ymin, xmax, ymax] from SAHI is the upright bounding box of the skewed text.
    
    # The problem: If we just rotate the upright bounding box, the actual text rect is smaller.
    # But usually YOLO oriented bounding boxes output center x, center y, width, height, theta.
    # The JSON here outputs standard [xmin, ymin, xmax, ymax] of the straight bounding box.
    rx1 = int(ncx - w/2)
    rx2 = int(ncx + w/2)
    ry1 = int(ncy - h/2)
    ry2 = int(ncy + h/2)
    
    final_crop = rotated_padded[max(0, ry1):min(rotated_padded.shape[0], ry2), max(0, rx1):min(rotated_padded.shape[1], rx2)]
    
    if final_crop.size == 0:
        return image_pil.crop(bbox) # fallback
        
    return Image.fromarray(final_crop)

