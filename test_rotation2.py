import cv2
import numpy as np
import os

def visualize_rotation(img_w, img_h, bbox, angle):
    # Create blank white image
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # Draw original bbox in red (representing the straight bounding box output by YOLO/SAHI)
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    
    # Simulate a rotated text box inside
    rect = ((cx, cy), (w, h*0.2), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(img, [box], 0, (255, 0, 0), -1) # Blue is the "text"

    cv2.imwrite(f"test_orig_{abs(angle)}.jpg", img)

    # Now run the pipeline logic
    pad = int(max(w, h) * 1.5)
    pxmin = max(0, int(cx - pad))
    pxmax = min(img.shape[1], int(cx + pad))
    pymin = max(0, int(cy - pad))
    pymax = min(img.shape[0], int(cy + pad))
    
    padded_crop = img[pymin:pymax, pxmin:pxmax]
    
    if padded_crop.size > 0:
        ncx = cx - pxmin
        ncy = cy - pymin
        
        M_padded = cv2.getRotationMatrix2D((ncx, ncy), angle, 1.0)
        rotated_padded = cv2.warpAffine(
            padded_crop, M_padded, (padded_crop.shape[1], padded_crop.shape[0]), 
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
        )
        
        cv2.imwrite(f"test_rotated_padded_{abs(angle)}.jpg", rotated_padded)

        # The math from pipeline:
        h_box, w_box = h, w
        if angle < -45:
            h_box, w_box = w_box, h_box
            
        rx1 = int(ncx - w_box/2)
        rx2 = int(ncx + w_box/2)
        ry1 = int(ncy - h_box/2)
        ry2 = int(ncy + h_box/2)
        
        final_crop_arr = rotated_padded[max(0, ry1):min(rotated_padded.shape[0], ry2), max(0, rx1):min(rotated_padded.shape[1], rx2)]
        cv2.imwrite(f"test_final_{abs(angle)}.jpg", final_crop_arr)

# Test with angle -30
visualize_rotation(800, 600, [300, 200, 500, 300], -30)

# Test with angle -70
visualize_rotation(800, 600, [300, 200, 500, 300], -70)
