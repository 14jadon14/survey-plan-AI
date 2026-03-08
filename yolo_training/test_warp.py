import cv2
import json
import numpy as np

def test_opencv_warp():
    with open('test_corners.json', 'r') as f:
        data = json.load(f)
        
    for idx in range(3):
        item = data[idx]
        corners = item['corners']
        tl, tr, br, bl = corners
        
        print(f"\n--- Box {idx} ({item['label']}) ---")
        
        # Calculate width and height
        w = int(np.linalg.norm(np.array(tr) - np.array(tl)))
        h = int(np.linalg.norm(np.array(bl) - np.array(tl)))
        
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        
        # Define destination points for a straight rectangle
        # Order MUST match src_pts: TL, TR, BR, BL
        dst_pts = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        print("Source Points:")
        print(src_pts)
        print("Dest Points:")
        print(dst_pts)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print("Perspective Transform Matrix:")
        print(M)
        
        # Check if determinant is negative (which implies a reflection/twist)
        det = np.linalg.det(M)
        print(f"Determinant: {det:.2f} (Negative means twisted/reflected)")
        
        # Let's also check PIL's expected order.
        # PIL Image.QUAD takes (x0, y0, x1, y1, x2, y2, x3, y3) which represents:
        # TOP-LEFT, BOTTOM-LEFT, BOTTOM-RIGHT, TOP-RIGHT
        # Wait. Look at pipeline.py line 58:
        # quad_data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
        # This matches the documentation for Image.QUAD destination mapping (TL, BL, BR, TR)
        # However, the SOURCE image mapping is backwards if we provide it in that order?

if __name__ == "__main__":
    test_opencv_warp()
