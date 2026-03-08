import json

def test_json():
    with open('test_corners_new.json', 'r') as f:
        data = json.load(f)
        
    all_passed = True
    
    for idx, item in enumerate(data):
        label = item.get('label', 'unknown')
        bbox = item.get('bbox')
        corners = item.get('corners')
        
        if not corners:
            # Some text blocks might not have OBB corners if they are purely HBB, just skip or note
            continue
            
        # 1. Test BBox Encapsulation
        # The bbox should tightly wrap the corners since SAHI shifted both!
        min_x = min(pt[0] for pt in corners)
        max_x = max(pt[0] for pt in corners)
        min_y = min(pt[1] for pt in corners)
        max_y = max(pt[1] for pt in corners)
        
        # Bbox from JSON is [xmin, ymin, xmax, ymax] (int)
        b_xmin, b_ymin, b_xmax, b_ymax = bbox
        
        # Allow a small floating point margin (SAHI might round or pad slightly)
        margin = 5
        
        x_match = (b_xmin <= min_x + margin) and (b_xmax >= max_x - margin)
        y_match = (b_ymin <= min_y + margin) and (b_ymax >= max_y - margin)
        
        # We also want to make sure it's tight, it shouldn't be hundreds of pixels off
        # If it was double shifted, the bbox would be off by hundreds of pixels.
        tight_x = abs(b_xmin - min_x) < 20 and abs(b_xmax - max_x) < 20
        tight_y = abs(b_ymin - min_y) < 20 and abs(b_ymax - max_y) < 20
        
        is_encapsulated = x_match and y_match and tight_x and tight_y
        
        # 2. Test Convex Poly (Hourglass twist check)
        def cross_product(p1, p2, p3):
            ax = p2[0] - p1[0]
            ay = p2[1] - p1[1]
            bx = p3[0] - p2[0]
            by = p3[1] - p2[1]
            return ax * by - ay * bx
            
        tl, tr, br, bl = corners
        cp1 = cross_product(bl, tl, tr)
        cp2 = cross_product(tl, tr, br)
        cp3 = cross_product(tr, br, bl)
        cp4 = cross_product(br, bl, tl)
        
        is_convex = (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or \
                    (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0)
                    
        if not is_encapsulated or not is_convex:
            all_passed = False
            print(f"FAILED Box {idx} ({label}):")
            if not is_encapsulated:
                print(f"  -> Encapsulation Failed! BBox: {bbox}, Corners MinMax: [{min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f}]")
            if not is_convex:
                print(f"  -> Geometry Failed! Polygon is Twisted (Hourglass).")
                
    if all_passed:
        print("SUCCESS! All boxes are convex (no twists) and perfectly encapsulated within their bounding boxes (no double-shift)!")

if __name__ == "__main__":
    test_json()
