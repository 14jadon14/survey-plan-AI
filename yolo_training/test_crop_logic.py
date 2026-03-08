import json
import math

def get_dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def test_crop_simulation():
    with open('test_corners.json', 'r') as f:
        data = json.load(f)
        
    for idx in range(3):
        corners = data[idx]['corners']
        tl, tr, br, bl = corners
        
        print(f"\n--- Box {idx} ({data[idx]['label']}) ---")
        # In pipeline.py, crop dimensions are calculated as:
        # w = int(math.hypot(tr[0]-tl[0], tr[1]-tl[1]))
        # h = int(math.hypot(bl[0]-tl[0], bl[1]-tl[1]))
        w = int(get_dist(tl, tr))
        h = int(get_dist(bl, tl))
        
        print(f"Pipeline sees W: {w}, H: {h}")
        
        # In pipeline.py, quad data is passed as:
        # quad_data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
        # BUT Image.QUAD documentation specifically states:
        # "maps a quadrilateral (defined by its four corners, TOP-LEFT, BOTTOM-LEFT, BOTTOM-RIGHT, and TOP-RIGHT) into a rectangle"
        
        print("Pipeline Quadrilateral Mapping:")
        print(f"Top-Left (Dest) -> Mapped from TL {tl}")
        print(f"Bottom-Left (Dest) -> Mapped from BL {bl}")
        print(f"Bottom-Right (Dest) -> Mapped from BR {br}")
        print(f"Top-Right (Dest) -> Mapped from TR {tr}")
        
        # The key to an hourglass occurs when edges cross.
        # This mapping looks correct according to PIL docs.
        # Wait, let's look at Box 2 (distance).
        
        print("\nIs this a crossed polygon? Let's check Box 2 (distance)")
        if idx == 2:
            print("TL:", tl)
            print("BL:", bl)
            print("Vector TL->BL: dx=", bl[0]-tl[0], "dy=", bl[1]-tl[1])
            print("TR:", tr)
            print("BR:", br)
            print("Vector TR->BR: dx=", br[0]-tr[0], "dy=", br[1]-tr[1])

if __name__ == "__main__":
    test_crop_simulation()
