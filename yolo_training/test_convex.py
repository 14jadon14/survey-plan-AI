import json
import math

def trace_polygon_mapping():
    with open('test_corners.json', 'r') as f:
        data = json.load(f)
        
    for idx in range(3):
        item = data[idx]
        corners = item['corners']
        tl, tr, br, bl = corners
        
        print(f"\n--- Box {idx} ({item['label']}) ---")
        
        # In a valid polygon, the edges should not cross.
        # Let's trace the perimeter: TL -> TR -> BR -> BL -> TL
        # We can calculate the sign of the cross product of sequential edges.
        # If the signs are all the same, the polygon is convex and not twisted.
        # If they alternate, the polygon is a self-intersecting hourglass ("bowtie").
        
        def cross_product(p1, p2, p3):
            # Vector A = p2 - p1
            # Vector B = p3 - p2
            ax = p2[0] - p1[0]
            ay = p2[1] - p1[1]
            bx = p3[0] - p2[0]
            by = p3[1] - p2[1]
            return ax * by - ay * bx
            
        cp1 = cross_product(bl, tl, tr)
        cp2 = cross_product(tl, tr, br)
        cp3 = cross_product(tr, br, bl)
        cp4 = cross_product(br, bl, tl)
        
        print("Cross Products (TL-corner, TR-corner, BR-corner, BL-corner):")
        print(f"{cp1:.2f}, {cp2:.2f}, {cp3:.2f}, {cp4:.2f}")
        
        if (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or \
           (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0):
            print("Status: POLYGON IS CONVEX AND HEALTHY (Not twisted)")
        else:
            print("Status: POLYGON IS SELF-INTERSECTING (Hourglass/Twisted!)")

if __name__ == "__main__":
    trace_polygon_mapping()
