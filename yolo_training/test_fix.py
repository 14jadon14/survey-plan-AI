import json

def fix_geometric_mapping():
    with open('test_corners.json', 'r') as f:
        data = json.load(f)
        
    for idx in range(3):
        item = data[idx]
        corners = item['corners'] # These are the broken mapped corners for box 2
        
        # We need the RAW corners to test our fix. Let's pretend the mapped corners are raw
        # but randomly shuffled to simulate what cv2.boxPoints gives us.
        box = corners
        
        print(f"\n--- Box {idx} ({item['label']}) ---")
        
        def edge_center(idxA, idxB):
            return ((box[idxA][0] + box[idxB][0]) / 2.0, (box[idxA][1] + box[idxB][1]) / 2.0)
            
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        # Find the absolute leftmost edge by center X
        leftmost_edge = min(edge_indices, key=lambda e: edge_center(e[0], e[1])[0])
        
        # When looking for the bottom edge, it CANNOT be the opposite edge to the leftmost!
        # If it's opposite, they don't share a corner, and our fallback triggers (which causes the twist).
        # We must find the lowest edge out of the TWO edges that actually touch the leftmost edge.
        
        # The edges touching `leftmost_edge` are the two edges that share an index with it.
        touching_edges = [
            e for e in edge_indices 
            if (e[0] in leftmost_edge or e[1] in leftmost_edge) and e != leftmost_edge
        ]
        
        # From the two adjacent edges, pick the one that is physically lower (highest Y)
        bottommost_edge = max(touching_edges, key=lambda e: edge_center(e[0], e[1])[1])
        
        bl_idx = None
        for i in leftmost_edge:
            if i in bottommost_edge:
                bl_idx = i
                break
                
        print(f"Leftmost Edge: {leftmost_edge}")
        print(f"Bottommost (Adjacent) Edge: {bottommost_edge}")
        print(f"Intersecting BL Index: {bl_idx}")
        
        # Sequence: BL -> TL -> TR -> BR
        tl_idx = (bl_idx + 1) % 4
        tr_idx = (bl_idx + 2) % 4
        br_idx = (bl_idx + 3) % 4
        
        tl = box[tl_idx].tolist()
        bl = box[bl_idx].tolist()
        tr = box[tr_idx].tolist()
        br = box[br_idx].tolist()
        
        print("Final Mapped Centers:")
        print(f"Left Edge (BL->TL) X: {((bl[0]+tl[0])/2):.2f} (Should be min X)")
        print(f"Bottom Edge (BL->BR) Y: {((bl[1]+br[1])/2):.2f} (Should be max Y of adjacents)")

if __name__ == "__main__":
    fix_geometric_mapping()
