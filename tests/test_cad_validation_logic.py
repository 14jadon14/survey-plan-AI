import os
import sys
import ezdxf

# Add app directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.services.cad_validator import CADValidator

def create_test_dxf(filename, layers=None, entities=None):
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    if layers:
        for layer in layers:
            doc.layers.new(name=layer)
            
    if entities:
        for layer, type, pts in entities:
            if type == 'LINE':
                msp.add_line(pts[0], pts[1], dxfattribs={'layer': layer})
            elif type == 'LWPOLYLINE':
                msp.add_lwpolyline(pts, dxfattribs={'layer': layer})
                
    doc.saveas(filename)
    return filename

def test_validation():
    test_file = "test_cad_validation.dxf"
    
    # 1. Valid File: Only allowed layers, no intersections, connected.
    print("\n--- TEST 1: Valid File ---")
    create_test_dxf(test_file, 
        layers=["NBBDRY", "NBSBDRY"],
        entities=[
            ("NBBDRY", "LINE", [(0,0), (10,0)]),
            ("NBBDRY", "LINE", [(10,0), (10,10)]),
            ("NBBDRY", "LINE", [(10,10), (0,10)]),
            ("NBBDRY", "LINE", [(0,10), (0,0)])
        ]
    )
    validator = CADValidator(test_file)
    res = validator.validate_all()
    print(f"Valid: {res['valid']}")
    print(f"Layers Found: {res['layers']['found_layers']}")
    print(f"Topology Gaps: {res['topology']['connectivity_gaps_count']}")
    
    # 2. Invalid File: Unauthorized layer
    print("\n--- TEST 2: Unauthorized Layer ---")
    create_test_dxf(test_file, 
        layers=["NBBDRY", "BAD_LAYER"],
        entities=[
            ("NBBDRY", "LINE", [(0,0), (10,0)]),
            ("BAD_LAYER", "LINE", [(10,0), (10,10)])
        ]
    )
    validator = CADValidator(test_file)
    res = validator.validate_all()
    print(f"Valid: {res['valid']} (Expected False)")
    print(f"Unauthorized: {res['layers']['unauthorized_layers']}")

    # 3. Invalid File: Intersection
    print("\n--- TEST 3: Intersection ---")
    create_test_dxf(test_file, 
        layers=["NBBDRY"],
        entities=[
            ("NBBDRY", "LINE", [(0,0), (10,10)]),
            ("NBBDRY", "LINE", [(0,10), (10,0)])
        ]
    )
    validator = CADValidator(test_file)
    res = validator.validate_all()
    print(f"Valid: {res['valid']} (Expected False)")
    print(f"Intersections: {res['topology']['intersections_count']}")

    # 4. Invalid File: Gap
    print("\n--- TEST 4: Connectivity Gap ---")
    create_test_dxf(test_file, 
        layers=["NBBDRY"],
        entities=[
            ("NBBDRY", "LINE", [(0,0), (10,0)]),
            ("NBBDRY", "LINE", [(10.01, 0), (10.01, 10)]) # Gap > 0.001
        ]
    )
    validator = CADValidator(test_file)
    res = validator.validate_all()
    print(f"Valid: {res['valid']} (Expected False)")
    print(f"Gaps: {res['topology']['connectivity_gaps_count']}")

    # 5. Corner Cross-Reference
    print("\n--- TEST 5: Corner Cross-Reference ---")
    create_test_dxf(test_file, 
        layers=["NBBDRY"],
        entities=[("NBBDRY", "LINE", [(100, 200), (300, 400)])]
    )
    validator = CADValidator(test_file)
    corners = [{"x": 100, "y": 200, "name": "P1"}, {"x": 300.0001, "y": 400.0001, "name": "P2"}]
    res = validator.verify_corners(corners)
    print(f"Valid: {res['valid']} (Expected True due to 0.001 tolerance)")
    print(f"Matches: {res['matches_count']}")

    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    test_validation()
