import ezdxf
import pandas as pd
import os

def create_valid_dxf(filename):
    doc = ezdxf.new()
    msp = doc.modelspace()
    doc.layers.new("NBBDRY")
    doc.layers.new("NBSBDRY")
    
    # Square boundary
    msp.add_line((0, 0), (100, 0), dxfattribs={"layer": "NBBDRY"})
    msp.add_line((100, 0), (100, 100), dxfattribs={"layer": "NBBDRY"})
    msp.add_line((100, 100), (0, 100), dxfattribs={"layer": "NBBDRY"})
    msp.add_line((0, 100), (0, 0), dxfattribs={"layer": "NBBDRY"})
    
    # Interior line
    msp.add_line((0, 0), (100, 100), dxfattribs={"layer": "NBSBDRY"})
    
    doc.saveas(filename)
    print(f"Created {filename}")

def create_invalid_layer_dxf(filename):
    doc = ezdxf.new()
    msp = doc.modelspace()
    doc.layers.new("NBBDRY")
    doc.layers.new("UNAUTHORIZED_LAYER")
    
    msp.add_line((0, 0), (50, 50), dxfattribs={"layer": "NBBDRY"})
    msp.add_line((50, 50), (100, 100), dxfattribs={"layer": "UNAUTHORIZED_LAYER"})
    
    doc.saveas(filename)
    print(f"Created {filename}")

def create_invalid_topology_dxf(filename):
    doc = ezdxf.new()
    msp = doc.modelspace()
    doc.layers.new("NBBDRY")
    
    # 1. Intersection (X shape)
    msp.add_line((0, 0), (10, 10), dxfattribs={"layer": "NBBDRY"})
    msp.add_line((0, 10), (10, 0), dxfattribs={"layer": "NBBDRY"})
    
    # 2. Gap (Dangling line)
    msp.add_line((20, 20), (30, 30), dxfattribs={"layer": "NBBDRY"})
    msp.add_line((30.01, 30.01), (40, 40), dxfattribs={"layer": "NBBDRY"})
    
    doc.saveas(filename)
    print(f"Created {filename}")

def create_csv(filename):
    data = [
        {"Point": "1", "Northing": 0, "Easting": 0, "Description": "CORNER"},
        {"Point": "2", "Northing": 0, "Easting": 100, "Description": "CORNER"},
        {"Point": "3", "Northing": 100, "Easting": 100, "Description": "CORNER"},
        {"Point": "4", "Northing": 100, "Easting": 0, "Description": "CORNER"},
        {"Point": "5", "Northing": 50, "Easting": 50, "Description": "RANDOM"} # Should fail match if not in CAD
    ]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename}")

if __name__ == "__main__":
    os.makedirs("test_data", exist_ok=True)
    create_valid_dxf("test_data/test_valid.dxf")
    create_invalid_layer_dxf("test_data/test_invalid_layers.dxf")
    create_invalid_topology_dxf("test_data/test_invalid_topology.dxf")
    create_csv("test_data/test_points.csv")
