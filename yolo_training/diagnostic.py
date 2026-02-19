
import os
import sys
import yaml
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # yolo_training
sys.path.append(parent_dir)

from src import config
from src.detection import dataset

def diagnose():
    print("=== STARTING DIAGNOSTICS ===")
    
    # 1. Check Data Path
    print(f"[Check] Data Directory: {config.CUSTOM_DATA_DIR}")
    if config.CUSTOM_DATA_DIR.exists():
        print(f"  - Exists. Contents: {[p.name for p in config.CUSTOM_DATA_DIR.iterdir()]}")
    else:
        print("  - DOES NOT EXIST!")

    # 2. Run Prepare Data
    print("\n[Check] Running dataset.prepare_data()...")
    try:
        obb_dir, classes = dataset.prepare_data()
        print(f"  - Return OBB Dir: {obb_dir}")
        print(f"  - Return Classes: {classes}")
        print(f"  - Classes Type: {type(classes)}")
        print(f"  - Classes Length: {len(classes)}")
    except Exception as e:
        print(f"  - FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Check YAML Generation
    print("\n[Check] Generating YAML...")
    try:
        yaml_path = dataset.create_yaml(obb_dir, classes)
        print(f"  - YAML Path: {yaml_path}")
        
        if yaml_path.exists():
            print("  - Reading YAML content:")
            with open(yaml_path, 'r') as f:
                content = f.read()
                print("------------------------------------------------")
                print(content)
                print("------------------------------------------------")
                
            # Verify YAML parsing
            data = yaml.safe_load(content)
            print(f"  - Parsed Names: {data.get('names')}")
            
            if not data.get('names'):
                print("  - [CRITICAL FAILURE] 'names' is empty in YAML!")
            else:
                print("  - 'names' looks valid.")
        else:
            print("  - [CRITICAL FAILURE] YAML file not created!")
            
    except Exception as e:
        print(f"  - FAILED to generate/read YAML: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
