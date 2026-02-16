import sys
import os
from unittest.mock import MagicMock

# Add current dir to sys.path
sys.path.append(os.getcwd())

# MOCK Dependencies
# This allows us to verify the package structure even if the user 
# doesn't have the heavy ML libraries installed locally.
sys.modules["ultralytics"] = MagicMock()
sys.modules["sahi"] = MagicMock()
sys.modules["sahi.slicing"] = MagicMock()
sys.modules["pybboxes"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["shapely"] = MagicMock()
sys.modules["shapely.geometry"] = MagicMock()

print("[INFO] Testing imports with MOCKED dependencies...")
try:
    from src import config
    print(f"[OK] config imported. Batch size: {config.BATCH_SIZE}")
    
    from src.detection import model
    # Mock return of YOLO model
    sys.modules["ultralytics"].YOLO.return_value = "MockedYOLO"
    m = model.get_model("test")
    print(f"[OK] model.get_model() returned: {m}")
    
    from src.detection import dataset
    print("[OK] dataset module imported.")
    
    from src.detection import train
    print("[OK] train module imported.")
    
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

print("\n[INFO] Checking functions...")
if hasattr(dataset, 'slice_coco_wrapper') or hasattr(dataset, 'process_single_image_obb') or hasattr(dataset, 'convert_to_obb_parallel'):
     print("[OK] Dataset logic found.")
else:
     print("[WARN] Dataset functions might be missing.")

print("\n[SUCCESS] logic structure verified.")
