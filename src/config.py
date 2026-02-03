from pathlib import Path
import os

# Hyperparameters
BATCH_SIZE = 16
IMGSZ = 640
EPOCHS = 5
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# Hybrid Training (Slices + Global Context)
HYBRID_TRAINING = True
GLOBAL_RESIZE_SIZE = 1024

# Paths
# Default to local relative paths, but can be overridden
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
CUSTOM_DATA_DIR = BASE_DIR / "custom_data"
SLICED_DATA_DIR = BASE_DIR / "sliced_dataset"
OBB_DATA_DIR = BASE_DIR / "survey_obb_dataset"
RUNS_DIR = BASE_DIR / "runs"

# Model
MODEL_NAME = "yolo26s-obb.pt"
PROJECT_NAME = "survey_plan_obb_run"
