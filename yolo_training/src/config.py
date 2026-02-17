from pathlib import Path
import os

# Hyperparameters
BATCH_SIZE = 8
IMGSZ = 1024
EPOCHS = 100
SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_HEIGHT_RATIO = 0.3
OVERLAP_WIDTH_RATIO = 0.3

# Hybrid Training (Slices + Global Context)
HYBRID_TRAINING = True
GLOBAL_RESIZE_SIZE = 1024

# Paths
# Default to local relative paths, but can be overridden
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CUSTOM_DATA_DIR = BASE_DIR / "custom_data"
SLICED_DATA_DIR = BASE_DIR / "sliced_dataset"
OBB_DATA_DIR = BASE_DIR / "survey_obb_dataset"
RUNS_DIR = BASE_DIR / "runs"

# Model
MODEL_NAME = "yolo26m-obb.pt" 
PROJECT_NAME = "survey_plan_obb_run"

# Inference / Document Parsing Integration
SAVE_JSON_FOR_DOC_PARSING = True
# Default to a generic Google Drive path, user can update.
# Windows typical path: "G:/My Drive/SurveyPlan/detected_assets.json"
GOOGLE_DRIVE_PATH = os.path.join(os.path.expanduser("~"), "Google Drive") # Placeholder
if not os.path.exists(GOOGLE_DRIVE_PATH):
    # Fallback if Google Drive not found strictly
    GOOGLE_DRIVE_PATH = RUNS_DIR 

JSON_OUTPUT_FILENAME = "marketing_plan_assets.json"
JSON_OUTPUT_PATH = os.path.join(GOOGLE_DRIVE_PATH, JSON_OUTPUT_FILENAME)

# Labels to consider as text for document parsing
# Add labels here that should be included in the JSON output
# Example: ["text_block", "header", "table"]
TEXT_LABELS = ["plan title", "lot number", "adj lot", "area", "coord table", "curve table", "line table", "notes", "curve data", "plan purpsoe", "title data", "plan date", "azimuth", "distance", "street", "text", "file num"] # Default list, user should update based on their model classes


