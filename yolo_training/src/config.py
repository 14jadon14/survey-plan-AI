from pathlib import Path
import os

# Hyperparameters
BATCH_SIZE = 16
IMGSZ = 640
EPOCHS = 10
SLICE_HEIGHT = 1280
SLICE_WIDTH = 1280
OVERLAP_HEIGHT_RATIO = 0.4
OVERLAP_WIDTH_RATIO = 0.4
ROTATION_DEGREES = 0.0 # Augmentation parameter

# Label Mapping System
# Maps multiple raw classes to a single target macro-class
# Format: {"target_class_name": ["raw_class_1", "raw_class_2"]}
LABEL_MAP = {
    "adj lot": ["adj lot"],
    "anbls valnum": ["anbls valnum"],
    "area": ["area"],
    "azimuth": ["azimuth"],
    "calc": ["calc"],
    "coord table": ["coord table"],
    "counsel stamp": ["counsel stamp"],
    "curve data": ["curve data"],
    "curve table": ["curve table"],
    "dev stamp": ["dev stamp"],
    "distance": ["distance"],
    "file num": ["file num"],
    "ip": ["ip"],
    "key plan": ["key plan"],
    "legend": ["legend"],
    "line table": ["line table"],
    "lot number": ["lot number"],
    "north arrow": ["north arrow"],
    "notes": ["notes"],
    "pin num": ["pin num"],
    "plan date": ["plan date"],
    "plan purpsoe": ["plan purpsoe"],
    "plan title": ["plan title"],
    "ribf": ["ribf"],
    "scale bar": ["scale bar"],
    "smf": ["smf"],
    "sms": ["sms"],
    "sqib": ["sqib"],
    "street": ["street"],
    "text": ["text"],
    "title data": ["title data"],
    "signature": ["owner signature", "nbls signature"]
}

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
MODEL_NAME = "yolo26l-obb.pt" 
PROJECT_NAME = "survey_plan_obb_run"

# Inference / Document Parsing Integration
SAVE_JSON_FOR_DOC_PARSING = True
# Default to a generic Google Drive path, user can update.
# Windows typical path: "G:/My Drive/SurveyPlan/detected_assets.json"
GOOGLE_DRIVE_PATH = os.path.join(os.path.expanduser("~"), "Google Drive") # Placeholder
if not os.path.exists(GOOGLE_DRIVE_PATH):
    # Fallback if Google Drive not found 

    GOOGLE_DRIVE_PATH = RUNS_DIR 

JSON_OUTPUT_FILENAME = "marketing_plan_assets.json"
JSON_OUTPUT_PATH = os.path.join(GOOGLE_DRIVE_PATH, JSON_OUTPUT_FILENAME)

# Labels to consider as text for document parsing
# Add labels here that should be included in the JSON output
# Example: ["text_block", "header", "table"]
TEXT_LABELS = ["plan title", "lot number", "adj lot", "area", "coord table", "curve table", "line table", "notes", "curve data", "plan purpsoe", "title data", "plan date", "azimuth", "distance", "street", "text", "file num"] # Default list, user should update based on their model classes


