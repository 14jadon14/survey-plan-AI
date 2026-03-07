from pathlib import Path
import os

# Hyperparameters
BATCH_SIZE = 16
IMGSZ = 1024
EPOCHS = 10
SLICE_HEIGHT = 2048
SLICE_WIDTH = 2048
OVERLAP_HEIGHT_RATIO = 0.4
OVERLAP_WIDTH_RATIO = 0.4
ROTATION_DEGREES = 2.0 # Augmentation parameter
NUM_WORKERS = 16 # Use 16 for Colab
AUG_FLIPLR = 0.0
CACHE_RAM = True
COS_LR = True
AMP = True
CLOSE_MOSAIC_EPOCHS = 1
PATIENCE = 20

# Inference Hyperparameters
CONF_THRESHOLD = 0.30
SAHI_POSTPROCESS_TYPE = "NMM"
SAHI_POSTPROCESS_MATCH_THRESHOLD = 0.15
SAHI_POSTPROCESS_MATCH_METRIC = "IOS"  # IOS = Intersection over Smaller area (better for diagonal/long objects)
PERFORM_STANDARD_PRED = True       # Run full-image prediction alongside slices
POSTPROCESS_CLASS_AGNOSTIC = True   # Merge overlapping boxes regardless of class
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

# Dataset Processing Hyperparameters
TRAIN_SPLIT_PCT = 0.8
DATASET_MAX_WORKERS = 4

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
    "ip": ["ip"],
    "key plan": ["key plan"],
    "legend": ["legend"],
    "line table": ["line table"],
    "lot number": ["lot number"],
    "north arrow": ["north arrow"],
    "notes": ["notes"],
    "pin num": ["pin num"],
    "plan title": ["plan title"],
    "ribf": ["ribf"],
    "scale bar": ["scale bar"],
    "smf": ["smf"],
    "sms": ["sms"],
    "sqib": ["sqib"],
    "street": ["street"],
    "text": ["text", "title data", "plan purpsoe", "file num", "plan date"],
    "signature": ["owner signature", "nbls signature"]
}

# Hybrid Training (Slices + Global Context)
HYBRID_TRAINING = True
GLOBAL_RESIZE_SIZE = 2048

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

JSON_OUTPUT_FILENAME = "crop_parameters.json"
JSON_OUTPUT_PATH = os.path.join(GOOGLE_DRIVE_PATH, JSON_OUTPUT_FILENAME)

# Labels to consider as text for document parsing
# Define labels to exclude from document parsing
EXCLUDED_FROM_TEXT = ["smf", "sms", "sqib", "ip", "ribf", "scale bar", "north arrow", "signature", "counsel stamp", "dev stamp", "anbls valnum", "pin num", "calc", "key plan", "legend"]

# Dynamically derive text labels but subtract the excluded ones
TEXT_LABELS = [label for label in LABEL_MAP.keys() if label not in EXCLUDED_FROM_TEXT]

