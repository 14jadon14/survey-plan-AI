import os
import sys
import json
import uuid
import tempfile
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import ezdxf

# Base directory setup to allow importing from brother directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import module paths
try:
    # Add paths specifically for pipeline and inference
    if os.path.join(BASE_DIR, 'yolo_training', 'src') not in sys.path:
        sys.path.append(os.path.join(BASE_DIR, 'yolo_training', 'src'))
    if os.path.join(BASE_DIR, 'document_understanding') not in sys.path:
        sys.path.append(os.path.join(BASE_DIR, 'document_understanding'))

    from detection.inference import run_inference
    from pipeline import DocumentParser
    
    # Initialize parser globally to avoid reloading the heavy model on every request
    # Note: If memory becomes an issue on local machine, we can initialize it lazily per request
    # For now, we will lazy-load it to avoid crashing the server on boot if the model isn't downloaded
    parser = None
except ImportError as e:
    print(f"[WARNING] Could not import AI modules: {e}. AI endpoints may fail.")
    run_inference = None
    DocumentParser = None
    parser = None

router = APIRouter()

def get_parser():
    global parser
    if parser is None and DocumentParser is not None:
        print("[INFO] Initializing DocumentParser...")
        parser = DocumentParser()
    return parser

@router.post("/upload/plan")
async def upload_plan(file: UploadFile = File(...)):
    """
    Accepts high resolution images (.jpg, .png, .tif), runs YOLO inference,
    and then parses the detected bounding box crops using Donut.
    """
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be JPG, PNG, or TIF.")
    
    if run_inference is None or DocumentParser is None:
         raise HTTPException(status_code=500, detail="AI pipeline modules failed to load.")

    try:
        # Save uploaded file to temp directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Define output directory for inference
        run_uuid = str(uuid.uuid4())
        output_dir = os.path.join(BASE_DIR, 'runs', 'inference', run_uuid)
        json_output_path = os.path.join(output_dir, "crop_parameters.json")

        # Configurable model path - ideally from environment variable
        # For now, default to where it's expected
        model_path = os.environ.get('YOLO_MODEL_PATH', os.path.join(BASE_DIR, 'yolo_training', 'runs', 'train', 'weights', 'best.pt'))
        
        # 1. Run YOLO Inference
        print(f"[INFO] Running inference on {temp_file_path} using model {model_path}")
        run_inference(
            model_path=model_path,
            source=temp_file_path,
            output_dir=output_dir,
            json_output=json_output_path,
            # using defaults for inference settings for now (can map from request later)
        )

        # 2. Read JSON output
        if not os.path.exists(json_output_path):
             return {"message": "Inference completed, but no objects were detected or JSON failed to save.", "results": []}

        with open(json_output_path, 'r') as f:
            detections = json.load(f)

        # 3. Parse with Donut
        bboxes = []
        angles = []
        
        for det in detections:
            bboxes.append(det.get("bbox"))
            angles.append(det.get("angle", 0))

        donut_parser = get_parser()
        if donut_parser and bboxes:
             print(f"[INFO] Running Donut parsing on {len(bboxes)} crops...")
             parsed_results = donut_parser.process_image(
                 image=temp_file_path,
                 bboxes=bboxes,
                 angles=angles
             )
             
             # Merge results back together
             for i, det in enumerate(detections):
                 if i < len(parsed_results):
                     det["parsed_content"] = parsed_results[i].get("parsed_content", "")
                     det["error"] = parsed_results[i].get("error", None)

        # We return the initial file URL so the frontend can serve it (in reality, we should serve statically or use a blob URL)
        # We will assume frontend serves the same local file or we stream it back.
        # For now, return path and detections.
        return {
            "message": "Processing complete",
            "file_path": temp_file_path,
            "detections": detections
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/cad")
async def upload_cad(file: UploadFile = File(...)):
    """
    Parses CAD files and verifies mandatory layers.
    """
    if not file.filename.lower().endswith(('.dxf', '.dwg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be DXF or DWG.")
    
    try:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Parse logic
        doc = ezdxf.readfile(temp_file_path)
        layers = [layer.dxf.name for layer in doc.layers]
        
        mandatory_layers = ["PLACEHOLDER_LAYER_1", "PLACEHOLDER_LAYER_2"]
        missing_layers = [layer for layer in mandatory_layers if layer not in layers]
        
        return {
            "valid": len(missing_layers) == 0,
            "layers_found": layers,
            "missing_layers": missing_layers,
            "message": "CAD file parsed successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CAD file: {e}")

@router.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Parses CSV for coordinate extraction.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be CSV.")
        
    try:
        df = pd.read_csv(file.file)
        
        # Determine coordinate columns heuristically or strictly based on common names
        # Just returning the whole parsed records for now for flexibility
        records = df.to_dict(orient="records")
        return {
            "message": "CSV parsed successfully.",
            "data": records
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {e}")

class SubjectLotInput(BaseModel):
    lot_number: str
    plan_number: str
    description: str = None

@router.post("/subject-lot")
async def define_subject_lot(lot_data: SubjectLotInput):
    """
    Receives subject lot configuration.
    """
    # Simply echo back for now, in future we might save this state
    return {
        "message": "Subject lot configured.",
        "data": lot_data.dict()
    }
