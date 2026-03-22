import os
import sys
import json
import uuid
import tempfile
import pandas as pd
from typing import List, Dict
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import ezdxf

# Base directory setup to allow importing from brother directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.services.cad_validator import CADValidator
from app.services.plan_evaluator import PlanEvaluator

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
        donut_model_path = os.environ.get('DONUT_MODEL_PATH', os.path.join(BASE_DIR, 'yolo_training', 'runs', 'donut_weights'))
        print(f"[INFO] Initializing DocumentParser with model from {donut_model_path}...")
        try:
            parser = DocumentParser(model_path=donut_model_path)
        except Exception as e:
            print(f"[WARNING] Failed to load local Donut model from {donut_model_path}. Falling back to default: {e}")
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
            
        print(f"[INFO] CAD file saved to {temp_file_path}. Initializing validator...")
        validator = CADValidator(temp_file_path)
        print(f"[INFO] Running validation sequence...")
        # standalone validation (without CSV corners)
        validation_results = validator.validate_all()
        print(f"[INFO] CAD Validation complete. Valid: {validation_results['valid']}")
        
        return {
            "valid": validation_results["valid"],
            "file_path": temp_file_path, # Return path so frontend can use it for cross-verify if needed
            "validation": validation_results,
            "message": "CAD file validated successfully."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to parse CAD file: {e}")

class CrossVerifyInput(BaseModel):
    cad_file_path: str
    csv_corners: List[Dict]

@router.post("/verify-cad-csv")
async def verify_cad_csv(data: CrossVerifyInput):
    """
    Cross-references CAD vertices with previously uploaded CSV corners.
    """
    if not os.path.exists(data.cad_file_path):
        raise HTTPException(status_code=404, detail="CAD file not found on server.")
        
    try:
        validator = CADValidator(data.cad_file_path)
        results = validator.verify_corners(data.csv_corners)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-verification failed: {e}")

@router.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...), corners: str = Form(None)):
    """
    Parses ASCII coordinate files (PNEZD format).
    Supports .csv, .txt, .asc, .xyz, .pts
    Handles comma or whitespace delimiters.
    """
    print("[INFO] PNEZD Upload Endpoint Active (Serialization Fix v4)")
    import math
    
    # Helper to ensure standard Python types for JSON recursively
    def force_native(obj):
        if isinstance(obj, dict):
            return {str(k): force_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [force_native(x) for x in obj]
        if pd.isna(obj):
            return None
        # Handle numpy scalars
        if hasattr(obj, 'item') and not isinstance(obj, (list, dict)):
            obj = obj.item()
        
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return float(obj)
        if isinstance(obj, (int, bool)):
            return obj
        if obj is None:
            return None
        return str(obj)

    ext = file.filename.lower().split('.')[-1]
    
    if ext not in ['csv', 'txt', 'asc', 'xyz', 'pts']:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Must be CSV or ASCII (txt, asc, xyz, pts).")
        
    try:
        content = await file.read()
        # Decode content
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
            
        lines = text.strip().split('\n')
        if not lines:
            raise ValueError("File is empty.")

        # Heuristic delimiter detection
        first_line = lines[0]
        if ',' in first_line:
            sep = ','
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = None # Whitespace (default for read_table/read_csv)

        # Read into dataframe
        from io import StringIO
        
        # Robust header detection: 
        # Only treat as header if the second and third columns are NOT numeric in the first line, 
        # but ARE numeric in the second line (if it exists).
        first_line_parts = [p.strip().strip('"').strip("'") for p in (first_line.split(sep) if sep else first_line.split())]
        
        def is_numeric(s):
            try:
                float(s)
                return True
            except (ValueError, TypeError):
                return False

        # PNEZD typically has Northing and Easting in indices 1 and 2
        # Check if indices 1 or 2 in first line are numeric
        line1_coords_numeric = any(is_numeric(first_line_parts[i]) for i in range(1, min(len(first_line_parts), 4)))
        
        # If the first line's coordinate area is numeric, it's NOT a header.
        is_header = not line1_coords_numeric and len(lines) > 1
        
        df = pd.read_csv(StringIO(text), sep=sep, engine='python', header=0 if is_header else None)
        
        # Normalize columns to PNEZD
        mapping = {}
        cols = [str(c).lower() for c in df.columns]
        
        # 1. Map Point ID (Point, ID, Name, Number)
        # Priority: Exact match, then contains
        for i, c in enumerate(cols):
            if any(k == c for k in ['p', 'pt', 'id', 'name', 'number', 'point']): mapping['point'] = i; break
        if 'point' not in mapping:
            for i, c in enumerate(cols):
                if any(k in c for k in ['point', 'pt', 'id', 'name', 'num']) and not any(k in c for k in ['north', 'east', 'elev']):
                    mapping['point'] = i; break

        # 2. Map Northing/Y (Priority: Exact match, then contains 'north' or 'n')
        # We need to be careful not to match 'note' or 'name' here.
        for i, c in enumerate(cols):
            if i in mapping.values(): continue
            if any(k == c for k in ['n', 'y', 'northing', 'north']): mapping['y'] = i; break
        if 'y' not in mapping:
             for i, c in enumerate(cols):
                if i in mapping.values(): continue
                # Match 'northing' or 'north' but avoid 'name'
                if ('north' in c or 'nort' in c) and 'name' not in c:
                    mapping['y'] = i; break
        
        # 3. Map Easting/X (Priority: Exact match, then contains 'east' or 'e')
        for i, c in enumerate(cols):
            if i in mapping.values(): continue
            if any(k == c for k in ['e', 'x', 'easting', 'east']): mapping['x'] = i; break
        if 'x' not in mapping:
            for i, c in enumerate(cols):
                if i in mapping.values(): continue
                # Match 'easting' or 'east' but avoid 'elev'
                if ('east' in c or 'east' in c) and 'elev' not in c:
                    mapping['x'] = i; break
            
        # 4. Map Elevation/Z
        for i, c in enumerate(cols):
            if i in mapping.values(): continue
            if any(k == c for k in ['z', 'elev', 'height', 'elevation']): mapping['z'] = i; break

        # 5. Map Description
        for i, c in enumerate(cols):
            if i in mapping.values(): continue
            if any(k in c for k in ['desc', 'code', 'note', 'remark', 'info']): mapping['desc'] = i; break

        # Fallback to positional mapping for PNEZD if heuristic failed
        if 'point' not in mapping and df.shape[1] >= 1: mapping['point'] = 0
        if 'y' not in mapping and df.shape[1] >= 2: mapping['y'] = 1
        if 'x' not in mapping and df.shape[1] >= 3: mapping['x'] = 2
        if 'z' not in mapping and df.shape[1] >= 4: mapping['z'] = 3
        if 'desc' not in mapping and df.shape[1] >= 5: mapping['desc'] = 4

        final_data = []
        for _, row in df.iterrows():
            record = {}
            # Point
            record['point'] = row.iloc[mapping['point']] if 'point' in mapping else ""
            # Y / Northing
            record['y'] = row.iloc[mapping['y']] if 'y' in mapping else None
            # X / Easting
            record['x'] = row.iloc[mapping['x']] if 'x' in mapping else None
            # Z / Elevation
            record['z'] = row.iloc[mapping['z']] if 'z' in mapping else 0.0
            # Description
            record['desc'] = row.iloc[mapping['desc']] if 'desc' in mapping else ""
            
            final_data.append(record)

        # Convert columns_mapped keys/values to standard strings
        mapped_info = {str(k): str(df.columns[v]) for k, v in mapping.items()}

        response_payload = {
            "message": f"PNEZD file parsed successfully using {sep if sep else 'whitespace'} delimiter.",
            "data": final_data,
            "columns_mapped": mapped_info
        }
        
        # Apply the nuclear cleaner to EVERY part of the response
        return force_native(response_payload)
    except Exception as e:
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Failed to parse coordinate file: {e}")

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

class EvaluatePlanInput(BaseModel):
    detections: List[Dict]
    csv_corners: Optional[List[Dict]] = None

@router.post("/evaluate-plan")
async def evaluate_plan(data: EvaluatePlanInput):
    """
    Evaluates a survey plan's correctness based on YOLO detections + Donut OCR.
    """
    try:
        evaluator = PlanEvaluator()
        results = evaluator.evaluate(data.detections, data.csv_corners)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Plan evaluation failed: {e}")
