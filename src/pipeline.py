"""
Pipeline orchestration module.
Ties together the detection (YOLO), parsing (Donut), and validation stages.
"""
import os
from PIL import Image
from src import config
from src.detection import model as detection_model
from src.detection import inference as detection_inference
from src.parsing import model as parsing_model
from src.parsing import processor as parsing_processor
from src.parsing import inference as parsing_inference

class SurveyPipeline:
    def __init__(self):
        print("[INFO] Initializing SurveyPipeline...")
        # Load Detection Model (SAHI)
        # Using the model path from config or default
        model_path = config.MODEL_NAME # Assuming this is a path or name that load_model handles
        if not os.path.exists(model_path) and not model_path.endswith(".pt"):
             # Fallback or check if it's a valid ultralytics name
             pass
             
        self.detection_model = detection_inference.load_model(model_path)
        
        # Load Parsing Model
        self.donut_processor, self.donut_model, self.device = parsing_model.load_interface(
            model_name_or_path=config.DONUT_MODEL_PATH
        )
        
    def process_image(self, image_path):
        """
        Runs the full pipeline on a single image.
        1. Detect objects (Surveys, Tables, etc.) using SAHI
        2. Crop objects (Rectify if OBB available)
        3. Parse content from crops
        """
        print(f"[INFO] Pipeline processing: {image_path}")
        results = {
            "image_path": image_path,
            "detections": []
        }
        
        # 1. Detection (SAHI)
        if self.detection_model:
            detection_result = detection_inference.predict_image(self.detection_model, image_path)
            
            # Load original image for cropping
            original_image = Image.open(image_path).convert("RGB")
            
            if detection_result:
                object_prediction_list = detection_result.object_prediction_list
                
                for obj in object_prediction_list:
                    # SAHI ObjectPrediction
                    # bbox is a ShiftedBox object
                    box = obj.bbox.to_xyxy() # [x1, y1, x2, y2]
                    score = obj.score.value
                    label = obj.category.name
                    
                    # Check for OBB angle if available (SAHI standard usually doesn't provide it)
                    # We'll default to 0 for now.
                    angle = 0
                    if hasattr(obj, 'mask') and obj.mask:
                        # logical place to check if SAHI supported specialized masks/OBB
                        pass

                    # 2. Rectify & Crop
                    # Calculate center, w, h from box
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    
                    obb = (cx, cy, w, h, angle)
                    
                    detection_entry = {
                        "label": label,
                        "confidence": score,
                        "box": box,
                        "angle": angle,
                        "parsed_data": None
                    }
                    
                    if self.donut_model:
                         # Construct prompt: <s_{label}>
                         # Sanitize label just in case
                         safe_label = label.lower().replace(" ", "_")
                         task_prompt = f"<s_{safe_label}>"
                         
                         print(f"  - Parsing detected {label} with prompt {task_prompt}...")
                         
                         # Use rectify_and_crop
                         crop = parsing_processor.rectify_and_crop(original_image, obb)
                         
                         # 3. Parsing Inference
                         parsed_json = parsing_inference.run_parsing(
                             self.donut_model, 
                             self.donut_processor, 
                             crop, 
                             self.device,
                             task_prompt=task_prompt
                         )
                         detection_entry["parsed_data"] = parsed_json
                    
                    results["detections"].append(detection_entry)
        
        return results

if __name__ == "__main__":
    # Simple test
    pipeline = SurveyPipeline()
    print("Pipeline initialized. Ready to process.")
