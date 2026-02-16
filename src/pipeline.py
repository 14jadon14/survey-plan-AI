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
                    
                    # Check for OBB angle
                    # SAHI ObjectPrediction might not expose angle directly in standard bbox.
                    # If using YOLOv8-OBB with SAHI, it often returns a mask (polygon) or we need to check extra attributes.
                    angle = 0
                    if obj.mask:
                        # If a mask/polygon is available, compute the OBB angle
                        import cv2
                        import numpy as np
                        try:
                            # obj.mask.to_mask() returns a binary mask or checking if it's a segmentation dict
                            # SAHI's Mask object strategies vary. Let's assume we can get a polygon or binary mask.
                            # If it's a boolean mask:
                            if hasattr(obj.mask, 'bool_mask'):
                                mask_arr = obj.mask.bool_mask.astype(np.uint8)
                                contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    largest_cnt = max(contours, key=cv2.contourArea)
                                    rect = cv2.minAreaRect(largest_cnt)
                                    # rect is ((cx, cy), (w, h), angle)
                                    angle = rect[2]
                        except Exception as e:
                            print(f"[WARN] Failed to extract angle from mask: {e}")
                    
                    # Alternatively check for 'angle' in extra_data if populated by custom inference
                    if hasattr(obj, 'extra_data') and obj.extra_data and 'angle' in obj.extra_data:
                        angle = obj.extra_data['angle']

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
                         # Construct prompt using project-specific token
                         # If detections have specific classes (e.g. 'bearing'), use them.
                         # If generic 'survey_item', use '<s_survey_parsing>'.
                         
                         safe_label = label.lower().replace(" ", "_")
                         
                         # Project-specific task token
                         task_prompt = f"<s_{safe_label}>"
                         
                         # If the label is generic, we can enforce a specific start token
                         # user requested "replace ... with a specialized project token, such as <s_survey_parsing>"
                         # We'll prepend or use it as the base. 
                         # Let's assume we want <s_survey_parsing> to be the start, and maybe class info follows?
                         # Or simply: if the model is fine-tuned on <s_survey_parsing>, we should use that.
                         # But since we have multi-class (bearing, distance), we likely want <s_bearing> etc.
                         # However, to explicitly satisfy the user's request:
                         # "replace the generic <s_cord-v2> ... with ... <s_survey_parsing>"
                         
                         # If we are parsing a specific field, maybe we use <s_survey_parsing> as the task, 
                         # and the model outputs {class: value}. 
                         # Let's pivot to using <s_survey_parsing> as the default for unknown or generic items,
                         # but keep class specific if valuable.
                         # Actually, let's just use <s_survey_parsing> for everything if that's the trained protocol,
                         # or stick to the dynamic one if we are confident.
                         # Given the prompt, I will set the default to <s_survey_parsing> in the inference call,
                         # OR pass it here.
                         
                         # The user likely implies the model is/will be trained on <s_survey_parsing>.
                         # I'll use that as the base prompt.
                         task_prompt = "<s_survey_parsing>"
                         
                         # If we really want class-specific, we could do "<s_survey_parsing> <s_bearing>" 
                         # but usually it's just one start token.
                         # I will set it to <s_survey_parsing> as requested.
                         
                         # Optimization: If we have the label, we can add it to the detection entry
                         # but pass the generic prompt to the model if it handles all types.
                         
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
