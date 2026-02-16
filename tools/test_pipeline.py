
import sys
import os
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())

from src.pipeline import SurveyPipeline
from src import config

def test_pipeline():
    print("Testing SurveyPipeline...")
    
    pipeline = SurveyPipeline()
    
    # Create a dummy image for testing if no real one exists
    # We want a large image to trigger SAHI slicing logic if possible, though pipeline handles it.
    img_path = "temp_test_image.jpg"
    
    # Create a white image with some text-like noise or shapes
    img = np.ones((2000, 2000, 3), dtype=np.uint8) * 255
    # Draw a box that LOOKS like a detection (YOLO won't detect it without weights, but we can mock detection if needed)
    # Since we are testing the pipeline FLOW, we rely on the model.
    # If the model is not found, the pipeline should just print warnings.
    
    cv2.imwrite(img_path, img)
    
    try:
        results = pipeline.process_image(img_path)
        print("Pipeline Execution Complete.")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_pipeline()
