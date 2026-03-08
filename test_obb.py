from ultralytics import YOLO
import torch

def test_obb_format():
    # Load model
    model_path = r"c:\Users\Jadon\Desktop\SurveyPlan AI\runs\obb\survey_plan_obb_run\weights\best.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        print("Model error:", e)
        return

    # Let's create a dummy image
    import numpy as np
    import cv2
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = model([img])
    
    for r in results:
        if hasattr(r, 'obb') and r.obb is not None:
            print("r.obb.data shape:", r.obb.data.shape)
            if len(r.obb.data) > 0:
                print("r.obb.data[0]:", r.obb.data[0])
            else:
                print("No detections, but data shape is", r.obb.data.shape)
                # Let's check what properties exist
                print("Properties of OBB:", dir(r.obb))

if __name__ == "__main__":
    test_obb_format()
