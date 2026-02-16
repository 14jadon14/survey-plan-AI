from ultralytics import YOLO
import os
from src import config

def get_model(model_name=None):
    """
    Returns a YOLO model instance.
    Checks if a local pretrained file exists, otherwise downloads.
    """
    name = model_name or config.MODEL_NAME
    print(f"[INFO] Loading model: {name}")
    
    # Check if we have a local weights file (optional optimization)
    if os.path.exists(name):
        return YOLO(name)
        
    return YOLO(name)
