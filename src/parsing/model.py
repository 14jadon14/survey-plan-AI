from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_interface(model_name_or_path="naver-clova-ix/donut-base", device=None):
    """
    Loads the Donut processor and model.
    """
    print(f"[INFO] Loading Donut model: {model_name_or_path}...")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = DonutProcessor.from_pretrained(model_name_or_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
        model.to(device)
        model.eval() # Set to evaluation mode
        
        print(f"[INFO] Donut model loaded successfully on {device}")
        return processor, model, device
    except Exception as e:
        print(f"[ERROR] Failed to load Donut model: {e}")
        return None, None, None
