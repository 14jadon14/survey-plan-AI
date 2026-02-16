from PIL import Image
import requests
from io import BytesIO
import torch
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.parsing import model as parsing_model
from src.parsing import inference as parsing_inference

def test_donut():
    print("[INFO] Starting Donut Integration Test...")
    
    # 1. Load Model
    processor, model, device = parsing_model.load_interface()
    
    if model is None:
        print("[FAIL] Model failed to load.")
        return
    
    # 2. Load Sample Image (Receipt)
    # Using a sample receipt image from the web for CORD dataset testing
    # url = "https://raw.githubusercontent.com/naver-clova-ix/donut/master/misc/sample_image_cord.png"
    # Alternative reliable URL for a receipt
    url = "https://cdn.discordapp.com/attachments/1083437255152504852/1151528640707166299/receipt.jpg"
    
    try:
        # Use a user agent to avoid blocking
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        print(f"[INFO] Loaded sample image from {url}")
    except Exception as e:
        print(f"[ERROR] Failed to download sample image: {e}")
        # Create a dummy image if download fails for testing flow
        print("[INFO] Creating dummy image for testing flow...")
        image = Image.new('RGB', (800, 1200), color = 'white')

    # 3. Run Inference
    print("[INFO] Running inference...")
    from src import config
    # CORD v2 usually uses <s_cord-v2> task start token, but base model might use none or different
    result = parsing_inference.run_parsing(model, processor, image, device=device, task_prompt=config.PARSING_TASK_PROMPT)
    
    # 4. Verification output
    if result:
        print("\n[SUCCESS] Extraction Result:")
        print(result)
    else:
        print("\n[FAIL] No result returned.")

if __name__ == "__main__":
    test_donut()
