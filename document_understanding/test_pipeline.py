
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Ensure we can import the pipeline from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pipeline import DocumentParser
except ImportError:
    # If running from root, might need to adjust path
    sys.path.append(os.path.join(os.getcwd(), 'document_understanding'))
    from pipeline import DocumentParser

def create_dummy_image(path="dummy_survey_plan.jpg"):
    """Creates a simple dummy image with text for testing."""
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    # Add some text acting as "survey content"
    d.text((100, 100), "LOT 1234", fill=(0, 0, 0))
    d.text((100, 150), "PLAN 5678", fill=(0, 0, 0))
    d.text((400, 300), "AREA: 500 sqm", fill=(0, 0, 0))
    
    # Draw a "box" helper
    d.rectangle([90, 90, 300, 200], outline="red", width=2)
    
    img.save(path)
    return path

def test_pipeline():
    print("Starting pipeline test...")
    
    # 1. Create dummy image
    image_path = create_dummy_image()
    print(f"Created dummy image at {image_path}")
    
    # 2. Define a bounding box (e.g., around the "LOT 1234" text)
    # Coordinates: [xmin, ymin, xmax, ymax]
    # "LOT 1234" is roughly at 100, 100.
    bbox = [90, 90, 300, 200]
    
    # 3. Initialize Parser
    print("Initializing DocumentParser...")
    try:
        parser = DocumentParser()
    except Exception as e:
        print(f"Failed to initialize parser (likely missing dependencies or model download issue): {e}")
        return

    # 4. Run processing
    print("Processing image with bbox...")
    results = parser.process_image(image_path, bboxes=[bbox])
    
    # 5. Output results
    for res in results:
        print(f"BBox: {res.get('bbox')}")
        print(f"Parsed Content: '{res.get('parsed_content')}'")
        
    print("Test complete.")

if __name__ == "__main__":
    test_pipeline()
