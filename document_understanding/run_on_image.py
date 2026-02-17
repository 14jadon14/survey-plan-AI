
import json
import argparse
import sys
import os

# Ensure we can import the pipeline from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import DocumentParser

def run_custom(image_path=None, bboxes=None, json_path=None):
    if json_path:
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found at {json_path}")
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Support two formats:
                # 1. Simple dict: {"image_path": "...", "bboxes": [[x,y,x,y], ...]}
                # 2. List of dicts: [{"image_path": "...", "bbox": ...}, ...]
                
                if isinstance(data, dict):
                    image_path = data.get("image_path", image_path)
                    bboxes = data.get("bboxes", bboxes)
                elif isinstance(data, list):
                    # Handle list of tasks, but for now let's just assume simple structure or single image
                    # Use the first entry if it's a list implementation, or iterate?
                    # The prompt implies "search a particular file path for a list of text and bbox coordinates"
                    # pointing to iterating through a list.
                    
                    # Let's iterate if it's a list, assuming each item has image_path and bbox
                    print("Initializing DocumentParser...")
                    parser = DocumentParser()
                    
                    for i, item in enumerate(data):
                        img_p = item.get("image_path", image_path)
                        bbox = item.get("bbox")
                        
                        if not img_p:
                            print(f"Skipping item {i}: No image path provided.")
                            continue
                            
                        # Robust path handling
                        if not os.path.exists(img_p):
                            # Try to find the file in common locations
                            filename = os.path.basename(img_p)
                            potential_paths = [
                                os.path.join(os.getcwd(), filename),
                                os.path.join("/content/custom_data", filename), # Colab specific
                                os.path.join("/content/data", filename),        # Colab specific
                                os.path.join("data", filename),
                                # Recursive search in custom_data if needed? 
                                # Let's try to find it in the same directory as the JSON if applicable
                                os.path.join(os.path.dirname(json_path), filename)
                            ]
                            
                            found = False
                            for p in potential_paths:
                                if os.path.exists(p):
                                    print(f"[WARN] Image not found at {img_p}. Found at {p}. using that.")
                                    img_p = p
                                    found = True
                                    break
                            
                            if not found:
                                # Try one more fallback: check if it's in a subdirectory of custom_data matching the original parent
                                # e.g. orig: /content/data/validation/images/img.jpg
                                # new: /content/custom_data/validation/images/img.jpg
                                # We can try to replace /content/data with /content/custom_data
                                if "/content/data" in img_p:
                                    new_p = img_p.replace("/content/data", "/content/custom_data")
                                    if os.path.exists(new_p):
                                         print(f"[WARN] Image not found at {img_p}. Found at {new_p}. using that.")
                                         img_p = new_p
                                         found = True
                                
                            if not found:
                                print(f"[ERROR] Image not found at {img_p} and fallback search failed.")
                                continue

                        print(f"\nProcessing item {i}: {img_p}")
                        try:
                            results = parser.process_image(img_p, bboxes=[bbox] if bbox else None)
                            for res in results:
                                 print(f"  Parsed: '{res.get('parsed_content')}'")
                        except Exception as e:
                            print(f"[ERROR] Failed to process {img_p}: {e}")
                    return # Done with list processing

        except Exception as e:
            print(f"Error reading JSON: {e}")
            return

    # Fallback to single image processing if not a list
    if not image_path:
         print("Error: No image path provided (via CLI or JSON).")
         return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("Initializing DocumentParser...")
    parser = DocumentParser()

    print(f"Processing {image_path}...")
    # Wrap bbox in a list if provided as single item
    if bboxes and isinstance(bboxes[0], int):
         bbox_list = [bboxes]
    else:
         bbox_list = bboxes

    results = parser.process_image(image_path, bboxes=bbox_list)
    
    for i, res in enumerate(results):
        bbox_str = f" (BBox: {res.get('bbox')})" if res.get('bbox') else ""
        print(f"\n--- Result {i+1}{bbox_str} ---")
        content = res.get('parsed_content')
        if content:
            print(content)
        else:
            print("<No content extracted>")
            if 'error' in res:
                print(f"Error: {res['error']}")
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Document Understanding pipeline on an image or batch from JSON.")
    parser.add_argument("--image_path", help="Path to the image file to process.")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=('xmin', 'ymin', 'xmax', 'ymax'),
                        help="Optional bounding box coordinates (xmin ymin xmax ymax).")
    parser.add_argument("--json", help="Path to a JSON file containing processing instructions.")
    
    args = parser.parse_args()
    
    if not args.image_path and not args.json:
        parser.print_help()
        sys.exit(1)
        
    run_custom(image_path=args.image_path, bboxes=args.bbox, json_path=args.json)
