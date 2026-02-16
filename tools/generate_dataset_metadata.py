
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.getcwd())

def generate_metadata(image_dir, label_dir, output_file):
    """
    Iterates through images and labels to generate a metadata.jsonl file for Donut training.
    Assumes labels are in a format that can be mapped to a JSON structure.
    For this utility, we'll assume the ground truth is already in a corresponding JSON or text file,
    OR we are generating a scaffold for the user to fill.
    
    If label_dir contains JSON files: we read them and use as ground_truth.
    If label_dir contains TXT files (YOLO format): we might convert them, but Donut inputs are usually transcription.
    
    Hypothesis: The user likely has OCR ground truth or needs to pair images with empty structural JSONs.
    Let's assume we are pairing images with their corresponding JSON labels.
    """
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    print(f"Generating metadata from {image_dir} and {label_dir}...")
    
    entries = []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    for img_path in tqdm(images):
        # Look for corresponding json label
        json_label_path = label_dir / (img_path.stem + ".json")
        
        ground_truth = ""
        
        if json_label_path.exists():
            with open(json_label_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Donut expects a stringified JSON in "gt_parse" key usually, inside "ground_truth"
                    # Format: {"gt_parse": { ... }}
                    # We wrap the content of the json file into the gt_parse structure
                    ground_truth = json.dumps({"gt_parse": data})
                except json.JSONDecodeError:
                    print(f"[WARN] Invalid JSON in {json_label_path}")
        else:
            # Fallback or empty if no label found (semi-supervised or prediction mode)
            # Create a placeholder
            ground_truth = json.dumps({"gt_parse": {"text": ""}})
        
        entry = {
            "file_name": img_path.name,
            "ground_truth": ground_truth
        }
        entries.append(entry)
        
    # Write to jsonl
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully generated {output_file} with {len(entries)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl for Donut training")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing ground truth JSON files")
    parser.add_argument("--output_file", type=str, default="metadata.jsonl", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    generate_metadata(args.image_dir, args.label_dir, args.output_file)
