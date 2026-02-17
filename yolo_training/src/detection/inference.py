import os
import glob
import argparse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import json
import sys

# Add src to path for config import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    import config
except ImportError:
    # Fallback if running from root or different structure
    print("[WARNING] Could not import config.py. Using default values.")
    config = None

def run_inference(model_path, source, output_dir, slice_wh=1024, overlap_ratio=0.2, conf_thres=0.60):
    """
    Runs SAHI sliced inference on a set of images.
    """
    print(f"[INFO] Initializing SAHI with model: {model_path}")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            confidence_threshold=conf_thres,
            device=device
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    # Gather images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    if os.path.isdir(source):
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(source, ext)))
    elif os.path.isfile(source):
        image_paths = [source]
    
    if not image_paths:
        print(f"[WARNING] No images found in {source}")
        return

    print(f"[INFO] Found {len(image_paths)} images to process.")
    
    # List to store JSON results
    json_results = []

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"[{i+1}/{len(image_paths)}] Processing {image_name}...")

        try:
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=slice_wh,
                slice_width=slice_wh,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                postprocess_type="NMS",
                postprocess_match_threshold=0.5
            )

            # Export visualization
            export_path = os.path.join(output_dir, "vis")
            os.makedirs(export_path, exist_ok=True)
            
            result.export_visuals(
                export_dir=export_path,
                file_name=image_name.replace(os.path.splitext(image_name)[1], '')
            )
            
            # Collect results for JSON if enabled
            if config and getattr(config, 'SAVE_JSON_FOR_DOC_PARSING', False):
                text_labels = getattr(config, 'TEXT_LABELS', [])
                
                for prediction in result.object_prediction_list:
                    label = prediction.category.name
                    
                    # Filter by label if TEXT_LABELS is defined and not empty
                    if text_labels and label not in text_labels:
                        continue

                    # SAHI bbox is [xmin, ymin, xmax, ymax] via to_xyxy()
                    # But verifying SAHI API: bbox is a BoundingBox object.
                    # It usually holds [minx, miny, maxx, maxy] in .to_xyxy()
                    bbox = prediction.bbox.to_xyxy()
                    # Convert to int list
                    bbox = [int(x) for x in bbox]
                    
                    score = float(prediction.score.value)
                    
                    json_results.append({
                        "image_path": os.path.abspath(image_path),
                        "bbox": bbox,
                        "label": label,
                        "score": score
                    })

        except Exception as e:
            print(f"[ERROR] Failed to process {image_name}: {e}")

    # Save JSON if enabled
    if config and getattr(config, 'SAVE_JSON_FOR_DOC_PARSING', False) and json_results:
        json_path = getattr(config, 'JSON_OUTPUT_PATH', os.path.join(output_dir, "detected_assets.json"))
        # Ensure directory exists for json
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        try:
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"[INFO] Saved detection results to {json_path}")
        except Exception as e:
             print(f"[ERROR] Failed to save JSON to {json_path}: {e}")

    print(f"[INFO] Inference complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run SAHI Sliced Inference using YOLO model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained YOLO .pt model')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--output_dir', type=str, default='runs/sahi_predict', help='Directory to save results')
    parser.add_argument('--slice_wh', type=int, default=1024, help='Slice width and height (square)')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap ratio (0-1)')
    parser.add_argument('--conf', type=float, default=0.60, help='Confidence threshold')
    
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        source=args.source,
        output_dir=args.output_dir,
        slice_wh=args.slice_wh,
        overlap_ratio=args.overlap,
        conf_thres=args.conf
    )

if __name__ == "__main__":
    main()
