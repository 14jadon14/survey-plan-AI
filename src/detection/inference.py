import os
import glob
import argparse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch


def load_model(model_path, conf_thres=0.60):
    """
    Loads the SAHI AutoDetectionModel.
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
        return detection_model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def predict_image(detection_model, image_path, slice_wh=1024, overlap_ratio=0.2):
    """
    Runs SAHI prediction on a single image.
    Returns: sahi.prediction.PredictionResult
    """
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
        return result
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None

def run_inference(model_path, source, output_dir, slice_wh=1024, overlap_ratio=0.2, conf_thres=0.60):
    """
    Runs SAHI sliced inference on a set of images.
    """
    detection_model = load_model(model_path, conf_thres)
    if detection_model is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    # Gather images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(source, ext)))
    
    # Also check if source is a direct file
    if os.path.isfile(source):
        image_paths = [source]

    if not image_paths:
        print(f"[WARNING] No images found in {source}")
        return

    print(f"[INFO] Found {len(image_paths)} images to process.")

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"[{i+1}/{len(image_paths)}] Processing {image_name}...")

        result = predict_image(detection_model, image_path, slice_wh, overlap_ratio)
        
        if result:

            # Export visualization
            export_path = os.path.join(output_dir, "vis")
            os.makedirs(export_path, exist_ok=True)
            
            result.export_visuals(
                export_dir=export_path,
                file_name=image_name.replace(os.path.splitext(image_name)[1], '')
            )
            
            # Optional: Save standard labels if needed (not in original requirement but good practice)
            # result.export_prediction_visuals(output_dir, file_name=image_name)

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
