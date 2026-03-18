import os
import glob
import argparse
import math
import numpy as np
import cv2
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
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


class OBBUltralyticsDetectionModel(UltralyticsDetectionModel):
    """
    Extends SAHI's UltralyticsDetectionModel to preserve OBB angle and rotated
    dimensions.  SAHI already stores the four corner points
    (result.obb.xyxyxyxy) in `masks_or_points`, but it never computes the
    angle from them.  We override
    `_create_object_prediction_list_from_original_predictions` to call
    cv2.minAreaRect on those corners and store
    {angle, rect_w, rect_h} in ObjectPrediction.extra_data.

    This adds zero inference overhead — the corner data was already computed
    by Ultralytics internally.
    """

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ):
        """Identical to the parent implementation, but populates extra_data
        with OBB geometry when this is an OBB model."""
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]

        original_predictions = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image = []

        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            if self.has_mask or self.is_obb:
                boxes = image_predictions[0].cpu().detach().numpy()
                masks_or_points = image_predictions[1].cpu().detach().numpy()
            else:
                boxes = image_predictions.data.cpu().detach().numpy()
                masks_or_points = None

            for pred_ind, prediction in enumerate(boxes):
                bbox = prediction[:4].tolist()
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                bbox = [max(0, coord) for coord in bbox]
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    continue

                segmentation = None
                extra_data = {}

                if masks_or_points is not None:
                    if self.has_mask:
                        # Segmentation mask — unchanged from parent
                        from sahi.utils.cv import get_coco_segmentation_from_bool_mask
                        bool_mask = masks_or_points[pred_ind]
                        bool_mask = cv2.resize(
                            bool_mask.astype(np.uint8),
                            (self._original_shape[1], self._original_shape[0])
                        )
                        segmentation = [get_coco_segmentation_from_bool_mask(bool_mask)]
                    else:
                        # OBB: xyxyxyxy corner points, shape (4, 2)
                        obb_points = masks_or_points[pred_ind]  # (4, 2)
                        segmentation = [obb_points.reshape(-1).tolist()]

                        # Derive angle from OBB corners with consistent ordering.
                        # In image coordinates: x increases rightward, y increases downward.
                        try:
                            pts = obb_points.reshape(4, 2)
                            
                            # We no longer perform geometric sorting for deskewing. 
                            # We just use the raw bounding box or minAreaRect points to get the tightest crop.
                            rect = cv2.minAreaRect(pts)
                            box = cv2.boxPoints(rect)
                            
                            # Export the 4 corners for the crop bypassing SAHI AABBs
                            # Add shift_amount to make coordinates relative to the full original image
                            shift_x = shift_amount[0]
                            shift_y = shift_amount[1]
                            corners = [
                                [float(p[0] + shift_x), float(p[1] + shift_y)] for p in box
                            ]
                            
                            # CRITICAL FIX 1: SAHI naively extracts `prediction[:4]` for bbox which is corrupted by OBB cx,cy format.
                            # CRITICAL FIX 2: Bbox sent to SAHI *must* be in local slice coordinates to avoid a double-shift.
                            min_x = max(0.0, float(min(p[0] for p in box)))
                            max_x = max(0.0, float(max(p[0] for p in box)))
                            min_y = max(0.0, float(min(p[1] for p in box)))
                            max_y = max(0.0, float(max(p[1] for p in box)))
                            bbox = [min_x, min_y, max_x, max_y]
                            
                            extra_data = {"corners": corners}
                        except Exception as e:
                            print(f"[WARN] OBBUltralyticsDetectionModel: failed to compute OBB geometry: {e}")

                    if segmentation is not None and len(segmentation) == 0:
                        continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=self._original_shape[:2] if full_shape is None else full_shape,
                )
                if extra_data:
                    object_prediction.extra_data = extra_data
                object_prediction_list.append(object_prediction)

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
        return object_prediction_list_per_image

def run_inference(model_path, source, output_dir, slice_wh=None, overlap_ratio=None, conf_thres=None, json_output=None):
    """
    Runs SAHI sliced inference on a set of images.
    Uses OBBUltralyticsDetectionModel to preserve OBB angle geometry in
    ObjectPrediction.extra_data without any additional inference cost.
    """
    print(f"[INFO] Initializing SAHI with model: {model_path}")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    try:
        detection_model = OBBUltralyticsDetectionModel(
            model_path=model_path,
            confidence_threshold=conf_thres if conf_thres else (getattr(config, 'CONF_THRESHOLD', 0.60) if config else 0.60),
            device=device
        )
        detection_model.load_model()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    # Gather images
    extensions = getattr(config, 'IMAGE_EXTENSIONS', ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']) if config else ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
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
            _slice_h = slice_wh if slice_wh else (getattr(config, 'SLICE_HEIGHT', 1280) if config else 1280)
            _slice_w = slice_wh if slice_wh else (getattr(config, 'SLICE_WIDTH', 1280) if config else 1280)
            _overlap_h = overlap_ratio if overlap_ratio else (getattr(config, 'OVERLAP_HEIGHT_RATIO', 0.4) if config else 0.4)
            _overlap_w = overlap_ratio if overlap_ratio else (getattr(config, 'OVERLAP_WIDTH_RATIO', 0.4) if config else 0.4)
            _perform_std_pred = getattr(config, 'PERFORM_STANDARD_PRED', True) if config else True

            if _perform_std_pred:
                print(f"  [INFO] Global context enabled (full-image standard prediction)")

            # CLEAR SAHI INTERNAL CACHES
            # SAHI's get_sliced_prediction calls perform_inference multiple times.
            # We must wipe the previous image's cached predictions from the model instance
            # to prevent cross-image contamination during the merge phase.
            detection_model._object_prediction_list_per_image = []
            
            # CLEAR RAW PREDICTIONS STATE LEAKAGE
            # Prevent metadata from previous plans from corrupting the current plan.
            detection_model._all_raw_predictions = []

            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=_slice_h,
                slice_width=_slice_w,
                overlap_height_ratio=_overlap_h,
                overlap_width_ratio=_overlap_w,
                perform_standard_pred=_perform_std_pred,
                postprocess_type=getattr(config, 'SAHI_POSTPROCESS_TYPE', 'NMM') if config else 'NMM',
                postprocess_match_metric=getattr(config, 'SAHI_POSTPROCESS_MATCH_METRIC', 'IOS') if config else 'IOS',
                postprocess_match_threshold=getattr(config, 'SAHI_POSTPROCESS_MATCH_THRESHOLD', 0.25) if config else 0.25,
                postprocess_class_agnostic=getattr(config, 'POSTPROCESS_CLASS_AGNOSTIC', True) if config else True,
                verbose=0,
            )

            # Export visualization
            export_path = os.path.join(output_dir, "vis")
            os.makedirs(export_path, exist_ok=True)
            
            result.export_visuals(
                export_dir=export_path,
                file_name=image_name.replace(os.path.splitext(image_name)[1], '')
            )
            # Remove raw OBB Extra Data recovery logic per user request. 
            # We now rely exclusively on the averaged SAHI bboxes.
            if hasattr(detection_model, '_all_raw_predictions'):
                detection_model._all_raw_predictions = []
            

            # Collect results for JSON output.
            # Always gather results — saving to disk is gated separately below.
            # This ensures the API path (Scenario 3) gets angle data even when
            # SAVE_JSON_FOR_DOC_PARSING is False.
            text_labels = getattr(config, 'TEXT_LABELS', []) if config else []
            
            for prediction in result.object_prediction_list:
                label = prediction.category.name
                
                # Filter by label if TEXT_LABELS is defined and not empty
                if text_labels and label not in text_labels:
                    continue

                bbox = prediction.bbox.to_xyxy()
                bbox = [int(x) for x in bbox]
                score = float(prediction.score.value)

                # Read angle/dims populated by OBBUltralyticsDetectionModel.extra_data
                corners = None
                if hasattr(prediction, 'extra_data') and prediction.extra_data:
                    corners = prediction.extra_data.get('corners', None)
                
                json_results.append({
                    "image_path": os.path.abspath(image_path),
                    "bbox": bbox,
                    "corners": corners,
                    "label": label,
                    "score": score
                })

        except Exception as e:
            print(f"[ERROR] Failed to process {image_name}: {e}")

        # --- INCREMENTAL JSON SAVE ---
        # Save JSON to disk after every image so that early termination (Ctrl+C in Colab)
        # still yields usable partial results.
        should_save_json = json_output or (config and getattr(config, 'SAVE_JSON_FOR_DOC_PARSING', False))
        if should_save_json and json_results:
            if json_output:
                json_path = json_output
            else:
                json_path = getattr(config, 'JSON_OUTPUT_PATH', os.path.join(output_dir, "crop_parameters.json"))
            
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            try:
                with open(json_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
                current_img_items = len([r for r in json_results if r['image_path'] == os.path.abspath(image_path)])
                print(f"  [DEBUG] Updated {json_path} (Current image detections: {current_img_items} | Total saved across all images: {len(json_results)})")
            except Exception as e:
                 print(f"[ERROR] Failed to save incremental JSON to {json_path}: {e}")
        # -----------------------------

    print(f"[INFO] Inference complete. Results saved to {output_dir}")

    # --- DONUT CROP EXPORT ---
    if getattr(config, 'SAVE_CROPS_FOR_LABELING', False) and json_results:
        import json as json_mod
        from PIL import Image
        donut_dir = getattr(config, 'DONUT_TUNING_DIR', "/content/drive/MyDrive/SurveyPlan AI/runs/donut_tuning")
        os.makedirs(donut_dir, exist_ok=True)
        metadata_path = os.path.join(donut_dir, "metadata.jsonl")
        
        print(f"[INFO] Exporting text crops for Donut labeling to: {donut_dir}")
        
        # Load existing counts from metadata.jsonl
        class_counts = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json_mod.loads(line)
                        label = record.get("label")
                        if label:
                            class_counts[label] = class_counts.get(label, 0) + 1
            except Exception as e:
                print(f"[WARNING] Could not parse existing metadata.jsonl for counts: {e}")
        
        max_per_class = getattr(config, 'MAX_CROPS_PER_CLASS', 40)
        
        grouped_by_img = {}
        for item in json_results:
            img_p = item["image_path"]
            if img_p not in grouped_by_img:
                grouped_by_img[img_p] = []
            grouped_by_img[img_p].append(item)
            
        crop_count = 0
        try:
            with open(metadata_path, 'a', encoding='utf-8') as mf:
                for img_p, items in grouped_by_img.items():
                    if not os.path.exists(img_p):
                        continue
                    try:
                        img = Image.open(img_p).convert("RGB")
                        base_name = os.path.splitext(os.path.basename(img_p))[0]
                        
                        for idx, item in enumerate(items):
                            bbox = item["bbox"]
                            label = item["label"]
                            
                            if class_counts.get(label, 0) >= max_per_class:
                                continue
                                
                            # PIL crop format: (left, upper, right, lower)
                            crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                            
                            crop_filename = f"{base_name}_{label.replace(' ', '_')}_{idx}.jpg"
                            crop_img.save(os.path.join(donut_dir, crop_filename))
                            
                            # Write template to metadata.jsonl with label for tracking
                            record = {
                                "file_name": crop_filename,
                                "label": label,
                                "ground_truth": json_mod.dumps({"gt_parse": {"text": ""}}, ensure_ascii=False)
                            }
                            mf.write(json_mod.dumps(record, ensure_ascii=False) + "\n")
                            class_counts[label] = class_counts.get(label, 0) + 1
                            crop_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to extract crop from {img_p}: {e}")
            print(f"[SUCCESS] Exported {crop_count} crops for labeling.")
        except Exception as e:
            print(f"[ERROR] Failed to save crops to {donut_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run SAHI Sliced Inference using YOLO model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained YOLO .pt model')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--output_dir', type=str, default='runs/sahi_predict', help='Directory to save results')
    parser.add_argument('--slice_wh', type=int, default=None, help='Slice width and height (square, defaults to config)')
    parser.add_argument('--overlap', type=float, default=None, help='Overlap ratio (0-1, defaults to config)')
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold (defaults to config)')
    parser.add_argument('--json_output', type=str, default=None, help='Specific path to save the JSON output')
    
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        source=args.source,
        output_dir=args.output_dir,
        slice_wh=args.slice_wh,
        overlap_ratio=args.overlap,
        conf_thres=args.conf,
        json_output=args.json_output
    )

if __name__ == "__main__":
    main()
