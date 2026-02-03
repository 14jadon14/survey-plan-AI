import os
import shutil
import json
import yaml
import multiprocessing
import concurrent.futures
import numpy as np
import cv2
import sys
from pathlib import Path
from sahi.slicing import slice_coco
from shapely.geometry import Polygon

from src import config

def process_single_image_obb(args):
    """Worker function to process a single image for OBB conversion"""
    img_info, annotations, src_img_dir, out_img_dir, out_lbl_dir = args
    
    # 1. Copy Image
    src_img_path = src_img_dir / img_info['file_name']
    dst_img_path = out_img_dir / img_info['file_name']
    
    if not dst_img_path.exists():
        if src_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
        else:
            return None # Skip missing images
            
    # 2. Convert Poly -> OBB
    lines = []
    img_w, img_h = img_info['width'], img_info['height']
    
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            # Handle potential nested lists in segmentation
            seg = ann['segmentation'][0]
            poly = np.array(seg).reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(poly)
            box = cv2.boxPoints(rect)

            # Normalize
            norm_box = []
            for pt in box:
                norm_box.extend([
                    max(0, min(1, pt[0] / img_w)),
                    max(0, min(1, pt[1] / img_h))
                ])
            lines.append(f"{ann['category_id']} {' '.join(f'{x:.6f}' for x in norm_box)}")

    # Write label file
    txt_path = out_lbl_dir / (Path(img_info['file_name']).stem + ".txt")
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
        
    return 1 # Success count

def convert_to_obb_parallel(json_file, out_img_dir, out_lbl_dir):
    out_img_dir = Path(out_img_dir)
    out_lbl_dir = Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(json_file) as f:
        data = json.load(f)

    cats = {c['id']: c['name'] for c in data['categories']}
    src_dir = Path(json_file).parent
    
    # Prepare args for parallel execution
    tasks = []
    for img in data['images']:
        img_anns = [a for a in data['annotations'] if a['image_id'] == img['id']]
        tasks.append((img, img_anns, src_dir, out_img_dir, out_lbl_dir))
        
    # Execute in parallel
    cpu_count = os.cpu_count() or 1
    print(f"  - Processing {len(tasks)} images with {cpu_count} cores...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_image_obb, tasks))
        
    return cats

def prepare_data(data_path_arg=None):
    """
    Orchestrates the data preparation pipeline.
    
    Args:
        data_path_arg (str/Path): Optional override for the raw data location.
                                  Useful if data is in /content/data on Colab.
    """
    
    # Determine paths
    raw_data_path = Path(data_path_arg) if data_path_arg else config.CUSTOM_DATA_DIR
    
    print(f"[INFO] Preparing data from: {raw_data_path}")
    
    # 1. Unzip Logic (Handled externally in Colab, but check locally)
    # If raw_data_path is expected to be populated but isn't, warn or fail.
    if not raw_data_path.exists():
        print(f"[WARN] Warning: Data path {raw_data_path} does not exist. Assuming it will be mounted or unzipped.")
        
    # 2. Train/Val Split
    # We essentially re-implement the logic or call the script if needed.
    # For robust modularity, let's check if 'train' and 'validation' folders exist in raw_data_path
    train_dir = raw_data_path / "train"
    val_dir = raw_data_path / "validation"
    
    if not train_dir.exists() or not val_dir.exists():
        print("Build step: Splitting data...")
        # Download script if missing
        split_script = Path("train_val_split.py")
        if not split_script.exists():
             os.system("wget -O train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py")
        
        # Run split
        cmd = f'{sys.executable} train_val_split.py --datapath="{raw_data_path}" --train_pct=0.9'
        os.system(cmd)
        
    # 3. Slicing & OBB Conversion
    detected_classes = {}
    
    # We need to map the split folder names to what the slicing logic expects
    # The split script creates: 'train/images', 'validation/images'
    # We want to process these.
    
    # Find JSON annotations. Usually in raw_data_path/train/labels.json or similar?
    # Wait, the original notebook expected 'train/_annotations.coco.json' inside the splits?
    # Actually, the original notebook unzipped to 'custom_data', then ran split.
    # The split script MOVES images. It doesn't handle JSONs usually?
    # Checked original notebook: The user has COCO json.
    
    # Let's assume the standard Roboflow/CVAT export structure where existing JSONs are used.
    # In the notebook, `found_img_dir` was detected.
    
    # For this refactor, let's assume the user provides a direct path to the JSON if it exists,
    # OR we look for it in standard locations.
    
    # Re-reading notebook logic:
    # splits = {'train': ...json_path, 'val': ...json_path}
    
    # Let's search for JSONs in the train/val directories
    splits = {}
    for split_name in ['train', 'validation']:
        d = raw_data_path / split_name
        json_files = list(d.glob("*.json"))
        if json_files:
            splits[split_name] = json_files[0]
            
    # If no JSONs found in subfolders, maybe they are in root?
    # Fallback: Check root for master JSON (common in Roboflow/CVAT exports before splitting)
    if not splits:
        print("[INFO] No split-specific JSONs found. Checking for master JSON...")
        # Look for typical names
        master_jsons = list(raw_data_path.glob("*.json"))
        if master_jsons:
             master_json = master_jsons[0]
             print(f"[INFO] Found master JSON: {master_json.name}. Will attempt to use it for both splits.")
             # We assume the images have been moved to train/images and validation/images by step 2
             # We can re-use the same JSON path for both, assuming the slicer just skips missing images.
             
             # Verify which split folders actually exist with images
             if (raw_data_path / "train").exists():
                 splits['train'] = master_json
             if (raw_data_path / "validation").exists():
                 splits['validation'] = master_json
             if (raw_data_path / "val").exists():
                 splits['val'] = master_json
        else:
             print("[ERROR] No JSON annotations found in data directory!")
             return obb_base, {}

    # Slicing
    sliced_base = config.SLICED_DATA_DIR
    obb_base = config.OBB_DATA_DIR
    
    for split_name, json_path in splits.items():
        print(f"Processing {split_name}...")
        sliced_out = sliced_base / split_name
        
        # Check cache
        if sliced_out.exists() and len(list(sliced_out.glob("*.json"))) > 0:
             print(f"  - Found existing slice at {sliced_out}")
             sliced_coco_path = str(list(sliced_out.glob("sliced_*_coco.json"))[0])
        else:
             print(f"  - Slicing to {sliced_out}...")
             img_dir = json_path.parent # Attributes image location to json location
             # Often in COCO exports, images are side-by-side with json
             
             slice_coco_dict, sliced_coco_path = slice_coco(
                coco_annotation_file_path=str(json_path),
                image_dir=str(img_dir),
                output_coco_annotation_file_name=f"sliced_{split_name}",
                output_dir=str(sliced_out),
                slice_height=config.SLICE_HEIGHT,
                slice_width=config.SLICE_WIDTH,
                overlap_height_ratio=config.OVERLAP_HEIGHT_RATIO,
                overlap_width_ratio=config.OVERLAP_WIDTH_RATIO,
                verbose=False
            )
            
        # OBB Conversion
        print(f"  - Converting to YOLO OBB...")
        classes = convert_to_obb_parallel(
            sliced_coco_path,
            obb_base / split_name / "images",
            obb_base / split_name / "labels"
        )
        detected_classes.update(classes)
        
    return obb_base, detected_classes

def create_yaml(obb_dir, classes):
    val_folder = 'validation' if (obb_dir / 'validation').exists() else 'val'
    
    yaml_config = {
        'path': str(obb_dir.absolute()),
        'train': 'train/images',
        'val': f'{val_folder}/images',
        'names': classes
    }
    
    yaml_path = config.BASE_DIR / 'survey_plan_obb.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f)
        
    return yaml_path
