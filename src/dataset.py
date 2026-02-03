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
    # Cap CPU count to avoid OOM on high-core machines during local dev
    max_workers = min(cpu_count, 4)
    
    print(f"  - Processing {len(tasks)} images with {max_workers} cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
             import urllib.request
             print("[INFO] Downloading train_val_split.py...")
             urllib.request.urlretrieve("https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py", "train_val_split.py")
        
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
             # Check raw_data_path (custom_data) AND config.BASE_DIR/data (train_val_split default)
             if (raw_data_path / "train").exists() or (config.BASE_DIR / "data" / "train").exists():
                 splits['train'] = master_json
             if (raw_data_path / "validation").exists() or (config.BASE_DIR / "data" / "validation").exists():
                 splits['validation'] = master_json
             if (raw_data_path / "val").exists() or (config.BASE_DIR / "data" / "val").exists():
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
        sliced_base.mkdir(parents=True, exist_ok=True) # Ensure parent exists for temp file

        
        # Check cache
        if sliced_out.exists() and len(list(sliced_out.glob("*.json"))) > 0:
             print(f"  - Found existing slice at {sliced_out}")
             sliced_coco_path = str(list(sliced_out.glob("sliced_*_coco.json"))[0])
        else:
             print(f"  - Slicing to {sliced_out}...")
             
             # Smart Image Dir Detection
             # If train_val_split moved images, they are in raw_data_path/split_name/images
             # OR they are in ./data/split_name/images (hardcoded in the script)
             potential_img_dir = raw_data_path / split_name / "images"
             potential_img_dir_2 = config.BASE_DIR / "data" / split_name / "images"
             
             if potential_img_dir.exists() and any(potential_img_dir.iterdir()):
                 img_dir = potential_img_dir
             elif potential_img_dir_2.exists() and any(potential_img_dir_2.iterdir()):
                 img_dir = potential_img_dir_2
             else:
                 img_dir = json_path.parent
                 
             print(f"  - Using image dir: {img_dir}")
             
             # SANITIZE JSON:
             # The JSON might contain relative paths from Label Studio (e.g., ../../media/...).
             # We need to strip these so SAHI looks for the basename in 'img_dir'.
             import tempfile
             
             with open(json_path, 'r') as jf:
                 coco_data = json.load(jf)
             
             # Get list of available images in the directory
             available_imgs = {p.name for p in img_dir.glob("*")}
             
             filtered_images = []
             kept_ids = set()
             
             for img in coco_data['images']:
                 # Handle both / and \ separators for cross-platform compatibility
                 raw_name = str(img['file_name'])
                 basename = raw_name.replace('\\', '/').split('/')[-1]
                 img['file_name'] = basename # Clean the name
                 
                 if basename in available_imgs:
                     filtered_images.append(img)
                     kept_ids.add(img['id'])
             
             # DEBUG: If filtering removed everything, show why
             if not filtered_images and coco_data['images']:
                 print(f"  - [ERROR] Filter removed all images! Debug info:")
                 print(f"    - Available on disk (first 5): {list(available_imgs)[:5]}")
                 print(f"    - JSON filenames (first 5 processed): {[str(img['file_name']).replace('\\', '/').split('/')[-1] for img in coco_data['images'][:5]]}")
                 print(f"    - JSON raw filenames (first 5): {[img['file_name'] for img in coco_data['images'][:5]]}")
             
             coco_data['images'] = filtered_images
             
             # Filter annotations (optional but cleaner)
             if 'annotations' in coco_data:
                 coco_data['annotations'] = [
                     ann for ann in coco_data['annotations'] 
                     if ann['image_id'] in kept_ids
                 ]
                 
             print(f"  - Filtered JSON: {len(filtered_images)} images found in {img_dir}")
                 
             # Save to temp file
             temp_json_path = sliced_out.parent / f"temp_clean_{split_name}.json"
             with open(temp_json_path, 'w') as jf:
                 json.dump(coco_data, jf)
             
             # If no images, skip slicing
             if not filtered_images:
                 print(f"  - [WARN] No images found for {split_name} split. Skipping.")
                 continue
             
             slice_coco_dict, sliced_coco_path = slice_coco(
                coco_annotation_file_path=str(temp_json_path),
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
