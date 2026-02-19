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
    
    # Safe print
    print(f"[INFO] Preparing data from: {repr(str(raw_data_path))}")
    
    # 1. Unzip Logic (Handled externally in Colab, but check locally)
    # If raw_data_path is expected to be populated but isn't, warn or fail.
    if not raw_data_path.exists():
        print(f"[WARN] Warning: Data path {repr(str(raw_data_path))} does not exist. Assuming it will be mounted or unzipped.")
        
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
        cmd = f'{sys.executable} train_val_split.py --datapath="{raw_data_path}" --train_pct=0.7'
        os.system(cmd)
        
    # 3. Slicing & OBB Conversion
    detected_classes = {}
    
    # We need to map the split folder names to what the slicing logic expects
    # The split script creates: 'train/images', 'validation/images'
    # We want to process these.
    
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
             
             # Verify which split folders actually exist with images
             # Check raw_data_path (custom_data) AND config.BASE_DIR/data (train_val_split default)
             if (raw_data_path / "train").exists() or (config.BASE_DIR / "data" / "train").exists() or Path("/content/data/train").exists():
                 splits['train'] = master_json
             if (raw_data_path / "validation").exists() or (config.BASE_DIR / "data" / "validation").exists() or Path("/content/data/validation").exists():
                 splits['validation'] = master_json
             if (raw_data_path / "val").exists() or (config.BASE_DIR / "data" / "val").exists() or Path("/content/data/val").exists():
                 splits['val'] = master_json

        else:
             print("[ERROR] No JSON annotations found in data directory!")
             return config.OBB_DATA_DIR, {}

    # Slicing
    sliced_base = config.SLICED_DATA_DIR
    obb_base = config.OBB_DATA_DIR
    
    for split_name, json_path in splits.items():
        print(f"Processing {split_name}...")
        sliced_out = sliced_base / split_name
        sliced_base.mkdir(parents=True, exist_ok=True) # Ensure parent exists for temp file

        temp_json_path = sliced_out.parent / f"temp_clean_{split_name}.json"

        # Smart Image Dir Detection
        potential_img_dir = raw_data_path / split_name / "images"
        potential_img_dir_2 = config.BASE_DIR / "data" / split_name / "images"
        potential_img_dir_3 = Path("/content/data") / split_name / "images"
        
        if potential_img_dir.exists() and any(potential_img_dir.iterdir()):
            img_dir = potential_img_dir
        elif potential_img_dir_2.exists() and any(potential_img_dir_2.iterdir()):
            img_dir = potential_img_dir_2
        elif potential_img_dir_3.exists() and any(potential_img_dir_3.iterdir()):
            img_dir = potential_img_dir_3
        else:
            img_dir = json_path.parent
            
        print(f"  - Using image dir: {repr(str(img_dir))}")

        # Check cache
        if sliced_out.exists() and len(list(sliced_out.glob("*.json"))) > 0 and temp_json_path.exists():
             print(f"  - Found existing slice at {repr(str(sliced_out))}")
             sliced_coco_path = str(list(sliced_out.glob("sliced_*_coco.json"))[0])
        else:
             print(f"  - Slicing to {repr(str(sliced_out))}...")
             
             # SANITIZE JSON
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
                 # Safe printing of debug info
                 print(f"    - Available on disk (first 5): {[repr(p) for p in list(available_imgs)[:5]]}")
             
             coco_data['images'] = filtered_images
             
             # Filter annotations
             if 'annotations' in coco_data:
                 coco_data['annotations'] = [
                     ann for ann in coco_data['annotations'] 
                     if ann['image_id'] in kept_ids
                 ]
                 
             print(f"  - Filtered JSON: {len(filtered_images)} images found in {repr(str(img_dir))}")
                 
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
        
        if config.HYBRID_TRAINING:
             print(f"  - [Hybrid] Generating Stream B (Global Context)...")
             global_json = generate_global_views(
                 json_path=temp_json_path, # Use the clean temp JSON
                 img_dir=img_dir,
                 out_img_dir=sliced_out / "global_images", 
                 target_size=config.GLOBAL_RESIZE_SIZE
             )
             
             print(f"  - [Hybrid] Converting Global to OBB...")
             # Output to the SAME obb_base folder to merge them
             classes_global = convert_to_obb_parallel(
                 global_json,
                 obb_base / split_name / "images",
                 obb_base / split_name / "labels"
             )
             detected_classes.update(classes_global)
             
    # 4. Final Safety Check
    if not detected_classes:
        # If we failed to detect classes from the conversion (maybe no annotations?),
        # try to pull them from the raw JSON directly as a fallback.
        print("[WARN] detected_classes is empty after processing. Attempting fallback...")
        fallback_json = None
        if splits:
            fallback_json = list(splits.values())[0]
        elif 'master_json' in locals():
            fallback_json = master_json
        
        if fallback_json:
            try:
                with open(fallback_json, 'r') as f:
                    data = json.load(f)
                    detected_classes = {c['id']: c['name'] for c in data.get('categories', [])}
                print(f"[INFO] Fallback: Loaded {len(detected_classes)} classes from JSON header: {detected_classes}")
            except Exception as e:
                print(f"[ERROR] Fallback failed: {e}")

    if not detected_classes:
        raise ValueError(f"Could not detect any classes from dataset at {raw_data_path}. Please check your JSON 'categories' field.")

    print(f"[INFO] Final Detected Classes: {detected_classes}")
    return obb_base, detected_classes

def generate_global_views(json_path, img_dir, out_img_dir, target_size=1024):
    """
    Stream B: Generates resized global views of the survey plans.
    """
    import cv2
    import numpy as np
    
    print(f"  - [Global] Generating global views at {target_size}x{target_size}...")
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    img_map = {img['id']: img for img in coco['images']}
    global_coco = {'categories': coco['categories'], 'images': [], 'annotations': []}
    
    processed_count = 0
    id_map = {} # old_id -> new_id
    
    for i, img in enumerate(coco['images']):
        old_id = img['id']
        new_id = i + 1
        id_map[old_id] = new_id
        
        fname = img['file_name']
        src_path = img_dir / fname
        
        if not src_path.exists():
            continue
            
        # Read
        original_img = cv2.imread(str(src_path))
        if original_img is None:
            continue
            
        h, w = original_img.shape[:2]
        
        # Resize
        resized_img = cv2.resize(original_img, (target_size, target_size))
        
        # Save
        new_fname = f"global_{fname}"
        dst_path = out_img_dir / new_fname
        cv2.imwrite(str(dst_path), resized_img)
        
        # Add to JSON
        global_coco['images'].append({
            'id': new_id,
            'file_name': new_fname,
            'width': target_size,
            'height': target_size
        })
        processed_count += 1

    # Process Annotations
    if 'annotations' in coco:
        for ann in coco['annotations']:
            if ann['image_id'] not in id_map:
                continue
                
            old_img_id = ann['image_id']
            img_info = img_map[old_img_id]
            orig_w, orig_h = img_info['width'], img_info['height']
            
            # Scale Factor
            sx = target_size / orig_w
            sy = target_size / orig_h
            
            new_ann = ann.copy()
            new_ann['image_id'] = id_map[old_img_id]
            new_ann['id'] = len(global_coco['annotations']) + 1
            
            # Scale Segmentation
            if 'segmentation' in ann:
                # segmentation is list of lists
                new_segs = []
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.float32)
                    poly[:, 0] *= sx
                    poly[:, 1] *= sy
                    new_segs.append(poly.flatten().tolist())
                new_ann['segmentation'] = new_segs
            
            # Scale Bbox (x, y, w, h)
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                new_ann['bbox'] = [x*sx, y*sy, w*sx, h*sy]
                
            global_coco['annotations'].append(new_ann)

    # Save Global JSON
    global_json_path = out_img_dir / "global_coco.json"
    with open(global_json_path, 'w') as f:
        json.dump(global_coco, f)
        
    print(f"  - [Global] Created {processed_count} global images.")
    return global_json_path

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
