import os
import json
import re
from pathlib import Path

# Mapping of raw keys to Donut schema keys
LEAF_KEY_MAP = {
    "azimuth": "az",
    "distance": "dist",
    "lot_number": "lot_id",
    "adj_lot": "adj_id",
    "area": "area_val",
    "notes": "text",
    "street": "street",
    "plan_title": "plan_title",
    "title_data": "title_data"
}

def extract_keys(filename):
    """
    Extracts the leaf key from a filename.
    """
    match = re.search(r'page-\d{4}_(.*)_\d+$', filename)
    if match:
        return match.group(1).replace('_', ' ') 
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[-2]
    return "text"

def parse_curve_data_block(content):
    """Parses a multi-line unstructured curve data block."""
    parsed = {}
    arc_match = re.search(r'[A|Arc][=:\s]?([\d.]+)', content, re.IGNORECASE)
    if arc_match: parsed["arc"] = arc_match.group(1)
    
    rad_match = re.search(r'[R|Rad][=:\s]?([\d.]+)', content, re.IGNORECASE)
    if rad_match: parsed["radius"] = rad_match.group(1)
    
    az_match = re.search(r'(\d{1,3}-\d{1,2}-\d{1,2})', content)
    if az_match: parsed["az"] = az_match.group(1)
    
    floats = re.findall(r'\b\d+\.\d+\b', content)
    for f in floats:
        if f != parsed.get("arc") and f != parsed.get("radius"):
            parsed["chord"] = f
            break
            
    if not parsed:
        parsed["text"] = content
    return parsed

def parse_tabular_data(table_type, content):
    """Parses tabular text into a list of row dicts."""
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    rows = []
    
    if "coord" in table_type:
        for line in lines:
            if "POINT" in line.upper() or "EAST" in line.upper() or "NORTH" in line.upper(): continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append({"pt_id": parts[0], "east": parts[1], "north": parts[2]})
                
    elif "curve" in table_type:
        for line in lines:
            if "CURVE" in line.upper() or "RADIUS" in line.upper(): continue
            parts = line.split()
            if len(parts) < 3: continue
            row_dict = {}
            
            az_idx = next((i for i, p in enumerate(parts) if '-' in p and len(p.split('-'))==3), -1)
            if az_idx != -1: row_dict["az"] = parts[az_idx]
            
            floats = []
            for p in parts:
                try: 
                    # check if float
                    float(p.replace(',',''))
                    floats.append(p)
                except ValueError: 
                    pass
                
            if len(floats) >= 3:
                # Heuristic mapping
                row_dict["radius"] = floats[0]
                row_dict["arc"] = floats[1]
                row_dict["chord"] = floats[2]
            elif len(floats) == 2:
                row_dict["radius"] = floats[0]
                row_dict["arc"] = floats[1]
                
            if row_dict:
                rows.append(row_dict)
            
    elif "line" in table_type:
        # Check if it is a vertical line table
        if len(lines) == 3 and '-' in lines[0] and '-' in lines[2] and '.' in lines[1]:
            rows.append({"line_id": lines[0], "dist": lines[1], "az": lines[2]})
        else:
            for line in lines:
                if "LINE" in line.upper(): continue
                parts = line.split()
                if len(parts) >= 2:
                    az_idx = next((i for i, p in enumerate(parts) if '-' in p and len(p.split('-'))==3), -1)
                    r = {}
                    if az_idx != -1: r["az"] = parts[az_idx]
                    
                    floats = [p for p in parts if '.' in p and '-' not in p]
                    if floats: r["dist"] = floats[0]
                    if r: rows.append(r)
                    
    return rows if rows else [{"text": content}] # fallback if completely unparseable

def process_dataset(dataset_dir, output_file=None):
    dataset_path = Path(dataset_dir)
    if output_file is None:
        output_file = dataset_path / "metadata.jsonl"
    else:
        output_file = Path(output_file)
        
    entries = []
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory {dataset_dir} does not exist.")
        return

    for subfolder in sorted(dataset_path.iterdir()):
        if subfolder.is_dir() and subfolder.name.startswith('s_'):
            root_key = subfolder.name[2:] 
            print(f"Processing folder: {subfolder.name} (root_key: {root_key})")
            
            for img_file in subfolder.iterdir():
                if img_file.suffix.lower() in img_exts:
                    txt_file = img_file.with_suffix('.txt')
                    if not txt_file.exists(): continue
                    
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                    except Exception as e:
                        print(f"Warning: Could not read {txt_file}: {e}")
                        continue
                    
                    leaf_key = extract_keys(img_file.stem)
                    original_leaf = leaf_key.replace(" ", "_")
                    
                    if "curve_table" in original_leaf or "line_table" in original_leaf:
                        continue
                        
                    schema_key = LEAF_KEY_MAP.get(original_leaf, original_leaf)
                    
                    gt_dict = {}
                    if root_key == "tabular_data":
                        rows = parse_tabular_data(original_leaf, content)
                        # Tabular data expects a "row" key containing a list
                        gt_dict = {"gt_parse": {root_key: {"row": rows}}}
                    elif root_key == "lot_geometry" and original_leaf == "curve_data":
                        parsed = parse_curve_data_block(content)
                        gt_dict = {"gt_parse": {root_key: parsed}}
                    else:
                        clean_content = content.replace('\n', ' ')
                        if schema_key == "az":
                            clean_content = clean_content.replace('°', '-').replace("'", '-').replace('"', '-').replace('.', '-')
                        gt_dict = {"gt_parse": {root_key: {schema_key: clean_content}}}
                    
                    entry = {
                        "file_name": img_file.name,
                        "label": original_leaf.replace("_", " "),
                        "ground_truth": json.dumps(gt_dict, ensure_ascii=False)
                    }
                    entries.append(entry)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\nSUCCESS: Generated {len(entries)} entries in {output_file}")
    except Exception as e:
        print(f"ERROR: Could not write to {output_file}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Donut metadata.jsonl from subfolders.")
    parser.add_argument("--dir", default="C:/Users/Jadon/Downloads/dataset", help="Dataset root directory")
    parser.add_argument("--out", help="Output JSONL file path (default: metadata.jsonl in dataset root)")
    args = parser.parse_args()
    
    process_dataset(args.dir, args.out)
