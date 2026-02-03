import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    print("[INFO] Starting Local Smoke Test...")
    
    base_dir = Path.cwd()
    data_zip = base_dir / "data" / "data.zip"
    custom_data = base_dir / "custom_data"
    
    # 0. Clean Clean Up Previous Runs
    # We must remove existing processed data so that the new limited dataset is processed.
    dirs_to_clean = [
        base_dir / "sliced_dataset", 
        base_dir / "survey_obb_dataset",
        base_dir / "data" / "train",
        base_dir / "data" / "validation",
        base_dir / "data" / "val"
    ]
    for p in dirs_to_clean:
        if p.exists():
            print(f"[INFO] cleaning up {p}...")
            shutil.rmtree(p, ignore_errors=True)
            
    # 1. Check Data
    if not custom_data.exists():
        if data_zip.exists():
            print(f"[INFO] Unzipping {data_zip} to {custom_data}...")
            # Using shutil for cross-platform support
            shutil.unpack_archive(str(data_zip), str(custom_data))
        else:
            print(f"[ERROR] Error: Could not find '{custom_data}' or '{data_zip}'.")
            print("   Please place 'data.zip' in this folder to run the test.")
            sys.exit(1)
    else:
        print(f"[INFO] Data found at {custom_data}")
        
    # 1.5 Limit Dataset for Memory Safety
    # We delete excess images to ensuring the test runs quickly and with low RAM
    print("[INFO] Optimizing dataset for local testing (keeping max 3 images per folder)...")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for root, dirs, files in os.walk(custom_data):
        img_count = 0
        for f in files:
            f_path = Path(root) / f
            if f_path.suffix.lower() in valid_exts:
                img_count += 1
                if img_count > 3:
                    try:
                        f_path.unlink()
                    except OSError:
                        pass
    print("[INFO] Dataset optimization complete.")
        
    print("[INFO] Dataset optimization complete.")
        
    # 2. Run Training (Minimal Config)
    print("\n[INFO] Running training script (1 Epoch, Batch 2)...")
    train_script = base_dir / "src" / "train.py"

    # --- REGRESSION TEST: Use Hybrid Mode ---
    # We rely on src/config.py having HYBRID_TRAINING = True by default now.
    
    # Construct command
    # python src/train.py --epochs 1 --batch 2 --data_path custom_data
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(base_dir) # Ensure src can be imported
    
    cmd = [
        sys.executable, 
        str(train_script),
        "--epochs", "1",
        "--batch", "2",
        "--imgsz", "160",
        "--data_path", str(custom_data)
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        
        # --- VERIFICATION ---
        print("\n[TEST] Verifying Hybrid Training Outputs...")
        obb_train = base_dir / "survey_obb_dataset" / "train" / "images"
        
        # Check for global images
        global_imgs = list(obb_train.glob("global_*.jpg")) + list(obb_train.glob("global_*.png"))
        if not global_imgs:
            print("[FAIL] Hybrid Training Logic Failed: No 'global_*' images found in output.")
            sys.exit(1)
        else:
            print(f"[PASS] Hybrid Training: Found {len(global_imgs)} global context images.")
            
        # Check for slices
        slice_imgs = list(obb_train.glob("*_slice_*.jpg")) + list(obb_train.glob("*_slice_*.png"))
        if not slice_imgs:
             print("[WARN] No slices found? (Maybe dataset too small for slicing or threshold high)")
        else:
             print(f"[PASS] Slicing: Found {len(slice_imgs)} sliced images.")

        print("\n[SUCCESS] Local Test PASSED! Hybrid Training logic verified.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Local Test FAILED with exit code {e.returncode}.")
        print("   Check the output above for errors.")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
