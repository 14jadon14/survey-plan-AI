import argparse
from src import config
from src.detection import dataset, model

import sys
import io

# Monkey-patch to fix UnicodeEncodeError in Colab/Jupyter with emojis/surrogates
# This ensures that ANY print from ANY library (Ultralytics, etc.) gets sanitized.
class SafeWriter(io.TextIOWrapper):
    def __init__(self, buffer, encoding='utf-8', errors='replace', **kwargs):
        super().__init__(buffer, encoding=encoding, errors=errors, **kwargs)

    def write(self, s):
        try:
            return super().write(s)
        except UnicodeEncodeError:
            # Fallback: strip incompatible characters
            return super().write(s.encode('utf-8', 'ignore').decode('utf-8'))

# Wrap standard streams
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = SafeWriter(sys.stdout.buffer)
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = SafeWriter(sys.stderr.buffer)

def main():
    parser = argparse.ArgumentParser(description="Train YOLO Model for Survey Plans")
    parser.add_argument('--data_path', type=str, default=None, help='Path to raw dataset (contains train/val folders)')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=config.IMGSZ, help='Image size')
    
    args = parser.parse_args()
    
    # 1. Prepare Data
    obb_dir, classes = dataset.prepare_data(args.data_path)
    print(f"[INFO] Data prepared at: {obb_dir}")
    print(f"[INFO] Classes: {classes}")
    
    # 2. Config
    yaml_path = dataset.create_yaml(obb_dir, classes)
    print(f"[INFO] Config saved to: {yaml_path}")
    
    # 3. Model
    yolo = model.get_model()
    
    # 4. Train
    print("[INFO] Starting training...")
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("[WARN]  CUDA not available. Training on CPU will be slow.")
    else:
        print(f"[INFO]  CUDA available: {torch.cuda.get_device_name(0)}")

    try:
        results = yolo.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=config.PROJECT_NAME,
            device=device,
            workers=8 if IN_COLAB else 0,
            degrees=45,
            fliplr=0.0,
            cache=True,
            cos_lr=True
        )
    except Exception as e:
        # Sanitize error message to remove emojis which crash Colab/Jupyter
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"[ERROR] Training failed: {error_msg}")
        # Re-raise nicely or exit? Exit to avoid ugly tracebacks in notebook loop
        exit(1)
    
    print("[INFO] Training complete!")

if __name__ == "__main__":
    main()
