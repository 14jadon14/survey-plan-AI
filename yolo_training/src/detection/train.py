import argparse
from src import config
from src.detection import dataset, model

def main():
    parser = argparse.ArgumentParser(description="Train YOLO Model for Survey Plans")
    parser.add_argument('--data_path', type=str, default=None, help='Path to raw dataset (contains train/val folders)')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=config.IMGSZ, help='Image size')
    
    args = parser.parse_args()
    
    # 1. Prepare Data
    obb_dir, classes = dataset.prepare_data(args.data_path)
    print(f"[INFO] Data prepared at: {repr(str(obb_dir))}")
    print(f"[INFO] Classes detected: {classes}")
    if not classes:
        print("[ERROR] Classes dictionary is EMPTY! This will cause training to fail.")
    
    # 2. Config
    yaml_path = dataset.create_yaml(obb_dir, classes)
    print(f"[INFO] Config saved to: {repr(str(yaml_path))}")
    with open(yaml_path, 'r') as f:
        print(f"[DEBUG] YAML Content:\n{f.read()}")
    
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
        print("⚠️  CUDA not available. Training on CPU will be slow.")
    else:
        print(f"✅  CUDA available: {torch.cuda.get_device_name(0)}")

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
        cos_lr=True,
        amp=True,
        close_mosaic=10
    )
    
    print("[INFO] Training complete!")

if __name__ == "__main__":
    main()
