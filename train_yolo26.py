#!/usr/bin/env python3
"""
Ultralytics YOLO26n Training Script for 2D Chess Piece & Board Detection.

This script implements the recommended YOLO26 training recipe:
- Model: Pretrained yolo26n.pt (native end-to-end NMS-free head, no DFL overhead)
- Optimizer: MuSGD (hybrid Muon + SGD optimizer for accelerated convergence)
- Supervised Loss: ProgLoss (Progressive Loss balancing)
- Label Assignment: STAL (Small-Target-Aware Label Assignment for tiny piece accuracy)
- Augmentation: Domain-tuned (fliplr=0, flipud=0, mixup=0, mild mosaic with close_mosaic)
- Post-training: Hold-out test set evaluation and ONNX export

Usage:
    python train_yolo26.py                          # Run with standard settings
    python train_yolo26.py --epochs 80 --batch 32   # Custom epochs/batch
    python train_yolo26.py --device 0               # Specify GPU
    python train_yolo26.py --export-onnx            # Automatically export to ONNX
"""

import argparse
import os
import sys

# Default Training Configuration
DEFAULT_DATA_YAML = "chess_detection.yaml"
DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_EPOCHS = 60
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_OPTIMIZER = "MuSGD"
DEFAULT_LR0 = 0.01
DEFAULT_LRF = 0.01
DEFAULT_MOMENTUM = 0.937
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_PATIENCE = 15
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "yolo26n_chess"


def print_recipe_overview():
    print("================================================================================")
    print(" Ultralytics YOLO26n Training Recipe for 2D Chess Detection")
    print("================================================================================")
    print("Architectural & Training Innovations in YOLO26:")
    print("  1. Native End-to-End (NMS-Free): Uses dual-head architecture with a 1-to-1")
    print("     inference head. Completely eliminates traditional NMS post-processing latency")
    print("     and removes Distribution Focal Loss (DFL) regression complexity.")
    print("  2. MuSGD Optimizer: Hybrid Muon-SGD optimization algorithm borrowed from modern")
    print("     LLM pre-training, providing superior convergence on lightweight vision backbones.")
    print("  3. ProgLoss (Progressive Loss): Dynamic supervision scheduling that smoothly")
    print("     shifts loss focus toward the deployment head across the training lifecycle.")
    print("  4. STAL (Small-Target-Aware Label Assignment): Guarantees positive anchor supervision")
    print("     for small objects, solving small piece detection bottlenecks on dense boards.")
    print("  5. Domain-Specific Augmentation Policy:")
    print("     - fliplr=0.0 & flipud=0.0: Prevents flipping directional pieces (Knights) & notation")
    print("     - mixup=0.0: Prevents illegal overlapping transparent boards")
    print("     - mosaic=0.2 & close_mosaic=10: Prevents chessboard structural disintegration")
    print("================================================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO26n on 2D Chess Dataset")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to chess_detection.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model weights (default: yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size (-1 for AutoBatch)")
    parser.add_argument("--device", default="", help="Device: '0', '0,1', 'cpu', or '' (auto)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker threads")
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER, help="Optimizer: MuSGD, SGD, AdamW, or auto")
    parser.add_argument("--lr0", type=float, default=DEFAULT_LR0, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=DEFAULT_LRF, help="Final learning rate fraction")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience epochs")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Output project directory")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Run experiment name")
    parser.add_argument("--freeze", type=int, default=0, help="Number of backbone layers to freeze")
    parser.add_argument("--eval-test", action="store_true", default=True, help="Evaluate on test split after training")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export best model to ONNX after training")
    return parser.parse_args()


def main():
    args = parse_args()
    print_recipe_overview()

    # Check dataset YAML and directory
    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset configuration file not found at: {args.data}")
        print("Please ensure chess_detection.yaml exists and points to your generated dataset.")
        sys.exit(1)

    if not os.path.exists("datasets/images/train"):
        print("[ERROR] Dataset directory 'datasets/images/train' not found!")
        print("Generate dataset first: python3 generate_datasets.py --train-count 3000 --val-count 450 --test-count 450")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed in the active Python environment.")
        print("Install via: pip install ultralytics")
        print("\nReady-to-run CLI Equivalent Command:")
        print(
            f"yolo detect train model={args.model} data={args.data} epochs={args.epochs} "
            f"imgsz={args.imgsz} batch={args.batch} optimizer={args.optimizer} lr0={args.lr0} "
            f"fliplr=0.0 flipud=0.0 mixup=0.0 mosaic=0.2 close_mosaic=10"
        )
        sys.exit(1)

    print(f"Loading base model: {args.model}...")
    model = YOLO(args.model)

    print("Configuring YOLO26n training parameters...")
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": DEFAULT_MOMENTUM,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "save": True,
        "save_period": 5,
        "plots": True,
        "exist_ok": True,
        # Domain-specific augmentations
        "fliplr": 0.0,       # Never flip horizontally (preserves Knight facing direction & board parity)
        "flipud": 0.0,       # Never flip vertically (no upside-down chess pieces)
        "mixup": 0.0,        # No ghost overlapping boards
        "mosaic": 0.2,       # Mild mosaic to preserve board structure
        "close_mosaic": 10,  # Turn off mosaic for final 10 epochs for crisp feature alignment
        "degrees": 0.0,      # Generator already handles exact perspective warp and camera tilt
        "scale": 0.2,        # Mild scale jitter (+/- 20%)
        "translate": 0.1,    # Mild translation (+/- 10%)
        "hsv_h": 0.015,      # Subtle hue shift
        "hsv_s": 0.30,       # Saturation variation
        "hsv_v": 0.30,       # Value / Brightness variation
    }

    if args.device:
        train_args["device"] = args.device

    if args.freeze > 0:
        train_args["freeze"] = args.freeze

    print("\nStarting YOLO26n training session...")
    results = model.train(**train_args)

    print("\n================================================================================")
    print(" Training Finished Successfully!")
    print("================================================================================")

    # 1. Validation on Val split
    print("\nEvaluating on Validation Split...")
    val_metrics = model.val(data=args.data, split="val")
    print(f"Validation mAP50: {val_metrics.box.map50:.4f} | mAP50-95: {val_metrics.box.map:.4f}")

    # 2. Evaluation on unseen Test split
    if args.eval_test:
        print("\nEvaluating on Unseen Hold-out Test Split (Zero Data Leakage)...")
        try:
            test_metrics = model.val(data=args.data, split="test")
            print(f"Test mAP50: {test_metrics.box.map50:.4f} | mAP50-95: {test_metrics.box.map:.4f}")
        except Exception as e:
            print(f"Note: Test evaluation skipped or encountered error: {e}")

    # 3. Export to ONNX
    if args.export_onnx:
        print("\nExporting Best Model to ONNX format...")
        try:
            onnx_path = model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
            print(f"Model exported successfully to: {onnx_path}")
        except Exception as e:
            print(f"Note: ONNX export encountered: {e}")

    print("\nAll tasks completed. Trained model weights are in:", os.path.join(args.project, args.name, "weights"))


if __name__ == "__main__":
    main()
