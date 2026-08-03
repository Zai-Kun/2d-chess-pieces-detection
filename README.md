# 2D Chess Board and Pieces Detection for YOLO26s / YOLO11 / YOLOv8

This repository provides a data generation pipeline and training configuration for detecting 2D chessboards and pieces in images with high precision.

---

## Key Improvements & Features

### 1. FEN Distribution (70% / 20% / 10%)
- **70% Chess.com Open Database**: Real game positions fetched from Chess.com's public API and cached in `assets/chess_com_fens.txt`.
- **20% Custom Realistic Generator**: Plausible legal/quasi-legal positions generated following standard piece counts and pawn placement rules.
- **10% Complete Bogus Generator**: Wild/chaotic piece arrangements (extreme counts, piece spam, unusual combinations) to prevent position bias and ensure YOLO learns visual piece features in any context.

### 2. Comprehensive Augmentations & Domain Noise
To maximize real-world generalization (photos of screens, low-res images, web screenshots):
- **Blur**: Gaussian blur, box blur, and directional motion blur.
- **JPEG Compression**: Simulated lossy web compression ($Q \in [20, 90]$).
- **Noise**: Gaussian sensor noise, ISO film grain, and salt-and-pepper noise.
- **Screen Scanlines / Moiré Patterns**: Simulates photographing a computer monitor.
- **Color Jitter & Lighting**: Hue, saturation, contrast, brightness, sharpness, and gamma adjustments.
- **Vignetting**: Edge lighting falloff and glare shadows across the board.
- **Geometric Distortions**:
  - Sub-tile piece micro-offsets (imperfect piece placement on squares).
  - Piece scaling ($80\% - 120\%$) and micro-rotation ($\pm 14^\circ$).
  - 3D Perspective warping (camera angle tilt) with homography matrix-transformed bounding boxes.
- **Background Compositing**: Placing boards on diverse real-world background scenes and noise textures.

---

## Dataset Generation Commands

1. **(Optional) Fetch / Refresh Chess.com FEN Database**:
   ```bash
   python3 fetch_chess_com_fens.py
   ```

2. **Generate Dataset (Images & YOLO Labels)**:
   ```bash
   python3 generate_datasets.py
   ```

3. **Visualize & Inspect Labels**:
   ```bash
   python3 visualize_labels.py --random --split train --count 5
   ```

---

## Training Recommended for YOLO26s

Train YOLO26s on `chess_detection.yaml` with the following CLI command:

```bash
yolo detect train data=chess_detection.yaml model=yolo26s.pt epochs=60 imgsz=640 batch=32 lr0=0.001 mosaic=0.5 mixup=0.1
```

Or in Python:

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")  # Or yolo11s.pt / yolov8s.pt
results = model.train(
    data="chess_detection.yaml",
    epochs=60,
    imgsz=640,
    batch=32,
    lr0=0.001,
    mosaic=0.5,
    mixup=0.1,
    patience=15,
)
```
