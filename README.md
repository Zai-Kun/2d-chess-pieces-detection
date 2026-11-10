# High-Quality 2D Chess Piece & Board Detection Dataset Generator
### Optimized for Ultralytics YOLO26n (`yolo26n.pt`)

An advanced, production-grade dataset generation pipeline and quality assurance suite designed to train **Ultralytics YOLO26n** for detecting 2D chess pieces and chessboards across real-world screenshots, mobile apps, streaming video frames, and photographed screens.

---

## 🌟 Key Features & Architectural Improvements

### 1. Tight & Accurate Bounding Boxes (Piece Geometry)
- **Sub-pixel Alpha Thresholding:** Instead of using loose square tiles or naive bounding boxes that capture drop shadows and transparent fringes, our pipeline analyzes the alpha channel with a calibrated threshold (`alpha >= 20`) to generate tight bounding boxes around the visible piece body.
- **Proportional Scaling & Baseline Alignment:** Natural piece height hierarchy (Pawns < Knights/Bishops/Rooks < Queens/Kings) is strictly preserved, and pieces remain realistically aligned along the tile baseline.
- **Micro-Jitter & In-Motion Variations:** Realistic positional micro-jitter ($\pm 1\text{--}2\text{ px}$) and occasional dragged/moving pieces with soft shadows simulate live gameplay.

### 2. High Visual Diversity & Synthetic-to-Real Gap Minimization
- **Four Composition Environments:**
  1. **Clean Full Board (85–100% frame):** Boards placed on a solid background sampled from a light tile, dark tile, or neutral grey — with no visible boundary — plus coordinate notation.
  2. **Web & Desktop Chess Interface (50–75% frame):** Simulates Chess.com, Lichess, and chess applications with dark/light themes, vertical evaluation bars, player avatar cards, rating badges, and digital clocks.
  3. **Embedded / Medium-Small Board Context (19–62% of canvas; ~120–400 px at 640):** Chessboards placed on diverse photographic backgrounds from `assets/random_noise_backgrounds` and clutter to train small-object detection (vital for YOLO26's STAL label assignment), while keeping pieces large enough to label reliably.
  4. **Photographed Screen / Physical Capture:** Simulates smartphone camera captures of computer monitors or phones with keystone perspective warp, subtle LCD sub-pixel moiré patterns, camera vignetting, and lighting gradients.
- **Realistic In-Game Overlays:**
  - **Move Highlights:** Subtle yellow-green, olive, cyan, or purple highlights on the from-square and to-square.
  - **In-Check Indicators:** Red radial indicators on the checked King.
  - **Legal Move Hints:** Translucent destination dots and capture rings.
  - **Analysis Arrows:** Semi-transparent arrows with distinct arrowheads.
  - **Coordinate Notation:** Inner corner coordinates or outer border coordinate labels.

### 3. Realistic Chess Position Generation
- **Opening Configurations:** Starting position and 25+ classic opening positions (Sicilian, Ruy Lopez, French, Caro-Kann, Queen's Gambit, King's Indian, English, etc.).
- **Middlegame Structures:** Realistic piece densities (16–28 pieces) with legal pawn structures strictly on ranks 2–7 and castled King positions.
- **Endgame Positions:** Sparse boards (3–10 pieces) ensuring the detector performs robustly when most tiles are empty.
- **Board Perspective Flipping:** 50% of positions are viewed from Black's perspective (flipped ranks and files), matching real apps when playing Black.

### 4. Mathematically Exact Transformations
- Any geometric transformation (translation, scaling, keystone perspective warp) projects bounding box corners through the exact $3 \times 3$ forward homography matrix $M$:
  $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim M \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- The enclosing axis-aligned bounding box is computed from the transformed corners and clamped to $[0, 1]$, guaranteeing **zero out-of-bounds errors** and mathematically correct labels.

### 5. Deterministic Zero-Leakage Dataset Splitting
- Piece sets (66 total), board designs (112 total), and scene/background images (2,900+ total) are partitioned into **strictly disjoint** `train`, `val`, and `test` subsets using a fixed random seed.
- **Zero Data Leakage:** Validation and test splits evaluate piece sets and board themes that the model has **never seen during training**, providing a genuine measure of out-of-domain generalization.

---

## 📁 Project Structure

```
.
├── assets/
│   ├── boards/                    # 112 board textures (wood, marble, glass, flat)
│   ├── pieces/                    # 66 piece sets (Neo, cburnett, Alpha, Classic, etc.)
│   ├── random_noise_backgrounds/  # 2,900+ realistic scene & clutter backgrounds
│   └── textures/                  # High-resolution wood and marble textures (board generation only)
├── datasets/                      # Generated dataset (images & labels)
│   ├── images/ {train, val, test}
│   └── labels/ {train, val, test}
├── chess_detection.yaml           # Ultralytics dataset configuration file
├── generate_datasets.py           # Main multiprocessing dataset generator
├── random_fen_gen.py              # Chess position & FEN generator
├── verify_dataset.py              # Automated QA and validation suite
├── visualize_labels.py            # Bounding box visual inspection tool
├── train_yolo26.py                # Ready-to-run YOLO26n training script
└── README.md                      # Documentation
```

---

## 🏷️ Class Mapping (`chess_detection.yaml`)

| Class ID | Name | Notation | Class ID | Name | Notation |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **0** | `black_pawn` | `bP` | **6** | `white_pawn` | `wP` |
| **1** | `black_rook` | `bR` | **7** | `white_rook` | `wR` |
| **2** | `black_knight` | `bN` | **8** | `white_knight` | `wN` |
| **3** | `black_bishop` | `bB` | **9** | `white_bishop` | `wB` |
| **4** | `black_queen` | `bQ` | **10** | `white_queen` | `wQ` |
| **5** | `black_king` | `bK` | **11** | `white_king` | `wK` |
| **12** | `chess_board` | `board` | — | — | — |

---

## 🚀 Quickstart: Dataset Generation

Generate a high-quality dataset with reproducible train/val/test splits:

```bash
# Standard generation (e.g. 5,000 train, 750 val, 750 test samples)
python3 generate_datasets.py --train-count 5000 --val-count 750 --test-count 750 --seed 42

# Fast test run (100 train, 25 val, 25 test samples)
python3 generate_datasets.py --train-count 100 --val-count 25 --test-count 25 --clean

# Higher resolution (800x800) for enhanced small-piece detection
python3 generate_datasets.py --imgsz 800 --train-count 3000 --val-count 500 --test-count 500

# Train-only dataset: when both val and test counts are 0, all piece sets,
# boards, and backgrounds go into train instead of being partitioned.
python3 generate_datasets.py --train-count 5000 --val-count 0 --test-count 0 --seed 42 --clean
```

### Generator Arguments:
- `--train-count`, `--val-count`, `--test-count`: Number of images to generate per split. If val and test are both `0`, all assets go to train.
- `--imgsz`: Output canvas resolution (default: `640`).
- `--board-size`: Base rendered board resolution (default: `640`).
- `--seed`: Random seed for deterministic asset partitioning **and sample content** (default: `42`). The same seed produces identical images and labels regardless of `--workers`.
- `--min-piece-px`: Redraw samples where any piece is smaller than this many pixels (default: `12`; `0` disables).
- `--max-sample-attempts`: Maximum redraw attempts per sample for `--min-piece-px` (default: `8`).
- `--workers`: Number of parallel CPU workers (default: all available cores).
- `--clean`: Wipes the output directory before generating.
- `--no-board-labels`: Omits class 12 (`chess_board`) if piece-only detection is desired.

---

## 🔍 Automated Quality Assurance (`verify_dataset.py`)

Run the comprehensive QA tool to verify dataset integrity, coordinate bounds, class balance, and lack of data leakage:

```bash
python3 verify_dataset.py --data-dir datasets
```

The QA script checks:
- ✅ **1:1 Pair Integrity:** Zero orphaned images or labels.
- ✅ **Boundary & Format Sanity:** All normalized coordinates within $(0.0, 1.0)$ with width/height $> 0$.
- ✅ **Aspect Ratio Sanity:** Identifies collapsed or distorted boxes.
- ✅ **Zero Data Leakage:** Verifies that no identical or duplicate image exists across splits.
- ✅ **Scale Statistics:** Reports breakdown across COCO object scales (Small $< 32^2$, Medium $32^2\text{--}96^2$, Large $\ge 96^2$).
- ✅ **Visual Montages:** Automatically saves labeled preview collages to `results/previews/preview_{split}.jpg`.

---

## 🎨 Visualizing Annotations (`visualize_labels.py`)

To inspect generated images with overlaid bounding boxes and class pills:

```bash
# Visualize a specific image ID from the training set
python3 visualize_labels.py 42 --split train

# Visualize 5 random images from the validation set
python3 visualize_labels.py --random --split val --count 5

# Visualize an explicit image file
python3 visualize_labels.py datasets/images/test/1.jpg
```
Output images are saved to `results/visualizations/` with color-coded boxes and labels.

---

## 🧠 Ultralytics YOLO26n Training Guide

### Why YOLO26 for 2D Chess Piece Detection?
Ultralytics YOLO26 introduces core advancements tailored for edge deployment and small-object detection:
1. **Native End-to-End (NMS-Free):** Features a 1-to-1 matching detection head at inference. This completely eliminates Non-Maximum Suppression (NMS) post-processing overhead and removes Distribution Focal Loss (DFL), lowering CPU latency by up to 43% while preventing duplicate box predictions on dense boards.
2. **MuSGD Optimizer:** Combines classical SGD momentum with Muon-style matrix updates (adapted from LLM training). It accelerates convergence and stabilizes gradient trajectories on lightweight vision backbones (`yolo26n`).
3. **ProgLoss (Progressive Loss Balancing):** Dynamically transitions supervision weights throughout training, focusing loss optimization directly onto the final inference head.
4. **STAL (Small-Target-Aware Label Assignment):** Explicitly preserves positive anchor coverage for small bounding boxes. On boards embedded in web pages or mobile screenshots, chess pieces often measure $20\text{--}35\text{ px}$; STAL prevents these targets from being dropped during training.

### Recommended Training Command

Run the dedicated training script:
```bash
python3 train_yolo26.py --epochs 60 --batch 16 --optimizer MuSGD --lr0 0.01
```

Or run via the Ultralytics CLI directly:
```bash
yolo detect train \
  model=yolo26n.pt \
  data=chess_detection.yaml \
  epochs=60 \
  imgsz=640 \
  batch=16 \
  optimizer=MuSGD \
  lr0=0.01 \
  lrf=0.01 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=3.0 \
  patience=15 \
  fliplr=0.0 \
  flipud=0.0 \
  mixup=0.0 \
  mosaic=0.2 \
  close_mosaic=10 \
  degrees=0.0 \
  scale=0.2 \
  translate=0.1
```

### Hyperparameter Reasoning Table

| Hyperparameter | Value | Technical Justification |
| :--- | :---: | :--- |
| **Model** | `yolo26n.pt` | Pretrained Nano model; optimal speed-accuracy tradeoff for real-time edge & browser detection. |
| **Optimizer** | `MuSGD` | YOLO26's official hybrid Muon-SGD optimizer; delivers faster convergence on lightweight backbones. |
| **Learning Rate (`lr0`)** | `0.01` | Recommended initial learning rate for MuSGD fine-tuning from COCO checkpoints. |
| **LR Schedule (`lrf`)** | `0.01` | Cosine annealing decaying to 1% of initial learning rate ($10^{-4}$). |
| **Epochs** | `60` | Sufficient for full convergence with pretrained weights; early stopping (`patience=15`) halts if validation plateaus. |
| **Image Size (`imgsz`)** | `640` | Standard resolution (use `800` if prioritizing very small embedded mobile boards). |
| **Horizontal Flip (`fliplr`)** | `0.0` | **Disabled.** In chess, Knights face a designated direction, board coordinates are oriented, and tile parity (light square on right corner) is fixed. |
| **Vertical Flip (`flipud`)** | `0.0` | **Disabled.** Upside-down 2D chess pieces do not exist. |
| **MixUp (`mixup`)** | `0.0` | **Disabled.** Blending two translucent chessboards causes illegal overlapping pieces on identical squares. |
| **Mosaic (`mosaic`)** | `0.2` | Kept low. Aggressive 4-image mosaic slicing disrupts chessboard grid regularity. |
| **Close Mosaic (`close_mosaic`)** | `10` | Disables mosaic during the final 10 epochs for pristine bounding box refinement. |
| **Rotation (`degrees`)** | `0.0` | Disabled in YOLO because camera tilts and keystone perspective warps are already rendered with exact homography box math in the dataset generator. |

---

## 📈 Evaluation & ONNX Deployment

After training finishes, evaluate on the unseen hold-out test set:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/yolo26n_chess/weights/best.pt")

# Evaluate generalization on unseen test set
metrics = model.val(data="chess_detection.yaml", split="test")
print("Test mAP50:", metrics.box.map50)
print("Test mAP50-95:", metrics.box.map)

# Export to ONNX for edge / web deployment
model.export(format="onnx", imgsz=640, dynamic=False)
```

The exported ONNX model features native end-to-end inference without requiring custom NMS plugins!
