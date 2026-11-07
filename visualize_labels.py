#!/usr/bin/env python3
"""
Draws YOLO-format labels onto their corresponding image to visually confirm
bounding boxes line up correctly with chess pieces and chessboards.

Usage:
    python visualize_labels.py <image_id>                     # e.g. python visualize_labels.py 42
    python visualize_labels.py <image_id> --split val         # look in val split
    python visualize_labels.py <image_id> --split test        # look in test split
    python visualize_labels.py /path/to/123.jpg               # explicit image path
    python visualize_labels.py --random --count 5             # 5 random train images
    python visualize_labels.py --random --split val --count 5 # 5 random val images
"""

import argparse
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

DATASETS_IMAGES_DIR = "datasets/images"
DATASETS_LABELS_DIR = "datasets/labels"

CLASS_NAMES = {
    "0": "bP", "1": "bR", "2": "bN", "3": "bB", "4": "bQ", "5": "bK",
    "6": "wP", "7": "wR", "8": "wN", "9": "wB", "10": "wQ", "11": "wK",
    "12": "board",
}

CLASS_FULL_NAMES = {
    "0": "Black Pawn", "1": "Black Rook", "2": "Black Knight",
    "3": "Black Bishop", "4": "Black Queen", "5": "Black King",
    "6": "White Pawn", "7": "White Rook", "8": "White Knight",
    "9": "White Bishop", "10": "White Queen", "11": "White King",
    "12": "Chess Board",
}

CLASS_COLORS = {
    "0": (180, 50, 50),
    "1": (200, 60, 60),
    "2": (220, 80, 80),
    "3": (170, 40, 130),
    "4": (140, 30, 180),
    "5": (90, 20, 140),
    "6": (40, 140, 200),
    "7": (30, 170, 140),
    "8": (40, 190, 80),
    "9": (200, 170, 40),
    "10": (230, 140, 30),
    "11": (250, 190, 40),
    "12": (60, 80, 230),
}


def get_default_font(size=12):
    ttf_candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    ]
    for path in ttf_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_labels(image_path, label_path, out_path):
    if not os.path.exists(image_path):
        print(f"  [ERROR] Image not found: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"  [ERROR] Label not found: {label_path}")
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    font = get_default_font(size=11)

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print("  Label file is empty.")
        return

    # Draw board outline first, pieces second
    lines_sorted = sorted(lines, key=lambda l: 0 if l.split()[0] == "12" else 1)

    for line in lines_sorted:
        tokens = line.split()
        if len(tokens) != 5:
            continue

        cid, xc, yc, w, h = tokens
        xc, yc, w, h = map(float, (xc, yc, w, h))

        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h

        color = CLASS_COLORS.get(cid, (255, 255, 0))

        if cid == "12":
            # Dashed or thin outline for chessboard
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.rectangle([x1, y1, x1 + 45, y1 + 14], fill=color)
            draw.text((x1 + 3, y1 + 1), "board", fill=(255, 255, 255), font=font)
        else:
            # Piece bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label_text = CLASS_NAMES.get(cid, cid)
            pill_h = 13
            pill_w = len(label_text) * 8 + 6
            pill_y1 = max(0, y1 - pill_h)
            draw.rectangle([x1, pill_y1, x1 + pill_w, pill_y1 + pill_h], fill=color)
            draw.text((x1 + 3, pill_y1), label_text, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path, quality=95)
    print(f"  Saved: {out_path} ({len(lines)} annotations)")


def resolve_paths(image_ref, split="train", base_dir="datasets"):
    if image_ref.endswith(".jpg") or os.path.sep in image_ref:
        img_path = image_ref
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    else:
        img_path = os.path.join(base_dir, "images", split, f"{image_ref}.jpg")
        lbl_path = os.path.join(base_dir, "labels", split, f"{image_ref}.txt")
    return img_path, lbl_path


def pick_random_ids(split="train", count=1, base_dir="datasets"):
    img_dir = os.path.join(base_dir, "images", split)
    if not os.path.exists(img_dir):
        raise SystemExit(f"Directory not found: {img_dir}")
    files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not files:
        raise SystemExit(f"No .jpg files found in {img_dir}")
    selected = random.sample(files, min(count, len(files)))
    return [os.path.splitext(os.path.basename(f))[0] for f in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_ref", nargs="?", help="Image ID (e.g. 42) or explicit file path")
    parser.add_argument("label_ref", nargs="?", help="Optional explicit label path")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split (default: train)")
    parser.add_argument("--random", action="store_true", help="Pick random image(s)")
    parser.add_argument("--count", type=int, default=1, help="Count of images to visualize with --random")
    parser.add_argument("--out-dir", default="results/visualizations", help="Output directory (default: results/visualizations)")
    parser.add_argument("--data-dir", default="datasets", help="Root datasets directory (default: datasets)")
    args = parser.parse_args()

    if args.random:
        ids = pick_random_ids(args.split, args.count, args.data_dir)
        pairs = [resolve_paths(i, args.split, args.data_dir) for i in ids]
    elif args.image_ref:
        if args.label_ref:
            pairs = [(args.image_ref, args.label_ref)]
        else:
            pairs = [resolve_paths(args.image_ref, args.split, args.data_dir)]
    else:
        parser.error("Specify an image ID, image path, or use --random")
        return

    for img_path, lbl_path in pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out_dir, f"{args.split}_{base}_labeled.jpg")
        draw_labels(img_path, lbl_path, out_path)


if __name__ == "__main__":
    main()
