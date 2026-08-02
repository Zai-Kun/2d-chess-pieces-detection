"""
Draws YOLO-format labels onto their corresponding image so you can visually
confirm bounding boxes line up correctly with the generated dataset.

Usage:
    python visualize_labels.py <image_id>
    python visualize_labels.py <image_id> --split val
    python visualize_labels.py /path/to/123.jpg /path/to/123.txt
    python visualize_labels.py --random               # pick a random train image
    python visualize_labels.py --random --split val --count 5

Output image(s) are saved next to a "_labeled" suffix, e.g. 123_labeled.jpg,
and opened for viewing if possible.
"""

import argparse
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

CLASS_COLORS = {}


def get_color(class_id):
    if class_id not in CLASS_COLORS:
        random.seed(hash(class_id) & 0xFFFFFFFF)  # stable color per class
        CLASS_COLORS[class_id] = tuple(random.randint(60, 255) for _ in range(3))
    return CLASS_COLORS[class_id]


def draw_labels(image_path, label_path, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if not os.path.exists(label_path):
        print(f"  No label file found at {label_path}")
        return

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print("  Label file is empty.")
        return

    for line in lines:
        tokens = line.split()
        if len(tokens) != 5:
            print(f"  Skipping malformed line: {line}")
            continue

        class_id, xc, yc, w, h = tokens
        xc, yc, w, h = map(float, (xc, yc, w, h))

        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h

        color = get_color(class_id)
        # Draw the board box thinner/dashed-ish (just thinner) so it doesn't
        # obscure piece boxes underneath it.
        width = 1 if class_id == "12" else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label_text = CLASS_NAMES.get(class_id, class_id)
        text_pos = (x1 + 2, max(0, y1 - 12))
        if font:
            draw.text(text_pos, label_text, fill=color, font=font)
        else:
            draw.text(text_pos, label_text, fill=color)

    image.save(out_path)
    print(f"  Saved: {out_path}  ({len(lines)} boxes)")


def resolve_paths(image_id, split):
    img_path = f"{DATASETS_IMAGES_DIR}/{split}/{image_id}.jpg"
    lbl_path = f"{DATASETS_LABELS_DIR}/{split}/{image_id}.txt"
    return img_path, lbl_path


def pick_random_ids(split, count):
    img_dir = f"{DATASETS_IMAGES_DIR}/{split}"
    ids = [f.split(".")[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if not ids:
        raise SystemExit(f"No images found in {img_dir}")
    return random.sample(ids, min(count, len(ids)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_ref", nargs="?", help="Image id (e.g. 42) or path to an image file")
    parser.add_argument("label_ref", nargs="?", help="Optional explicit label path (only used with an explicit image path)")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Which dataset split to look in (default: train)")
    parser.add_argument("--random", action="store_true", help="Pick random image(s) instead of specifying an id")
    parser.add_argument("--count", type=int, default=1, help="How many random images to visualize (with --random)")
    parser.add_argument("--out-dir", default=None, help="Directory to save labeled images (default: alongside source image)")
    args = parser.parse_args()

    if args.random:
        ids = pick_random_ids(args.split, args.count)
        pairs = [resolve_paths(i, args.split) for i in ids]
    elif args.image_ref and (args.image_ref.endswith(".jpg") or os.path.sep in args.image_ref):
        img_path = args.image_ref
        lbl_path = args.label_ref or img_path.rsplit(".", 1)[0] + ".txt"
        # try to infer label path in the mirrored labels dir if not given
        if not args.label_ref and DATASETS_IMAGES_DIR in img_path:
            lbl_path = img_path.replace(DATASETS_IMAGES_DIR, DATASETS_LABELS_DIR).rsplit(".", 1)[0] + ".txt"
        pairs = [(img_path, lbl_path)]
    elif args.image_ref:
        pairs = [resolve_paths(args.image_ref, args.split)]
    else:
        parser.error("Provide an image id, an image path, or use --random")
        return

    for img_path, lbl_path in pairs:
        print(f"Image: {img_path}")
        if not os.path.exists(img_path):
            print(f"  Image not found, skipping.")
            continue

        base = os.path.basename(img_path).rsplit(".", 1)[0]
        out_dir = args.out_dir or os.path.dirname(img_path) or "."
        out_path = os.path.join(out_dir, f"{base}_labeled.jpg")

        draw_labels(img_path, lbl_path, out_path)


if __name__ == "__main__":
    main()
