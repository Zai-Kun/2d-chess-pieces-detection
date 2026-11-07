#!/usr/bin/env python3
"""
Automated Quality Assurance (QA) Suite for 2D Chess Piece Detection Datasets.

Performs comprehensive validation:
1. Label & Format Integrity: Checks 5-token YOLO format, class IDs, numeric validity.
2. Boundary & Coordinate Verification: Ensures all boxes are strictly within [0.0, 1.0],
   width/height > 0, and flags out-of-image or collapsed boxes.
3. Aspect Ratio & Scale Sanity: Checks for pathological piece dimensions.
4. Pair & Orphan Integrity: Verifies 1:1 match between images and labels.
5. Image Integrity: Verifies all images load cleanly and match valid dimensions.
6. Duplicate Detection: Detects identical/duplicate images across splits and within splits.
7. Class Balance & Statistics: Computes class distributions, imbalance ratios,
   and COCO object scale breakdown (small, medium, large).
8. Visual Previews: Renders high-quality preview collages with bounding boxes.
"""

import argparse
import glob
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = [
    "black_pawn",    # 0
    "black_rook",    # 1
    "black_knight",  # 2
    "black_bishop",  # 3
    "black_queen",   # 4
    "black_king",    # 5
    "white_pawn",    # 6
    "white_rook",    # 7
    "white_knight",  # 8
    "white_bishop",  # 9
    "white_queen",   # 10
    "white_king",    # 11
    "chess_board",   # 12
]

CLASS_SHORT = {
    0: "bP", 1: "bR", 2: "bN", 3: "bB", 4: "bQ", 5: "bK",
    6: "wP", 7: "wR", 8: "wN", 9: "wB", 10: "wQ", 11: "wK",
    12: "board",
}

CLASS_COLORS = {
    0: (120, 50, 50),
    1: (180, 60, 60),
    2: (210, 80, 80),
    3: (160, 40, 120),
    4: (140, 30, 180),
    5: (90, 20, 140),
    6: (40, 140, 180),
    7: (30, 160, 120),
    8: (40, 180, 80),
    9: (180, 160, 40),
    10: (220, 140, 30),
    11: (240, 180, 40),
    12: (70, 70, 220),
}


def get_default_font(size=11):
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


def compute_image_hash(img_path):
    """Computes a 64-bit difference hash (dhash) for fast duplicate detection."""
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = np.array(gray)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return tuple(diff.flatten())
    except Exception:
        return None


def validate_label_file(label_path, img_w=640, img_h=640, max_class=12):
    """Validates a single YOLO label file and returns boxes and errors/warnings."""
    errors = []
    warnings = []
    boxes = []

    try:
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception as e:
        return boxes, [f"Failed to read file: {e}"], []

    if not lines:
        warnings.append("Empty label file (0 annotations)")
        return boxes, errors, warnings

    for idx, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) != 5:
            errors.append(f"Line {idx}: Expected 5 tokens, got {len(tokens)} ('{line}')")
            continue

        try:
            cid = int(tokens[0])
            xc, yc, w, h = map(float, tokens[1:])
        except ValueError as e:
            errors.append(f"Line {idx}: Token conversion error: {e}")
            continue

        if cid < 0 or cid > max_class:
            errors.append(f"Line {idx}: Class ID {cid} out of range [0, {max_class}]")

        if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0):
            errors.append(f"Line {idx}: Center ({xc:.4f}, {yc:.4f}) outside (0.0, 1.0)")

        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"Line {idx}: Dimension w={w:.4f}, h={h:.4f} outside (0.0, 1.0]")

        # Boundaries
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0

        eps = 1e-3
        if x1 < -eps or y1 < -eps or x2 > 1.0 + eps or y2 > 1.0 + eps:
            errors.append(f"Line {idx}: Box boundary [({x1:.3f}, {y1:.3f}) to ({x2:.3f}, {y2:.3f})] exceeds image bounds")

        # Minimum pixel size check
        pix_w = w * img_w
        pix_h = h * img_h
        if pix_w < 4 or pix_h < 4:
            warnings.append(f"Line {idx}: Tiny box ({pix_w:.1f}x{pix_h:.1f} px)")

        # Aspect ratio check
        if h > 0:
            aspect = w / h
            if cid != 12 and (aspect < 0.15 or aspect > 3.5):
                warnings.append(f"Line {idx}: Unusual aspect ratio w/h={aspect:.2f} for class {cid}")
            elif cid == 12 and (aspect < 0.70 or aspect > 1.40):
                warnings.append(f"Line {idx}: Non-square chessboard box aspect ratio w/h={aspect:.2f}")

        boxes.append((cid, xc, yc, w, h))

    return boxes, errors, warnings


def render_preview_grid(sample_pairs, out_path, grid_cols=4, thumb_size=320):
    """Generates an annotated image montage from a list of (img_path, label_path) pairs."""
    if not sample_pairs:
        return

    n_samples = len(sample_pairs)
    grid_rows = math.ceil(n_samples / grid_cols)
    grid_w = grid_cols * thumb_size
    grid_h = grid_rows * thumb_size

    montage = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    font = get_default_font(size=10)

    for idx, (img_path, lbl_path) in enumerate(sample_pairs):
        row = idx // grid_cols
        col = idx % grid_cols
        ox = col * thumb_size
        oy = row * thumb_size

        try:
            im = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            iw, ih = im.size

            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                for line in lines:
                    toks = line.split()
                    if len(toks) == 5:
                        cid = int(toks[0])
                        xc, yc, w, h = map(float, toks[1:])
                        x1 = (xc - w / 2.0) * iw
                        y1 = (yc - h / 2.0) * ih
                        x2 = (xc + w / 2.0) * iw
                        y2 = (yc + h / 2.0) * ih

                        color = CLASS_COLORS.get(cid, (255, 255, 0))
                        box_width = 1 if cid == 12 else 2
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

                        # Small label pill for non-board classes
                        if cid != 12:
                            lbl_text = CLASS_SHORT.get(cid, str(cid))
                            draw.rectangle([x1, max(0, y1 - 12), x1 + 20, max(0, y1)], fill=color)
                            draw.text((x1 + 2, max(0, y1 - 12)), lbl_text, fill=(255, 255, 255), font=font)

            thumb = im.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
            montage.paste(thumb, (ox, oy))

            # Overlay filename
            m_draw = ImageDraw.Draw(montage)
            base_name = os.path.basename(img_path)
            m_draw.rectangle([ox, oy, ox + 60, oy + 16], fill=(0, 0, 0, 180))
            m_draw.text((ox + 4, oy + 2), base_name, fill=(255, 255, 255), font=font)

        except Exception as e:
            print(f"Warning: Failed rendering preview for {img_path}: {e}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    montage.save(out_path, quality=90)
    print(f"  Preview collage saved to: {out_path}")


def verify_dataset(data_dir="datasets", splits=("train", "val", "test"), preview=True, preview_count=8, preview_out="results/previews"):
    """Runs full validation across splits."""
    print("================================================================================")
    print(" 2D Chess Detection Dataset Quality Assurance (QA) Suite")
    print("================================================================================")
    print(f"Target Dataset Directory: {os.path.abspath(data_dir)}")
    print(f"Splits to Verify: {list(splits)}")

    all_passed = True
    overall_stats = {}
    split_hashes = defaultdict(dict)

    for split in splits:
        img_dir = os.path.join(data_dir, "images", split)
        lbl_dir = os.path.join(data_dir, "labels", split)

        print(f"\n--------------------------------------------------------------------------------")
        print(f" Validating Split: [{split.upper()}]")
        print(f"--------------------------------------------------------------------------------")

        if not os.path.exists(img_dir):
            print(f"  [ERROR] Image directory does not exist: {img_dir}")
            all_passed = False
            continue

        if not os.path.exists(lbl_dir):
            print(f"  [ERROR] Label directory does not exist: {lbl_dir}")
            all_passed = False
            continue

        image_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

        print(f"  Found {len(image_files)} images, {len(label_files)} label files.")

        if len(image_files) == 0:
            print(f"  [WARN] No images found in {img_dir}!")
            all_passed = False
            continue

        # 1. Check for orphaned files
        img_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        lbl_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

        orphaned_images = [f for stem, f in img_stems.items() if stem not in lbl_stems]
        orphaned_labels = [f for stem, f in lbl_stems.items() if stem not in img_stems]

        if orphaned_images:
            print(f"  [FAIL] {len(orphaned_images)} orphaned image(s) with no corresponding label file!")
            all_passed = False
        else:
            print(f"  [PASS] 0 orphaned images (all images have matching labels).")

        if orphaned_labels:
            print(f"  [FAIL] {len(orphaned_labels)} orphaned label(s) with no corresponding image file!")
            all_passed = False
        else:
            print(f"  [PASS] 0 orphaned labels (all labels have matching images).")

        # 2. Check each label and image pair
        split_class_counts = Counter()
        split_box_sizes = {"small": 0, "medium": 0, "large": 0}
        total_boxes = 0
        total_errors = 0
        total_warnings = 0

        sample_pairs = []

        for stem, img_path in img_stems.items():
            if stem not in lbl_stems:
                continue
            lbl_path = lbl_stems[stem]
            sample_pairs.append((img_path, lbl_path))

            # Image sanity
            try:
                with Image.open(img_path) as im:
                    iw, ih = im.size
            except Exception as e:
                print(f"  [FAIL] Corrupted image file {img_path}: {e}")
                total_errors += 1
                continue

            # Duplicate hash check
            im_hash = compute_image_hash(img_path)
            if im_hash:
                if im_hash in split_hashes[split]:
                    total_warnings += 1
                else:
                    split_hashes[split][im_hash] = img_path

            # Label validation
            boxes, errors, warnings = validate_label_file(lbl_path, img_w=iw, img_h=ih)
            if errors:
                total_errors += len(errors)
                if total_errors <= 5:
                    for err in errors:
                        print(f"  [LABEL ERROR] {os.path.basename(lbl_path)}: {err}")
            total_warnings += len(warnings)

            total_boxes += len(boxes)
            for cid, xc, yc, w, h in boxes:
                split_class_counts[cid] += 1
                # COCO object scale definition
                pix_area = (w * iw) * (h * ih)
                if pix_area < 32 ** 2:
                    split_box_sizes["small"] += 1
                elif pix_area < 96 ** 2:
                    split_box_sizes["medium"] += 1
                else:
                    split_box_sizes["large"] += 1

        # Summary for this split
        print(f"\n  Split [{split}] Quality Summary:")
        print(f"    - Total Images: {len(image_files)}")
        print(f"    - Total Annotations: {total_boxes} (avg {total_boxes / max(1, len(image_files)):.1f} per image)")
        print(f"    - Errors: {total_errors}")
        print(f"    - Warnings: {total_warnings}")

        if total_errors == 0:
            print(f"    - Label Format & Boundary Checks: [PASS]")
        else:
            print(f"    - Label Format & Boundary Checks: [FAIL] ({total_errors} errors)")
            all_passed = False

        # Class Breakdown
        print(f"\n    Class Representation Breakdown:")
        for cid in range(13):
            cname = CLASS_NAMES[cid]
            cnt = split_class_counts[cid]
            pct = (cnt / max(1, total_boxes)) * 100
            print(f"      {cid:2d} | {cname:<14} : {cnt:6d} ({pct:5.1f}%)")

        # Imbalance check (In standard chess, pawns are naturally ~8x-12x more frequent than queens)
        piece_counts = [split_class_counts[cid] for cid in range(12)]
        if piece_counts and min(piece_counts) > 0:
            imbalance = max(piece_counts) / min(piece_counts)
            print(f"    - Piece Class Imbalance Ratio (Max/Min): {imbalance:.2f}x (Natural chess prior: ~8-12x)")
            if imbalance > 16.0:
                print(f"      [WARN] Severely elevated class imbalance detected ({imbalance:.2f}x)")
        elif piece_counts:
            missing = [CLASS_NAMES[i] for i, cnt in enumerate(piece_counts) if cnt == 0]
            print(f"      [WARN] Classes with 0 instances: {missing}")

        # Scale Breakdown
        print(f"\n    Object Scale Distribution (COCO Standard):")
        print(f"      - Small  (area < 32^2)     : {split_box_sizes['small']:6d} ({split_box_sizes['small']*100/max(1, total_boxes):.1f}%)")
        print(f"      - Medium (32^2 <= a < 96^2): {split_box_sizes['medium']:6d} ({split_box_sizes['medium']*100/max(1, total_boxes):.1f}%)")
        print(f"      - Large  (area >= 96^2)    : {split_box_sizes['large']:6d} ({split_box_sizes['large']*100/max(1, total_boxes):.1f}%)")

        # 3. Generate Visual Preview Collage
        if preview and sample_pairs:
            preview_subset = random.sample(sample_pairs, min(preview_count, len(sample_pairs)))
            out_preview = os.path.join(preview_out, f"preview_{split}.jpg")
            render_preview_grid(preview_subset, out_preview, grid_cols=4, thumb_size=320)

        overall_stats[split] = {
            "images": len(image_files),
            "boxes": total_boxes,
            "errors": total_errors,
            "warnings": total_warnings,
            "classes": split_class_counts,
        }

    # Cross-split duplicate detection (data leakage check)
    print("\n--------------------------------------------------------------------------------")
    print(" Cross-Split Data Leakage Verification")
    print("--------------------------------------------------------------------------------")
    if "train" in split_hashes and "val" in split_hashes:
        train_h = set(split_hashes["train"].keys())
        val_h = set(split_hashes["val"].keys())
        leakage = train_h.intersection(val_h)
        if leakage:
            print(f"  [FAIL] Data leakage: {len(leakage)} duplicate image(s) shared between Train and Val!")
            all_passed = False
        else:
            print(f"  [PASS] 0 duplicate images between Train and Val (No Leakage).")

    if "train" in split_hashes and "test" in split_hashes:
        train_h = set(split_hashes["train"].keys())
        test_h = set(split_hashes["test"].keys())
        leakage = train_h.intersection(test_h)
        if leakage:
            print(f"  [FAIL] Data leakage: {len(leakage)} duplicate image(s) shared between Train and Test!")
            all_passed = False
        else:
            print(f"  [PASS] 0 duplicate images between Train and Test (No Leakage).")

    print("\n================================================================================")
    if all_passed:
        print(" QA RESULT: ALL CHECKS PASSED SUCCESSFULLY [STATUS: READY FOR YOLO26n]")
    else:
        print(" QA RESULT: ISSUES DETECTED - PLEASE REVIEW LOGS ABOVE")
    print("================================================================================\n")

    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description="2D Chess Detection Dataset QA & Verification Tool")
    parser.add_argument("--data-dir", default="datasets", help="Root directory containing images/ and labels/")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to verify (e.g. train val test)")
    parser.add_argument("--no-preview", action="store_true", help="Disable generating visual preview collages")
    parser.add_argument("--preview-count", type=int, default=8, help="Number of images in preview collage")
    parser.add_argument("--preview-out", default="results/previews", help="Directory for preview collages")
    args = parser.parse_args()

    exit_code = verify_dataset(
        data_dir=args.data_dir,
        splits=args.splits,
        preview=not args.no_preview,
        preview_count=args.preview_count,
        preview_out=args.preview_out,
    )
    exit(exit_code)


if __name__ == "__main__":
    main()
