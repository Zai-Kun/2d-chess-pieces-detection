#!/usr/bin/env python3
"""Evaluate a YOLO chess detector and save enlarged annotated images.

Placement-only FEN convention used by this script:
- Exactly eight rows, separated by '/'.
- The first FEN row is the top row of the image.
- Files are ordered left-to-right.
- Black pieces are lowercase; white pieces are uppercase.
- POV, side to move, castling, en-passant, and move counters are ignored.

Examples:
    python eval.py -m chess_detection.onnx -i screenshot
    python eval.py -m chess_detection.onnx -i screenshot -o predictions --scale 3
    python eval.py -m chess_detection.onnx -i screenshot --suppress-overlaps
    python eval.py -m chess_detection.onnx -i screenshot --refined-search --board-padding 0.06
    python eval.py -m chess_detection.onnx -i screenshot --match-fen
    python eval.py -m chess_detection.onnx -i screenshot --generate-fen
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2


# Keep these class IDs synchronized with the model's data.yaml.
CLASS_LABELS = {
    0: "p",   # black pawn
    1: "r",   # black rook
    2: "n",   # black knight
    3: "b",   # black bishop
    4: "q",   # black queen
    5: "k",   # black king
    6: "P",   # white pawn
    7: "R",   # white rook
    8: "N",   # white knight
    9: "B",   # white bishop
    10: "Q",  # white queen
    11: "K",  # white king
}
PIECE_CLASSES = tuple(CLASS_LABELS)
BOARD_CLASS = 12
VALID_FEN_PIECES = set("prnbqkPRNBQK")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# OpenCV colors are BGR.
CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (0, 255, 255),
    3: (255, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (128, 0, 255),
    7: (0, 128, 255),
    8: (128, 255, 0),
    9: (255, 128, 0),
    10: (255, 0, 128),
    11: (0, 255, 128),
    12: (255, 255, 255),
}


@dataclass(frozen=True)
class Detection:
    """A detection expressed in the original input image's coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cls: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chess YOLO detections, save annotated images, and optionally "
            "generate or compare placement-only FENs."
        )
    )
    parser.add_argument("-m", "--model", required=True, help="YOLO model path (.pt or .onnx).")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input image or image directory.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("eval_output"),
        help="Output directory (default: ./eval_output).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (default: 640).")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence (default: 0.25).")
    parser.add_argument("--scale", type=float, default=3.0, help="Output enlargement factor (default: 3).")
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively.")

    parser.add_argument(
        "--suppress-overlaps", action="store_true",
        help="Suppress overlapping detections, keeping the higher-confidence detection.",
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.70,
        help="IoU threshold for --suppress-overlaps (default: 0.70).",
    )

    parser.add_argument(
        "--refined-search", action="store_true",
        help="Detect the board first, crop around it, then detect pieces within the crop.",
    )
    parser.add_argument(
        "--board-padding", type=float, default=0.05,
        help="Crop padding as a fraction of board width/height (default: 0.05 = 5%%).",
    )

    fen_group = parser.add_mutually_exclusive_group()
    fen_group.add_argument(
        "--match-fen", action="store_true",
        help="Compare detections against fens.json in the input directory.",
    )
    fen_group.add_argument(
        "--generate-fen", action="store_true",
        help="Add missing image entries to fens.json in the input directory, preserving existing entries.",
    )

    args = parser.parse_args()

    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than zero")
    if not math.isfinite(args.conf) or not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be a finite number greater than zero")
    if not math.isfinite(args.overlap_threshold) or not 0.0 <= args.overlap_threshold <= 1.0:
        parser.error("--overlap-threshold must be between 0 and 1")
    if not math.isfinite(args.board_padding) or args.board_padding < 0:
        parser.error("--board-padding must be a finite, non-negative number")

    return args


def is_relative_to(path: Path, parent: Path) -> bool:
    """Path.is_relative_to compatibility helper."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_images(source: Path, recursive: bool, excluded_dir: Path | None = None) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image format: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    images: list[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if excluded_dir is not None and is_relative_to(resolved, excluded_dir):
            continue
        images.append(resolved)
    return sorted(images)


def image_key(image_path: Path, input_root: Path) -> str:
    """Return a stable POSIX-style key for fens.json and result reports."""
    return image_path.relative_to(input_root).as_posix()


def normalize_json_image_key(key: str) -> str:
    """Normalize JSON keys so nested paths use forward slashes on every OS."""
    normalized = key.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Invalid image key in fens.json: {key!r}")
    return pure.as_posix()


def detections_from_result(result: Any) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        coords = box.xyxy[0].detach().cpu().tolist()
        confidence = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())
        x1, y1, x2, y2 = (float(value) for value in coords)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(x1, y1, x2, y2, confidence, class_id))
    return detections


def run_inference(
    model: Any,
    image: Any,
    args: argparse.Namespace,
    classes: list[int] | None = None,
) -> list[Detection]:
    """Run inference while leaving custom overlap suppression to this script."""
    results = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.99,
        max_det=1000,
        classes=classes,
        verbose=False,
    )
    if not results:
        return []
    return detections_from_result(results[0])


def intersection_over_union(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_overlapping_detections(
    detections: list[Detection], threshold: float
) -> list[Detection]:
    """Greedy class-agnostic NMS using IoU; highest confidence wins."""
    ordered = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if any(intersection_over_union(candidate, accepted) > threshold for accepted in kept):
            continue
        kept.append(candidate)
    return kept


def best_board_detection(detections: list[Detection]) -> Detection | None:
    boards = [detection for detection in detections if detection.cls == BOARD_CLASS]
    return max(boards, key=lambda detection: detection.confidence, default=None)


def crop_to_board(
    image: Any, board: Detection, padding_fraction: float
) -> tuple[Any, int, int]:
    height, width = image.shape[:2]
    board_width = board.x2 - board.x1
    board_height = board.y2 - board.y1
    if board_width <= 0 or board_height <= 0:
        raise ValueError("Detected board has invalid dimensions")

    pad_x = board_width * padding_fraction
    pad_y = board_height * padding_fraction
    crop_x1 = max(0, int(math.floor(board.x1 - pad_x)))
    crop_y1 = max(0, int(math.floor(board.y1 - pad_y)))
    crop_x2 = min(width, int(math.ceil(board.x2 + pad_x)))
    crop_y2 = min(height, int(math.ceil(board.y2 + pad_y)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Detected board produced an invalid crop")
    return image[crop_y1:crop_y2, crop_x1:crop_x2], crop_x1, crop_y1


def infer_image(
    model: Any, image: Any, args: argparse.Namespace
) -> tuple[list[Detection], Detection | None]:
    """Infer an image and return detections plus the chosen board box.

    All returned detection coordinates are in the original image coordinate space.
    """
    if not args.refined_search:
        detections = run_inference(model, image, args)
        return detections, best_board_detection(detections)

    # First pass: detect only the board in the full image.
    board_detections = run_inference(model, image, args, classes=[BOARD_CLASS])
    board = best_board_detection(board_detections)
    if board is None:
        print("  [WARN] Refined search: board not detected; falling back to full-image inference.")
        detections = run_inference(model, image, args)
        return detections, best_board_detection(detections)

    crop, offset_x, offset_y = crop_to_board(image, board, args.board_padding)
    # The board has already been detected; this pass only looks for pieces.
    crop_detections = run_inference(model, crop, args, classes=list(PIECE_CLASSES))
    translated = [board]
    for detection in crop_detections:
        translated.append(
            Detection(
                detection.x1 + offset_x,
                detection.y1 + offset_y,
                detection.x2 + offset_x,
                detection.y2 + offset_y,
                detection.confidence,
                detection.cls,
            )
        )
    return translated, board


def parse_placement_fen(value: str, *, source: str = "FEN") -> tuple[str, list[list[str]]]:
    """Validate a placement-only or full FEN and return placement plus 8x8 grid."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: expected a non-empty FEN string")

    # A full FEN may have side-to-move and other fields; placement is field 1.
    placement = value.strip().split()[0]
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError(f"{source}: expected exactly 8 rows, got {len(rows)}")

    grid: list[list[str]] = []
    for row_number, encoded_row in enumerate(rows, start=1):
        decoded: list[str] = []
        previous_was_digit = False
        for character in encoded_row:
            if character in "12345678":
                if previous_was_digit:
                    raise ValueError(
                        f"{source}: row {row_number} has adjacent empty-square digits; "
                        "combine them into one digit"
                    )
                decoded.extend(["."] * int(character))
                previous_was_digit = True
            elif character in VALID_FEN_PIECES:
                decoded.append(character)
                previous_was_digit = False
            else:
                raise ValueError(
                    f"{source}: invalid character {character!r} in row {row_number}"
                )

        if len(decoded) != 8:
            raise ValueError(
                f"{source}: row {row_number} expands to {len(decoded)} squares, expected 8"
            )
        grid.append(decoded)

    return placement, grid


def grid_to_placement_fen(grid: list[list[str]]) -> str:
    """Compress an 8x8 grid into a placement-only FEN."""
    if len(grid) != 8 or any(len(row) != 8 for row in grid):
        raise ValueError("FEN generation requires an 8x8 grid")

    encoded_rows: list[str] = []
    for row in grid:
        encoded: list[str] = []
        empty_count = 0
        for piece in row:
            if piece == ".":
                empty_count += 1
            else:
                if piece not in VALID_FEN_PIECES:
                    raise ValueError(f"Invalid piece character in grid: {piece!r}")
                if empty_count:
                    encoded.append(str(empty_count))
                    empty_count = 0
                encoded.append(piece)
        if empty_count:
            encoded.append(str(empty_count))
        encoded_rows.append("".join(encoded) or "8")
    return "/".join(encoded_rows)


def detections_to_grid(
    detections: list[Detection], board: Detection
) -> list[list[str]]:
    """Assign piece detections to board squares using their bounding-box centers."""
    board_width = board.x2 - board.x1
    board_height = board.y2 - board.y1
    if board_width <= 0 or board_height <= 0:
        raise ValueError("Cannot map pieces: board detection has invalid dimensions")

    # Each square keeps the highest-confidence detection assigned to it.
    squares: list[list[Detection | None]] = [[None for _ in range(8)] for _ in range(8)]
    for detection in detections:
        if detection.cls not in CLASS_LABELS:
            continue

        center_x, center_y = detection.center
        # Do not clamp detections outside the board onto its edge squares.
        if not (board.x1 <= center_x < board.x2 and board.y1 <= center_y < board.y2):
            continue

        col = int((center_x - board.x1) / board_width * 8)
        row = int((center_y - board.y1) / board_height * 8)
        if not (0 <= row < 8 and 0 <= col < 8):
            continue

        current = squares[row][col]
        if current is None or detection.confidence > current.confidence:
            squares[row][col] = detection

    return [
        [CLASS_LABELS[square.cls] if square is not None else "." for square in row]
        for row in squares
    ]


def placement_fen(detections: list[Detection], board: Detection | None) -> str | None:
    """Create the 8-row placement FEN, or None if no board was detected."""
    if board is None:
        return None
    return grid_to_placement_fen(detections_to_grid(detections, board))


def compare_placements(predicted: str, expected: str) -> tuple[int, list[str], float]:
    """Compare two validated placement strings and return square-level differences."""
    _, predicted_grid = parse_placement_fen(predicted, source="Predicted FEN")
    _, expected_grid = parse_placement_fen(expected, source="Expected FEN")

    differences: list[str] = []
    for row in range(8):
        # The first image row is represented as rank 8; row order is not POV-adjusted.
        rank = 8 - row
        for col in range(8):
            if predicted_grid[row][col] != expected_grid[row][col]:
                file_name = "abcdefgh"[col]
                predicted_piece = predicted_grid[row][col]
                expected_piece = expected_grid[row][col]
                differences.append(
                    f"{file_name}{rank} (predicted: {predicted_piece}, expected: {expected_piece})"
                )

    count = len(differences)
    return count, differences, (count / 64.0) * 100.0


def draw_piece_label(
    image: Any,
    detection: Detection,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    label = CLASS_LABELS.get(detection.cls)
    if label is None:
        return

    confidence_text = f"{detection.confidence:.2f}"
    box_width, box_height = x2 - x1, y2 - y1
    padding = max(2, int(round(4 * scale)))
    max_width = box_width - 2 * padding
    max_height = box_height - 2 * padding
    if max_width <= 0 or max_height <= 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    main_thickness = max(1, int(round(scale)))
    confidence_thickness = max(1, int(round(scale * 0.7)))
    font_scale = 2.0
    min_font_scale = 0.05

    while font_scale >= min_font_scale:
        (label_w, label_h), label_base = cv2.getTextSize(
            label, font, font_scale, main_thickness
        )
        confidence_scale = font_scale * 0.55
        (conf_w, conf_h), conf_base = cv2.getTextSize(
            confidence_text, font, confidence_scale, confidence_thickness
        )
        gap = max(1, int(round(font_scale * scale * 0.5)))
        total_height = label_h + label_base + gap + conf_h + conf_base
        if max(label_w, conf_w) <= max_width and total_height <= max_height:
            break
        font_scale -= 0.05

    if font_scale < min_font_scale:
        return

    (label_w, label_h), label_base = cv2.getTextSize(
        label, font, font_scale, main_thickness
    )
    confidence_scale = font_scale * 0.55
    (conf_w, conf_h), conf_base = cv2.getTextSize(
        confidence_text, font, confidence_scale, confidence_thickness
    )
    gap = max(1, int(round(font_scale * scale * 0.5)))
    total_height = label_h + label_base + gap + conf_h + conf_base
    top = y1 + (box_height - total_height) // 2
    label_x = x1 + (box_width - label_w) // 2
    label_baseline = top + label_h
    conf_x = x1 + (box_width - conf_w) // 2
    conf_baseline = label_baseline + label_base + gap + conf_h

    # Do not draw text if any part would escape the detection box.
    if not (
        label_x >= x1 + padding
        and label_x + label_w <= x2 - padding
        and label_baseline - label_h >= y1 + padding
        and label_baseline + label_base <= y2 - padding
        and conf_x >= x1 + padding
        and conf_x + conf_w <= x2 - padding
        and conf_baseline - conf_h >= y1 + padding
        and conf_baseline + conf_base <= y2 - padding
    ):
        return

    cv2.putText(
        image, label, (label_x, label_baseline), font, font_scale,
        color, main_thickness, cv2.LINE_AA,
    )
    cv2.putText(
        image, confidence_text, (conf_x, conf_baseline), font, confidence_scale,
        color, confidence_thickness, cv2.LINE_AA,
    )


def draw_detections(image: Any, detections: list[Detection], scale: float) -> None:
    """Draw the board first, then pieces so the board outline stays behind them."""
    height, width = image.shape[:2]
    ordered = sorted(detections, key=lambda detection: detection.cls != BOARD_CLASS)

    for detection in ordered:
        x1, y1, x2, y2 = (
            int(round(value * scale))
            for value in (detection.x1, detection.y1, detection.x2, detection.y2)
        )
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = CLASS_COLORS.get(detection.cls, (255, 255, 255))
        is_board = detection.cls == BOARD_CLASS
        thickness = max(1, int(round((3 if is_board else 2) * scale)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if not is_board:
            draw_piece_label(image, detection, x1, y1, x2, y2, scale, color)


def load_fens_json(input_root: Path) -> tuple[Path, dict[str, str]]:
    fen_path = input_root / "fens.json"
    if not fen_path.is_file():
        raise FileNotFoundError(f"fens.json was not found: {fen_path}")

    try:
        with fen_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {fen_path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected {fen_path} to contain a JSON object mapping image paths to FEN strings")

    normalized: dict[str, str] = {}
    for raw_key, raw_fen in raw_data.items():
        if not isinstance(raw_key, str) or not isinstance(raw_fen, str):
            raise ValueError("Every fens.json entry must map a string image path to a string FEN")
        key = normalize_json_image_key(raw_key)
        if key in normalized:
            raise ValueError(f"Duplicate image key after path normalization in fens.json: {key}")
        placement, _ = parse_placement_fen(raw_fen, source=f"fens.json[{raw_key!r}]")
        normalized[key] = placement

    return fen_path, normalized


def resolve_reference_key(
    image_path: Path,
    input_root: Path,
    expected_fens: dict[str, str],
    basename_counts: dict[str, int],
) -> str | None:
    """Prefer relative paths; allow basename-only keys when the basename is unique."""
    relative_key = image_key(image_path, input_root)
    if relative_key in expected_fens:
        return relative_key

    basename = image_path.name
    if basename_counts.get(basename, 0) == 1 and basename in expected_fens:
        return basename
    return None


def make_output_path(
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    is_directory_input: bool,
    used_output_paths: set[Path],
) -> Path:
    """Create an output path that preserves subdirectories and avoids collisions."""
    if is_directory_input:
        relative = image_path.relative_to(input_root)
        candidate = (output_dir / relative).with_suffix(".jpg")
    else:
        # A single-file input may share its parent with the output directory.
        # Always use a distinct suffix in that case, avoiding both the input
        # itself and a same-stem sibling with another image extension.
        if output_dir == input_root:
            candidate = output_dir / f"{image_path.stem}_annotated.jpg"
        else:
            candidate = output_dir / f"{image_path.stem}.jpg"

    if candidate in used_output_paths:
        candidate = candidate.with_name(f"{candidate.stem}_{image_path.suffix.lstrip('.')}" + candidate.suffix)
    if candidate in used_output_paths:
        raise ValueError(f"Output filename collision for input image: {image_path}")

    used_output_paths.add(candidate)
    return candidate


def warn_if_model_class_names_differ(model: Any) -> None:
    """Warn if model metadata appears inconsistent with this script's class mapping."""
    names = getattr(model, "names", None)
    if not isinstance(names, (dict, list, tuple)):
        return

    expected_names = {
        **{class_id: f"black_{piece}" for class_id, piece in zip(range(6), "pawn rook knight bishop queen king".split())},
        **{class_id: f"white_{piece}" for class_id, piece in zip(range(6, 12), "pawn rook knight bishop queen king".split())},
        BOARD_CLASS: "chess_board",
    }
    mismatches: list[str] = []
    for class_id, expected in expected_names.items():
        try:
            actual = names[class_id]
        except (KeyError, IndexError, TypeError):
            mismatches.append(f"class {class_id} missing (expected {expected})")
            continue
        if str(actual) != expected:
            mismatches.append(f"class {class_id}: model={actual!r}, script expects {expected!r}")
    if mismatches:
        print("[WARN] Model class names do not match the script's class mapping:")
        for mismatch in mismatches:
            print(f"       {mismatch}")
        print("       Check that the model was trained with the expected 13-class order.")


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    is_directory_input = input_path.is_dir()
    input_root = input_path if is_directory_input else input_path.parent

    # Output equal to the input directory would overwrite inputs and is not safe.
    if is_directory_input and output_dir == input_root:
        print("[ERROR] Output directory cannot be the same as the input directory.", file=sys.stderr)
        return 2

    # Only exclude the output directory if it is inside the input directory.
    excluded_output = output_dir if is_directory_input and is_relative_to(output_dir, input_root) else None

    fen_target_path = input_root / "fens.json"

    try:
        image_paths = collect_images(input_path, args.recursive, excluded_output)
        if not image_paths:
            print(f"[ERROR] No supported images found in: {input_path}", file=sys.stderr)
            return 1

        expected_fens: dict[str, str] = {}
        existing_fens: dict[str, str] = {}
        if args.match_fen:
            fen_path, expected_fens = load_fens_json(input_root)
            print(f"[INFO] Loaded {len(expected_fens)} reference FENs from {fen_path}")
        elif args.generate_fen and fen_target_path.exists():
            # Load existing references so generation can append only missing images.
            _, existing_fens = load_fens_json(input_root)
            print(
                f"[INFO] Found existing {fen_target_path} with "
                f"{len(existing_fens)} entries; existing entries will be preserved."
            )

        from ultralytics import YOLO

        print(f"[INFO] Loading model: {args.model}")
        model = YOLO(str(Path(args.model).expanduser()))
        warn_if_model_class_names_differ(model)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Count basenames so recursive mode can safely support legacy basename keys
        # only when there is no ambiguity.
        basename_counts: dict[str, int] = {}
        for image_path in image_paths:
            basename_counts[image_path.name] = basename_counts.get(image_path.name, 0) + 1

        generated_fens: dict[str, str] = dict(existing_fens)
        new_fen_count = 0
        fen_results: dict[str, dict[str, Any]] = {}
        processed = 0
        failed = 0
        compared = 0
        matched = 0
        missing_reference = 0
        board_not_detected = 0
        used_output_paths: set[Path] = set()

        print(f"[INFO] Input:  {input_path}")
        print(f"[INFO] Output: {output_dir}")
        print(f"[INFO] Images: {len(image_paths)}")

        for index, image_path in enumerate(image_paths, start=1):
            print(f"\n[{index}/{len(image_paths)}] {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print("  [WARN] Could not read image; skipping.")
                failed += 1
                continue

            try:
                detections, board = infer_image(model, image, args)

                if args.suppress_overlaps:
                    before = len(detections)
                    detections = suppress_overlapping_detections(detections, args.overlap_threshold)
                    removed = before - len(detections)
                    if removed:
                        print(f"  Suppressed {removed} overlapping detection(s) using IoU.")
                    # FEN generation must use the board that remains after suppression.
                    board = best_board_detection(detections)

                predicted_fen = placement_fen(detections, board)
                relative_key = image_key(image_path, input_root)

                if args.generate_fen:
                    existing_key = resolve_reference_key(
                        image_path, input_root, existing_fens, basename_counts
                    )
                    if existing_key is not None:
                        print(f"  [FEN] Preserving existing entry: {existing_key}")
                    elif predicted_fen is None:
                        print("  [WARN] Board not detected; no FEN generated for this image.")
                    else:
                        generated_fens[relative_key] = predicted_fen
                        new_fen_count += 1
                        print(f"  [FEN] Added: {relative_key} -> {predicted_fen}")

                if args.match_fen:
                    reference_key = resolve_reference_key(
                        image_path, input_root, expected_fens, basename_counts
                    )
                    if reference_key is None:
                        missing_reference += 1
                        fen_results[relative_key] = {
                            "status": "missing_reference",
                            "expected": None,
                            "predicted": predicted_fen,
                            "match": None,
                        }
                        print("  [WARN] No unambiguous matching entry in fens.json.")
                    elif predicted_fen is None:
                        board_not_detected += 1
                        fen_results[relative_key] = {
                            "status": "board_not_detected",
                            "expected": expected_fens[reference_key],
                            "predicted": None,
                            "match": None,
                        }
                        print("  [WARN] Board not detected; this image cannot be compared.")
                    else:
                        expected_fen = expected_fens[reference_key]
                        diff_count, differences, percentage = compare_placements(
                            predicted_fen, expected_fen
                        )
                        is_match = diff_count == 0
                        compared += 1
                        matched += int(is_match)
                        fen_results[relative_key] = {
                            "status": "match" if is_match else "mismatch",
                            "reference_key": reference_key,
                            "expected": expected_fen,
                            "predicted": predicted_fen,
                            "match": is_match,
                            "different_squares": diff_count,
                            "percentage_different": percentage,
                            "differences": differences,
                        }
                        print(f"  FEN {'MATCH' if is_match else 'MISMATCH'}: {diff_count}/64 squares differ.")
                        if differences:
                            print("  " + "; ".join(differences))

                # Resize the original image first; all detection coordinates still
                # refer to the original image and are scaled only while drawing.
                height, width = image.shape[:2]
                output_width = max(1, int(round(width * args.scale)))
                output_height = max(1, int(round(height * args.scale)))
                if output_width == width and output_height == height:
                    annotated = image.copy()
                else:
                    interpolation = cv2.INTER_CUBIC if args.scale > 1 else cv2.INTER_AREA
                    annotated = cv2.resize(
                        image, (output_width, output_height), interpolation=interpolation
                    )

                draw_detections(annotated, detections, args.scale)

                destination = make_output_path(
                    image_path, input_root, output_dir, is_directory_input, used_output_paths
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(destination), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]
                ):
                    raise OSError(f"Could not write annotated image: {destination}")

                print(f"  Detections: {len(detections)}")
                print(f"  Saved: {destination}")
                processed += 1

            except Exception as exc:
                print(f"  [ERROR] Failed to process image: {exc}", file=sys.stderr)
                failed += 1

        if args.generate_fen:
            if new_fen_count:
                # Atomic replacement avoids leaving a partially-written fens.json.
                temporary_fen_path = fen_target_path.with_name(f"{fen_target_path.name}.tmp")
                with temporary_fen_path.open("w", encoding="utf-8") as file:
                    json.dump(generated_fens, file, indent=2, ensure_ascii=False)
                    file.write("\n")
                temporary_fen_path.replace(fen_target_path)
                print(
                    f"\n[INFO] Added {new_fen_count} new FEN entr"
                    f"{'y' if new_fen_count == 1 else 'ies'}; "
                    f"{len(generated_fens)} total: {fen_target_path}"
                )
            elif existing_fens:
                print(
                    f"\n[INFO] No new FEN entries needed; "
                    f"all matching images already have entries ({len(existing_fens)} total)."
                )
            else:
                print("\n[WARN] No FENs were generated because no input image had a detected board.")

        if args.match_fen:
            mismatched = compared - matched
            report = {
                "summary": {
                    "images_processed": processed,
                    "compared": compared,
                    "matched": matched,
                    "mismatched": mismatched,
                    "missing_reference": missing_reference,
                    "board_not_detected": board_not_detected,
                    "accuracy": (matched / compared) if compared else None,
                    "accuracy_percent": (matched / compared * 100.0) if compared else None,
                    "fen_format": "placement only; top image row first; no POV or game-state fields",
                },
                "images": fen_results,
            }
            report_path = output_dir / "fen_match_results.json"
            with report_path.open("w", encoding="utf-8") as file:
                json.dump(report, file, indent=2, ensure_ascii=False)
                file.write("\n")

            print("\n[FEN SUMMARY]")
            print(
                f"Compared: {compared} | Matches: {matched} | Mismatches: {mismatched} | "
                f"Missing references: {missing_reference} | Board not detected: {board_not_detected}"
            )
            if compared:
                print(f"Accuracy: {matched / compared:.2%}")
            else:
                print("Accuracy: N/A (no images were compared)")
            print(f"Report: {report_path}")

        print(
            f"\n[DONE] {processed} processed, {failed} failed. "
            f"Output: {output_dir}"
        )
        return 0 if processed > 0 and failed == 0 else (1 if failed else 0)

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
