#!/usr/bin/env python3
"""
High-Quality 2D Chess Piece & Board Dataset Generator for Ultralytics YOLO26n.

Key features:
1. Tight, Antialiased Bounding Boxes: Computes exact alpha bounding boxes (thresholding
   out translucent fringes and soft shadows) for all pieces.
2. Realistic Visual Overlays: Move highlights, check indicators, legal move dots,
   analysis arrows, inner & outer board coordinates.
3. Realistic Contexts & Environments:
   - Clean full-board views (85-100% frame) on a randomized light/dark/neutral-grey solid background
   - Web / App interfaces (Dark/Light themes, Eval bar, player cards, clocks)
   - Embedded / small boards on realistic background clutter (tests small-object detection / STAL)
   - Physical screen captures (perspective keystone warp, subtle moire, vignette, lighting gradients)
4. Mathematically Exact Transformations: Perspective warps and rotations transform every
   bounding box corner via the exact projective homography matrix.
5. Deterministic Zero-Leakage Splitting: Partitions piece sets, boards, and backgrounds
   into disjoint Train, Validation, and Test subsets with a fixed seed.
6. Multiprocessing & In-Memory Sprite Caching for high generation throughput.
"""

import argparse
import math
import multiprocessing
import os
import random
import shutil
import warnings
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from random_fen_gen import generate_fen

# Suppress PIL palette transparency conversion warning
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# ==============================================================================
# Class Definitions & Label Mappings
# ==============================================================================

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

FEN_TO_PIECE_ID = {
    "p": 0, "r": 1, "n": 2, "b": 3, "q": 4, "k": 5,
    "P": 6, "R": 7, "N": 8, "B": 9, "Q": 10, "K": 11,
}

FEN_TO_PIECE = {
    "p": "bP", "r": "bR", "n": "bN", "b": "bB", "q": "bQ", "k": "bK",
    "P": "wP", "R": "wR", "N": "wN", "B": "wB", "Q": "wQ", "K": "wK",
}

# Directories
BOARDS_DIR = "assets/boards"
PIECES_DIR = "assets/pieces"
BACKGROUND_NOISE_DIR = "assets/random_noise_backgrounds"
DATASETS_IMAGES_DIR = "datasets/images"
DATASETS_LABELS_DIR = "datasets/labels"

# Default Generation Settings
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BOARD_SIZE = 640
DEFAULT_ALPHA_THRESHOLD = 20  # Filters out translucent antialiasing fringe and soft shadows
DEFAULT_SEED = 42
DEFAULT_MIN_PIECE_PX = 12  # Reject/redraw samples with pieces smaller than this
DEFAULT_MAX_SAMPLE_ATTEMPTS = 8  # Redraw attempts per sample before accepting the last one

# Font setup
_CACHED_FONTS = {}


def get_default_font(size=14):
    """Loads a readable TTF font if available on the system, otherwise PIL default."""
    if size in _CACHED_FONTS:
        return _CACHED_FONTS[size]
    ttf_candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in ttf_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _CACHED_FONTS[size] = font
                return font
            except Exception:
                pass
    font = ImageFont.load_default()
    _CACHED_FONTS[size] = font
    return font


# ==============================================================================
# Bounding Box & Geometry Utilities
# ==============================================================================

def get_piece_tight_bbox(img_rgba, alpha_threshold=DEFAULT_ALPHA_THRESHOLD):
    """
    Computes the tight bounding box of visible piece pixels, filtering out
    transparent padding, sub-pixel antialiasing fringes, and faint drop shadows.
    Returns (x1, y1, x2, y2) in local pixel coordinates.
    """
    alpha = np.array(img_rgba.getchannel("A"))
    mask = alpha >= alpha_threshold
    if not np.any(mask):
        return None
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)


def find_perspective_coeffs(src_pts, dst_pts):
    """
    Computes backward mapping coefficients for PIL's Image.transform(PERSPECTIVE).
    Maps dst_pts -> src_pts.
    """
    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        matrix.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    A = np.array(matrix, dtype=float)
    B = np.array(dst_pts).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res


def find_forward_matrix(src_pts, dst_pts):
    """
    Computes forward 3x3 projective homography matrix mapping src_pts -> dst_pts.
    Used to project bounding box corners accurately.
    """
    matrix = []
    for (x, y), (X, Y) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array(dst_pts).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.append(res, 1.0).reshape(3, 3)


def transform_box(box, M, img_w, img_h, min_size=5):
    """
    Transforms an axis-aligned box (x1, y1, x2, y2) through forward homography M,
    computes the new enclosing axis-aligned bounding box, clamps to image boundaries,
    and returns normalized YOLO coordinates (xc, yc, w, h).
    Returns None if the transformed box is too small or entirely out of bounds.
    """
    x1, y1, x2, y2 = box
    corners = np.array([
        [x1, y1, 1.0],
        [x2, y1, 1.0],
        [x2, y2, 1.0],
        [x1, y2, 1.0],
    ]).T  # shape (3, 4)

    transformed = M @ corners
    denom = transformed[2, :]
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)
    transformed /= denom

    xs = transformed[0, :]
    ys = transformed[1, :]

    xmin = max(0.0, float(np.min(xs)))
    ymin = max(0.0, float(np.min(ys)))
    xmax = min(float(img_w), float(np.max(xs)))
    ymax = min(float(img_h), float(np.max(ys)))

    w = xmax - xmin
    h = ymax - ymin
    if w < min_size or h < min_size:
        return None

    # Check that the box was not clipped away to nothingness
    orig_w = x2 - x1
    orig_h = y2 - y1
    if orig_w > 0 and orig_h > 0:
        if (w * h) / (orig_w * orig_h) < 0.20:
            return None

    xc = (xmin + xmax) / (2.0 * img_w)
    yc = (ymin + ymax) / (2.0 * img_h)
    norm_w = w / float(img_w)
    norm_h = h / float(img_h)

    return xc, yc, norm_w, norm_h


def yolo_label_str(class_id, xc, yc, w, h):
    """Formats a YOLO label line with 6 decimal places."""
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ==============================================================================
# Piece & Board Loading with In-Memory Caching
# ==============================================================================

# Worker-local cache: piece_set_name -> {char: (Image, (local_x1, local_y1, local_x2, local_y2))}
_PIECE_CACHE = {}


def load_piece_sprites(piece_set_name, tile_size):
    """
    Loads all 12 piece sprites for a piece set, resizes to tile_size x tile_size,
    and precomputes tight alpha bounding boxes. Cached per worker process.
    """
    cache_key = (piece_set_name, tile_size)
    if cache_key in _PIECE_CACHE:
        return _PIECE_CACHE[cache_key]

    sprites = {}
    set_dir = os.path.join(PIECES_DIR, piece_set_name)
    for char, fname in FEN_TO_PIECE.items():
        path = os.path.join(set_dir, f"{fname}.png")
        if not os.path.exists(path):
            continue
        try:
            im = Image.open(path).convert("RGBA")
            # Resize using high-quality Lanczos resampling
            im_resized = im.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            bbox = get_piece_tight_bbox(im_resized, DEFAULT_ALPHA_THRESHOLD)
            if bbox is None:
                bbox = (0, 0, tile_size, tile_size)
            sprites[char] = (im_resized, bbox)
        except Exception as e:
            print(f"Warning: Failed loading piece {path}: {e}")

    _PIECE_CACHE[cache_key] = sprites
    return sprites


def load_raw_board(board_file, board_size):
    """Loads a board texture and scales it to board_size x board_size."""
    path = os.path.join(BOARDS_DIR, board_file)
    try:
        return Image.open(path).convert("RGB").resize((board_size, board_size), Image.Resampling.BICUBIC)
    except Exception as e:
        print(f"Warning: Failed loading board {path}: {e}")
        # Fallback procedural checkerboard
        return generate_procedural_board(board_size)


def generate_procedural_board(board_size=DEFAULT_BOARD_SIZE):
    """Generates a clean procedural checkerboard in case of missing assets."""
    board = Image.new("RGB", (board_size, board_size))
    draw = ImageDraw.Draw(board)
    tile = board_size // 8
    light_colors = [(240, 217, 181), (238, 238, 210), (240, 240, 240), (232, 235, 239)]
    dark_colors = [(181, 136, 99), (118, 150, 86), (120, 140, 160), (125, 135, 150)]
    light = random.choice(light_colors)
    dark = random.choice(dark_colors)
    for r in range(8):
        for c in range(8):
            color = light if (r + c) % 2 == 0 else dark
            draw.rectangle([c * tile, r * tile, (c + 1) * tile, (r + 1) * tile], fill=color)
    return board


def extract_tile_colors(raw_board_img, tile_size, border_px=0):
    """
    Extracts representative (R, G, B) colors for light and dark tiles
    from the raw checkerboard image.
    Tile (col 0, row 0) is Light; Tile (col 1, row 0) is Dark.
    Samples from the central 50% region of the squares.
    `border_px` offsets sampling past an outer coordinate frame, if present.
    """
    arr = np.array(raw_board_img)
    pad = max(1, tile_size // 4)
    y1 = border_px + pad
    y2 = border_px + tile_size - pad
    # Light square (col 0, row 0)
    light_crop = arr[y1:y2, border_px + pad : border_px + tile_size - pad]
    light_rgb = tuple(int(round(c)) for c in light_crop.mean(axis=(0, 1))[:3])
    # Dark square (col 1, row 0)
    dark_crop = arr[y1:y2, border_px + tile_size + pad : border_px + 2 * tile_size - pad]
    dark_rgb = tuple(int(round(c)) for c in dark_crop.mean(axis=(0, 1))[:3])
    return light_rgb, dark_rgb


def get_random_solid_background(light_color, dark_color):
    """
    Randomly selects between three solid background variations:
    1. Background matching the light tile color
    2. Background matching the dark tile color
    3. A neutral grey background (128, 128, 128)
    """
    variant = random.choice(["light", "dark", "grey"])
    if variant == "light":
        return light_color
    elif variant == "dark":
        return dark_color
    else:
        return (128, 128, 128)


# ==============================================================================
# Realistic Visual Overlays: Highlights, Arrows, Coordinates
# ==============================================================================

HIGHLIGHT_COLORS = [
    (186, 202, 68, 130),   # Chess.com yellow-green
    (135, 158, 90, 130),   # Lichess olive green
    (64, 169, 255, 120),   # Cyan / Blue
    (168, 85, 247, 120),   # Lavender / Purple
    (250, 204, 21, 130),   # Warm Amber / Gold
]

CHECK_COLOR = (235, 50, 50, 150)  # Red check highlight


def draw_arrow(overlay_draw, p1, p2, color=(40, 180, 80, 175), width=8, head_size=22):
    """Renders a smooth analysis arrow with a distinct triangular head."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 15:
        return
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    h = min(head_size, dist * 0.40)
    w = h * 0.85

    xs, ys = x2 - ux * h, y2 - uy * h

    # Arrow shaft line
    overlay_draw.line([(x1, y1), (xs, ys)], fill=color, width=width)
    # Triangular arrow head
    pt1 = (x2, y2)
    pt2 = (xs + nx * w, ys + ny * w)
    pt3 = (xs - nx * w, ys - ny * w)
    overlay_draw.polygon([pt1, pt2, pt3], fill=color)


def apply_board_overlays(board_img, fen, tile_size, flipped=False):
    """
    Adds realistic visual UI elements to the board:
    - Last move highlights (from_sq, to_sq)
    - Checked king red highlight
    - Legal move dots and capture rings
    - Analysis arrows
    """
    overlay = Image.new("RGBA", board_img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    # Collect occupied squares: map (col, row) -> char
    occupied = {}
    kings = {}
    for r_idx, rank in enumerate(fen.split()[0].split("/")):
        c_idx = 0
        for char in rank:
            if char.isdigit():
                c_idx += int(char)
            else:
                occupied[(c_idx, r_idx)] = char
                if char in ("K", "k"):
                    kings[char] = (c_idx, r_idx)
                c_idx += 1

    # 1. Last move square highlights (45% chance)
    if random.random() < 0.45 and occupied:
        hl_color = random.choice(HIGHLIGHT_COLORS)
        # Pick a random occupied piece as the destination square
        to_pos = random.choice(list(occupied.keys()))
        # Pick an adjacent or related square as the origin square
        c, r = to_pos
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1), (1, 2), (-1, 2)]
        from_candidates = [(c + dc, r + dr) for dc, dr in deltas if 0 <= c + dc < 8 and 0 <= r + dr < 8]
        from_pos = random.choice(from_candidates) if from_candidates else to_pos

        for col, row in [from_pos, to_pos]:
            x1, y1 = col * tile_size, row * tile_size
            ov_draw.rectangle([x1, y1, x1 + tile_size, y1 + tile_size], fill=hl_color)

    # 2. King in check highlight (15% chance)
    if random.random() < 0.15 and kings:
        checked_king = random.choice(list(kings.values()))
        kc, kr = checked_king
        kx1, ky1 = kc * tile_size, kr * tile_size
        ov_draw.rectangle([kx1, ky1, kx1 + tile_size, ky1 + tile_size], fill=CHECK_COLOR)

    # 3. Legal move dots & capture rings (55% chance)
    if random.random() < 0.55:
        empty_squares = [(c, r) for c in range(8) for r in range(8) if (c, r) not in occupied]
        if empty_squares:
            dot_color = random.choice([
                (40, 40, 40, 110),
                (20, 130, 60, 110),
                (50, 50, 50, 90),
            ])
            dot_r = int(tile_size * random.uniform(0.13, 0.16))
            num_dots = min(random.randint(3, 9), len(empty_squares))
            for c, r in random.sample(empty_squares, num_dots):
                cx = c * tile_size + tile_size // 2
                cy = r * tile_size + tile_size // 2
                ov_draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=dot_color)

        # Target capture ring on enemy piece (1 to 3 rings)
        if occupied and random.random() < 0.65:
            num_rings = min(random.randint(1, 3), len(occupied))
            for tc, tr in random.sample(list(occupied.keys()), num_rings):
                cx = tc * tile_size + tile_size // 2
                cy = tr * tile_size + tile_size // 2
                ring_r = int(tile_size * 0.42)
                ring_color = random.choice([
                    (30, 30, 30, 100),
                    (220, 50, 50, 120),
                    (30, 140, 60, 120),
                ])
                ov_draw.ellipse(
                    [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
                    outline=ring_color,
                    width=max(3, int(tile_size * 0.045)),
                )

    # 4. Analysis arrows (60% chance)
    if random.random() < 0.60:
        num_arrows = random.choices([1, 2, 3], weights=[0.45, 0.35, 0.20])[0]
        used_pairs = set()
        arrow_palette = [
            (40, 180, 80, 185),   # Engine Best: Green
            (59, 130, 246, 185),  # 2nd Best: Blue
            (245, 140, 30, 185),  # 3rd Best: Orange
            (239, 68, 68, 185),   # Threat/Blunder: Red
        ]
        for a_idx in range(num_arrows):
            if occupied and random.random() < 0.80:
                c1, r1 = random.choice(list(occupied.keys()))
            else:
                c1, r1 = random.randint(0, 7), random.randint(0, 7)
            c2, r2 = random.randint(0, 7), random.randint(0, 7)
            if (c1, r1) != (c2, r2) and (c1, r1, c2, r2) not in used_pairs:
                used_pairs.add((c1, r1, c2, r2))
                color = arrow_palette[a_idx % len(arrow_palette)]
                p1 = (c1 * tile_size + tile_size // 2, r1 * tile_size + tile_size // 2)
                p2 = (c2 * tile_size + tile_size // 2, r2 * tile_size + tile_size // 2)
                draw_arrow(
                    ov_draw,
                    p1,
                    p2,
                    color=color,
                    width=max(4, int(tile_size * 0.09)),
                    head_size=max(16, int(tile_size * 0.30)),
                )

    return Image.alpha_composite(board_img.convert("RGBA"), overlay).convert("RGB")


def render_board_coordinates(board_img, tile_size, flipped=False, style="inner", frame_color=(45, 42, 38)):
    """
    Renders standard rank ('1'-'8') and file ('a'-'h') coordinate labels.
    - 'inner': Drawn in corners of border tiles (Chess.com / Lichess modern style)
    - 'outer': Border frame with centered letters and numbers
    `frame_color` matches the solid background so no visible boundary is added.
    """
    font = get_default_font(size=max(10, int(tile_size * 0.16)))

    if style == "inner":
        draw = ImageDraw.Draw(board_img)
        files = ["h", "g", "f", "e", "d", "c", "b", "a"] if flipped else ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"] if flipped else ["8", "7", "6", "5", "4", "3", "2", "1"]

        for col in range(8):
            for row in range(8):
                is_light = (row + col) % 2 == 0
                text_color = (130, 95, 65) if is_light else (225, 210, 195)
                # File letter on bottom rank (row 7)
                if row == 7:
                    x = (col + 1) * tile_size - int(tile_size * 0.18)
                    y = (row + 1) * tile_size - int(tile_size * 0.22)
                    draw.text((x, y), files[col], fill=text_color, font=font)
                # Rank number on left file (col 0)
                if col == 0:
                    x = col * tile_size + int(tile_size * 0.05)
                    y = row * tile_size + int(tile_size * 0.05)
                    draw.text((x, y), ranks[row], fill=text_color, font=font)
        return board_img, 0  # 0 border added

    elif style == "outer":
        border_px = max(16, int(tile_size * 0.25))
        new_size = board_img.width + 2 * border_px
        framed = Image.new("RGB", (new_size, new_size), frame_color)
        framed.paste(board_img, (border_px, border_px))
        draw = ImageDraw.Draw(framed)

        files = ["h", "g", "f", "e", "d", "c", "b", "a"] if flipped else ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"] if flipped else ["8", "7", "6", "5", "4", "3", "2", "1"]

        for idx in range(8):
            center_pos = border_px + idx * tile_size + tile_size // 2 - 4
            # Top and bottom file letters
            draw.text((center_pos, border_px // 2 - 6), files[idx], fill=(180, 180, 180), font=font)
            draw.text((center_pos, new_size - border_px + 2), files[idx], fill=(180, 180, 180), font=font)
            # Left and right rank numbers
            draw.text((border_px // 2 - 4, center_pos), ranks[idx], fill=(180, 180, 180), font=font)
            draw.text((new_size - border_px + 4, center_pos), ranks[idx], fill=(180, 180, 180), font=font)
        return framed, border_px

    return board_img, 0


# ==============================================================================
# Piece Rendering & Exact Bounding Box Placement
# ==============================================================================

def render_pieces_on_board(board_img, piece_sprites, fen, tile_size, border_offset=0):
    """
    Places chess pieces from FEN onto the board image.
    Calculates exact tight bounding boxes (class_id, x1, y1, x2, y2)
    in board image pixel coordinates.
    """
    piece_boxes = []

    for r_idx, rank in enumerate(fen.split()[0].split("/")):
        c_idx = 0
        for char in rank:
            if char.isdigit():
                c_idx += int(char)
            else:
                if char in piece_sprites:
                    class_id = FEN_TO_PIECE_ID[char]
                    sprite, (lx1, ly1, lx2, ly2) = piece_sprites[char]

                    # Tile base coordinates
                    base_x = border_offset + c_idx * tile_size
                    base_y = border_offset + r_idx * tile_size

                    # Subtle natural placement jitter (+/- 1-2 pixels)
                    jx = random.randint(-2, 2)
                    jy = random.randint(-2, 2)

                    # Subtle piece scaling variation (93% - 103%)
                    piece_to_paste = sprite
                    bx1, by1, bx2, by2 = lx1, ly1, lx2, ly2

                    if random.random() < 0.30:
                        scale = random.uniform(0.92, 1.04)
                        nw = max(10, int(tile_size * scale))
                        nh = max(10, int(tile_size * scale))
                        piece_to_paste = sprite.resize((nw, nh), Image.Resampling.LANCZOS)
                        # Recompute tight bbox for scaled piece
                        tb = get_piece_tight_bbox(piece_to_paste, DEFAULT_ALPHA_THRESHOLD)
                        if tb:
                            bx1, by1, bx2, by2 = tb
                        # Align bottom baseline
                        off_x = (tile_size - nw) // 2
                        off_y = tile_size - nh
                        paste_x = base_x + off_x + jx
                        paste_y = base_y + off_y + jy
                    else:
                        paste_x = base_x + jx
                        paste_y = base_y + jy

                    # Paste with alpha channel as mask
                    board_img.paste(piece_to_paste, (paste_x, paste_y), piece_to_paste)

                    # Compute exact board-level coordinates
                    abs_x1 = max(0, paste_x + bx1)
                    abs_y1 = max(0, paste_y + by1)
                    abs_x2 = min(board_img.width, paste_x + bx2)
                    abs_y2 = min(board_img.height, paste_y + by2)

                    if abs_x2 > abs_x1 and abs_y2 > abs_y1:
                        piece_boxes.append((class_id, abs_x1, abs_y1, abs_x2, abs_y2))

                c_idx += 1

    return board_img, piece_boxes


# ==============================================================================
# Composition Environments (Full Board, Web/App UI, Clutter, Screen Photo)
# ==============================================================================

UI_THEME_COLORS = [
    # Dark mode variants
    {"bg": (38, 36, 33), "card": (49, 46, 43), "text": (220, 220, 220), "accent": (129, 182, 76)},
    {"bg": (22, 21, 18), "card": (30, 29, 26), "text": (210, 210, 210), "accent": (86, 120, 68)},
    {"bg": (48, 46, 43), "card": (60, 58, 55), "text": (240, 240, 240), "accent": (135, 158, 90)},
    {"bg": (26, 26, 26), "card": (36, 36, 36), "text": (230, 230, 230), "accent": (59, 130, 246)},
    # Light mode variants
    {"bg": (240, 240, 240), "card": (255, 255, 255), "text": (40, 40, 40), "accent": (74, 144, 226)},
    {"bg": (248, 249, 250), "card": (255, 255, 255), "text": (33, 37, 41), "accent": (118, 150, 86)},
]

UI_ACCENT_COLORS = [
    (59, 130, 246),   # blue
    (34, 197, 94),    # green
    (239, 68, 68),    # red
    (234, 179, 8),    # amber
    (168, 85, 247),   # purple
    (6, 182, 212),    # cyan
    (249, 115, 22),   # orange
    (236, 72, 153),   # pink
]

UI_CLUTTER_WORDS = [
    "Play", "Games", "Live", "Analysis", "Players", "Chat", "Settings",
    "Profile", "Search", "Home", "Puzzles", "Watch", "Learn", "Train",
    "Review", "Board", "Engine", "Score", "Depth", "Match", "Lobby",
    "Friends", "History", "Open", "Share", "Save", "Apply", "Cancel",
    "Options", "Notifications", "New Game", "Resign", "Draw", "Rematch",
    "Online", "Ranked", "Casual", "Tournament", "Broadcast", "Stats",
    "Move", "Best", "Eval", "Tactics", "Study", "Library", "Menu",
]


def draw_random_ui_clutter(canvas, canvas_size, exclude_rects=None, theme=None):
    """
    Draws randomized UI-like distractor elements on the background.

    This is called before the chessboard is pasted, so clutter never occludes
    pieces or corrupts labels. Elements are intentionally varied in type, size,
    opacity, and placement: cards, windows, toolbars, sidebars, chat bubbles,
    charts, buttons, toggles, clocks, progress bars, tabs, and text blocks.
    """
    if theme is None:
        theme = random.choice(UI_THEME_COLORS)

    scale = max(0.5, canvas_size / 640.0)

    def s(value):
        return max(1, int(round(value * scale)))

    def jitter(color, amount=18):
        return tuple(max(0, min(255, c + random.randint(-amount, amount))) for c in color[:3])

    def rgba(color, alpha):
        return jitter(color) + (max(0, min(255, alpha)),)

    def font(size):
        return get_default_font(max(6, s(size)))

    def text(x, y, value, size=10, color=None, alpha=180):
        draw.text((x, y), value, font=font(size), fill=rgba(color or theme["text"], alpha))

    def centered_text(x1, y1, x2, y2, value, size=9, color=None, alpha=190):
        f = font(size)
        bbox = draw.textbbox((0, 0), value, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x1 + max(0, (x2 - x1 - tw) // 2), y1 + max(0, (y2 - y1 - th) // 2)),
            value,
            font=f,
            fill=rgba(color or theme["text"], alpha),
        )

    def overlaps(x1, y1, x2, y2):
        margin = s(7)
        for ex1, ey1, ex2, ey2 in exclude_rects or []:
            if not (x2 + margin < ex1 or x1 - margin > ex2 or y2 + margin < ey1 or y1 - margin > ey2):
                return True
        return False

    def free_rect(w, h, pad=None):
        pad = s(5) if pad is None else pad
        w = max(1, min(w, canvas_size - 2 * pad))
        h = max(1, min(h, canvas_size - 2 * pad))
        max_x = max(pad, canvas_size - w - pad)
        max_y = max(pad, canvas_size - h - pad)
        for _ in range(24):
            x1 = random.randint(pad, max_x)
            y1 = random.randint(pad, max_y)
            x2, y2 = x1 + w, y1 + h
            if not overlaps(x1, y1, x2, y2):
                return x1, y1, x2, y2
        return None

    def rounded_panel(x1, y1, x2, y2, fill_alpha=None, radius=None):
        radius = radius or s(random.randint(4, 9))
        radius = max(1, min(radius, max(1, (x2 - x1) // 3), max(1, (y2 - y1) // 3)))
        draw.rounded_rectangle(
            [x1 + s(3), y1 + s(3), x2 + s(3), y2 + s(3)],
            radius=radius,
            fill=(0, 0, 0, random.randint(25, 55)),
        )
        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=radius,
            fill=rgba(theme["card"], fill_alpha or random.randint(140, 220)),
            outline=rgba(theme["text"], random.randint(20, 65)),
            width=max(1, s(1)),
        )
        return radius

    def text_lines(x1, y1, x2, y2=None, max_lines=4):
        if y2 is None:
            y2 = canvas_size - s(4)
        count = random.randint(1, max_lines)
        line_h = max(2, s(3))
        gap = max(4, s(10))
        for i in range(count):
            yy = y1 + i * gap
            if yy + line_h > y2:
                break
            width = random.randint(max(4, int((x2 - x1) * 0.30)), max(6, int((x2 - x1) * 0.88)))
            draw.rounded_rectangle(
                [x1, yy, min(x2, x1 + width), yy + line_h],
                radius=max(1, line_h // 2),
                fill=rgba(theme["text"], random.randint(70, 145)),
            )

    def draw_panel():
        w, h = random.randint(s(70), s(250)), random.randint(s(55), s(175))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2)
        text(x1 + s(9), y1 + s(7), random.choice(UI_CLUTTER_WORDS), size=random.randint(9, 14))
        text_lines(x1 + s(9), y1 + s(30), x2 - s(9), y2 - s(9), max_lines=4)

    def draw_window():
        w, h = random.randint(s(110), s(290)), random.randint(s(80), s(210))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(170, 230))
        bar_y = y1 + s(22)
        draw.line([x1, bar_y, x2, bar_y], fill=rgba(theme["text"], 35), width=max(1, s(1)))
        for i, c in enumerate([(239, 68, 68), (234, 179, 8), (34, 197, 94)]):
            cx = x1 + s(10) + i * s(12)
            draw.ellipse([cx, y1 + s(7), cx + s(6), y1 + s(13)], fill=rgba(c, 190))
        text(x1 + s(48), y1 + s(5), random.choice(UI_CLUTTER_WORDS), size=random.randint(9, 13))
        text_lines(x1 + s(10), bar_y + s(12), x2 - s(10), y2 - s(10), max_lines=5)

    def draw_toolbar():
        w = random.randint(s(220), max(s(230), canvas_size - s(20)))
        h = random.randint(s(24), s(46))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(150, 215), radius=s(5))
        draw.ellipse([x1 + s(8), y1 + s(7), x1 + s(24), y1 + s(23)], fill=rgba(theme["accent"], 190))
        text(x1 + s(32), y1 + s(6), random.choice(UI_CLUTTER_WORDS), size=random.randint(9, 12))
        for i in range(random.randint(2, 5)):
            rx = x2 - s(18) - i * s(36)
            if rx < x1 + s(90):
                break
            draw.rounded_rectangle(
                [rx, y1 + s(7), rx + s(28), y1 + s(17)],
                radius=s(3),
                fill=rgba(theme["text"], random.randint(45, 90)),
            )

    def draw_sidebar():
        w = random.randint(s(55), s(140))
        h = random.randint(s(150), max(s(160), int(canvas_size * 0.72)))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(150, 220))
        rows = random.randint(3, 7)
        row_h = max(s(12), (y2 - y1 - s(20)) // max(1, rows))
        for i in range(rows):
            ry = y1 + s(10) + i * row_h
            if ry + s(10) > y2 - s(5):
                break
            draw.ellipse([x1 + s(7), ry, x1 + s(19), ry + s(12)], fill=rgba(theme["accent"], random.randint(120, 200)))
            width = random.randint(s(14), max(s(16), (x2 - x1) - s(28)))
            draw.rounded_rectangle(
                [x1 + s(24), ry + s(2), x1 + s(24) + width, ry + s(8)],
                radius=s(3),
                fill=rgba(theme["text"], random.randint(70, 150)),
            )

    def draw_chat():
        w, h = random.randint(s(100), s(230)), random.randint(s(60), s(145))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(150, 220))
        text(x1 + s(9), y1 + s(7), random.choice(["Chat", "Messages", "Live Chat", "Comments"]), size=random.randint(9, 13))
        for i in range(random.randint(1, 3)):
            by = y1 + s(30) + i * s(28)
            if by + s(20) > y2 - s(5):
                break
            draw.ellipse([x1 + s(8), by + s(2), x1 + s(20), by + s(14)], fill=rgba(theme["accent"], 180))
            text_lines(x1 + s(26), by + s(2), x2 - s(8), by + s(18), max_lines=2)

    def draw_button():
        w, h = random.randint(s(52), s(120)), random.randint(s(22), s(40))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        accent = random.choice(UI_ACCENT_COLORS + [theme["accent"]])
        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=s(random.randint(4, 10)),
            fill=rgba(accent, random.randint(150, 225)),
            outline=rgba(theme["text"], 45),
            width=max(1, s(1)),
        )
        centered_text(x1, y1, x2, y2, random.choice(UI_CLUTTER_WORDS), size=random.randint(8, 11))

    def draw_chart():
        w, h = random.randint(s(100), s(245)), random.randint(s(70), s(165))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(155, 225))
        text(x1 + s(9), y1 + s(7), random.choice(["Evaluation", "Accuracy", "Stats", "Activity"]), size=random.randint(8, 12))
        base = y2 - s(14)
        left = x1 + s(12)
        right = x2 - s(10)
        bars = random.randint(5, 12)
        gap = max(2, (right - left) // max(1, bars))
        for i in range(bars):
            bx = left + i * gap
            bh = random.randint(s(6), max(s(8), int((base - y1 - s(30)) * random.uniform(0.15, 1.0))))
            draw.rectangle(
                [bx, base - bh, bx + max(2, gap - s(2)), base],
                fill=rgba(random.choice(UI_ACCENT_COLORS), random.randint(130, 210)),
            )
        draw.line([left, base, right, base], fill=rgba(theme["text"], 80), width=max(1, s(1)))

    def draw_toggle():
        w, h = random.randint(s(60), s(120)), random.randint(s(20), s(32))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        track_w = s(30)
        track_x2 = x1 + track_w
        draw.rounded_rectangle(
            [x1, y1 + s(4), track_x2, y2 - s(4)],
            radius=max(1, (y2 - y1) // 2),
            fill=rgba(random.choice(UI_ACCENT_COLORS), random.randint(130, 190)),
        )
        knob_x = track_x2 - s(10) if random.random() < 0.5 else x1 + s(2)
        draw.ellipse([knob_x, y1 + s(5), knob_x + s(10), y1 + s(15)], fill=rgba(theme["text"], 220))
        text(track_x2 + s(8), y1 + s(6), random.choice(UI_CLUTTER_WORDS), size=random.randint(7, 10))

    def draw_progress():
        w, h = random.randint(s(80), s(230)), random.randint(s(8), s(18))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        draw.rounded_rectangle([x1, y1, x2, y2], radius=max(1, h // 2), fill=rgba(theme["text"], 35))
        fill_w = random.randint(max(2, (x2 - x1) // 8), max(3, (x2 - x1) - 2))
        draw.rounded_rectangle(
            [x1, y1, x1 + fill_w, y2],
            radius=max(1, h // 2),
            fill=rgba(random.choice(UI_ACCENT_COLORS), random.randint(150, 220)),
        )

    def draw_eval_bar():
        if random.random() < 0.5:
            w, h = random.randint(s(8), s(16)), random.randint(s(90), s(260))
        else:
            w, h = random.randint(s(90), s(260)), random.randint(s(8), s(16))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        draw.rectangle([x1, y1, x2, y2], fill=rgba((35, 35, 35), 220))
        ratio = random.uniform(0.15, 0.85)
        if h > w:
            split = y1 + int(h * ratio)
            draw.rectangle([x1, split, x2, y2], fill=rgba((245, 245, 245), 225))
        else:
            split = x1 + int(w * ratio)
            draw.rectangle([x1, y1, split, y2], fill=rgba((245, 245, 245), 225))
        draw.rectangle([x1, y1, x2, y2], outline=rgba(theme["text"], 60), width=max(1, s(1)))

    def draw_clock():
        w, h = random.randint(s(62), s(110)), random.randint(s(28), s(50))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(175, 235), radius=s(4))
        value = f"{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        centered_text(x1, y1, x2, y2, value, size=random.randint(10, 15))

    def draw_text_block():
        w = random.randint(s(80), s(260))
        h = random.randint(s(24), s(70))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        if random.random() < 0.5:
            text(x1, y1, random.choice(UI_CLUTTER_WORDS), size=random.randint(9, 14), alpha=random.randint(130, 210))
        text_lines(x1, y1 + s(16), x2, y2, max_lines=3)

    def draw_tabs():
        count = random.randint(2, 5)
        w = random.randint(s(70), s(220))
        h = random.randint(s(22), s(38))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        tab_w = max(s(24), (x2 - x1) // count)
        for i in range(count):
            tx1 = x1 + i * tab_w
            if tx1 + tab_w > x2:
                break
            active = i == random.randrange(count)
            draw.rounded_rectangle(
                [tx1, y1, tx1 + tab_w - s(2), y2],
                radius=s(4),
                fill=rgba(theme["accent"] if active else theme["card"], random.randint(120, 210)),
                outline=rgba(theme["text"], 45),
                width=max(1, s(1)),
            )
            centered_text(tx1, y1, tx1 + tab_w - s(2), y2, random.choice(UI_CLUTTER_WORDS), size=random.randint(7, 10))

    def draw_modal():
        w, h = random.randint(s(170), s(310)), random.randint(s(110), s(220))
        r = free_rect(w, h)
        if not r:
            return
        x1, y1, x2, y2 = r
        rounded_panel(x1, y1, x2, y2, fill_alpha=random.randint(190, 240), radius=s(8))
        text(x1 + s(12), y1 + s(10), random.choice(UI_CLUTTER_WORDS), size=random.randint(10, 15))
        text_lines(x1 + s(12), y1 + s(36), x2 - s(12), y2 - s(45), max_lines=4)
        for i in range(random.randint(1, 2)):
            bw = s(random.randint(45, 75))
            bh = s(random.randint(20, 28))
            bx = x2 - s(12) - (i + 1) * bw - i * s(8)
            by = y2 - s(12) - bh
            draw.rounded_rectangle(
                [bx, by, bx + bw, by + bh],
                radius=s(4),
                fill=rgba(random.choice(UI_ACCENT_COLORS + [theme["card"]]), random.randint(140, 220)),
                outline=rgba(theme["text"], 45),
                width=max(1, s(1)),
            )
            centered_text(bx, by, bx + bw, by + bh, random.choice(UI_CLUTTER_WORDS), size=random.randint(7, 10))

    overlay = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    density = min(30, max(7, int(random.randint(10, 24) * scale)))
    drawers = [
        (draw_panel, 14),
        (draw_window, 8),
        (draw_toolbar, 5),
        (draw_sidebar, 6),
        (draw_chat, 10),
        (draw_button, 10),
        (draw_chart, 7),
        (draw_toggle, 6),
        (draw_progress, 6),
        (draw_eval_bar, 5),
        (draw_clock, 8),
        (draw_text_block, 6),
        (draw_tabs, 5),
        (draw_modal, 5),
    ]
    choices, weights = zip(*drawers)
    for _ in range(density):
        random.choices(choices, weights=weights)[0]()

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def compose_full_board(
    board_img,
    piece_boxes,
    border_px=0,
    base_board_size=DEFAULT_BOARD_SIZE,
    canvas_size=DEFAULT_IMAGE_SIZE,
    bg_color=None,
):
    """
    Composition 1: Full-board view.
    Board fills 85-98% of the canvas. Center-aligned with subtle background padding.
    The background is randomly matched to a light tile, dark tile, or neutral grey.
    The chess_board bounding box strictly bounds the 64 checkerboard squares.
    """
    if bg_color is None:
        light_color, dark_color = extract_tile_colors(
            board_img, base_board_size // 8, border_px=border_px
        )
        bg_color = get_random_solid_background(light_color, dark_color)

    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    bw, bh = board_img.size

    # Fit within canvas
    scale = min((canvas_size - 10) / bw, (canvas_size - 10) / bh)
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    resized_board = board_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    ox = (canvas_size - new_w) // 2
    oy = (canvas_size - new_h) // 2
    canvas.paste(resized_board, (ox, oy))

    # Scale piece boxes
    scaled_boxes = []
    sx = new_w / bw
    sy = new_h / bh
    for cid, x1, y1, x2, y2 in piece_boxes:
        scaled_boxes.append((cid, ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy))

    # Bounding box strictly for the 64 checkerboard squares (excluding outer coordinate borders)
    chk_x1 = ox + border_px * sx
    chk_y1 = oy + border_px * sy
    chk_x2 = ox + (border_px + base_board_size) * sx
    chk_y2 = oy + (border_px + base_board_size) * sy
    board_box = (12, chk_x1, chk_y1, chk_x2, chk_y2)
    return canvas, scaled_boxes, board_box


def compose_web_ui(
    board_img, piece_boxes, border_px=0, base_board_size=DEFAULT_BOARD_SIZE, canvas_size=DEFAULT_IMAGE_SIZE
):
    """
    Composition 2: Realistic Web / Desktop Chess Interface.
    Simulates Chess.com or Lichess with eval bar, player names, clocks, and sidebar.
    The chess_board bounding box strictly bounds the 64 checkerboard squares.
    """
    theme = random.choice(UI_THEME_COLORS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), theme["bg"])
    font = get_default_font(size=12)

    # Board size occupies 55% - 72% of canvas height
    target_board_size = random.randint(int(canvas_size * 0.55), int(canvas_size * 0.72))
    bw, bh = board_img.size
    resized_board = board_img.resize((target_board_size, target_board_size), Image.Resampling.LANCZOS)

    # Position board slightly to the left to leave room for sidebar/eval bar
    board_ox = random.randint(40, max(42, canvas_size - target_board_size - 30))
    board_oy = (canvas_size - target_board_size) // 2

    # Add random UI clutter to the background before the board is pasted.
    canvas = draw_random_ui_clutter(
        canvas,
        canvas_size,
        exclude_rects=[(board_ox, board_oy, board_ox + target_board_size, board_oy + target_board_size)],
        theme=theme,
    )
    canvas.paste(resized_board, (board_ox, board_oy))
    draw = ImageDraw.Draw(canvas)

    # 1. Vertical Evaluation Bar on the left of the board
    eval_w = max(10, int(target_board_size * 0.035))
    eval_x = board_ox - eval_w - 6
    if eval_x >= 4:
        draw.rectangle([eval_x, board_oy, eval_x + eval_w, board_oy + target_board_size], fill=(35, 35, 35))
        eval_score = random.uniform(-3.5, 3.5)
        # Sigmoid height mapping
        white_ratio = 1.0 / (1.0 + math.exp(-eval_score / 2.5))
        white_h = int(target_board_size * white_ratio)
        draw.rectangle(
            [eval_x, board_oy + target_board_size - white_h, eval_x + eval_w, board_oy + target_board_size],
            fill=(245, 245, 245),
        )

    # 2. Player Info Cards (Top and Bottom)
    top_y = board_oy - 28
    bot_y = board_oy + target_board_size + 6
    if top_y >= 4 and bot_y + 24 <= canvas_size:
        # Opponent card (top)
        draw.rectangle([board_ox, top_y, board_ox + 22, top_y + 22], fill=theme["card"])
        draw.text((board_ox + 28, top_y + 4), "Opponent (1850)", fill=theme["text"], font=font)
        # Top clock
        draw.rounded_rectangle(
            [board_ox + target_board_size - 56, top_y, board_ox + target_board_size, top_y + 22],
            radius=3,
            fill=theme["card"],
        )
        draw.text((board_ox + target_board_size - 50, top_y + 4), "04:52", fill=(255, 255, 255), font=font)

        # Player card (bottom)
        draw.rectangle([board_ox, bot_y, board_ox + 22, bot_y + 22], fill=theme["card"])
        draw.text((board_ox + 28, bot_y + 4), "Player (1920)", fill=theme["text"], font=font)
        # Bottom clock
        draw.rounded_rectangle(
            [board_ox + target_board_size - 56, bot_y, board_ox + target_board_size, bot_y + 22],
            radius=3,
            fill=theme["card"],
        )
        draw.text((board_ox + target_board_size - 50, bot_y + 4), "03:18", fill=(255, 255, 255), font=font)

    # Scale piece boxes
    scaled_boxes = []
    sx = target_board_size / bw
    sy = target_board_size / bh
    for cid, x1, y1, x2, y2 in piece_boxes:
        scaled_boxes.append((cid, board_ox + x1 * sx, board_oy + y1 * sy, board_ox + x2 * sx, board_oy + y2 * sy))

    # Bounding box strictly for the 64 checkerboard squares (excluding outer coordinate borders)
    chk_x1 = board_ox + border_px * sx
    chk_y1 = board_oy + border_px * sy
    chk_x2 = board_ox + (border_px + base_board_size) * sx
    chk_y2 = board_oy + (border_px + base_board_size) * sy
    board_box = (12, chk_x1, chk_y1, chk_x2, chk_y2)
    return canvas, scaled_boxes, board_box


def get_embedded_board_size(canvas_size):
    """
    Chooses a medium/small embedded board size as a fraction of the canvas.
    The old fixed 80-100 px range made pieces only ~8 px at 640 px, which is
    too small to label and learn reliably. This keeps the board embedded while
    giving pieces enough pixels to supervise.
    """
    min_size = max(120, int(round(canvas_size * 0.19)))
    max_size = int(round(canvas_size * 0.62))

    # Keep the board inside the canvas for unusually small resolutions.
    min_size = min(min_size, canvas_size - 8)
    max_size = min(max_size, canvas_size - 4)
    max_size = max(max_size, min_size + 1)

    mode = int(round(canvas_size * 0.375))
    mode = min(max(mode, min_size), max_size)
    return int(random.triangular(min_size, max_size, mode))


def compose_embedded_board(
    board_img,
    piece_boxes,
    border_px=0,
    base_board_size=DEFAULT_BOARD_SIZE,
    bg_file=None,
    canvas_size=DEFAULT_IMAGE_SIZE,
):
    """
    Composition 3: Small or embedded board on realistic photographic background.
    Tests detection robustness on small chessboards and cluttered scenes (STAL).

    The board occupies roughly 19-62% of the canvas (about 120-400 px at 640),
    instead of the previous fixed 80-100 px which produced unlearnably tiny pieces.
    The chess_board bounding box strictly bounds the 64 checkerboard squares.
    """
    canvas = None
    if bg_file:
        candidate = bg_file if os.path.isabs(bg_file) else os.path.join(BACKGROUND_NOISE_DIR, bg_file)
        if os.path.exists(candidate):
            try:
                canvas = (
                    Image.open(candidate)
                    .convert("RGB")
                    .resize((canvas_size, canvas_size), Image.Resampling.BICUBIC)
                )
            except Exception:
                pass
    if canvas is None:
        canvas = Image.new("RGB", (canvas_size, canvas_size), (45, 42, 38))

    # Medium/small embedded board: larger than the old 80-100 px range.
    target_board_size = get_embedded_board_size(canvas_size)
    bw, bh = board_img.size
    resized_board = board_img.resize((target_board_size, target_board_size), Image.Resampling.LANCZOS)

    max_x = canvas_size - target_board_size
    max_y = canvas_size - target_board_size
    ox = random.randint(0, max_x)
    oy = random.randint(0, max_y)

    # Add random UI clutter over the photo/texture background before pasting
    # the board, so it acts as a distractor without occluding any labels.
    canvas = draw_random_ui_clutter(
        canvas,
        canvas_size,
        exclude_rects=[(ox, oy, ox + target_board_size, oy + target_board_size)],
        theme=random.choice(UI_THEME_COLORS),
    )
    canvas.paste(resized_board, (ox, oy))

    scaled_boxes = []
    sx = target_board_size / bw
    sy = target_board_size / bh
    for cid, x1, y1, x2, y2 in piece_boxes:
        scaled_boxes.append((cid, ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy))

    # Bounding box strictly for the 64 checkerboard squares (excluding outer coordinate borders)
    chk_x1 = ox + border_px * sx
    chk_y1 = oy + border_px * sy
    chk_x2 = ox + (border_px + base_board_size) * sx
    chk_y2 = oy + (border_px + base_board_size) * sy
    board_box = (12, chk_x1, chk_y1, chk_x2, chk_y2)
    return canvas, scaled_boxes, board_box


def compose_multi_board(
    main_board_img,
    main_piece_boxes,
    main_border_px,
    main_base_size,
    sec_board_img,
    sec_piece_boxes,
    sec_border_px,
    sec_base_size,
    canvas_size=DEFAULT_IMAGE_SIZE,
):
    """
    Composition 4: Realistic Multi-Board Tournament / Analysis Broadcast View.
    Contains:
    - Primary active game board (54-62% canvas height/width)
    - Secondary variation or thumbnail board (28-34% canvas), either empty or with pieces
    - Broadcast top ribbon and sidebar container cards
    Both boards receive exact 64-square chess_board bounding boxes (excluding outer coordinate borders).
    """
    theme = random.choice(UI_THEME_COLORS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), theme["bg"])

    # Add randomized UI clutter behind the broadcast layout.
    canvas = draw_random_ui_clutter(canvas, canvas_size, theme=theme)
    draw = ImageDraw.Draw(canvas)
    font = get_default_font(size=11)

    # Top header bar (stream / tournament broadcast ribbon)
    header_h = 34
    draw.rectangle([0, 0, canvas_size, header_h], fill=theme["card"])
    draw.text((16, 10), "CHESS TOURNAMENT BROADCAST • LIVE", fill=theme["text"], font=font)
    draw.text((canvas_size - 80, 10), "MATCH #1", fill=theme["accent"], font=font)

    # 1. Main board: 54% - 62% of canvas (tile size 42-50 px)
    main_size = random.randint(int(canvas_size * 0.54), int(canvas_size * 0.62))
    bw1, bh1 = main_board_img.size
    resized_main = main_board_img.resize((main_size, main_size), Image.Resampling.LANCZOS)
    main_ox = 18
    main_oy = header_h + 12 + (canvas_size - header_h - 12 - main_size) // 2
    canvas.paste(resized_main, (main_ox, main_oy))

    # Scale main pieces
    sx1 = main_size / bw1
    sy1 = main_size / bh1
    scaled_boxes = []
    for cid, x1, y1, x2, y2 in main_piece_boxes:
        scaled_boxes.append((cid, main_ox + x1 * sx1, main_oy + y1 * sy1, main_ox + x2 * sx1, main_oy + y2 * sy1))

    # Main board box (strictly 64 checkerboard squares)
    m_chk_x1 = main_ox + main_border_px * sx1
    m_chk_y1 = main_oy + main_border_px * sy1
    m_chk_x2 = main_ox + (main_border_px + main_base_size) * sx1
    m_chk_y2 = main_oy + (main_border_px + main_base_size) * sy1
    main_board_box = (12, m_chk_x1, m_chk_y1, m_chk_x2, m_chk_y2)

    # 2. Secondary board in right sidebar: 28% - 34% of canvas (tile size ~23-27 px)
    sec_max_w = canvas_size - (main_ox + main_size) - 24
    sec_size = min(int(canvas_size * 0.33), sec_max_w)
    if sec_size < 160:
        sec_size = 160
    bw2, bh2 = sec_board_img.size
    resized_sec = sec_board_img.resize((sec_size, sec_size), Image.Resampling.LANCZOS)
    sec_ox = canvas_size - sec_size - 14
    sec_oy = main_oy + (main_size - sec_size) // 2

    # Draw container card for secondary board
    card_pad = 6
    draw.rounded_rectangle(
        [sec_ox - card_pad, sec_oy - 22, sec_ox + sec_size + card_pad, sec_oy + sec_size + card_pad],
        radius=4,
        fill=theme["card"],
    )
    card_title = "Board 2 (Live Game)" if sec_piece_boxes else "Board 2 (Analysis / Variation)"
    draw.text((sec_ox, sec_oy - 18), card_title, fill=theme["text"], font=font)
    canvas.paste(resized_sec, (sec_ox, sec_oy))

    # Scale secondary pieces (if any)
    sx2 = sec_size / bw2
    sy2 = sec_size / bh2
    for cid, x1, y1, x2, y2 in sec_piece_boxes:
        scaled_boxes.append((cid, sec_ox + x1 * sx2, sec_oy + y1 * sy2, sec_ox + x2 * sx2, sec_oy + y2 * sy2))

    # Secondary board box (strictly 64 checkerboard squares)
    s_chk_x1 = sec_ox + sec_border_px * sx2
    s_chk_y1 = sec_oy + sec_border_px * sy2
    s_chk_x2 = sec_ox + (sec_border_px + sec_base_size) * sx2
    s_chk_y2 = sec_oy + (sec_border_px + sec_base_size) * sy2
    sec_board_box = (12, s_chk_x1, s_chk_y1, s_chk_x2, s_chk_y2)

    return canvas, scaled_boxes, [main_board_box, sec_board_box]


# ==============================================================================
# Photometric / Sensor Augmentations
# ==============================================================================

def apply_screen_moire(img, strength=3.5):
    """Adds subtle periodic LCD subpixel grid pattern simulating screen photos."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    freq = random.uniform(0.35, 0.55)
    y = np.arange(h).reshape(-1, 1)
    x = np.arange(w).reshape(1, -1)
    pattern = (np.sin(x * freq) + np.sin(y * freq)) * (strength / 2.0)
    arr = np.clip(arr + pattern[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_vignette(img, strength=0.30):
    """Applies radial brightness falloff towards corners."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    y = np.linspace(-1, 1, h).reshape(-1, 1)
    x = np.linspace(-1, 1, w).reshape(1, -1)
    dist_sq = x ** 2 + y ** 2
    vignette = 1.0 - (strength * 0.5 * dist_sq)
    vignette = np.clip(vignette, 0.0, 1.0)[:, :, np.newaxis]
    arr = np.clip(arr * vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_photometric_effects(img):
    """
    Applies realistic camera / stream artifacts without altering geometry:
    - Brightness, Contrast, Saturation adjustments
    - Resolution downsampling / upsampling (simulating 360p/480p streams)
    - Defocus / Gaussian blur
    - JPEG compression artifacts
    """
    img = img.convert("RGB")

    # Brightness (0.80 - 1.20)
    if random.random() < 0.40:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))

    # Contrast (0.80 - 1.20)
    if random.random() < 0.40:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.20))

    # Saturation (0.75 - 1.25)
    if random.random() < 0.35:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.75, 1.25))

    # Video stream downsampling / upsampling (simulate low-bandwidth streams)
    if random.random() < 0.25:
        w, h = img.size
        scale = random.uniform(0.50, 0.85)
        sw, sh = max(10, int(w * scale)), max(10, int(h * scale))
        img = img.resize((sw, sh), Image.Resampling.BILINEAR)
        img = img.resize((w, h), Image.Resampling.BICUBIC)

    # Slight blur (out-of-focus or motion blur)
    if random.random() < 0.20:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))

    # JPEG compression artifacts
    if random.random() < 0.45:
        quality = random.randint(45, 90)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            img = compressed.convert("RGB")

    return img


# ==============================================================================
# Deterministic Zero-Leakage Dataset Partitioning
# ==============================================================================

def split_assets(seed=DEFAULT_SEED, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, holdout_assets=True):
    """
    Deterministically partitions available boards, piece sets, and background noise
    into disjoint Train, Validation, and Test subsets with ZERO data leakage.

    If `holdout_assets` is False, no partition is performed: every piece set,
    board, and background is assigned to Train, leaving Val/Test empty.
    """
    rng = random.Random(seed)

    all_pieces = sorted(os.listdir(PIECES_DIR))
    rng.shuffle(all_pieces)
    if holdout_assets:
        np_train = int(len(all_pieces) * train_ratio)
        np_val = int(len(all_pieces) * val_ratio)
        train_pieces = all_pieces[:np_train]
        val_pieces = all_pieces[np_train : np_train + np_val]
        test_pieces = all_pieces[np_train + np_val :]
    else:
        train_pieces = all_pieces
        val_pieces = []
        test_pieces = []

    all_boards = sorted(os.listdir(BOARDS_DIR))
    rng.shuffle(all_boards)
    if holdout_assets:
        nb_train = int(len(all_boards) * train_ratio)
        nb_val = int(len(all_boards) * val_ratio)
        train_boards = all_boards[:nb_train]
        val_boards = all_boards[nb_train : nb_train + nb_val]
        test_boards = all_boards[nb_train + nb_val :]
    else:
        train_boards = all_boards
        val_boards = []
        test_boards = []

    # Discover available background images. Textures are intentionally not used
    # as backgrounds here: they are reserved for board generation.
    all_bgs = []
    if os.path.exists(BACKGROUND_NOISE_DIR):
        all_bgs.extend([f for f in os.listdir(BACKGROUND_NOISE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not all_bgs:
        raise RuntimeError(
            f"No background images found in '{BACKGROUND_NOISE_DIR}'. "
            "Add .jpg/.jpeg/.png scene/background images; textures are not used as backgrounds."
        )

    rng.shuffle(all_bgs)
    if holdout_assets:
        ng_train = max(1, int(len(all_bgs) * train_ratio))
        ng_val = max(1, int(len(all_bgs) * val_ratio))
        train_bgs = all_bgs[:ng_train]
        val_bgs = all_bgs[ng_train : ng_train + ng_val]
        test_bgs = all_bgs[ng_train + ng_val :]
        if not test_bgs:
            test_bgs = val_bgs.copy()
    else:
        train_bgs = all_bgs
        val_bgs = []
        test_bgs = []

    return {
        "train": {"pieces": train_pieces, "boards": train_boards, "bgs": train_bgs},
        "val": {"pieces": val_pieces, "boards": val_boards, "bgs": val_bgs},
        "test": {"pieces": test_pieces, "boards": test_boards, "bgs": test_bgs},
    }


# ==============================================================================
# Single Image Generation Pipeline
# ==============================================================================

def _generate_single_sample_once(
    img_id,
    split_name,
    piece_set_name,
    board_file,
    bg_file,
    canvas_size,
    base_board_size,
    make_board_label,
    images_dir,
    labels_dir,
    pieces_pool,
    boards_pool,
    min_piece_px=DEFAULT_MIN_PIECE_PX,
):
    """
    Builds one sample and saves it. Returns False if a piece is below
    `min_piece_px`, so the caller can redraw with a different deterministic seed.
    """
    # Deterministic position generation
    fen = generate_fen(strategy="auto", allow_flip=True)
    flipped = random.random() < 0.50

    # 1. Load board & piece sprites
    tile_size = base_board_size // 8
    board_img = load_raw_board(board_file, base_board_size)
    light_tile_color, dark_tile_color = extract_tile_colors(board_img, tile_size)
    piece_sprites = load_piece_sprites(piece_set_name, tile_size)

    # 2. Apply realistic move highlights, arrows, dots
    board_img = apply_board_overlays(board_img, fen, tile_size, flipped=flipped)

    # 3. Choose Composition Environment (Clean 2D digital screen compositions only)
    # Bias toward embedded boards on real background photos; keep clean solid
    # backgrounds rare.
    comp_type = random.choices(["full", "ui", "embedded", "multi"], weights=[0.05, 0.30, 0.40, 0.25])[0]

    # 4. Solid tile-matched background for clean full-board views only.
    # Other compositions keep their photographic / themed backgrounds.
    if comp_type == "full":
        bg_color = get_random_solid_background(light_tile_color, dark_tile_color)
    else:
        bg_color = (45, 42, 38)

    # 5. Add coordinates / border (outer frame blends into the full-board background)
    coord_style = random.choices(["none", "inner", "outer"], weights=[0.25, 0.50, 0.25])[0]
    board_img, border_px = render_board_coordinates(
        board_img, tile_size, flipped=flipped, style=coord_style, frame_color=bg_color
    )

    # 6. Render pieces and get tight bounding boxes
    board_img, piece_boxes = render_pieces_on_board(
        board_img, piece_sprites, fen, tile_size, border_offset=border_px
    )

    if comp_type == "full":
        canvas, scaled_boxes, board_box = compose_full_board(
            board_img,
            piece_boxes,
            border_px=border_px,
            base_board_size=base_board_size,
            canvas_size=canvas_size,
            bg_color=bg_color,
        )
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])

    elif comp_type == "ui":
        canvas, scaled_boxes, board_box = compose_web_ui(
            board_img,
            piece_boxes,
            border_px=border_px,
            base_board_size=base_board_size,
            canvas_size=canvas_size,
        )
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])

    elif comp_type == "embedded":
        canvas, scaled_boxes, board_box = compose_embedded_board(
            board_img,
            piece_boxes,
            border_px=border_px,
            base_board_size=base_board_size,
            bg_file=bg_file,
            canvas_size=canvas_size,
        )
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])

    else:  # "multi" - Multi-board tournament / analysis broadcast view
        sec_board_file = random.choice(boards_pool)
        sec_tile_size = base_board_size // 8
        sec_board_raw = load_raw_board(sec_board_file, base_board_size)
        is_sec_empty = random.random() < 0.50

        if is_sec_empty:
            sec_piece_boxes = []
            sec_coord_style = random.choices(["none", "inner", "outer"], weights=[0.3, 0.5, 0.2])[0]
            sec_board_img, sec_border_px = render_board_coordinates(
                sec_board_raw, sec_tile_size, flipped=random.random() < 0.5, style=sec_coord_style
            )
        else:
            sec_piece_set = random.choice(pieces_pool)
            sec_fen = generate_fen(strategy="endgame", allow_flip=True)
            sec_sprites = load_piece_sprites(sec_piece_set, sec_tile_size)
            sec_board_img = apply_board_overlays(sec_board_raw, sec_fen, sec_tile_size)
            sec_coord_style = random.choices(["none", "inner", "outer"], weights=[0.3, 0.5, 0.2])[0]
            sec_board_img, sec_border_px = render_board_coordinates(
                sec_board_img, sec_tile_size, flipped=random.random() < 0.5, style=sec_coord_style
            )
            sec_board_img, sec_piece_boxes = render_pieces_on_board(
                sec_board_img, sec_sprites, sec_fen, sec_tile_size, border_offset=sec_border_px
            )

        canvas, scaled_boxes, board_boxes = compose_multi_board(
            board_img,
            piece_boxes,
            border_px,
            base_board_size,
            sec_board_img,
            sec_piece_boxes,
            sec_border_px,
            base_board_size,
            canvas_size=canvas_size,
        )
        all_boxes = scaled_boxes + (board_boxes if make_board_label else [])

    # Reject samples where any piece is too small to supervise reliably.
    # This mainly protects against embedded/multi-board compositions.
    if min_piece_px > 0:
        for cid, x1, y1, x2, y2 in all_boxes:
            if cid == 12:  # chess_board
                continue
            x1_c = max(0.0, min(float(canvas_size), float(x1)))
            y1_c = max(0.0, min(float(canvas_size), float(y1)))
            x2_c = max(0.0, min(float(canvas_size), float(x2)))
            y2_c = max(0.0, min(float(canvas_size), float(y2)))
            if (x2_c - x1_c) < min_piece_px or (y2_c - y1_c) < min_piece_px:
                return False

    # Convert to YOLO format
    yolo_labels = []
    for cid, x1, y1, x2, y2 in all_boxes:
        x1_c = max(0.0, min(float(canvas_size), float(x1)))
        y1_c = max(0.0, min(float(canvas_size), float(y1)))
        x2_c = max(0.0, min(float(canvas_size), float(x2)))
        y2_c = max(0.0, min(float(canvas_size), float(y2)))
        w_box = x2_c - x1_c
        h_box = y2_c - y1_c
        if w_box >= 3 and h_box >= 3:
            xc = (x1_c + x2_c) / (2.0 * canvas_size)
            yc = (y1_c + y2_c) / (2.0 * canvas_size)
            w = w_box / float(canvas_size)
            h = h_box / float(canvas_size)
            yolo_labels.append(yolo_label_str(cid, xc, yc, w, h))

    # 6. Apply photometric effects (color, contrast, blur, jpeg)
    canvas = apply_photometric_effects(canvas)

    # 7. Save image and label file
    img_path = os.path.join(images_dir, f"{img_id}.jpg")
    lbl_path = os.path.join(labels_dir, f"{img_id}.txt")

    canvas.save(img_path, format="JPEG", quality=95)
    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_labels) + "\n")

    return True


def generate_single_sample(args):
    """
    Worker entry point for one sample.

    The task carries its own deterministic seed, so the same dataset seed
    reproduces identical images/labels regardless of worker count or scheduling.
    Samples that contain pieces below `min_piece_px` are redrawn with a new
    deterministic attempt seed. The final attempt always accepts the sample so
    the requested dataset count is preserved.
    """
    (
        img_id,
        split_name,
        piece_set_name,
        board_file,
        bg_file,
        canvas_size,
        base_board_size,
        make_board_label,
        images_dir,
        labels_dir,
        pieces_pool,
        boards_pool,
        task_seed,
        min_piece_px,
        max_attempts,
    ) = args

    min_piece_px = max(0, int(min_piece_px))
    max_attempts = max(1, int(max_attempts))

    for attempt in range(max_attempts):
        random.seed(f"{task_seed}:attempt:{attempt}")
        accept_small = attempt == max_attempts - 1
        saved = _generate_single_sample_once(
            img_id,
            split_name,
            piece_set_name,
            board_file,
            bg_file,
            canvas_size,
            base_board_size,
            make_board_label,
            images_dir,
            labels_dir,
            pieces_pool,
            boards_pool,
            min_piece_px=0 if accept_small else min_piece_px,
        )
        if saved:
            return True

    return False


# ==============================================================================
# Main Generation Coordinator
# ==============================================================================

def generate_split_dataset(
    split_name,
    count,
    asset_partition,
    images_dir,
    labels_dir,
    canvas_size=DEFAULT_IMAGE_SIZE,
    board_size=DEFAULT_BOARD_SIZE,
    make_board_label=True,
    num_workers=None,
    start_id=1,
    seed=DEFAULT_SEED,
    min_piece_px=DEFAULT_MIN_PIECE_PX,
    max_attempts=DEFAULT_MAX_SAMPLE_ATTEMPTS,
):
    """Generates `count` samples for a given split using its assigned asset partition."""
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    pieces_list = asset_partition["pieces"]
    boards_list = asset_partition["boards"]
    bgs_list = asset_partition["bgs"]

    if not pieces_list or not boards_list or not bgs_list:
        raise ValueError(f"Empty asset partition for split {split_name}!")

    # Per-split RNG so train/val/test task lists stay reproducible independently
    # of generation order and counts in the other splits.
    task_rng = random.Random(f"{seed}:{split_name}:assets")

    tasks = []
    for idx in range(count):
        sample_id = start_id + idx
        piece_set = task_rng.choice(pieces_list)
        board_file = task_rng.choice(boards_list)
        bg_file = task_rng.choice(bgs_list)
        task_seed = f"{seed}:{split_name}:{sample_id}"

        tasks.append((
            sample_id,
            split_name,
            piece_set,
            board_file,
            bg_file,
            canvas_size,
            board_size,
            make_board_label,
            images_dir,
            labels_dir,
            pieces_list,
            boards_list,
            task_seed,
            min_piece_px,
            max_attempts,
        ))

    workers = num_workers or max(1, multiprocessing.cpu_count())
    print(f"Generating {count} samples for [{split_name}] using {workers} worker processes...")

    with multiprocessing.Pool(processes=workers) as pool:
        # Use imap_unordered with chunksize for fast multiprocessing
        chunk = max(1, count // (workers * 4))
        completed = 0
        for _ in pool.imap_unordered(generate_single_sample, tasks, chunksize=chunk):
            completed += 1
            if completed % max(1, (count // 10)) == 0 or completed == count:
                print(f"  [{split_name}] Progress: {completed}/{count} ({completed * 100 // count}%)")


def main():
    parser = argparse.ArgumentParser(
        description="High-Quality 2D Chess Detection Dataset Generator for Ultralytics YOLO26"
    )
    parser.add_argument("--train-count", type=int, default=1000, help="Number of training samples to generate")
    parser.add_argument("--val-count", type=int, default=200, help="Number of validation samples to generate")
    parser.add_argument("--test-count", type=int, default=200, help="Number of test samples to generate")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Output image resolution (default 640)")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE, help="Base board resolution (default 640)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible asset split and sample content")
    parser.add_argument("--workers", type=int, default=None, help="Number of multiprocessing workers")
    parser.add_argument("--out-dir", type=str, default="datasets", help="Root dataset output directory")
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory before generation")
    parser.add_argument(
        "--min-piece-px",
        type=int,
        default=DEFAULT_MIN_PIECE_PX,
        help=f"Redraw samples where any piece is smaller than this many pixels (default {DEFAULT_MIN_PIECE_PX}; 0 disables)",
    )
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=DEFAULT_MAX_SAMPLE_ATTEMPTS,
        help=f"Maximum redraw attempts per sample for --min-piece-px (default {DEFAULT_MAX_SAMPLE_ATTEMPTS})",
    )
    parser.add_argument(
        "--no-board-labels", action="store_true", help="Do not generate labels for the chessboard class (12)"
    )
    args = parser.parse_args()

    if args.min_piece_px < 0:
        parser.error("--min-piece-px must be >= 0")
    if args.max_sample_attempts < 1:
        parser.error("--max-sample-attempts must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("================================================================================")
    print(" 2D Chess Piece & Board Dataset Generator (Optimized for Ultralytics YOLO26n)")
    print("================================================================================")
    print(f"Target counts: Train={args.train_count}, Val={args.val_count}, Test={args.test_count}")
    print(f"Output Resolution: {args.imgsz}x{args.imgsz} | Random Seed: {args.seed}")
    print(f"Output Directory: {args.out_dir}")

    if args.clean and os.path.exists(args.out_dir):
        print(f"Cleaning existing directory: {args.out_dir}")
        shutil.rmtree(args.out_dir)

    # 1. Partition assets deterministically with zero data leakage.
    # When no validation or test samples are requested, skip the split entirely
    # so every piece set and board is used for training.
    holdout_assets = args.val_count > 0 or args.test_count > 0
    partitions = split_assets(seed=args.seed, holdout_assets=holdout_assets)
    print("\nDeterministic Asset Partitioning (Zero Data Leakage):")
    for sname in ["train", "val", "test"]:
        part = partitions[sname]
        print(f"  [{sname}] Piece sets: {len(part['pieces'])}, Boards: {len(part['boards'])}, Backgrounds: {len(part['bgs'])}")

    make_board_label = not args.no_board_labels

    # 2. Generate Train split
    if args.train_count > 0:
        generate_split_dataset(
            split_name="train",
            count=args.train_count,
            asset_partition=partitions["train"],
            images_dir=os.path.join(args.out_dir, "images", "train"),
            labels_dir=os.path.join(args.out_dir, "labels", "train"),
            canvas_size=args.imgsz,
            board_size=args.board_size,
            make_board_label=make_board_label,
            num_workers=args.workers,
            start_id=1,
            seed=args.seed,
            min_piece_px=args.min_piece_px,
            max_attempts=args.max_sample_attempts,
        )

    # 3. Generate Validation split
    if args.val_count > 0:
        generate_split_dataset(
            split_name="val",
            count=args.val_count,
            asset_partition=partitions["val"],
            images_dir=os.path.join(args.out_dir, "images", "val"),
            labels_dir=os.path.join(args.out_dir, "labels", "val"),
            canvas_size=args.imgsz,
            board_size=args.board_size,
            make_board_label=make_board_label,
            num_workers=args.workers,
            start_id=1,
            seed=args.seed,
            min_piece_px=args.min_piece_px,
            max_attempts=args.max_sample_attempts,
        )

    # 4. Generate Test split
    if args.test_count > 0:
        generate_split_dataset(
            split_name="test",
            count=args.test_count,
            asset_partition=partitions["test"],
            images_dir=os.path.join(args.out_dir, "images", "test"),
            labels_dir=os.path.join(args.out_dir, "labels", "test"),
            canvas_size=args.imgsz,
            board_size=args.board_size,
            make_board_label=make_board_label,
            num_workers=args.workers,
            start_id=1,
            seed=args.seed,
            min_piece_px=args.min_piece_px,
            max_attempts=args.max_sample_attempts,
        )

    print("\nDataset generation completed successfully!")
    print(f"Dataset root: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
