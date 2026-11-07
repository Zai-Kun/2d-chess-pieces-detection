#!/usr/bin/env python3
"""
High-Quality 2D Chess Piece & Board Dataset Generator for Ultralytics YOLO26n.

Key features:
1. Tight, Antialiased Bounding Boxes: Computes exact alpha bounding boxes (thresholding
   out translucent fringes and soft shadows) for all pieces.
2. Realistic Visual Overlays: Move highlights, check indicators, legal move dots,
   analysis arrows, inner & outer board coordinates.
3. Realistic Contexts & Environments:
   - Clean full-board views (85-100% frame)
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
TEXTURES_DIR = "assets/textures"
DATASETS_IMAGES_DIR = "datasets/images"
DATASETS_LABELS_DIR = "datasets/labels"

# Default Generation Settings
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BOARD_SIZE = 640
DEFAULT_ALPHA_THRESHOLD = 20  # Filters out translucent antialiasing fringe and soft shadows
DEFAULT_SEED = 42

# Font setup
_CACHED_FONT = None


def get_default_font(size=14):
    """Loads a readable TTF font if available on the system, otherwise PIL default."""
    global _CACHED_FONT
    if _CACHED_FONT is not None:
        return _CACHED_FONT
    ttf_candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in ttf_candidates:
        if os.path.exists(path):
            try:
                _CACHED_FONT = ImageFont.truetype(path, size)
                return _CACHED_FONT
            except Exception:
                pass
    _CACHED_FONT = ImageFont.load_default()
    return _CACHED_FONT


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

    # 3. Legal move dots & capture rings (20% chance)
    if random.random() < 0.20:
        empty_squares = [(c, r) for c in range(8) for r in range(8) if (c, r) not in occupied]
        if empty_squares:
            dot_color = (40, 40, 40, 95)
            dot_r = int(tile_size * 0.14)
            for c, r in random.sample(empty_squares, min(random.randint(2, 5), len(empty_squares))):
                cx = c * tile_size + tile_size // 2
                cy = r * tile_size + tile_size // 2
                ov_draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=dot_color)

        # Target capture ring on enemy piece
        if occupied and random.random() < 0.5:
            tc, tr = random.choice(list(occupied.keys()))
            cx = tc * tile_size + tile_size // 2
            cy = tr * tile_size + tile_size // 2
            ring_r = int(tile_size * 0.42)
            ov_draw.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r], outline=(30, 30, 30, 90), width=3)

    # 4. Analysis arrow (20% chance)
    if random.random() < 0.20:
        c1, r1 = random.randint(0, 7), random.randint(0, 7)
        c2, r2 = random.randint(0, 7), random.randint(0, 7)
        if (c1, r1) != (c2, r2):
            arrow_color = random.choice([
                (40, 180, 80, 180),   # Green
                (245, 140, 30, 180),  # Orange
                (59, 130, 246, 180),  # Blue
                (239, 68, 68, 180),   # Red
            ])
            p1 = (c1 * tile_size + tile_size // 2, r1 * tile_size + tile_size // 2)
            p2 = (c2 * tile_size + tile_size // 2, r2 * tile_size + tile_size // 2)
            draw_arrow(ov_draw, p1, p2, color=arrow_color, width=max(4, int(tile_size * 0.10)))

    return Image.alpha_composite(board_img.convert("RGBA"), overlay).convert("RGB")


def render_board_coordinates(board_img, tile_size, flipped=False, style="inner"):
    """
    Renders standard rank ('1'-'8') and file ('a'-'h') coordinate labels.
    - 'inner': Drawn in corners of border tiles (Chess.com / Lichess modern style)
    - 'outer': Border frame with centered letters and numbers
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
        framed = Image.new("RGB", (new_size, new_size), (45, 42, 38))
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


def compose_full_board(board_img, piece_boxes, canvas_size=DEFAULT_IMAGE_SIZE):
    """
    Composition 1: Full-board view.
    Board fills 85-100% of the canvas. Center-aligned with subtle background padding.
    """
    canvas = Image.new("RGB", (canvas_size, canvas_size), (30, 28, 26))
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

    board_box = (12, ox, oy, ox + new_w, oy + new_h)
    return canvas, scaled_boxes, board_box


def compose_web_ui(board_img, piece_boxes, canvas_size=DEFAULT_IMAGE_SIZE):
    """
    Composition 2: Realistic Web / Desktop Chess Interface.
    Simulates Chess.com or Lichess with eval bar, player names, clocks, and sidebar.
    """
    theme = random.choice(UI_THEME_COLORS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), theme["bg"])
    draw = ImageDraw.Draw(canvas)
    font = get_default_font(size=12)

    # Board size occupies 55% - 75% of canvas height
    target_board_size = random.randint(int(canvas_size * 0.55), int(canvas_size * 0.75))
    bw, bh = board_img.size
    resized_board = board_img.resize((target_board_size, target_board_size), Image.Resampling.LANCZOS)

    # Position board slightly to the left to leave room for sidebar/eval bar
    board_ox = random.randint(40, max(42, canvas_size - target_board_size - 30))
    board_oy = (canvas_size - target_board_size) // 2
    canvas.paste(resized_board, (board_ox, board_oy))

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

    board_box = (12, board_ox, board_oy, board_ox + target_board_size, board_oy + target_board_size)
    return canvas, scaled_boxes, board_box


def compose_embedded_board(board_img, piece_boxes, bg_file, canvas_size=DEFAULT_IMAGE_SIZE):
    """
    Composition 3: Small or embedded board on realistic photographic background.
    Tests detection robustness on small chessboards and cluttered scenes (STAL).
    """
    bg_path = os.path.join(BACKGROUND_NOISE_DIR, bg_file)
    try:
        canvas = Image.open(bg_path).convert("RGB").resize((canvas_size, canvas_size), Image.Resampling.BICUBIC)
    except Exception:
        canvas = Image.new("RGB", (canvas_size, canvas_size), (60, 60, 60))

    # Board size occupies 28% to 65% of frame
    target_board_size = random.randint(int(canvas_size * 0.28), int(canvas_size * 0.65))
    bw, bh = board_img.size
    resized_board = board_img.resize((target_board_size, target_board_size), Image.Resampling.LANCZOS)

    max_x = canvas_size - target_board_size
    max_y = canvas_size - target_board_size
    ox = random.randint(0, max_x)
    oy = random.randint(0, max_y)
    canvas.paste(resized_board, (ox, oy))

    scaled_boxes = []
    sx = target_board_size / bw
    sy = target_board_size / bh
    for cid, x1, y1, x2, y2 in piece_boxes:
        scaled_boxes.append((cid, ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy))

    board_box = (12, ox, oy, ox + target_board_size, oy + target_board_size)
    return canvas, scaled_boxes, board_box


def compose_photo_capture(canvas_img, all_boxes, canvas_size=DEFAULT_IMAGE_SIZE):
    """
    Composition 4: Simulates taking a phone photo of a computer or mobile screen.
    Applies:
    - Keystone perspective warp (up to +/- 5% corner displacement)
    - Camera vignette
    - Screen moire pattern
    - Mathematical transformation of bounding boxes via forward homography matrix M.
    """
    # Perspective jitter corners
    w, h = canvas_size, canvas_size
    jitter = int(canvas_size * 0.055)
    src_corners = [(0, 0), (w, 0), (w, h), (0, h)]
    dst_corners = [
        (random.randint(0, jitter), random.randint(0, jitter)),
        (w - random.randint(0, jitter), random.randint(0, jitter)),
        (w - random.randint(0, jitter), h - random.randint(0, jitter)),
        (random.randint(0, jitter), h - random.randint(0, jitter)),
    ]

    # Backward coeffs for PIL
    coeffs = find_perspective_coeffs(dst_corners, src_corners)
    warped_img = canvas_img.transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR
    )

    # Forward matrix for bounding box projection
    M = find_forward_matrix(src_corners, dst_corners)

    # Project every bounding box
    projected_boxes = []
    for cid, x1, y1, x2, y2 in all_boxes:
        yolo_box = transform_box((x1, y1, x2, y2), M, w, h)
        if yolo_box is not None:
            xc, yc, bw, bh = yolo_box
            projected_boxes.append((cid, xc, yc, bw, bh))

    # Apply screen moire (subtle sinusoidal line interference)
    if random.random() < 0.60:
        warped_img = apply_screen_moire(warped_img, strength=random.uniform(2.5, 5.0))

    # Apply camera vignette (falloff at corners)
    if random.random() < 0.70:
        warped_img = apply_vignette(warped_img, strength=random.uniform(0.20, 0.45))

    return warped_img, projected_boxes


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

def split_assets(seed=DEFAULT_SEED, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Deterministically partitions available boards, piece sets, and background noise
    into disjoint Train, Validation, and Test subsets with ZERO data leakage.
    """
    rng = random.Random(seed)

    all_pieces = sorted(os.listdir(PIECES_DIR))
    rng.shuffle(all_pieces)
    np_train = int(len(all_pieces) * train_ratio)
    np_val = int(len(all_pieces) * val_ratio)
    train_pieces = all_pieces[:np_train]
    val_pieces = all_pieces[np_train : np_train + np_val]
    test_pieces = all_pieces[np_train + np_val :]

    all_boards = sorted(os.listdir(BOARDS_DIR))
    rng.shuffle(all_boards)
    nb_train = int(len(all_boards) * train_ratio)
    nb_val = int(len(all_boards) * val_ratio)
    train_boards = all_boards[:nb_train]
    val_boards = all_boards[nb_train : nb_train + nb_val]
    test_boards = all_boards[nb_train + nb_val :]

    all_bgs = sorted(os.listdir(BACKGROUND_NOISE_DIR))
    rng.shuffle(all_bgs)
    ng_train = int(len(all_bgs) * train_ratio)
    ng_val = int(len(all_bgs) * val_ratio)
    train_bgs = all_bgs[:ng_train]
    val_bgs = all_bgs[ng_train : ng_train + ng_val]
    test_bgs = all_bgs[ng_train + ng_val :]

    return {
        "train": {"pieces": train_pieces, "boards": train_boards, "bgs": train_bgs},
        "val": {"pieces": val_pieces, "boards": val_boards, "bgs": val_bgs},
        "test": {"pieces": test_pieces, "boards": test_boards, "bgs": test_bgs},
    }


# ==============================================================================
# Single Image Generation Pipeline
# ==============================================================================

def generate_single_sample(args):
    """Worker task that generates a single sample image and its YOLO label file."""
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
    ) = args

    # Deterministic position generation
    fen = generate_fen(strategy="auto", allow_flip=True)
    flipped = random.random() < 0.50

    # 1. Load board & piece sprites
    tile_size = base_board_size // 8
    board_img = load_raw_board(board_file, base_board_size)
    piece_sprites = load_piece_sprites(piece_set_name, tile_size)

    # 2. Apply realistic move highlights, arrows, dots
    board_img = apply_board_overlays(board_img, fen, tile_size, flipped=flipped)

    # 3. Add coordinates / border
    coord_style = random.choices(["none", "inner", "outer"], weights=[0.25, 0.50, 0.25])[0]
    board_img, border_px = render_board_coordinates(board_img, tile_size, flipped=flipped, style=coord_style)

    # 4. Render pieces and get tight bounding boxes
    board_img, piece_boxes = render_pieces_on_board(
        board_img, piece_sprites, fen, tile_size, border_offset=border_px
    )

    # 5. Choose Composition Environment
    comp_type = random.choices(["full", "ui", "embedded", "photo"], weights=[0.25, 0.35, 0.25, 0.15])[0]

    if comp_type == "full":
        canvas, scaled_boxes, board_box = compose_full_board(board_img, piece_boxes, canvas_size)
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])
        # Convert to YOLO format
        yolo_labels = []
        for cid, x1, y1, x2, y2 in all_boxes:
            xc = (x1 + x2) / (2.0 * canvas_size)
            yc = (y1 + y2) / (2.0 * canvas_size)
            w = (x2 - x1) / float(canvas_size)
            h = (y2 - y1) / float(canvas_size)
            yolo_labels.append(yolo_label_str(cid, xc, yc, w, h))

    elif comp_type == "ui":
        canvas, scaled_boxes, board_box = compose_web_ui(board_img, piece_boxes, canvas_size)
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])
        yolo_labels = []
        for cid, x1, y1, x2, y2 in all_boxes:
            xc = (x1 + x2) / (2.0 * canvas_size)
            yc = (y1 + y2) / (2.0 * canvas_size)
            w = (x2 - x1) / float(canvas_size)
            h = (y2 - y1) / float(canvas_size)
            yolo_labels.append(yolo_label_str(cid, xc, yc, w, h))

    elif comp_type == "embedded":
        canvas, scaled_boxes, board_box = compose_embedded_board(board_img, piece_boxes, bg_file, canvas_size)
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])
        yolo_labels = []
        for cid, x1, y1, x2, y2 in all_boxes:
            xc = (x1 + x2) / (2.0 * canvas_size)
            yc = (y1 + y2) / (2.0 * canvas_size)
            w = (x2 - x1) / float(canvas_size)
            h = (y2 - y1) / float(canvas_size)
            yolo_labels.append(yolo_label_str(cid, xc, yc, w, h))

    else:  # 'photo' mode with perspective keystone distortion
        # First compose as full or embedded
        base_canvas, scaled_boxes, board_box = compose_full_board(board_img, piece_boxes, canvas_size)
        all_boxes = scaled_boxes + ([board_box] if make_board_label else [])
        canvas, projected_boxes = compose_photo_capture(base_canvas, all_boxes, canvas_size)
        yolo_labels = []
        for cid, xc, yc, w, h in projected_boxes:
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
):
    """Generates `count` samples for a given split using its assigned asset partition."""
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    pieces_list = asset_partition["pieces"]
    boards_list = asset_partition["boards"]
    bgs_list = asset_partition["bgs"]

    if not pieces_list or not boards_list or not bgs_list:
        raise ValueError(f"Empty asset partition for split {split_name}!")

    tasks = []
    for idx in range(count):
        sample_id = start_id + idx
        piece_set = random.choice(pieces_list)
        board_file = random.choice(boards_list)
        bg_file = random.choice(bgs_list)

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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible asset split")
    parser.add_argument("--workers", type=int, default=None, help="Number of multiprocessing workers")
    parser.add_argument("--out-dir", type=str, default="datasets", help="Root dataset output directory")
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory before generation")
    parser.add_argument(
        "--no-board-labels", action="store_true", help="Do not generate labels for the chessboard class (12)"
    )
    args = parser.parse_args()

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

    # 1. Partition assets deterministically with zero data leakage
    partitions = split_assets(seed=args.seed)
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
        )

    print("\nDataset generation completed successfully!")
    print(f"Dataset root: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
