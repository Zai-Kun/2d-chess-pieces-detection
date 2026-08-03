import io
import math
import multiprocessing
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from random_fen_gen import generate_fen

# ---------------------------------------------------------------------------
# Global Settings & Hyperparameters for Dataset Generation
# ---------------------------------------------------------------------------
BOARD_SIZE = 640
TILE_SIZE = BOARD_SIZE // 8
PIECE_CANVAS_SCALE = 1.3
PIECE_CANVAS_SIZE = int(TILE_SIZE * PIECE_CANVAS_SCALE)
VARIATIONS = 4

BOARDS_DIR = "assets/boards"
PIECES_DIR = "assets/pieces"
BACKGROUND_NOISE_DIR = "assets/random_noise_backgrounds"
DATASETS_IMAGES_DIR = "datasets/images"
DATASETS_LABELS_DIR = "datasets/labels"
DATA_SPLIT = 0.8  # 80% train, 20% val

MAKE_LABELS_FOR_CHESSBOARD = True
GENERATE_IMAGES_WITH_BACKGROUND_NOISE = True

# Augmentation Probabilities (tuned for maximum YOLO26s generalization)
PROB_PIECE_RESIZE = 0.40
PROB_PIECE_ROTATE = 0.30
PROB_PIECE_OFFSET = 0.50

PROB_PERSPECTIVE_WARP = 0.25
PROB_COLOR_JITTER = 0.60
PROB_BLUR = 0.45
PROB_JPEG_COMPRESSION = 0.60
PROB_NOISE = 0.40
PROB_SCREEN_SCANLINES = 0.25
PROB_VIGNETTE = 0.35
PROB_RANDOM_LINES = 0.30

FEN_TO_PIECE = {
    "p": "bP", "r": "bR", "n": "bN", "b": "bB", "q": "bQ", "k": "bK",
    "P": "wP", "R": "wR", "N": "wN", "B": "wB", "Q": "wQ", "K": "wK",
}
FEN_CHAR_ORDER = list(FEN_TO_PIECE.keys())


# ---------------------------------------------------------------------------
# Image Augmentation Utility Functions
# ---------------------------------------------------------------------------
def apply_jpeg_compression(img: Image.Image, min_q: int = 20, max_q: int = 90) -> Image.Image:
    """Applies realistic JPEG compression artifacts (simulates web uploads and low-bitrate compression)."""
    quality = random.randint(min_q, max_q)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_blur(img: Image.Image) -> Image.Image:
    """Applies blur (Gaussian blur, box blur, or motion blur)."""
    blur_choice = random.choice(["gaussian", "box", "motion"])
    if blur_choice == "gaussian":
        radius = random.uniform(0.5, 2.5)
        return img.filter(ImageFilter.GaussianBlur(radius))
    elif blur_choice == "box":
        radius = random.randint(1, 2)
        return img.filter(ImageFilter.BoxBlur(radius))
    else:  # motion blur
        try:
            import cv2
            arr = np.array(img)
            size = random.choice([3, 5, 7])
            kernel = np.zeros((size, size), dtype=np.float32)
            if random.random() < 0.5:
                kernel[int((size - 1) / 2), :] = 1.0
            else:
                kernel[:, int((size - 1) / 2)] = 1.0
            kernel /= size
            blurred = cv2.filter2D(arr, -1, kernel)
            return Image.fromarray(blurred)
        except Exception:
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.8, 2.0)))


def apply_noise(img: Image.Image) -> Image.Image:
    """Applies Gaussian noise or salt-and-pepper grain."""
    arr = np.array(img, dtype=np.float32)
    h, w, c = arr.shape
    
    if random.random() < 0.7:
        std = random.uniform(5.0, 22.0)
        noise = np.random.normal(0, std, (h, w, c))
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    else:
        prob = random.uniform(0.005, 0.02)
        rnd = np.random.rand(h, w)
        arr[rnd < prob / 2] = 0
        arr[rnd > 1 - prob / 2] = 255
        noisy_arr = np.clip(arr, 0, 255).astype(np.uint8)
        
    return Image.fromarray(noisy_arr)


def apply_screen_scanlines(img: Image.Image) -> Image.Image:
    """Simulates screen Moiré pattern / scanlines (photographing a computer screen)."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    line_spacing = random.choice([2, 3, 4, 5])
    alpha = random.randint(15, 45)
    
    for y in range(0, h, line_spacing):
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha), width=1)
        
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def apply_color_jitter(img: Image.Image) -> Image.Image:
    """Applies brightness, contrast, saturation, and sharpness color jitter."""
    if random.random() < 0.7:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)
        
    if random.random() < 0.7:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)
        
    if random.random() < 0.7:
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Color(img).enhance(factor)
        
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.8)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    return img


def apply_vignette(img: Image.Image) -> Image.Image:
    """Applies lighting falloff / vignetting shadow effect across image edges."""
    w, h = img.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    
    vignette = 1 - np.clip((radius - 0.5) / 0.8, 0, 0.6)
    vignette = np.stack([vignette] * 3, axis=-1)
    
    arr = np.array(img, dtype=np.float32) * vignette
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_random_lines_and_spots(board: Image.Image) -> Image.Image:
    """Adds random overlay lines or spots."""
    overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = board.size
    num_elements = random.randint(1, 8)

    for _ in range(num_elements):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(20, 120),
        )
        if random.random() < 0.5:
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 6))
        else:
            x, y = random.randint(0, width), random.randint(0, height)
            rad = random.randint(2, 10)
            draw.ellipse([(x, y), (x + rad, y + rad)], fill=color)

    return Image.alpha_composite(board.convert("RGBA"), overlay).convert("RGB")


def apply_perspective_transform(img: Image.Image, piece_labels: List[Tuple]) -> Tuple[Image.Image, List[Tuple]]:
    """Applies slight perspective warping to board and updates bounding boxes."""
    w, h = img.size
    max_shift = int(w * 0.08)
    
    dx1, dy1 = random.randint(0, max_shift), random.randint(0, max_shift)
    dx2, dy2 = random.randint(-max_shift, 0), random.randint(0, max_shift)
    dx3, dy3 = random.randint(-max_shift, 0), random.randint(-max_shift, 0)
    dx4, dy4 = random.randint(0, max_shift), random.randint(-max_shift, 0)

    src_quad = [(0, 0), (w, 0), (w, h), (0, h)]
    dst_quad = [(dx1, dy1), (w + dx2, dy2), (w + dx3, h + dy3), (dx4, h + dy4)]

    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.matrix(pb, dtype=np.float64).reshape(8, 1)
        res = A.I * B
        return np.array(res).reshape(8)

    try:
        coeffs = find_coeffs(dst_quad, src_quad)
        warped_img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        
        A_fwd = []
        for p1, p2 in zip(src_quad, dst_quad):
            A_fwd.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            A_fwd.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        res_fwd = np.matrix(A_fwd, dtype=np.float64).I * np.matrix(dst_quad, dtype=np.float64).reshape(8, 1)
        c = np.array(res_fwd).flatten()
        H = np.array([[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], 1.0]])

        transformed_labels = []
        for class_id, x, y, bw, bh in piece_labels:
            pts = np.array([
                [x, y, 1],
                [x + bw, y, 1],
                [x + bw, y + bh, 1],
                [x, y + bh, 1]
            ]).T
            trans_pts = H @ pts
            trans_pts /= trans_pts[2, :]
            
            x_min = max(0, np.min(trans_pts[0, :]))
            y_min = max(0, np.min(trans_pts[1, :]))
            x_max = min(w, np.max(trans_pts[0, :]))
            y_max = min(h, np.max(trans_pts[1, :]))
            
            if x_max > x_min and y_max > y_min:
                transformed_labels.append((class_id, x_min, y_min, x_max - x_min, y_max - y_min))

        return warped_img, transformed_labels
    except Exception:
        return img, piece_labels


# ---------------------------------------------------------------------------
# Asset Loading & Piece Processing
# ---------------------------------------------------------------------------
def load_pieces(piece_set_name: str) -> dict:
    pieces = {}
    for f, p in FEN_TO_PIECE.items():
        img_path = f"{PIECES_DIR}/{piece_set_name}/{p}.png"
        with Image.open(img_path) as img:
            rgba = img.convert("RGBA")
            pieces[f] = rgba.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
    return pieces


def load_board(board_file: str) -> Image.Image:
    board_path = f"{BOARDS_DIR}/{board_file}"
    with Image.open(board_path) as img:
        rgb = img.convert("RGB")
        return rgb.resize((BOARD_SIZE, BOARD_SIZE), Image.BILINEAR)


def yolo_label(x, y, w, h, img_w, img_h, class_id) -> str:
    """Converts absolute pixel box to normalized YOLO format line."""
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    norm_w = min(max(norm_w, 0.0001), 1.0)
    norm_h = min(max(norm_h, 0.0001), 1.0)
    
    return f"{class_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}"


def labels_to_yolo_lines(piece_labels, img_w, img_h, x_bias=0.0, y_bias=0.0, scale=1.0) -> List[str]:
    lines = []
    for class_id, x, y, w, h in piece_labels:
        fx = x * scale + x_bias
        fy = y * scale + y_bias
        fw = w * scale
        fh = h * scale
        if fw <= 0 or fh <= 0:
            continue
        lines.append(yolo_label(fx, fy, fw, fh, img_w, img_h, class_id))
    return lines


def _apply_random_resize(piece_image: Image.Image) -> Image.Image:
    bbox = piece_image.getbbox()
    if not bbox:
        return piece_image

    piece_content = piece_image.crop(bbox)
    content_width, content_height = piece_content.size

    scale_factor = random.uniform(0.80, 1.20)
    new_width = min(int(content_width * scale_factor), PIECE_CANVAS_SIZE)
    new_height = min(int(content_height * scale_factor), PIECE_CANVAS_SIZE)
    if new_width <= 0 or new_height <= 0:
        return piece_image

    resized_content = piece_content.resize((new_width, new_height))
    canvas = Image.new("RGBA", (PIECE_CANVAS_SIZE, PIECE_CANVAS_SIZE), (0, 0, 0, 0))
    paste_x = (PIECE_CANVAS_SIZE - new_width) // 2
    paste_y = (PIECE_CANVAS_SIZE - new_height) // 2
    canvas.paste(resized_content, (paste_x, paste_y), resized_content)
    return canvas


# ---------------------------------------------------------------------------
# Core Image Generation Function
# ---------------------------------------------------------------------------
def generate_image(board: Image.Image, piece_set: dict, fen: str) -> Tuple[Image.Image, List[Tuple]]:
    """Draws pieces onto board according to FEN, with micro-jitters & augmentations."""
    piece_labels = []
    board = board.copy()

    for row, fen_rank in enumerate(fen.split()[0].split("/")):
        file_index = 0
        for char in fen_rank:
            if char.isdigit():
                file_index += int(char)
                continue

            if char not in piece_set:
                file_index += 1
                continue

            x, y = file_index * TILE_SIZE, row * TILE_SIZE
            piece_image = piece_set[char].copy()

            if random.random() < PROB_PIECE_RESIZE:
                piece_image = _apply_random_resize(piece_image)

            if random.random() < PROB_PIECE_ROTATE:
                piece_image = piece_image.rotate(random.uniform(-14, 14), expand=True)

            offset_x = (piece_image.width - TILE_SIZE) // 2
            offset_y = (piece_image.height - TILE_SIZE) // 2
            
            shift_x = 0
            shift_y = 0
            if random.random() < PROB_PIECE_OFFSET:
                shift_x = random.randint(-int(TILE_SIZE * 0.07), int(TILE_SIZE * 0.07))
                shift_y = random.randint(-int(TILE_SIZE * 0.07), int(TILE_SIZE * 0.07))

            paste_x = x - offset_x + shift_x
            paste_y = y - offset_y + shift_y

            board.paste(piece_image, (paste_x, paste_y), piece_image)

            bbox = piece_image.getbbox()
            if bbox:
                bx1, by1, bx2, by2 = bbox
                abs_x1 = max(0, paste_x + bx1)
                abs_y1 = max(0, paste_y + by1)
                abs_x2 = min(BOARD_SIZE, paste_x + bx2)
                abs_y2 = min(BOARD_SIZE, paste_y + by2)
                if abs_x2 > abs_x1 and abs_y2 > abs_y1:
                    class_id = str(FEN_CHAR_ORDER.index(char))
                    piece_labels.append(
                        (class_id, abs_x1, abs_y1, abs_x2 - abs_x1, abs_y2 - abs_y1)
                    )

            file_index += 1

    if random.random() < PROB_RANDOM_LINES:
        board = add_random_lines_and_spots(board)

    if random.random() < PROB_PERSPECTIVE_WARP:
        board, piece_labels = apply_perspective_transform(board, piece_labels)

    if random.random() < PROB_COLOR_JITTER:
        board = apply_color_jitter(board)

    if random.random() < PROB_VIGNETTE:
        board = apply_vignette(board)

    if random.random() < PROB_SCREEN_SCANLINES:
        board = apply_screen_scanlines(board)

    if random.random() < PROB_BLUR:
        board = apply_blur(board)

    if random.random() < PROB_NOISE:
        board = apply_noise(board)

    if random.random() < PROB_JPEG_COMPRESSION:
        board = apply_jpeg_compression(board, min_q=25, max_q=88)

    return board, piece_labels


# ---------------------------------------------------------------------------
# Parallel Worker Generation Functions
# ---------------------------------------------------------------------------
def generate_images_worker(args):
    boards, pieces, images_dir, labels_dir, variations, image_id = args[0:6]
    for board_image in boards:
        for _ in range(variations):
            fen = generate_fen()
            img_path = f"{images_dir}/{image_id}.jpg"
            label_path = f"{labels_dir}/{image_id}.txt"

            image, piece_labels = generate_image(board_image, pieces, fen)
            image.save(img_path, "JPEG", quality=92)

            lines = labels_to_yolo_lines(piece_labels, BOARD_SIZE, BOARD_SIZE)
            if MAKE_LABELS_FOR_CHESSBOARD:
                lines.append(
                    yolo_label(0, 0, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, "12")
                )

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            image_id += 1


def generate_images_with_background_noise_worker(args):
    images_dir, labels_dir, boards, piece_sets, background, variations, image_id = args

    bg_path = f"{BACKGROUND_NOISE_DIR}/{background}"
    with Image.open(bg_path) as img:
        bg_img = img.convert("RGB").resize((BOARD_SIZE, BOARD_SIZE))

    original_bg_size = bg_img.width

    for _ in range(variations):
        board_img = random.choice(boards)
        board_size_random = random.randint(320, BOARD_SIZE)
        scale_factor = board_size_random / original_bg_size
        max_pos = original_bg_size - board_size_random

        bg_img_copy = bg_img.copy()
        random_x = random.randint(0, max_pos)
        random_y = random.randint(0, max_pos)

        pieces = random.choice(piece_sets)
        fen = generate_fen()
        chessboard, piece_labels = generate_image(board_img, pieces, fen)
        chessboard = chessboard.resize((board_size_random, board_size_random))

        bg_img_copy.paste(chessboard, (random_x, random_y))

        if random.random() < PROB_COLOR_JITTER:
            bg_img_copy = apply_color_jitter(bg_img_copy)
        if random.random() < PROB_BLUR:
            bg_img_copy = apply_blur(bg_img_copy)
        if random.random() < PROB_NOISE:
            bg_img_copy = apply_noise(bg_img_copy)
        if random.random() < PROB_JPEG_COMPRESSION:
            bg_img_copy = apply_jpeg_compression(bg_img_copy, min_q=25, max_q=85)

        labels = labels_to_yolo_lines(
            piece_labels,
            BOARD_SIZE,
            BOARD_SIZE,
            x_bias=random_x,
            y_bias=random_y,
            scale=scale_factor,
        )

        if MAKE_LABELS_FOR_CHESSBOARD:
            labels.append(
                yolo_label(
                    random_x,
                    random_y,
                    board_size_random,
                    board_size_random,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    "12",
                )
            )

        bg_img_copy.save(f"{images_dir}/{image_id}.jpg", "JPEG", quality=92)
        with open(f"{labels_dir}/{image_id}.txt", "w") as f:
            f.write("\n".join(labels))

        image_id += 1


# ---------------------------------------------------------------------------
# Dataset Generation Pipeline
# ---------------------------------------------------------------------------
def generate_datasets(images_dir, labels_dir, boards, piece_sets, variations):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    current_id = get_next_image_id(images_dir)
    tasks = [
        (
            boards,
            piece_set,
            images_dir,
            labels_dir,
            variations,
            current_id + (idx * len(boards) * variations),
        )
        for idx, piece_set in enumerate(piece_sets)
    ]
    num_workers = max(1, multiprocessing.cpu_count())
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(generate_images_worker, tasks)


def run_generate_datasets_with_background_noise(
    images_dir, labels_dir, boards, piece_sets, backgrounds, variations
):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    current_id = get_next_image_id(images_dir)

    tasks = [
        (
            images_dir,
            labels_dir,
            boards,
            piece_sets,
            backgrounds[idx],
            variations,
            current_id + (idx * variations),
        )
        for idx in range(len(backgrounds))
    ]

    num_workers = max(1, multiprocessing.cpu_count())
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(generate_images_with_background_noise_worker, tasks)


def split_data(boards, pieces_sets, split):
    train_boards = boards[: int(len(boards) * split)]
    val_boards = boards[int(len(boards) * split) :]

    train_piece_sets = pieces_sets[: int(len(pieces_sets) * split)]
    val_piece_sets = pieces_sets[int(len(pieces_sets) * split) :]

    return train_boards, val_boards, train_piece_sets, val_piece_sets


def randomize_and_split_data(boards, pieces_sets, split):
    boards = boards[:]
    pieces_sets = pieces_sets[:]
    random.shuffle(boards)
    random.shuffle(pieces_sets)
    return split_data(boards, pieces_sets, split)


def get_next_image_id(dir_path):
    ids = []
    for f in os.listdir(dir_path):
        name, dot, ext = f.partition(".")
        if dot and name.isdigit():
            ids.append(int(name))
    return max(ids) + 1 if ids else 1


def main():
    print("Loading board and piece assets...")
    boards = [load_board(board) for board in os.listdir(BOARDS_DIR)]
    piece_sets = [load_pieces(piece_set) for piece_set in os.listdir(PIECES_DIR)]

    train_boards, val_boards, train_piece_sets, val_piece_sets = (
        randomize_and_split_data(boards, piece_sets, DATA_SPLIT)
    )

    print(f"Loaded {len(boards)} boards and {len(piece_sets)} piece sets.")
    print(f"Train split: {len(train_boards)} boards, {len(train_piece_sets)} piece sets.")
    print(f"Val split: {len(val_boards)} boards, {len(val_piece_sets)} piece sets.")

    print("\nGenerating training dataset (clean + board augmentations)...")
    generate_datasets(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        train_boards,
        train_piece_sets,
        VARIATIONS,
    )
    print("Training dataset generated.")

    print("\nGenerating validation dataset...")
    generate_datasets(
        DATASETS_IMAGES_DIR + "/val",
        DATASETS_LABELS_DIR + "/val",
        val_boards,
        val_piece_sets,
        VARIATIONS,
    )
    print("Validation dataset generated.")

    if not GENERATE_IMAGES_WITH_BACKGROUND_NOISE:
        print("Dataset generation completed!")
        return

    print("\nGenerating images with background noise and scene compositing...")

    backgrounds = os.listdir(BACKGROUND_NOISE_DIR)
    random.shuffle(backgrounds)
    train_backgrounds, val_backgrounds = (
        backgrounds[: int(len(backgrounds) * DATA_SPLIT)],
        backgrounds[int(len(backgrounds) * DATA_SPLIT) :],
    )

    run_generate_datasets_with_background_noise(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        train_boards,
        train_piece_sets,
        train_backgrounds,
        VARIATIONS,
    )
    print("Training dataset with background noise generated.")

    run_generate_datasets_with_background_noise(
        DATASETS_IMAGES_DIR + "/val",
        DATASETS_LABELS_DIR + "/val",
        val_boards,
        val_piece_sets,
        val_backgrounds,
        VARIATIONS,
    )
    print("Validation dataset with background noise generated.")

    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    main()
