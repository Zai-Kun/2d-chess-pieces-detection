import multiprocessing
import os
import random

from PIL import Image, ImageDraw

from random_fen_gen import generate_fen

# ---------------------------------------------------------------------------
# Distortions (toned down from the original — see notes below each change)
# ---------------------------------------------------------------------------
ADD_RANDOM_DISTORTIONS = True
# Was 0.6 for BOTH lines and noise independently, meaning ~84% of images got
# at least one distortion and ~36% got both. Lowered so the majority of
# images stay clean and distortions read as occasional noise, not the norm.
DISTORTION_PROBABILITY = 0.35

ROTATE_RANDOMLY = True
ROTATE_PROBABILITY = 0.20

RESIZE_RANDOMLY = True
RESIZE_PROBABILITY = 0.4
# Canvas used when resizing a piece, as a multiple of TILE_SIZE. Previously
# the resized content was capped to TILE_SIZE, which silently clipped the
# "up to 120%" upscale case back down to ~100% for most sprites (since
# sprites usually already fill most of their tile). A slightly larger
# working canvas lets the upscale augmentation actually take effect.
PIECE_CANVAS_SCALE = 1.3

GENRATE_IMAGES_WITH_BACKGROUND_NOISE = True

MAKE_LABELS_FOR_CHESSBOARD = True
BOARD_SIZE = 640
TILE_SIZE = BOARD_SIZE // 8
PIECE_CANVAS_SIZE = int(TILE_SIZE * PIECE_CANVAS_SCALE)
VARIATIONS = 4
BOARDS_DIR = "assets/boards"
PIECES_DIR = "assets/pieces"
BACKGROUND_NOISE_DIR = "assets/random_noise_backgrounds"
DATASETS_IMAGES_DIR = "datasets/images"
DATASETS_LABELS_DIR = "datasets/labels"
DATA_SPLIT = 0.7

FEN_TO_PIECE = {
    "p": "bP",
    "r": "bR",
    "n": "bN",
    "b": "bB",
    "q": "bQ",
    "k": "bK",
    "P": "wP",
    "R": "wR",
    "N": "wN",
    "B": "wB",
    "Q": "wQ",
    "K": "wK",
}
FEN_CHAR_ORDER = list(FEN_TO_PIECE.keys())


# for debugging
def draw_yolo_boxes(image, labels):
    """
    Draws bounding boxes on a PIL image using YOLO format labels.

    :param image: PIL Image object.
    :param labels: List of label strings in YOLO format.
                   Each label should be "class_id x_center y_center width height"
                   with values normalized (0 to 1).
    :return: PIL Image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    class_colors = {}

    for label in labels:
        label = label.replace("\n", " ").strip()
        tokens = label.split()

        if len(tokens) != 5:
            print(f"Skipping malformed label: {label}")
            continue

        try:
            class_id = int(tokens[0])
            x_center, y_center, w, h = map(float, tokens[1:])
        except ValueError as e:
            print(f"Error converting tokens for label '{label}': {e}")
            continue

        x1 = int((x_center - w / 2) * img_width)
        y1 = int((y_center - h / 2) * img_height)
        x2 = int((x_center + w / 2) * img_width)
        y2 = int((y_center + h / 2) * img_height)

        if class_id not in class_colors:
            class_colors[class_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        draw.rectangle([x1, y1, x2, y2], outline=class_colors[class_id], width=3)

    return image


def load_pieces(piece_set):
    return {
        f: Image.open(f"{PIECES_DIR}/{piece_set}/{p}.png")
        .convert("RGBA")
        .resize((TILE_SIZE, TILE_SIZE))
        for f, p in FEN_TO_PIECE.items()
    }


def load_board(board_file):
    return (
        Image.open(f"{BOARDS_DIR}/{board_file}")
        .convert("RGB")
        .resize((BOARD_SIZE, BOARD_SIZE))
    )


def yolo_label(x, y, w, h, img_w, img_h, class_id):
    """Convert an absolute pixel box to a YOLO format label string."""
    x_center, y_center = x + w / 2, y + h / 2
    return f"{class_id} {x_center / img_w:.6f} {y_center / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def labels_to_yolo_lines(piece_labels, img_w, img_h, x_bias=0.0, y_bias=0.0, scale=1.0):
    """
    Converts a list of (class_id, x, y, w, h) absolute-pixel boxes (computed
    in the original BOARD_SIZE x BOARD_SIZE coordinate space) into YOLO
    label lines, applying an optional scale + offset transform. This is used
    both for the plain dataset (scale=1, bias=0) and for the
    background-noise dataset, where the whole composited board gets resized
    and pasted at a random offset onto a background image.
    """
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


def _apply_random_resize(piece_image):
    """Randomly rescales a piece's actual (non-transparent) content between
    80% and 120% of its original size, centered on a slightly padded
    canvas so upscaling isn't silently clipped back down."""
    bbox = piece_image.getbbox()
    if not bbox:
        return piece_image

    piece_content = piece_image.crop(bbox)
    content_width, content_height = piece_content.size

    scale_factor = random.uniform(0.8, 1.2)
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


def add_random_lines(board, max_lines=12, max_thickness=10, max_opacity=140):
    """Randomly decides whether to add lines, and if so, how many and their properties."""
    if random.random() <= DISTORTION_PROBABILITY:
        overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = board.size
        num_lines = random.randint(1, max_lines)

        for _ in range(num_lines):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(30, max_opacity),
            )
            thickness = random.randint(1, max_thickness)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

        return Image.alpha_composite(board.convert("RGBA"), overlay).convert("RGB")

    return board


def add_random_noise(board, max_points=40, max_radius=12, max_opacity=140):
    """Randomly decides whether to add noise points, and if so, how many.
    Unlike the original version this composites with alpha instead of
    drawing fully opaque circles directly onto the board, so noise can no
    longer completely blot out a piece it happens to land on."""
    if random.random() <= DISTORTION_PROBABILITY:
        overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = board.size
        num_points = random.randint(1, max_points)

        for _ in range(num_points):
            x, y = random.randint(0, width), random.randint(0, height)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(30, max_opacity),
            )
            radius = random.randint(1, max_radius)
            draw.ellipse([(x, y), (x + radius, y + radius)], fill=color)

        return Image.alpha_composite(board.convert("RGBA"), overlay).convert("RGB")

    return board


def generate_image(board, piece_set, fen):
    """
    Draws pieces for the given FEN onto the board.

    Returns (board, piece_labels) where piece_labels is a list of
    (class_id, x, y, w, h) tuples in absolute pixel coordinates within the
    BOARD_SIZE x BOARD_SIZE space, tightly fit to the actual rendered
    (non-transparent) pixels of each piece — not the full tile.
    """
    piece_labels = []

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

            if RESIZE_RANDOMLY and random.random() < RESIZE_PROBABILITY:
                piece_image = _apply_random_resize(piece_image)

            if ROTATE_RANDOMLY and random.random() < ROTATE_PROBABILITY:
                piece_image = piece_image.rotate(random.uniform(-20, 20), expand=True)

            # Works uniformly whether the piece was resized, rotated, both,
            # or neither: for an untouched piece (still TILE_SIZE canvas)
            # this offset is simply 0.
            offset_x = (piece_image.width - TILE_SIZE) // 2
            offset_y = (piece_image.height - TILE_SIZE) // 2
            paste_x, paste_y = x - offset_x, y - offset_y

            board.paste(piece_image, (paste_x, paste_y), piece_image)

            # Tight bbox of what was actually drawn, in board coordinates.
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

    if ADD_RANDOM_DISTORTIONS:
        board = add_random_lines(board)
        board = add_random_noise(board)

    return board, piece_labels


def generate_images(args):
    boards, pieces, images_dir, labels_dir, variations, image_id = args[0:6]
    for board_image in boards:
        for _ in range(variations):
            fen = generate_fen()
            img_path = f"{images_dir}/{image_id}.jpg"
            label_path = f"{labels_dir}/{image_id}.txt"

            image, piece_labels = generate_image(board_image.copy(), pieces, fen)
            image.save(img_path, "JPEG", quality=95)

            lines = labels_to_yolo_lines(piece_labels, BOARD_SIZE, BOARD_SIZE)
            if MAKE_LABELS_FOR_CHESSBOARD:
                lines.append(
                    yolo_label(0, 0, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, "12")
                )

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            image_id += 1


def generate_images_with_background_noise(args):
    images_dir, labels_dir, boards, piece_sets, background, variations, image_id = args

    bg_img = (
        Image.open(f"{BACKGROUND_NOISE_DIR}/{background}")
        .convert("RGB")
        .resize((BOARD_SIZE, BOARD_SIZE))
    )
    original_bg_size = bg_img.width

    for _ in range(variations):
        board_img = random.choice(boards).copy()
        board_size_random = random.randint(350, BOARD_SIZE)
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

        # piece_labels are in original BOARD_SIZE-space; scale + offset them
        # to match the resized-and-repositioned chessboard.
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

        bg_img_copy.save(f"{images_dir}/{image_id}.jpg", "JPEG", quality=95)
        with open(f"{labels_dir}/{image_id}.txt", "w") as f:
            f.write("\n".join(labels))

        image_id += 1


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
        pool.map(generate_images, tasks)


def genrate_datasets_with_background_noise(
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
        pool.map(generate_images_with_background_noise, tasks)


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
    """Returns the next safe image id to use, based on the highest existing
    numeric filename rather than the file count. Using file count breaks
    (and can silently overwrite existing image/label pairs) if any files
    were ever deleted, moved, or if a previous run was interrupted."""
    ids = []
    for f in os.listdir(dir_path):
        name, dot, ext = f.partition(".")
        if dot and name.isdigit():
            ids.append(int(name))
    return max(ids) + 1 if ids else 1


def main():
    boards = [load_board(board) for board in os.listdir(BOARDS_DIR)]
    piece_sets = [load_pieces(piece_set) for piece_set in os.listdir(PIECES_DIR)]

    # Split ONCE and reuse the same train/val assets for both the plain
    # dataset and the background-noise dataset. Previously this split was
    # re-randomized independently for each dataset type, so a board/piece
    # set could land in "val" for one dataset and "train" for the other —
    # leaking assets between the two and making val metrics unreliable.
    train_boards, val_boards, train_piece_sets, val_piece_sets = (
        randomize_and_split_data(boards, piece_sets, DATA_SPLIT)
    )

    print("Boards and pieces loaded.")

    generate_datasets(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        train_boards,
        train_piece_sets,
        VARIATIONS,
    )
    print("Training dataset generated.")

    generate_datasets(
        DATASETS_IMAGES_DIR + "/val",
        DATASETS_LABELS_DIR + "/val",
        val_boards,
        val_piece_sets,
        VARIATIONS,
    )
    print("Validation dataset generated.")

    if not GENRATE_IMAGES_WITH_BACKGROUND_NOISE:
        return

    print("Generating images with background noise...")

    backgrounds = os.listdir(BACKGROUND_NOISE_DIR)
    random.shuffle(backgrounds)
    train_backgrounds, val_backgrounds = (
        backgrounds[: int(len(backgrounds) * DATA_SPLIT)],
        backgrounds[int(len(backgrounds) * DATA_SPLIT) :],
    )

    genrate_datasets_with_background_noise(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        train_boards,
        train_piece_sets,
        train_backgrounds,
        VARIATIONS,
    )
    print("Training dataset with background noise generated.")

    genrate_datasets_with_background_noise(
        DATASETS_IMAGES_DIR + "/val",
        DATASETS_LABELS_DIR + "/val",
        val_boards,
        val_piece_sets,
        val_backgrounds,
        VARIATIONS,
    )
    print("Validation dataset with background noise generated.")


if __name__ == "__main__":
    main()
