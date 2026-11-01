import multiprocessing
import os
import random
from io import BytesIO

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from random_fen_gen import generate_fen

ADD_RANDOM_DISTORTIONS = True
DISTORTION_PROBABILITY = 0.6

# Screenshot-quality augmentations. These do not change image dimensions,
# so the YOLO bounding boxes remain valid.
ENABLE_IMAGE_QUALITY_AUGMENTATIONS = True
JPEG_COMPRESSION_PROBABILITY = 0.45
JPEG_QUALITY_RANGE = (45, 90)
DOWNSCALE_PROBABILITY = 0.20
DOWNSCALE_RANGE = (0.55, 0.90)
BLUR_PROBABILITY = 0.15
BLUR_RADIUS_RANGE = (0.3, 1.0)
BRIGHTNESS_PROBABILITY = 0.25
BRIGHTNESS_RANGE = (0.75, 1.25)
CONTRAST_PROBABILITY = 0.25
CONTRAST_RANGE = (0.75, 1.25)
COLOR_PROBABILITY = 0.20
COLOR_RANGE = (0.70, 1.30)

ROTATE_RANDOMLY = False
ROTATE_PROBABILITY = 0.20

RESIZE_RANDOMLY = True
RESIZE_PROBABILITY = 0.4

GENRATE_IMAGES_WITH_BACKGROUND_NOISE = True

MAKE_LABELS_FOR_CHESSBOARD = True
BOARD_SIZE = 640
TILE_SIZE = BOARD_SIZE // 8
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
        # Remove any stray newlines and extra spaces.
        label = label.replace("\n", " ").strip()
        tokens = label.split()  # splits on whitespace and ignores extra spaces

        if len(tokens) != 5:
            print(f"Skipping malformed label: {label}")
            continue

        try:
            class_id = int(tokens[0])
            x_center, y_center, w, h = map(float, tokens[1:])
        except ValueError as e:
            print(f"Error converting tokens for label '{label}': {e}")
            continue

        # Convert normalized YOLO coordinates to pixel coordinates.
        x1 = int((x_center - w / 2) * img_width)
        y1 = int((y_center - h / 2) * img_height)
        x2 = int((x_center + w / 2) * img_width)
        y2 = int((y_center + h / 2) * img_height)

        # If this class id does not yet have a color, assign one.
        if class_id not in class_colors:
            class_colors[class_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        # Draw the bounding box (no text, just the rectangle).
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
    """Convert chess piece position to YOLO format."""
    x_center, y_center = x + w / 2, y + h / 2
    return f"{class_id} {x_center / img_w:.6f} {y_center / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def piece_boxes_to_yolo_labels(
    piece_boxes,
    img_w=BOARD_SIZE,
    img_h=BOARD_SIZE,
    scale=1.0,
    x_offset=0,
    y_offset=0,
):
    """Convert visible piece boxes (class_id, x1, y1, x2, y2) to YOLO labels."""
    labels = []
    for class_id, x1, y1, x2, y2 in piece_boxes:
        x1 = x_offset + x1 * scale
        y1 = y_offset + y1 * scale
        x2 = x_offset + x2 * scale
        y2 = y_offset + y2 * scale

        labels.append(
            yolo_label(x1, y1, x2 - x1, y2 - y1, img_w, img_h, class_id)
        )

    return labels


def add_random_lines(board, max_lines=20, max_thickness=20, max_opacity=200):
    """Randomly decides whether to add lines, and if so, how many and their properties."""
    if random.random() <= DISTORTION_PROBABILITY:
        overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = board.size
        num_lines = random.randint(1, max_lines)  # Random number of lines

        for _ in range(num_lines):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(50, max_opacity),  # Random opacity
            )
            thickness = random.randint(1, max_thickness)  # Random thickness
            draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

        return Image.alpha_composite(board.convert("RGBA"), overlay).convert("RGB")

    return board  # Return original board if no lines are added


def add_random_noise(board, max_points=200, max_radius=25):
    """Randomly decides whether to add noise points, and if so, how many."""
    if random.random() <= DISTORTION_PROBABILITY:
        draw = ImageDraw.Draw(board)
        width, height = board.size
        num_points = random.randint(1, max_points)  # Random number of noise points

        for _ in range(num_points):
            x, y = random.randint(0, width), random.randint(0, height)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            radius = random.randint(1, max_radius)  # Random size of noise
            draw.ellipse([(x, y), (x + radius, y + radius)], fill=color)

    return board  # Return board with or without noise


def apply_random_image_effects(image):
    """Simulate screenshot quality issues without changing image dimensions.

    These effects (color/brightness/contrast, blur, resampling and JPEG
    compression) do not change object positions, so existing labels stay valid.
    """
    if not ENABLE_IMAGE_QUALITY_AUGMENTATIONS:
        return image

    image = image.convert("RGB")

    if random.random() < BRIGHTNESS_PROBABILITY:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(*BRIGHTNESS_RANGE)
        )

    if random.random() < CONTRAST_PROBABILITY:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(*CONTRAST_RANGE)
        )

    if random.random() < COLOR_PROBABILITY:
        image = ImageEnhance.Color(image).enhance(random.uniform(*COLOR_RANGE))

    if random.random() < DOWNSCALE_PROBABILITY:
        width, height = image.size
        scale = random.uniform(*DOWNSCALE_RANGE)
        small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(small_size, Image.Resampling.BILINEAR)
        image = image.resize((width, height), Image.Resampling.BICUBIC)

    if random.random() < BLUR_PROBABILITY:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(*BLUR_RADIUS_RANGE))
        )

    if random.random() < JPEG_COMPRESSION_PROBABILITY:
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=random.randint(*JPEG_QUALITY_RANGE),
            optimize=True,
        )
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")

    return image


def generate_image(board, piece_set, fen):
    """Draw a position and return its image plus visible piece bounding boxes.

    Each box is (class_id, x1, y1, x2, y2) in board-image pixel coordinates.
    """
    piece_boxes = []

    for row, fen_rank in enumerate(fen.split()[0].split("/")):
        file_index = 0
        for char in fen_rank:
            if char.isdigit():
                file_index += int(char)
            else:
                if char in piece_set:
                    x, y = file_index * TILE_SIZE, row * TILE_SIZE
                    piece_id = str(list(FEN_TO_PIECE.keys()).index(char))
                    piece_image = piece_set[char].copy()

                    if RESIZE_RANDOMLY and random.random() < RESIZE_PROBABILITY:
                        # Crop using alpha so transparent padding is excluded.
                        bbox = piece_image.getchannel("A").getbbox()
                        if bbox:
                            piece_content = piece_image.crop(bbox)
                            content_width, content_height = piece_content.size
                            scale_factor = random.uniform(0.8, 1.2)

                            new_width = min(
                                int(content_width * scale_factor), TILE_SIZE
                            )
                            new_height = min(
                                int(content_height * scale_factor), TILE_SIZE
                            )

                            resized_content = piece_content.resize(
                                (new_width, new_height)
                            )

                            resized_piece = Image.new(
                                "RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0)
                            )
                            paste_x = (TILE_SIZE - new_width) // 2
                            paste_y = (TILE_SIZE - new_height) // 2
                            resized_piece.paste(
                                resized_content, (paste_x, paste_y), resized_content
                            )
                            piece_image = resized_piece

                    if ROTATE_RANDOMLY and random.random() < ROTATE_PROBABILITY:
                        piece_image = piece_image.rotate(
                            random.uniform(-20, 20), expand=True
                        )
                        piece_width, piece_height = piece_image.size
                        offset_x = (piece_width - TILE_SIZE) // 2
                        offset_y = (piece_height - TILE_SIZE) // 2
                        paste_x, paste_y = x - offset_x, y - offset_y
                    else:
                        paste_x, paste_y = x, y

                    board.paste(piece_image, (paste_x, paste_y), piece_image)

                    # Measure the actual non-transparent pixels AFTER all transforms.
                    alpha_bbox = piece_image.getchannel("A").getbbox()
                    if alpha_bbox:
                        left = max(0, paste_x + alpha_bbox[0])
                        top = max(0, paste_y + alpha_bbox[1])
                        right = min(board.width, paste_x + alpha_bbox[2])
                        bottom = min(board.height, paste_y + alpha_bbox[3])

                        if left < right and top < bottom:
                            piece_boxes.append(
                                (piece_id, left, top, right, bottom)
                            )

                file_index += 1

    if ADD_RANDOM_DISTORTIONS:
        board = add_random_lines(board, max_lines=20, max_thickness=20, max_opacity=180)
        board = add_random_noise(board, max_points=40, max_radius=35)

    return board, piece_boxes


def generate_images(args):
    boards, pieces, images_dir, labels_dir, variations, image_id = args[0:6]
    for board_image in boards:
        for _ in range(variations):
            fen = generate_fen()
            img_path = f"{images_dir}/{image_id}.jpg"
            label_path = f"{labels_dir}/{image_id}.txt"

            image, piece_boxes = generate_image(board_image.copy(), pieces, fen)
            image = apply_random_image_effects(image)
            image.save(img_path, "JPEG", quality=95)

            labels = piece_boxes_to_yolo_labels(piece_boxes)
            if MAKE_LABELS_FOR_CHESSBOARD:
                labels.append(
                    yolo_label(
                        0, 0, BOARD_SIZE, BOARD_SIZE,
                        BOARD_SIZE, BOARD_SIZE, "12"
                    )
                )

            with open(label_path, "w") as f:
                f.write("\n".join(labels))

            image_id += 1


def generate_images_with_background_noise(args):
    images_dir, labels_dir, boards, piece_sets, background, variations, image_id = args

    bg_img = (
        Image.open(f"{BACKGROUND_NOISE_DIR}/{background}")
        .convert("RGB")
        .resize((BOARD_SIZE, BOARD_SIZE))
    )

    for _ in range(variations):
        board_img = random.choice(boards).copy()
        board_size_random = random.randint(200, BOARD_SIZE)
        scale_factor = board_size_random / BOARD_SIZE
        max_pos = BOARD_SIZE - board_size_random

        bg_img_copy = bg_img.copy()
        random_x = random.randint(0, max_pos)
        random_y = random.randint(0, max_pos)

        pieces = random.choice(piece_sets)
        fen = generate_fen()
        chessboard, piece_boxes = generate_image(board_img, pieces, fen)
        chessboard = chessboard.resize((board_size_random, board_size_random))
        bg_img_copy.paste(chessboard, (random_x, random_y))

        # Scale the measured piece boxes along with the resized chessboard,
        # then offset them to the chessboard's position in the background.
        labels = piece_boxes_to_yolo_labels(
            piece_boxes,
            img_w=BOARD_SIZE,
            img_h=BOARD_SIZE,
            scale=scale_factor,
            x_offset=random_x,
            y_offset=random_y,
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

        bg_img_copy = apply_random_image_effects(bg_img_copy)
        bg_img_copy.save(f"{images_dir}/{image_id}.jpg", "JPEG", quality=95)
        with open(f"{labels_dir}/{image_id}.txt", "w") as f:
            f.write("\n".join(labels))

        image_id += 1


def generate_datasets(images_dir, labels_dir, boards, piece_sets, variations):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Get last saved image index
    current_id = get_next_image_id(images_dir)
    # Create tasks
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

    # Use only the backgrounds assigned to this split by main().
    current_id = get_next_image_id(images_dir)

    tasks = [
        (
            images_dir,
            labels_dir,
            boards,
            piece_sets,
            background,
            variations,
            current_id + (idx * variations),
        )
        for idx, background in enumerate(backgrounds)
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
    boards = boards.copy()
    pieces_sets = pieces_sets.copy()
    random.shuffle(boards)
    random.shuffle(pieces_sets)
    return split_data(boards, pieces_sets, split)


def main():
    boards = [load_board(board) for board in os.listdir(BOARDS_DIR)]
    piece_sets = [load_pieces(piece_set) for piece_set in os.listdir(PIECES_DIR)]
    train_boards, val_boards, train_piece_sets, val_piece_sets = (
        randomize_and_split_data(boards, piece_sets, DATA_SPLIT)
    )

    print("Boards and pieces loaded.")

    # generate training dataset
    generate_datasets(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        train_boards,
        train_piece_sets,
        VARIATIONS,
    )

    print("Training dataset generated.")

    # generate validation dataset
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

    # Split backgrounds once too; each background is assigned to only one split.
    backgrounds = os.listdir(BACKGROUND_NOISE_DIR)
    random.shuffle(backgrounds)
    split_index = int(len(backgrounds) * DATA_SPLIT)
    train_backgrounds = backgrounds[:split_index]
    val_backgrounds = backgrounds[split_index:]

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


def get_next_image_id(dir_path):
    image_ids = []
    for filename in os.listdir(dir_path):
        stem, _ = os.path.splitext(filename)
        if stem.isdigit():
            image_ids.append(int(stem))
    return max(image_ids, default=0) + 1


if __name__ == "__main__":
    main()
