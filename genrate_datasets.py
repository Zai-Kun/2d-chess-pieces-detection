import multiprocessing
import os
import random

from PIL import Image, ImageDraw

from random_fen_gen import generate_fen

ADD_RANDOM_DISTORTIONS = True
DISTORTION_PROBABILITY = 0.5

ROTATE_RANDOMLY = True
ROTATE_PROBABILITY = 0.25

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


def fen_to_yolo_labels(
    fen,
    x_bias: int | float = 0,
    y_bias: int | float = 0,
    img_w=BOARD_SIZE,
    img_h=BOARD_SIZE,
    tile_size: int | float = TILE_SIZE,
):
    """Converts FEN to YOLO labels with scaling awareness."""
    labels = []
    for row, fen_rank in enumerate(fen.split()[0].split("/")):
        file_index = 0
        for char in fen_rank:
            if char.isdigit():
                file_index += int(char)
            else:
                if char in FEN_TO_PIECE:
                    piece_id = str(list(FEN_TO_PIECE.keys()).index(char))
                    # Calculate absolute coordinates using CURRENT tile size
                    x = (file_index * tile_size) + x_bias
                    y = (row * tile_size) + y_bias
                    # Normalize against background dimensions
                    labels.append(
                        yolo_label(x, y, tile_size, tile_size, img_w, img_h, piece_id)
                    )
                file_index += 1
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


def generate_image(board, piece_set, fen):
    # Draw pieces on the board
    for row, fen_rank in enumerate(fen.split()[0].split("/")):
        file_index = 0
        for char in fen_rank:
            if char.isdigit():
                file_index += int(char)
            else:
                if char in piece_set:
                    x, y = file_index * TILE_SIZE, row * TILE_SIZE
                    piece_image = piece_set[
                        char
                    ].copy()  # Copy to avoid modifying the original

                    if RESIZE_RANDOMLY and random.random() < RESIZE_PROBABILITY:
                        # Get the bounding box of the actual piece content (non-transparent pixels)
                        bbox = piece_image.getbbox()
                        if bbox:
                            piece_content = piece_image.crop(bbox)
                            content_width, content_height = piece_content.size

                            # Scale factor (random between 80% and 120% of original size)
                            scale_factor = random.uniform(0.8, 1.2)

                            # Ensure the resized content does not exceed TILE_SIZE
                            new_width = min(
                                int(content_width * scale_factor), TILE_SIZE
                            )
                            new_height = min(
                                int(content_height * scale_factor), TILE_SIZE
                            )

                            # Resize only the non-transparent content
                            resized_content = piece_content.resize(
                                (new_width, new_height)
                            )

                            # Create a new transparent image of TILE_SIZE and paste resized content centered
                            resized_piece = Image.new(
                                "RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0)
                            )
                            paste_x = (TILE_SIZE - new_width) // 2
                            paste_y = (TILE_SIZE - new_height) // 2
                            resized_piece.paste(
                                resized_content, (paste_x, paste_y), resized_content
                            )

                            piece_image = resized_piece  # Use resized image

                    if ROTATE_RANDOMLY and random.random() < ROTATE_PROBABILITY:
                        rotated_piece = piece_image.rotate(
                            random.uniform(-20, 20), expand=True
                        )

                        piece_width, piece_height = rotated_piece.size
                        offset_x = (piece_width - TILE_SIZE) // 2
                        offset_y = (piece_height - TILE_SIZE) // 2

                        board.paste(
                            rotated_piece, (x - offset_x, y - offset_y), rotated_piece
                        )
                    else:
                        board.paste(piece_image, (x, y), piece_image)

                file_index += 1

    # Add random distortions
    if ADD_RANDOM_DISTORTIONS:
        board = add_random_lines(board, max_lines=15, max_thickness=6, max_opacity=180)
        board = add_random_noise(board, max_points=150, max_radius=6)

    return board


def generate_images(args):
    boards, pieces, images_dir, labels_dir, variations, image_id = args[0:6]
    for board_image in boards:
        for _ in range(variations):
            fen = generate_fen()
            img_path = f"{images_dir}/{image_id}.jpg"
            label_path = f"{labels_dir}/{image_id}.txt"

            # Save image
            generate_image(board_image.copy(), pieces, fen).save(
                img_path, "JPEG", quality=95
            )

            # Save label
            with open(label_path, "w") as f:
                for label in fen_to_yolo_labels(fen):
                    f.write(label + "\n")
                if MAKE_LABELS_FOR_CHESSBOARD:
                    f.write(
                        yolo_label(
                            0, 0, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, "12"
                        )
                        + "\n"
                    )

            image_id += 1


def generate_images_with_background_noise(args):
    images_dir, labels_dir, boards, piece_sets, background, variations, image_id = args

    # Load and prepare background image
    bg_img = (
        Image.open(f"{BACKGROUND_NOISE_DIR}/{background}")
        .convert("RGB")
        .resize((BOARD_SIZE, BOARD_SIZE))
    )
    original_bg_size = bg_img.width

    for _ in range(variations):
        # Select and resize chessboard
        board_img = random.choice(boards).copy()
        board_size_random = random.randint(250, BOARD_SIZE)
        scale_factor = board_size_random / original_bg_size
        scaled_tile = TILE_SIZE * scale_factor
        max_pos = original_bg_size - board_size_random

        # Copy background and determine random position
        bg_img_copy = bg_img.copy()
        random_x = random.randint(0, max_pos)
        random_y = random.randint(0, max_pos)

        # Generate chessboard with pieces
        pieces = random.choice(piece_sets)
        fen = generate_fen()
        chessboard = generate_image(board_img, pieces, fen).resize(
            (board_size_random, board_size_random)
        )

        # Overlay chessboard onto background
        bg_img_copy.paste(chessboard, (random_x, random_y))

        # Convert FEN to YOLO labels
        labels = fen_to_yolo_labels(
            fen, x_bias=random_x, y_bias=random_y, tile_size=scaled_tile
        )

        # Optionally label the entire chessboard
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

        # Save image and labels
        resized_bg = bg_img_copy
        resized_bg.save(f"{images_dir}/{image_id}.jpg", "JPEG", quality=95)
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

    current_id = get_next_image_id(
        images_dir
    )  # i hope that images dir and labels dir have the same number of files
    backgrounds = os.listdir(BACKGROUND_NOISE_DIR)

    # images_dir, labels_dir, boards, piece_sets, background, variations, image_id = args
    tasks = [
        (
            images_dir,
            labels_dir,
            boards,
            piece_sets,
            backgrounds[idx],
            VARIATIONS,
            current_id + (idx * VARIATIONS),
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

    train_boards, val_boards, train_piece_sets, val_piece_sets = (
        randomize_and_split_data(boards, piece_sets, DATA_SPLIT)
    )

    backgrounds = os.listdir(BACKGROUND_NOISE_DIR)
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


def get_next_image_id(dir_path):
    lst_dir = os.listdir(dir_path)
    if not lst_dir:
        return 1
    return len(lst_dir)


if __name__ == "__main__":
    main()
