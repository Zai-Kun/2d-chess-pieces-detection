import multiprocessing
import os
import random

from PIL import Image, ImageDraw

from random_fen_gen import generate_fen

ADD_RANDOM_DISTORTIONS = True
DISTORTION_PROBABILITY = 0.25

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
VALIDATION_SIZE = 1

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


def add_random_lines(board, max_lines=10, max_thickness=5, max_opacity=200):
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


def add_random_noise(board, max_points=200, max_radius=5):
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
                    board.paste(piece_set[char], (x, y), piece_set[char])
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


def genrate_images_with_background_noise(args):
    images_dir, labels_dir, boards, piece_sets, background, variations, image_id = args

    bg_img = Image.open(f"{BACKGROUND_NOISE_DIR}/{background}").convert("RGB")
    original_bg_size = bg_img.width  # Assume square background
    scale_factor = BOARD_SIZE / original_bg_size
    max_pos = original_bg_size - BOARD_SIZE
    scaled_tile = TILE_SIZE * scale_factor

    for _ in range(variations):
        bg_img_copy = bg_img.copy()
        random_x = random.randint(0, max_pos)
        random_y = random.randint(0, max_pos)

        # 3. Create chessboard with pieces
        board_img = random.choice(boards).copy()
        pieces = random.choice(piece_sets)
        fen = generate_fen()
        chessboard = generate_image(board_img, pieces, fen)

        # 4. Paste onto background and resize
        bg_img_copy.paste(chessboard, (random_x, random_y))
        resized_bg = bg_img_copy.resize((BOARD_SIZE, BOARD_SIZE))

        # 5. Calculate scaled values for label generation
        random_scaled_x = random_x * scale_factor
        random_scaled_y = random_y * scale_factor

        # 6. Generate CORRECT labels for resized image
        labels = fen_to_yolo_labels(
            fen,
            x_bias=random_scaled_x,
            y_bias=random_scaled_y,
            tile_size=scaled_tile,
        )

        # 7. Add chessboard label if needed
        if MAKE_LABELS_FOR_CHESSBOARD:
            labels.append(
                yolo_label(
                    random_scaled_x,
                    random_scaled_y,
                    BOARD_SIZE * scale_factor,
                    BOARD_SIZE * scale_factor,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    "12",
                )
            )

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
    num_workers = max(1, multiprocessing.cpu_count() - 2)
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

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(genrate_images_with_background_noise, tasks)


def main():
    # Load boards and shuffle them
    boards = [load_board(board) for board in os.listdir(BOARDS_DIR)]
    random.shuffle(boards)

    # Load piece sets and shuffle them
    piece_sets = [load_pieces(piece_set) for piece_set in os.listdir(PIECES_DIR)]
    random.shuffle(piece_sets)

    print("Boards and pieces loaded.")

    # generate training dataset
    generate_datasets(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        boards,
        piece_sets,
        VARIATIONS,
    )

    print("Training dataset generated.")

    # generate validation dataset
    generate_datasets(
        DATASETS_IMAGES_DIR + "/val",
        DATASETS_LABELS_DIR + "/val",
        boards[: int(len(boards) * VALIDATION_SIZE)],
        piece_sets[: int(len(piece_sets) * VALIDATION_SIZE)],
        VARIATIONS,
    )

    print("Validation dataset generated.")

    if not GENRATE_IMAGES_WITH_BACKGROUND_NOISE:
        return

    print("Generating images with background noise...")

    genrate_datasets_with_background_noise(
        DATASETS_IMAGES_DIR + "/train",
        DATASETS_LABELS_DIR + "/train",
        boards,
        piece_sets,
        os.listdir(BACKGROUND_NOISE_DIR),
        VARIATIONS,
    )

    print("Training dataset with background noise generated.")

    genrate_datasets_with_background_noise(
        DATASETS_IMAGES_DIR + "/val",
        DATASETS_LABELS_DIR + "/val",
        boards,
        piece_sets,
        os.listdir(BACKGROUND_NOISE_DIR),
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
