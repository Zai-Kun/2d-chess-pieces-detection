import os
import random
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count

BOARDS_DIR = "boards"
BACKGROUND_DIR = "random_noise_backgrounds"

os.makedirs(BACKGROUND_DIR, exist_ok=True)  # Ensure output directory exists

# Function to get the average RGB color of a cell
def get_avg_rgb(cell):
    cell_np = np.array(cell)
    return tuple(cell_np.mean(axis=(0, 1)).astype(int))  # Convert to integer RGB

# Function to create a solid color image with random size
def create_color_image(color, filename):
    size = random.randint(660, 1280)  # Random size between 660 and 1280
    img = Image.new("RGB", (size, size), color)  # Create solid color image
    img.save(os.path.join(BACKGROUND_DIR, filename))

# Function to process a single board image
def process_board(board_filename):
    board_path = os.path.join(BOARDS_DIR, board_filename)
    
    # Open board image
    board = Image.open(board_path).convert("RGB")

    # Compute cell size (assuming 8x8 grid)
    cell_size = board.size[0] // 8

    # Crop first two cells
    cell1 = board.crop((0, 0, cell_size, cell_size))
    cell2 = board.crop((cell_size, 0, cell_size * 2, cell_size))

    # Get average RGB colors
    avg_color1 = get_avg_rgb(cell1)
    avg_color2 = get_avg_rgb(cell2)

    # Compute combined average color
    combined_cells = np.concatenate(
        (np.array(cell1).reshape(-1, 3), np.array(cell2).reshape(-1, 3)), axis=0
    )
    avg_color_combined = tuple(combined_cells.mean(axis=0).astype(int))

    # Create and save images with random sizes
    create_color_image(avg_color1, f"{board_filename}_cell1.jpg")
    create_color_image(avg_color2, f"{board_filename}_cell2.jpg")
    create_color_image(avg_color_combined, f"{board_filename}_combined.jpg")

# Run multiprocessing pool
if __name__ == "__main__":
    board_files = os.listdir(BOARDS_DIR)
    
    # Use a pool with the number of available CPU cores
    with Pool(cpu_count()) as pool:
        pool.map(process_board, board_files)
