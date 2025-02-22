import os
import random
from PIL import Image
from multiprocessing import Pool, cpu_count

BOARDS_DIR = "boards"
BACKGROUND_DIR = "random_noise_backgrounds"
NUM_IMAGES = len(os.listdir(BACKGROUND_DIR))

os.makedirs(BACKGROUND_DIR, exist_ok=True)

def create_color_image(index):
    color = tuple(random.randint(0, 255) for _ in range(3))
    filename = f"background_{index}.jpg"
    img = Image.new("RGB", (640, 640), color)
    img.save(os.path.join(BACKGROUND_DIR, filename))

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        pool.map(create_color_image, range(NUM_IMAGES))
