from PIL import Image, ImageDraw, ImageEnhance
import random
import os

TEXTURES_DIR = "textures"

def load_random_texture_crop(tile_size):
    """Pick a random texture file and crop a random patch from it, sized to one tile."""
    files = os.listdir(TEXTURES_DIR)
    tex = Image.open(f"{TEXTURES_DIR}/{random.choice(files)}").convert("RGB")

    # pick a random crop region so different tiles from the same texture look varied
    tw, th = tex.size
    if tw < tile_size or th < tile_size:
        tex = tex.resize((max(tile_size, tw), max(tile_size, th)))
        tw, th = tex.size
    x = random.randint(0, tw - tile_size)
    y = random.randint(0, th - tile_size)
    return tex.crop((x, y, x + tile_size, y + tile_size))


def generate_textured_board(size=640, texture_opacity=0.6):
    tile = size // 8
    light = tuple(random.randint(180, 255) for _ in range(3))
    dark = tuple(random.randint(20, 140) for _ in range(3))

    board = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(board)

    # base flat checker pattern first
    for row in range(8):
        for col in range(8):
            color = light if (row + col) % 2 == 0 else dark
            draw.rectangle([col * tile, row * tile, (col + 1) * tile, (row + 1) * tile], fill=color)

    if random.random() < 0.7:  # not every board gets a texture — keep some flat ones too
        texture_crop = load_random_texture_crop(size)  # one texture patch covering the whole board
        board = Image.blend(board, texture_crop, alpha=texture_opacity)

    return board

for i in range(80):
    generate_textured_board().save(f"boards/___{i}.png")
