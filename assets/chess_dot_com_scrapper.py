import aiohttp
import asyncio
import os

ALLOW_3D = False
BOARDS_PATH = 'boards'
PIECES_PATH = 'pieces'

async def download_image(url, path):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            with open(path, 'wb') as f:
                f.write(await response.read())

async def fetch_data(url, json_data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json_data) as response:
            return await response.json()

# boardStyles[n]["name"  | "image"]
async def fetch_board_styles():
    json_data = {
        'platform': 'WEB',
        'boardSize': 99999,
    }
    url = 'https://www.chess.com/rpc/chesscom.themes.v1.ThemesService/ListBoardStyles'
    return await fetch_data(url, json_data)

# pieceSets[n]["name"  | "images"] images = dict
async def fetch_pieces_sets():
    json_data = {
        'platform': 'WEB',
        'piecesSize': 99999,
    }
    url = 'https://www.chess.com/rpc/chesscom.themes.v1.ThemesService/ListPieceSets'
    return await fetch_data(url, json_data)

async def save_board_styles():
    board_styles = await fetch_board_styles()
    tasks = []
    for board_style in board_styles["boardStyles"]:
        light = board_style['coordinateColorLight'].replace("#", "")
        dark = board_style['coordinateColorDark'].replace("#", "")
        board_name = f"{board_style['name']}_{light}_{dark}".lower()
        
        board_image = board_style['image']
        tasks.append(download_image(board_image, f"{BOARDS_PATH}/{board_name}.png"))

    await asyncio.gather(*tasks)

async def save_pieces_styles():
    pieces_styles = await fetch_pieces_sets()
    tasks = []
    for piece_style in pieces_styles["pieceSets"]:
        if piece_style["perspective"] != "TOP_DOWN" and not ALLOW_3D:
            continue
        
        set_name = piece_style['name']
        os.makedirs(f"{PIECES_PATH}/{set_name}", exist_ok=True)

        piece_images = piece_style['images']
        for piece_name, piece_image in piece_images.items():
            # converts piece_name from blackKnight -> bK
            color = piece_name[:5]  # 'black' or 'white'
            piece = piece_name[5:]  # 'Knight', 'Rook', etc.
            short_color = 'b' if color == 'black' else 'w'
            short_piece = piece[0].upper() if piece != 'Knight' else 'N'
            piece_name = short_color + short_piece

            tasks.append(download_image(piece_image, f"{PIECES_PATH}/{set_name}/{piece_name}.png"))
    await asyncio.gather(*tasks)

async def main():
    await save_board_styles()
    await save_pieces_styles()

if __name__ == "__main__":
    asyncio.run(main())
