import asyncio
import os
from io import BytesIO
import json

import aiofiles
import aiohttp
import xxhash
from PIL import Image
import random

# Constants
OUTPUT_PATH = "random_noise_backgrounds"
BASE_URL = "https://randomwordgenerator.com"
MAX_CONCURRENT_REQUESTS = 100  # Limit concurrent requests

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def download_image(session, url, resize=True):
    async with semaphore:
        async with session.get(BASE_URL + url) as response:
            if response.status != 200:
                print(f"Failed to download image: {url}")
                return

            data = await response.read()
            file_hash = xxhash.xxh32(data).hexdigest()
            path = os.path.join(OUTPUT_PATH, f"{file_hash}.jpg")

            if resize:
                img = Image.open(BytesIO(data)).convert("RGB")
                max_size = max(img.size)
                if max_size <= 640:
                    max_size = random.randint(640, 1280)
                img = img.resize((max_size, max_size))
                img.save(path, "JPEG", quality=90)
            else:
                async with aiofiles.open(path, "wb") as f:
                    await f.write(data)

            # print(f"Downloaded: {url} -> {path}")

async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL + "/json/pictures.php?category=all") as response:
            if response.status != 200:
                print("Failed to fetch images")
                return
            data = json.loads(await response.text())
            tasks = [download_image(session, img["image_url"]) for img in data["data"]]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
