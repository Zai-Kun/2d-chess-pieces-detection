import asyncio
import json
import os
from io import BytesIO

import aiofiles
import aiohttp
import xxhash
from PIL import Image

# Constants
PAGES_TO_DOWNLOAD = 500
OUTPUT_PATH = "random_noise_backgrounds"
BASE_URL = "https://pixabay.com/photos/search"
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent requests

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Headers for request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0 Pixabay",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.5",
    "x-fetch-bootstrap": "1",
    "DNT": "1",
    "Sec-GPC": "1",
    "Alt-Used": "pixabay.com",
    "Connection": "keep-alive",
}

# Create a semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def fetch_json(session, url, params):
    async with semaphore:  # Ensure we don't exceed 50 concurrent requests
        async with session.get(url, params=params, headers=HEADERS) as response:
            if response.status == 200:
                try:
                    return json.loads(await response.text())
                except json.JSONDecodeError:
                    print(f"JSON decoding failed for {url}: {await response.text()}")
            else:
                print(f"Failed to fetch {url}: HTTP {response.status}")

async def download_image(session, url, resize=True):
    async with semaphore:  # Ensure we don't exceed 50 concurrent downloads
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to download image: {url}")
                return

            data = await response.read()
            file_hash = xxhash.xxh32(data).hexdigest()
            path = os.path.join(OUTPUT_PATH, f"{file_hash}.jpg")

            if resize:
                img = Image.open(BytesIO(data)).convert("RGB")
                min_size = min(img.size)
                img = img.resize((min_size, min_size))
                img.save(path, "JPEG", quality=90)
            else:
                async with aiofiles.open(path, "wb") as f:
                    await f.write(data)

            print(f"Downloaded: {url} -> {path}")

async def download_page(session, page):
    params = {
        "order": "ec",
        "pagi": str(page),
    }

    data = await fetch_json(session, BASE_URL, params)
    if not data or "page" not in data or "results" not in data["page"]:
        return

    tasks = []
    for item in data["page"]["results"]:
        try:
            width, height = item.get("width", 0), item.get("height", 0)
            img_url = item.get("sources", {}).get("2x")

            if img_url and width > 640 and height > 640:
                tasks.append(download_image(session, img_url))
        except KeyError:
            continue

    await asyncio.gather(*tasks)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [download_page(session, i) for i in range(PAGES_TO_DOWNLOAD)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
