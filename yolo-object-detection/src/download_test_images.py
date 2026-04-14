"""
Download a curated set of diverse test images for evaluating the YOLO model.

Fetches public-domain / freely-licensed images from Wikimedia Commons that
cover many of the 600+ Open Images V7 classes (animals, food, vehicles,
household items, sports, etc.).
"""

import logging
import sys
import time
import urllib.request
from pathlib import Path

USER_AGENT = "YOLO-Test-Downloader/1.0 (educational project; contact: local-user@example.com)"
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Stable image sources:
#  - Ultralytics official sample images (hosted on their CDN)
#  - GitHub raw assets from public CV repos
IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    "dog.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg",
    "horses.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg",
    "person.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg",
    "eagle.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg",
    "kite.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg",
    "giraffe.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/giraffe.jpg",
    "scream.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/scream.jpg",
    "fruit.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "messi.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
    "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "baboon.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "aero1.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aero1.jpg",
    "aero3.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aero3.jpg",
    "graf1.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/graf1.png",
    "stuff.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/stuff.jpg",
    "basketball1.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png",
    "basketball2.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball2.png",
    # Random photos from picsum.photos (using fixed seed IDs for reproducibility)
    "random_001.jpg": "https://picsum.photos/id/1/800/600",
    "random_food.jpg": "https://picsum.photos/id/292/800/600",
    "random_dog.jpg": "https://picsum.photos/id/237/800/600",
    "random_office.jpg": "https://picsum.photos/id/119/800/600",
    "random_cars.jpg": "https://picsum.photos/id/133/800/600",
    "random_nature.jpg": "https://picsum.photos/id/15/800/600",
    "random_city.jpg": "https://picsum.photos/id/164/800/600",
    "random_people.jpg": "https://picsum.photos/id/177/800/600",
    "random_kitchen.jpg": "https://picsum.photos/id/225/800/600",
    "random_room.jpg": "https://picsum.photos/id/249/800/600",
}


def download_image(url: str, dest: Path) -> bool:
    """Download a single image from a URL to dest. Returns True on success."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            dest.write_bytes(data)
            size_kb = len(data) / 1024
            logger.info("Downloaded %s (%.1f KB)", dest.name, size_kb)
            return True
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < MAX_RETRIES:
                backoff = 5 * attempt
                logger.warning("Rate-limited on %s (attempt %d). Backing off %ds.", dest.name, attempt, backoff)
                time.sleep(backoff)
                continue
            logger.error("Failed to download %s: %s", dest.name, error)
            return False
        except Exception as error:
            logger.error("Failed to download %s: %s", dest.name, error)
            return False
    return False


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "input" / "test_set"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving images to %s", output_dir)

    succeeded = 0
    skipped = 0
    failed = 0

    for filename, url in IMAGES.items():
        dest = output_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("Already exists: %s", filename)
            skipped += 1
            continue
        if download_image(url, dest):
            succeeded += 1
        else:
            failed += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("Done. Downloaded=%d, Skipped=%d, Failed=%d", succeeded, skipped, failed)
    logger.info("Run detection on any image, e.g.:")
    logger.info("  python src/detect_image.py --image input/test_set/dog_park.jpg --output output/dog_park.jpg")


if __name__ == "__main__":
    sys.exit(main())
