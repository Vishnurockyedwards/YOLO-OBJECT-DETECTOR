"""
Command-line image detection script for YOLO object detection.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import MODEL_PATH, CONFIDENCE_THRESHOLD
    from .detector import YOLODetector
    from .visualizer import DetectionVisualizer
except ImportError:
    from config import MODEL_PATH, CONFIDENCE_THRESHOLD
    from detector import YOLODetector
    from visualizer import DetectionVisualizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects in a single image using YOLOv8.")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the YOLO model weights file.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Minimum confidence threshold for detections.")
    parser.add_argument("--output", default=None, help="Path to save the output image with detections.")
    parser.add_argument("--device", default="cpu", help="Device for inference: cpu or cuda.")
    return parser.parse_args()


def summarize_detections(detections: list) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for det in detections:
        summary[det.class_name] = summary.get(det.class_name, 0) + 1
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    detector = YOLODetector(model_path=args.model, confidence=args.confidence, device=args.device)
    visualizer = DetectionVisualizer()

    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise ValueError(f"Failed to load image: {image_path}")

    start_time = time.time()
    detections = detector.detect(image)
    elapsed_ms = (time.time() - start_time) * 1000.0

    annotated = visualizer.draw_detections(image, detections)

    if detections:
        logger.info("Detected %d objects in %0.1f ms", len(detections), elapsed_ms)
        summary = summarize_detections(detections)
        logger.info("Detection summary: %s", summary)
    else:
        logger.info("No objects detected in %s", image_path)

    if args.output:
        output_path = Path(args.output)
        visualizer.save_result(annotated, str(output_path))
        logger.info("Saved annotated image to %s", output_path)
    else:
        output_path = image_path.parent / f"detected_{image_path.name}"
        visualizer.save_result(annotated, str(output_path))
        logger.info("Saved annotated image to %s", output_path)

    visualizer.show_result(annotated, window_name="YOLO Image Detection")


if __name__ == "__main__":
    main()
