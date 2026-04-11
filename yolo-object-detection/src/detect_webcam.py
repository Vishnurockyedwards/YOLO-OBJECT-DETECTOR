"""
Command-line real-time webcam detection script for YOLO object detection.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run real-time webcam object detection using YOLOv8.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the YOLO model weights file.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Minimum confidence threshold for detections.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--device", default="cpu", help="Device for inference: cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise IOError(
            f"Unable to open webcam device {args.camera}. "
            "Check the camera index and ensure the camera is available."
        )

    detector = YOLODetector(model_path=args.model, confidence=args.confidence, device=args.device)
    visualizer = DetectionVisualizer()

    paused = False
    frame_count = 0
    last_time = time.time()
    fps = 0.0
    saved_count = 0

    while True:
        if not paused:
            ret, frame = capture.read()
            if not ret:
                logger.warning("Webcam frame read failed; exiting loop.")
                break
            frame_count += 1

            start_time = time.time()
            detections = detector.detect(frame)
            fps = 1.0 / max((time.time() - start_time), 1e-6)

            annotated = visualizer.draw_detections(frame, detections)
            annotated = visualizer.draw_info_overlay(annotated, fps=fps, num_detections=len(detections))

            cv2.imshow("YOLO Webcam Detection", annotated)
        else:
            key_frame = cv2.waitKey(100) & 0xFF
            if key_frame == ord("p"):
                paused = not paused
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            timestamp = int(time.time())
            save_path = output_dir / f"webcam_{timestamp}_{saved_count}.jpg"
            visualizer.save_result(annotated, str(save_path))
            saved_count += 1
            logger.info("Saved current frame to %s", save_path)
        if key == ord("+") or key == ord("="):
            args.confidence = min(0.99, args.confidence + 0.05)
            detector.confidence = args.confidence
            logger.info("Increased confidence threshold to %.2f", args.confidence)
        if key == ord("-"):
            args.confidence = max(0.01, args.confidence - 0.05)
            detector.confidence = args.confidence
            logger.info("Decreased confidence threshold to %.2f", args.confidence)
        if key == ord("p"):
            paused = not paused
            logger.info("Paused=%s", paused)

    capture.release()
    cv2.destroyAllWindows()
    logger.info("Released webcam and closed all windows.")


if __name__ == "__main__":
    main()
