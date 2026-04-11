"""
Command-line video detection script for YOLO object detection.
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
    parser = argparse.ArgumentParser(description="Detect objects in a video file using YOLOv8.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the YOLO model weights file.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Minimum confidence threshold for detections.")
    parser.add_argument("--output", default=None, help="Path to save the output video file.")
    parser.add_argument("--device", default="cpu", help="Device for inference: cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_path = Path(args.output) if args.output else video_path.parent / f"detected_{video_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))

    detector = YOLODetector(model_path=args.model, confidence=args.confidence, device=args.device)
    visualizer = DetectionVisualizer()

    total_frames = 0
    total_detections = 0
    start_time = time.time()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_start = time.time()
        detections = detector.detect(frame)
        total_detections += len(detections)
        total_frames += 1

        fps_frame = 1.0 / max((time.time() - frame_start), 1e-6)
        annotated = visualizer.draw_detections(frame, detections)
        annotated = visualizer.draw_info_overlay(annotated, fps=fps_frame, num_detections=len(detections))
        writer.write(annotated)
        cv2.imshow("YOLO Video Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.time() - start_time
    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    average_fps = total_frames / max(elapsed, 1e-6)
    logger.info("Finished video processing")
    logger.info("Total frames: %d", total_frames)
    logger.info("Total detections: %d", total_detections)
    logger.info("Average FPS: %.2f", average_fps)
    logger.info("Output saved to %s", output_path)


if __name__ == "__main__":
    main()
