"""
Educational OpenCV DNN detection pipeline using YOLOv3/YOLOv4.

This script demonstrates the low-level mechanics of object detection with
OpenCV's DNN module, without relying on Ultralytics. It loads Darknet model
files, creates a blob, runs a forward pass, parses raw outputs, and applies
custom confidence filtering and NMS.

Download URLs:
- YOLOv3 config: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
- YOLOv3 weights: https://pjreddie.com/media/files/yolov3.weights
- COCO names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .detector import Detection
    from .postprocessor import PostProcessor
    from .visualizer import DetectionVisualizer
except ImportError:
    from detector import Detection
    from postprocessor import PostProcessor
    from visualizer import DetectionVisualizer

logger = logging.getLogger(__name__)


def load_class_names(names_path: Path) -> Dict[int, str]:
    """
    Load COCO class names from a file.

    Args:
        names_path: Path to the .names file.

    Returns:
        A mapping from class index to name.
    """
    if not names_path.exists():
        logger.warning("Class names file not found: %s", names_path)
        return {}

    with names_path.open("r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]

    return {index: name for index, name in enumerate(names)}


def load_darknet_model(cfg_path: Path, weights_path: Path) -> cv2.dnn_Net:
    """
    Load a Darknet YOLO model from cfg and weights files.

    Args:
        cfg_path: Path to the Darknet configuration file.
        weights_path: Path to the Darknet weights file.

    Returns:
        OpenCV DNN network object.
    """
    if not cfg_path.exists():
        raise FileNotFoundError(f"Darknet config file not found: {cfg_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Darknet weights file not found: {weights_path}")

    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    logger.info("Loaded Darknet model: %s and %s", cfg_path, weights_path)
    return net


def create_blob(image: np.ndarray, target_size: int = 416) -> np.ndarray:
    """
    Convert an image into a DNN blob for Darknet YOLO inference.

    The blob scales pixel values to [0, 1], converts BGR to RGB, and preserves
    aspect ratio by setting crop=False. This matches the preprocessing expected
    by the Darknet model.

    Args:
        image: BGR input image.
        target_size: Square target size for the network.

    Returns:
        4D blob tensor with shape (1, 3, target_size, target_size).
    """
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=(target_size, target_size),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    logger.info("Created blob with shape %s", blob.shape)
    return blob


def get_output_layer_names(net: cv2.dnn_Net) -> List[str]:
    """
    Get the output layer names from the network.

    YOLO returns predictions from several output layers. OpenCV provides the
    unconnected output layer IDs, which we convert to layer names.

    Args:
        net: OpenCV DNN network.

    Returns:
        List of output layer names.
    """
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    return [layer_names[i - 1] for i in unconnected.flatten()]


def parse_darknet_outputs(
    outputs: List[np.ndarray],
    image_width: int,
    image_height: int,
    conf_threshold: float,
    class_names: Dict[int, str],
) -> List[Tuple[Tuple[float, float, float, float], float, int, str]]:
    """
    Parse raw YOLO output tensors into bounding boxes and scores.

    Each output row contains: [center_x, center_y, width, height, objectness, class_scores...].
    We multiply objectness by the highest class score to produce the final confidence.

    Args:
        outputs: List of output arrays from the network.
        image_width: Original image width.
        image_height: Original image height.
        conf_threshold: Minimum confidence threshold.
        class_names: Mapping of class IDs to names.

    Returns:
        List of tuples containing bbox, confidence, class_id, and class_name.
    """
    results: List[Tuple[Tuple[float, float, float, float], float, int, str]] = []

    for output in outputs:
        # Each detection is a row of values for one anchor box.
        for detection in output:
            scores = detection[5:]
            if scores.size == 0:
                continue

            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            objectness = float(detection[4])
            confidence = objectness * class_score

            if confidence < conf_threshold:
                continue

            center_x = float(detection[0]) * image_width
            center_y = float(detection[1]) * image_height
            width = float(detection[2]) * image_width
            height = float(detection[3]) * image_height

            x1 = max(0.0, center_x - width / 2.0)
            y1 = max(0.0, center_y - height / 2.0)
            x2 = min(image_width - 1.0, center_x + width / 2.0)
            y2 = min(image_height - 1.0, center_y + height / 2.0)

            class_name = class_names.get(class_id, str(class_id))
            results.append(((x1, y1, x2, y2), confidence, class_id, class_name))

    logger.info("Parsed %d raw detections from network outputs", len(results))
    return results


def build_detections(
    parsed_results: List[Tuple[Tuple[float, float, float, float], float, int, str]]
) -> List[Detection]:
    """Build Detection objects from parsed output tuples."""
    return [
        Detection(
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
        )
        for bbox, confidence, class_id, class_name in parsed_results
    ]


def detect_with_opencv_dnn(
    image: np.ndarray,
    net: cv2.dnn_Net,
    output_layers: List[str],
    class_names: Dict[int, str],
    conf_threshold: float,
    nms_threshold: float,
    target_size: int,
) -> Tuple[List["cv2.UMat"], Dict[str, float]]:
    """
    Run the full OpenCV DNN detection pipeline on an image.

    Args:
        image: Input image array in BGR format.
        net: Loaded Darknet network.
        output_layers: Names of output layers for forward pass.
        class_names: COCO class name mapping.
        conf_threshold: Confidence threshold.
        nms_threshold: NMS IoU threshold.
        target_size: Size used for blob creation.

    Returns:
        Tuple of cleaned detections and timing metrics.
    """
    metrics: Dict[str, float] = {}

    start_pre = time.time()
    blob = create_blob(image, target_size=target_size)
    net.setInput(blob)
    metrics["preprocessing_ms"] = (time.time() - start_pre) * 1000.0

    start_infer = time.time()
    outputs = net.forward(output_layers)
    metrics["inference_ms"] = (time.time() - start_infer) * 1000.0

    parsed = parse_darknet_outputs(outputs, image.shape[1], image.shape[0], conf_threshold, class_names)
    raw_detections = build_detections(parsed)
    metrics["raw_count"] = len(raw_detections)

    start_post = time.time()
    processor = PostProcessor()
    cleaned_detections = processor.process(raw_detections, conf_threshold, nms_threshold)
    metrics["postprocessing_ms"] = (time.time() - start_post) * 1000.0
    metrics["cleaned_count"] = len(cleaned_detections)

    return cleaned_detections, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects with OpenCV DNN and YOLO Darknet weights.")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument("--cfg", default="models/yolov3.cfg", help="Path to Darknet cfg file.")
    parser.add_argument("--weights", default="models/yolov3.weights", help="Path to Darknet weights file.")
    parser.add_argument("--names", default="models/coco.names", help="Path to COCO class names file.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--output", default=None, help="Path to save the annotated output image.")
    parser.add_argument("--size", type=int, default=416, help="Network input size for blob creation.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    cfg_path = Path(args.cfg)
    weights_path = Path(args.weights)
    names_path = Path(args.names)

    class_names = load_class_names(names_path)
    net = load_darknet_model(cfg_path, weights_path)
    output_layers = get_output_layer_names(net)

    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise ValueError(f"Failed to read image: {image_path}")

    detections, metrics = detect_with_opencv_dnn(
        image=image,
        net=net,
        output_layers=output_layers,
        class_names=class_names,
        conf_threshold=args.confidence,
        nms_threshold=args.nms,
        target_size=args.size,
    )

    logger.info("Preprocessing time: %.2f ms", metrics["preprocessing_ms"])
    logger.info("Inference time: %.2f ms", metrics["inference_ms"])
    logger.info("Post-processing time: %.2f ms", metrics["postprocessing_ms"])
    logger.info("Raw detections: %d", int(metrics["raw_count"]))
    logger.info("Cleaned detections: %d", metrics["cleaned_count"])

    visualizer = DetectionVisualizer()
    annotated = visualizer.draw_detections(image, detections)
    output_path = Path(args.output) if args.output else image_path.parent / f"dnn_{image_path.name}"
    visualizer.save_result(annotated, str(output_path))
    logger.info("Saved annotated image to %s", output_path)
    visualizer.show_result(annotated, window_name="OpenCV DNN YOLO Detection")


if __name__ == "__main__":
    main()
