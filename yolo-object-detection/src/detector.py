"""
Core YOLO detector implementation using Ultralytics YOLOv8.

This module defines a detection class and a detector wrapper that loads a
YOLOv8 model, runs inference, and converts results into structured detection
objects for downstream visualization and evaluation.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

try:
    from .config import CLASSES_FILE, INPUT_SIZE
except ImportError:
    from config import CLASSES_FILE, INPUT_SIZE

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single object detection result."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    YOLO detector wrapper around Ultralytics YOLOv8.

    This class keeps configuration values and provides methods for single-image
    and batch inference. It also exposes a simple model summary for inspection.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        confidence: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO weights file, or a known model name.
            confidence: Minimum confidence score for detections.
            nms_threshold: IoU threshold for built-in non-max suppression.
            device: Device to run inference on, e.g. 'cpu' or 'cuda'.
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.input_size = INPUT_SIZE
        self.device = device

        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(str(self.model_path))
        except Exception as error:
            logger.error("Failed to load YOLO model", exc_info=True)
            raise RuntimeError(
                f"Unable to load YOLO model from {self.model_path}. "
                "Check the path or internet connection for auto-download."
            ) from error

        self.class_names = self._load_class_names()
        logger.info(f"YOLODetector initialized with model={self.model_path}, confidence={confidence}, nms_threshold={nms_threshold}")

    def _load_class_names(self) -> Dict[int, str]:
        """
        Load COCO class names from the configured classes file.

        If the file is missing, fall back to the model's built-in names.
        """
        try:
            class_file = Path(CLASSES_FILE)
            if class_file.exists():
                with class_file.open("r", encoding="utf-8") as handle:
                    names = [line.strip() for line in handle if line.strip()]
                    logger.info(f"Loaded {len(names)} class names from {class_file}")
                    return {index: name for index, name in enumerate(names)}
        except Exception:
            logger.warning("Failed to load class names from CLASSES_FILE, using model names")

        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            return {int(k): v for k, v in self.model.names.items()}

        logger.warning("Unable to load class names; falling back to empty names")
        return {}

    def _parse_results(self, results: Any) -> List[Detection]:
        """
        Convert Ultralytics detection results into a list of Detection objects.
        """
        if not results or len(results) == 0:
            return []

        detections: List[Detection] = []
        result = results[0]

        if not hasattr(result, "boxes") or result.boxes is None:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
        confidences = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
        class_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []

        for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            class_name = self.class_names.get(class_id, str(class_id))
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )

        return detections

    def detect(self, image: Union[str, Path, Any]) -> List[Detection]:
        """
        Run YOLO detection on a single image.

        Args:
            image: Either a file path or a numpy image array.

        Returns:
            A list of Detection objects with coordinates in original image pixels.
        """
        try:
            raw_results = self.model.predict(
                source=image,
                imgsz=self.input_size,
                conf=0.0,
                iou=0.0,
                device=self.device,
                verbose=False,
            )
            raw_count = len(raw_results[0].boxes) if raw_results and raw_results[0].boxes is not None else 0

            filtered_results = self.model.predict(
                source=image,
                imgsz=self.input_size,
                conf=self.confidence,
                iou=0.0,
                device=self.device,
                verbose=False,
            )
            filtered_count = len(filtered_results[0].boxes) if filtered_results and filtered_results[0].boxes is not None else 0

            final_results = self.model.predict(
                source=image,
                imgsz=self.input_size,
                conf=self.confidence,
                iou=self.nms_threshold,
                device=self.device,
                verbose=False,
            )
            final_count = len(final_results[0].boxes) if final_results and final_results[0].boxes is not None else 0

            logger.info(
                "Detection summary: raw=%d, after_confidence=%d, after_nms=%d",
                raw_count,
                filtered_count,
                final_count,
            )

            detections = self._parse_results(final_results)
            return detections

        except Exception as error:
            logger.error("YOLO inference failed", exc_info=True)
            raise RuntimeError("YOLO detection failed. Check the input image and model configuration.") from error

    def detect_batch(self, images: List[Union[str, Path, Any]]) -> List[List[Detection]]:
        """
        Run YOLO detection on a batch of images.

        Args:
            images: List of file paths or numpy image arrays.

        Returns:
            A list of detection lists, one per image.
        """
        batch_detections: List[List[Detection]] = []
        for image in images:
            batch_detections.append(self.detect(image))
        return batch_detections

    def get_model_info(self) -> None:
        """
        Print model architecture summary, parameter count, and supported classes.
        """
        try:
            parameter_count = sum(p.numel() for p in self.model.model.parameters())
            layer_count = len(self.model.model.model) if hasattr(self.model.model, "model") else len(self.model.model)
            class_count = len(self.class_names)

            print("Model summary")
            print("--------------")
            print(f"Weights path: {self.model_path}")
            print(f"Input size: {self.input_size}")
            print(f"Device: {self.device}")
            print(f"Layer blocks: {layer_count}")
            print(f"Parameter count: {parameter_count:,}")
            print(f"Supported classes: {class_count}")
            print("Classes:")
            for class_id, class_name in list(self.class_names.items())[:20]:
                print(f"  {class_id}: {class_name}")
        except Exception:
            logger.exception("Failed to print model info")
            print("Unable to print detailed model info.")
