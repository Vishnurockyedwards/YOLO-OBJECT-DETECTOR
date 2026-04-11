"""
Post-processing utilities for YOLO detections.

This module includes manual implementations of IoU calculation, confidence
filtering, and Non-Maximum Suppression (NMS). It also provides a helper to
compare custom NMS against Ultralytics built-in NMS behavior.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .detector import Detection
except ImportError:
    from detector import Detection

logger = logging.getLogger(__name__)


class PostProcessor:
    """Manual post-processing for YOLO detections."""

    def compute_iou(self, box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            box_a: Bounding box A as (x1, y1, x2, y2).
            box_b: Bounding box B as (x1, y1, x2, y2).

        Returns:
            IoU value between 0.0 and 1.0.
        """
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
        area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
        union_area = area_a + area_b - intersection_area

        if union_area <= 0.0:
            return 0.0

        iou = intersection_area / union_area
        return float(iou)

    def apply_confidence_filter(self, detections: List[Detection], threshold: float) -> List[Detection]:
        """
        Remove detections below the confidence threshold.

        Args:
            detections: List of Detection objects.
            threshold: Minimum confidence to keep a detection.

        Returns:
            Filtered list of detections.
        """
        filtered = [det for det in detections if det.confidence >= threshold]
        removed = len(detections) - len(filtered)
        logger.info("Confidence filter removed %d detections below %.2f", removed, threshold)
        return filtered

    def apply_nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """
        Apply Non-Maximum Suppression (NMS) to remove redundant overlapping boxes.

        Args:
            detections: List of Detection objects.
            iou_threshold: IoU threshold above which a detection is suppressed.

        Returns:
            List of detections after NMS.
        """
        if not detections:
            return []

        # Sort by confidence descending so the strongest detections are kept first.
        sorted_detections = sorted(detections, key=lambda det: det.confidence, reverse=True)
        keep: List[Detection] = []

        while sorted_detections:
            current = sorted_detections.pop(0)
            keep.append(current)

            remaining: List[Detection] = []
            for candidate in sorted_detections:
                if candidate.class_id != current.class_id:
                    remaining.append(candidate)
                    continue

                iou = self.compute_iou(current.bbox, candidate.bbox)
                if iou <= iou_threshold:
                    remaining.append(candidate)
                else:
                    logger.debug(
                        "Suppressed box %s with IoU=%.3f against kept box %s",
                        candidate.bbox,
                        iou,
                        current.bbox,
                    )
            sorted_detections = remaining

        logger.info("NMS reduced detections from %d to %d", len(detections), len(keep))
        return keep

    def process(self, raw_detections: List[Detection], conf_threshold: float, nms_threshold: float) -> List[Detection]:
        """
        Run confidence filtering followed by NMS.

        Args:
            raw_detections: Raw detections before filtering.
            conf_threshold: Confidence threshold to apply.
            nms_threshold: IoU threshold for NMS.

        Returns:
            Cleaned detection list.
        """
        filtered = self.apply_confidence_filter(raw_detections, conf_threshold)
        cleaned = self.apply_nms(filtered, nms_threshold)
        return cleaned


def compare_with_builtin(image: str, detector: Any, conf_threshold: float = 0.5, nms_threshold: float = 0.45) -> Dict[str, int]:
    """
    Compare custom NMS with Ultralytics built-in NMS.

    This helper runs raw detections through the custom post-processor and compares
    the counts against Ultralytics' built-in NMS output.

    Args:
        image: Path or array for the input image.
        detector: An instance of YOLODetector.
        conf_threshold: Confidence threshold used for both methods.
        nms_threshold: IoU threshold used for both methods.

    Returns:
        A dictionary with counts for raw, custom, and built-in detections.
    """
    processor = PostProcessor()

    try:
        raw_results = detector.model.predict(
            source=image,
            imgsz=detector.input_size,
            conf=0.0,
            iou=0.0,
            device=detector.device,
            verbose=False,
        )
        raw_detections = []
        if raw_results and raw_results[0].boxes is not None:
            for bbox, conf, cls in zip(
                raw_results[0].boxes.xyxy.cpu().numpy(),
                raw_results[0].boxes.conf.cpu().numpy(),
                raw_results[0].boxes.cls.cpu().numpy().astype(int),
            ):
                raw_detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=detector.class_names.get(int(cls), str(int(cls))),
                    )
                )

        custom_detections = processor.process(raw_detections, conf_threshold, nms_threshold)

        builtin_results = detector.model.predict(
            source=image,
            imgsz=detector.input_size,
            conf=conf_threshold,
            iou=nms_threshold,
            device=detector.device,
            verbose=False,
        )
        builtin_count = len(builtin_results[0].boxes) if builtin_results and builtin_results[0].boxes is not None else 0

        comparison = {
            "raw_count": len(raw_detections),
            "custom_count": len(custom_detections),
            "builtin_count": builtin_count,
        }

        print("Comparison summary")
        print("------------------")
        print(f"Raw detections: {comparison['raw_count']}")
        print(f"Custom NMS detections: {comparison['custom_count']}")
        print(f"Built-in NMS detections: {comparison['builtin_count']}")

        if comparison["custom_count"] != comparison["builtin_count"]:
            print("Note: custom NMS result differs from built-in NMS. Check threshold and class grouping.")
        else:
            print("Custom NMS matches built-in NMS count.")

        return comparison
    except Exception as error:
        logger.exception("Failed to compare custom NMS with built-in NMS")
        raise RuntimeError("Unable to compare custom and built-in NMS.") from error
