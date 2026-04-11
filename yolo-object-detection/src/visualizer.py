"""
Visualization utilities for YOLO detections.

This module draws bounding boxes, labels, overlays, and optional grid
information on images returned by the detector.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .detector import Detection
except ImportError:
    from detector import Detection

logger = logging.getLogger(__name__)


class DetectionVisualizer:
    """Draw detection annotations and overlays on images."""

    def __init__(self, class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> None:
        """
        Initialize the visualizer.

        Args:
            class_colors: Optional mapping from class_id to BGR color tuple.
                If not provided, a default palette is used.
        """
        default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        self.class_colors = class_colors or {}
        self.default_colors = default_colors
        logger.info("DetectionVisualizer initialized with %d predefined colors", len(self.default_colors))

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return a BGR color for the given class id."""
        if class_id in self.class_colors:
            return self.class_colors[class_id]

        return self.default_colors[class_id % len(self.default_colors)]

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels for detections on an image.

        Args:
            image: BGR image array.
            detections: List of Detection objects.

        Returns:
            Annotated image.
        """
        annotated = image.copy()
        if detections is None:
            return annotated

        for detection in detections:
            color = self._get_color(detection.class_id)
            x1, y1, x2, y2 = map(int, detection.bbox)
            label = f"{detection.class_name}: {int(detection.confidence * 100)}%"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

            # Measure text size and draw label background
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_width, label_height = text_size
            label_top = max(y1 - label_height - baseline, 0)
            label_bottom = y1
            label_left = x1
            label_right = x1 + label_width + 8

            cv2.rectangle(annotated, (label_left, label_top), (label_right, label_bottom), color, thickness=cv2.FILLED)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - baseline if y1 - baseline > 0 else y1 + label_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            logger.debug("Drew detection %s at %s", label, detection.bbox)

        return annotated

    def draw_info_overlay(self, image: np.ndarray, fps: float, num_detections: int) -> np.ndarray:
        """
        Draw a semi-transparent info bar showing FPS and detection count.

        Args:
            image: BGR image array.
            fps: Current frames per second.
            num_detections: Number of detections in the frame.

        Returns:
            Image with overlay.
        """
        overlay = image.copy()
        height, width = image.shape[:2]
        bar_height = 30

        cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 0, 0), thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

        text = f"FPS: {fps:.1f} | Detections: {num_detections}"
        cv2.putText(
            image,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return image

    def draw_detection_grid(self, image: np.ndarray, grid_size: int = 32) -> np.ndarray:
        """
        Optionally draw a grid overlay on the image.

        This helps visualize the grid cells that YOLO implicitly uses when
        predicting bounding boxes.

        Args:
            image: BGR image array.
            grid_size: Size of each grid cell in pixels.

        Returns:
            Image with grid overlay.
        """
        annotated = image.copy()
        height, width = annotated.shape[:2]

        for x in range(0, width, grid_size):
            cv2.line(annotated, (x, 0), (x, height), (255, 255, 255), 1, lineType=cv2.LINE_AA)
        for y in range(0, height, grid_size):
            cv2.line(annotated, (0, y), (width, y), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        logger.info("Drew detection grid with cell size %d", grid_size)
        return annotated

    def save_result(self, image: np.ndarray, output_path: str) -> None:
        """
        Save the annotated image to disk.

        Args:
            image: BGR image array.
            output_path: Destination file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(path), image):
            logger.error("Failed to save image to %s", path)
            raise IOError(f"Could not save image to {path}")

        logger.info("Saved annotated image to %s", path)

    def show_result(self, image: np.ndarray, window_name: str = "Detection") -> None:
        """
        Display the image in an OpenCV window and wait for a key press.

        Args:
            image: BGR image array.
            window_name: Window title.
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
