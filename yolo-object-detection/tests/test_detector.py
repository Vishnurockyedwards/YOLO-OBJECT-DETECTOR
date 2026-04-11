import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.detector import YOLODetector, Detection


class TestYOLODetector(unittest.TestCase):
    @patch("src.detector.YOLO")
    @patch("src.detector.CLASSES_FILE", Path("does_not_exist.names"))
    def test_default_init_loads_model_and_class_names(self, mock_yolo):
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "car"}
        mock_yolo.return_value = mock_model

        detector = YOLODetector(model_path="yolov8n.pt", confidence=0.25, nms_threshold=0.5)

        self.assertEqual(detector.confidence, 0.25)
        self.assertEqual(detector.nms_threshold, 0.5)
        self.assertEqual(detector.class_names[0], "person")
        self.assertEqual(detector.class_names[1], "car")
        mock_yolo.assert_called_once_with(str(Path("yolov8n.pt")))

    @patch("src.detector.YOLO")
    @patch("src.detector.CLASSES_FILE", Path("does_not_exist.names"))
    def test_parse_results_returns_detection_objects(self, mock_yolo):
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 10, 10]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "person"}
        mock_yolo.return_value = mock_model

        detector = YOLODetector(model_path="yolov8n.pt")
        detections = detector.detect("dummy.jpg")

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], Detection)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(detections[0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
