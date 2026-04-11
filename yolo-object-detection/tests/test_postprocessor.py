import unittest

from src.detector import Detection
from src.postprocessor import PostProcessor


class TestPostProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = PostProcessor()

    def test_compute_iou_overlapping(self) -> None:
        box_a = (0.0, 0.0, 10.0, 10.0)
        box_b = (5.0, 5.0, 15.0, 15.0)
        iou = self.processor.compute_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 0.14285714285714285, places=6)

    def test_compute_iou_identical(self) -> None:
        box = (10.0, 10.0, 20.0, 20.0)
        iou = self.processor.compute_iou(box, box)
        self.assertEqual(iou, 1.0)

    def test_compute_iou_non_overlapping(self) -> None:
        box_a = (0.0, 0.0, 10.0, 10.0)
        box_b = (20.0, 20.0, 30.0, 30.0)
        iou = self.processor.compute_iou(box_a, box_b)
        self.assertEqual(iou, 0.0)

    def test_apply_confidence_filter(self) -> None:
        detections = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.4, class_id=0, class_name="person"),
            Detection(bbox=(10, 10, 20, 20), confidence=0.7, class_id=1, class_name="bicycle"),
        ]
        filtered = self.processor.apply_confidence_filter(detections, threshold=0.5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].confidence, 0.7)

    def test_apply_nms_removes_duplicates(self) -> None:
        detections = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=(1, 1, 11, 11), confidence=0.8, class_id=0, class_name="person"),
            Detection(bbox=(50, 50, 60, 60), confidence=0.85, class_id=0, class_name="person"),
            Detection(bbox=(0, 0, 10, 10), confidence=0.7, class_id=1, class_name="car"),
        ]
        kept = self.processor.apply_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(kept), 3)
        kept_classes = [det.class_id for det in kept]
        self.assertIn(0, kept_classes)
        self.assertIn(1, kept_classes)

    def test_process_combines_filter_and_nms(self) -> None:
        detections = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.4, class_id=0, class_name="person"),
            Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=(1, 1, 11, 11), confidence=0.8, class_id=0, class_name="person"),
        ]
        processed = self.processor.process(detections, conf_threshold=0.5, nms_threshold=0.5)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
