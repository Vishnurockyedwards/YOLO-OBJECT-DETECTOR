import unittest
import numpy as np

from src.preprocessor import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = ImagePreprocessor(target_size=64)

    def test_preprocess_returns_expected_shape_and_metadata(self) -> None:
        image = np.full((32, 16, 3), 255, dtype=np.uint8)
        processed, metadata = self.preprocessor.preprocess(image)

        self.assertEqual(processed.shape, (64, 64, 3))
        self.assertAlmostEqual(metadata["scale_factor"], 2.0)
        self.assertEqual(metadata["resized_width"], 32)
        self.assertEqual(metadata["resized_height"], 64)
        self.assertEqual(metadata["original_width"], 16)
        self.assertEqual(metadata["original_height"], 32)
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)

    def test_load_image_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_image("nonexistent_file.jpg")


if __name__ == "__main__":
    unittest.main()
