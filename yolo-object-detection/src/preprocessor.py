"""
Image preprocessing utilities for YOLO object detection.

This module handles all image transformations needed to prepare raw images
for YOLO model inference, including loading, resizing with letterboxing,
normalization, and blob creation for OpenCV DNN.
"""

import logging
from pathlib import Path
from typing import Tuple, Union, Dict, Any
import cv2
import numpy as np

from .config import INPUT_SIZE

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing for YOLO object detection.

    This class provides methods to load images, resize them while maintaining
    aspect ratio (letterboxing), normalize pixel values, and create blobs for
    neural network inference. YOLO models expect square input images, but
    real-world images have varying aspect ratios. Letterboxing preserves
    object proportions by padding with gray instead of stretching.
    """

    def __init__(self, target_size: int = INPUT_SIZE):
        """
        Initialize the image preprocessor.

        Args:
            target_size: The target square size for model input (e.g., 640 for YOLOv8).
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size {target_size}x{target_size}")

    def load_image(self, path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Load an image from disk and validate it.

        Args:
            path: Path to the image file.

        Returns:
            A tuple containing:
            - The loaded image as a numpy array (BGR format, uint8)
            - Original dimensions as (height, width)

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image couldn't be loaded or is empty.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image from {path}. Check file format and integrity.")

        if image.size == 0:
            raise ValueError(f"Loaded image is empty: {path}")

        original_height, original_width = image.shape[:2]
        logger.info(f"Loaded image: {path} -> shape {image.shape}")

        return image, (original_height, original_width)

    def resize_image(self, image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize image using letterboxing to maintain aspect ratio.

        YOLO models expect square inputs, but real images vary in aspect ratio.
        Instead of stretching (which distorts objects), we resize maintaining
        aspect ratio and pad with gray to fill the square. This preserves
        object proportions and spatial relationships.

        Args:
            image: Input image as numpy array.
            target_size: Target square size.

        Returns:
            A tuple containing:
            - Resized and padded image
            - Metadata dict with scale factors and padding info for coordinate mapping
        """
        original_height, original_width = image.shape[:2]

        # Calculate scale factor to fit image within target_size while maintaining aspect ratio
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        logger.info(f"Resized image: {original_height}x{original_width} -> {new_height}x{new_width}")

        # Create square canvas filled with gray (128, 128, 128)
        square_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)

        # Calculate padding offsets to center the resized image
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2

        # Place resized image on the square canvas
        square_image[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized

        # Metadata for mapping detections back to original coordinates
        metadata = {
            'scale_factor': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'resized_width': new_width,
            'resized_height': new_height,
            'original_width': original_width,
            'original_height': original_height
        }

        logger.info(f"Letterboxed to {target_size}x{target_size}, scale={scale:.3f}, padding=({pad_left}, {pad_top})")
        return square_image, metadata

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values from 0-255 uint8 to 0.0-1.0 float32.

        Neural networks typically work with normalized inputs for better
        numerical stability and training convergence.

        Args:
            image: Input image as uint8 numpy array.

        Returns:
            Normalized image as float32 numpy array.
        """
        # Convert to float32 and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        logger.info(f"Normalized image: dtype {image.dtype} -> {normalized.dtype}, range [0, 255] -> [0.0, 1.0]")
        return normalized

    def to_blob(self, image: np.ndarray) -> np.ndarray:
        """
        Create a 4D blob for OpenCV DNN inference.

        This method wraps cv2.dnn.blobFromImage to create the proper input
        format for neural networks. It handles BGR-to-RGB conversion and
        mean subtraction as expected by many pre-trained models.

        Args:
            image: Normalized image as float32 numpy array.

        Returns:
            4D blob tensor with shape (1, 3, height, width) for batch input.
        """
        # cv2.dnn.blobFromImage expects uint8 input, so convert back if needed
        if image.dtype != np.uint8:
            # Scale back to 0-255 for blobFromImage
            blob_input = (image * 255).astype(np.uint8)
        else:
            blob_input = image

        # Create blob: swapRB=True converts BGR to RGB, includes mean subtraction
        blob = cv2.dnn.blobFromImage(
            blob_input,
            scalefactor=1/255.0,  # Normalize to [0, 1]
            size=(self.target_size, self.target_size),
            mean=(0, 0, 0),  # No mean subtraction for YOLO
            swapRB=True,  # Convert BGR to RGB
            crop=False
        )

        logger.info(f"Created blob: shape {blob.shape}, dtype {blob.dtype}")
        return blob

    def preprocess(self, path_or_image: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete preprocessing pipeline for YOLO inference.

        This method runs the full preprocessing chain: load (if needed) ->
        resize with letterboxing -> normalize -> return processed image and metadata.

        Args:
            path_or_image: Either a file path (str/Path) or numpy array image.

        Returns:
            A tuple containing:
            - Processed image ready for model inference (normalized float32)
            - Metadata dict with original dimensions, scale factors, and padding offsets
              needed to map detection coordinates back to original image space.
        """
        # Handle input: load if path, use directly if array
        if isinstance(path_or_image, (str, Path)):
            image, original_size = self.load_image(path_or_image)
            metadata = {'original_height': original_size[0], 'original_width': original_size[1]}
        else:
            image = path_or_image
            metadata = {'original_height': image.shape[0], 'original_width': image.shape[1]}
            logger.info(f"Using provided image array: shape {image.shape}")

        # Resize with letterboxing
        resized_image, resize_metadata = self.resize_image(image, self.target_size)
        metadata.update(resize_metadata)

        # Normalize
        processed_image = self.normalize(resized_image)

        logger.info(f"Preprocessing complete: final shape {processed_image.shape}")
        return processed_image, metadata