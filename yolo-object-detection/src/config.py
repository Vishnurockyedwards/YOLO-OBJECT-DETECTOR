"""
Configuration file for YOLO Object Detection project.

This module contains all configurable parameters used throughout the project.
"""

from pathlib import Path

# Model configuration
MODEL_PATH = Path("models/yolov8n.pt")  # Path to YOLOv8 nano model weights
CLASSES_FILE = Path("models/coco.names")  # Path to COCO class names file

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score to keep a detection
NMS_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression

# Model input configuration
INPUT_SIZE = 640  # Model input dimensions (square, pixels)

# Visualization colors (BGR format for OpenCV)
# Mapping class IDs to distinct colors for drawing bounding boxes
COLORS = {
    0: (255, 0, 0),    # person - red
    1: (0, 255, 0),    # bicycle - green
    2: (0, 0, 255),    # car - blue
    3: (255, 255, 0),  # motorcycle - cyan
    4: (255, 0, 255),  # airplane - magenta
    5: (0, 255, 255),  # bus - yellow
    6: (128, 0, 0),    # train - dark red
    7: (0, 128, 0),    # truck - dark green
    8: (0, 0, 128),    # boat - dark blue
    9: (128, 128, 0),  # traffic light - olive
    # Add more colors as needed for other classes
}

# Default colors for classes not in the dictionary above
DEFAULT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (64, 64, 64)
]