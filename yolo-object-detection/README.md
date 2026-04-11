# YOLO Object Detection Project

This project implements a real-time object detection system using YOLO (You Only Look Once) with Ultralytics YOLOv8. It can detect and classify multiple objects in images and live webcam video, drawing labeled bounding boxes with confidence scores.

## Features

- Real-time object detection using YOLOv8
- Support for images, video files, and webcam streams
- Configurable confidence and NMS thresholds
- Visualization with bounding boxes and labels
- Educational implementation with detailed comments

## Tech Stack

- Python 3.10+
- OpenCV for image/video processing
- Ultralytics YOLOv8
- NumPy for array operations

## Setup Instructions

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. The YOLOv8 model weights will be automatically downloaded on first use

## Usage Examples

### Detect objects in an image
```bash
python src/detect_image.py --image input/sample.jpg --output output/result.jpg
```

### Real-time webcam detection
```bash
python src/detect_webcam.py
```

### Detect objects in a video file
```bash
python src/detect_video.py --video input/sample.mp4 --output output/result.mp4
```

## Project Structure

```
yolo-object-detection/
├── README.md
├── requirements.txt
├── models/                    # downloaded weights go here
├── input/                     # sample test images
├── output/                    # saved detection results
├── src/
│   ├── __init__.py
│   ├── config.py              # all configurable parameters
│   ├── detector.py            # core YOLODetector class
│   ├── preprocessor.py        # image preprocessing utilities
│   ├── postprocessor.py       # NMS, filtering, IoU logic
│   ├── visualizer.py          # drawing boxes, labels, overlays
│   ├── detect_image.py        # script: detect on a single image
│   ├── detect_video.py        # script: detect on video file
│   ├── detect_webcam.py       # script: real-time webcam detection
│   └── detect_opencv_dnn.py   # bonus: raw OpenCV DNN approach
├── notebooks/
│   └── exploration.ipynb      # Jupyter notebook for experimentation
└── tests/
    ├── test_detector.py
    ├── test_preprocessor.py
    └── test_postprocessor.py
```