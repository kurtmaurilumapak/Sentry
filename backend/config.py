"""Configuration settings for the backend."""

import os

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Get the directory where the backend is located
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "..", "src", "assets", "model")

# Available models with display names
AVAILABLE_MODELS = {
    "yolo11s": {
        "name": "YOLO11s",
        "path": os.path.join(MODEL_DIR, "yolo11s.pt"),
    },
    "yolov8s": {
        "name": "YOLOv8s",
        "path": os.path.join(MODEL_DIR, "yolov8s.pt"),
    },
}

# Default model
DEFAULT_MODEL = "yolo11s"

# Legacy MODEL_PATHS for backward compatibility
MODEL_PATHS = [
    os.path.join(MODEL_DIR, "yolo11s.pt"),
    os.path.join(MODEL_DIR, "yolov8s.pt"),
]

# CORS settings
CORS_ORIGINS = ["*"]

