"""
Video processing with YOLO tracking - YOLOSpotlight style.
Auto-detects NVIDIA CUDA or falls back to AMD/CPU optimized mode.
Supports multiple model selection.
"""

import cv2
import os
import base64
import numpy as np
from pathlib import Path
import torch

from ultralytics import YOLO
from config import AVAILABLE_MODELS, DEFAULT_MODEL


def detect_device():
    """
    Auto-detect the best available device.
    Returns: (device_string, device_name, is_gpu)
    """
    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        print(f"  CUDA version: {torch.version.cuda}")
        return 'cuda:0', gpu_name, True
    
    # Check for Apple MPS (Metal Performance Shaders)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ Apple Silicon GPU detected (MPS)")
        return 'mps', 'Apple Silicon', True
    
    # Fallback to CPU (optimized for AMD)
    print("✓ Using CPU (optimized for AMD/Intel)")
    print("  Tip: For NVIDIA GPU acceleration, install CUDA toolkit")
    
    # Optimize CPU threading for AMD/Intel systems
    cpu_count = os.cpu_count() or 4
    threads = min(cpu_count, 8)
    torch.set_num_threads(threads)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(max(2, threads // 2))
    
    return 'cpu', f'CPU ({threads} threads)', False


# Detect device at startup
DEVICE, DEVICE_NAME, HAS_GPU = detect_device()
print(f"  Device: {DEVICE} ({DEVICE_NAME})")


class VideoProcessor:
    """
    YOLO Interactive Object Tracker - Web version.
    Auto-detects GPU (NVIDIA/Apple) or uses optimized CPU.
    """
    
    def __init__(self):
        self.current_model_id = None
        self.model = None
        self.names = {}
        self.device = DEVICE
        self.device_name = DEVICE_NAME
        self.has_gpu = HAS_GPU
        
        # Load default model
        self.load_model(DEFAULT_MODEL)
        print(f"Ready on: {self.device_name}")
    
    def get_available_models(self):
        """Get list of available models."""
        models = []
        for model_id, info in AVAILABLE_MODELS.items():
            exists = os.path.exists(info['path'])
            models.append({
                'id': model_id,
                'name': info['name'],
                'available': exists,
                'active': model_id == self.current_model_id
            })
        return models
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model by ID."""
        if model_id not in AVAILABLE_MODELS:
            print(f"ERROR: Unknown model: {model_id}")
            return False
        
        model_info = AVAILABLE_MODELS[model_id]
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return False
        
        print(f"Loading model: {model_info['name']} from {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.current_model_id = model_id
        print(f"Model loaded. Classes: {self.names}")
        return True
        
        # Annotation settings (like YOLOSpotlight)
        self.ann = None
        self.current_data = None
    
    def reset_tracker(self):
        """Reset the tracker state for a new video."""
        if self.current_model_id:
            self.load_model(self.current_model_id)
            print(f"Tracker reset (model: {self.current_model_id})")
    
    def get_device_info(self):
        """Get current device information."""
        return {
            'device': self.device,
            'name': self.device_name,
            'has_gpu': self.has_gpu
        }
    
    def process_frame(self, im0: np.ndarray, conf: float = 0.25) -> tuple:
        """
        Process a single frame with YOLO tracking.
        Auto-optimized for GPU or CPU.
        """
        if self.model is None:
            return im0, []
        
        # Use larger size for GPU, smaller for CPU
        imgsz = 640 if self.has_gpu else 480
        
        # Object tracking with device-optimized settings
        results = self.model.track(
            im0, 
            persist=True, 
            conf=conf, 
            verbose=False,
            imgsz=imgsz,
            device=self.device,
            half=self.has_gpu  # FP16 for GPU acceleration
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                ids = result.boxes.id
                if ids is not None:
                    ids = ids.cpu().tolist()
                else:
                    ids = list(range(len(boxes)))
                
                # Draw boxes directly with OpenCV (optimized)
                for i, (box, cls, conf_val) in enumerate(zip(boxes, clss, confs)):
                    obj_id = ids[i] if i < len(ids) else i
                    class_name = self.names[int(cls)]
                    label = f"{class_name} {conf_val:.2f}"
                    
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Colors in BGR for OpenCV
                    if class_name.lower() == 'soldier':
                        color = (0, 0, 255)  # RED
                    else:
                        color = (255, 0, 0)  # BLUE for civilian/other
                    
                    # Draw rectangle
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(im0, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(im0, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    detections.append({
                        'id': int(obj_id),
                        'class': class_name,
                        'confidence': float(conf_val),
                        'box': box.tolist()
                    })
        
        return im0, detections
    
    def process_frame_base64(self, frame_base64: str, conf: float = 0.25) -> tuple:
        """
        Process a base64 encoded frame.
        
        Args:
            frame_base64: Base64 encoded image
            conf: Confidence threshold
            
        Returns:
            Tuple of (annotated_frame_base64, detections_list)
        """
        # Decode base64 to image
        img_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if im0 is None:
            return None, []
        
        # Process frame
        annotated_frame, detections = self.process_frame(im0, conf)
        
        # Encode back to base64 (high quality for full resolution)
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return annotated_base64, detections


# Global instance
video_processor = VideoProcessor()
