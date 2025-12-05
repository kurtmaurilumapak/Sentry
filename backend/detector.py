"""YOLO model loading and detection logic."""

import os
from typing import List, Optional
from PIL import Image
import torch

from ultralytics import YOLO

from config import MODEL_PATHS
from schemas import Detection


def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


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
    
    # Fallback to CPU
    print("✓ Using CPU")
    return 'cpu', 'CPU', False


class YOLODetector:
    """Handles YOLO model loading and inference."""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
        self.device, self.device_name, self.has_gpu = detect_device()
    
    def find_model(self) -> Optional[str]:
        """Find the model file from possible locations."""
        for path in MODEL_PATHS:
            if os.path.exists(path):
                return os.path.abspath(path)
        return None
    
    def load(self) -> bool:
        """Load the YOLO model. Returns True if successful."""
        if self.model is not None:
            return True
        
        self.model_path = self.find_model()
        
        if self.model_path is None:
            print("ERROR: Could not find model file. Tried:")
            for path in MODEL_PATHS:
                print(f"  - {os.path.abspath(path)}")
            return False
        
        print(f"Loading model from: {self.model_path}")
        print(f"Using device: {self.device} ({self.device_name})")
        self.model = YOLO(self.model_path)
        # Ensure model is on the correct device
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
            self.model.model.to(self.device)
            # Verify model is on GPU if CUDA is available
            if self.has_gpu and torch.cuda.is_available():
                next_param = next(self.model.model.parameters(), None)
                if next_param is not None:
                    device_location = next_param.device
                    print(f"Model device verified: {device_location}")
        
        print("=" * 50)
        print("MODEL LOADED SUCCESSFULLY")
        print(f"Device: {self.device_name}")
        print("Available classes:")
        for idx, name in self.model.names.items():
            print(f"  {idx}: {name}")
        print("=" * 50)
        
        return True
    
    def get_classes(self) -> dict:
        """Get the model's class names."""
        if self.model is None:
            self.load()
        return self.model.names if self.model else {}
    
    def cleanup(self):
        """Release model and clear GPU memory."""
        print("[*] Cleaning up YOLODetector...")
        self.model = None
        self.model_path = None
        
        if self.has_gpu:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[OK] YOLODetector GPU memory released")
            except Exception as e:
                print(f"[WARNING] YOLODetector cleanup error: {e}")
    
    def detect(self, image: Image.Image, confidence_threshold: float = 0.25) -> List[Detection]:
        """Run detection on an image and return list of Detection objects."""
        if self.model is None:
            if not self.load():
                return []
        
        # Convert device string to format YOLO expects (0 for cuda:0, 'cpu' for cpu)
        yolo_device = 0 if self.has_gpu and 'cuda' in self.device else self.device
        
        # Run inference with device specified (with OOM fallback)
        try:
            results = self.model(image, verbose=False, conf=confidence_threshold, device=yolo_device)
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM during detection - clearing cache and retrying...")
            clear_gpu_memory()
            # Try again with cleared memory
            try:
                results = self.model(image, verbose=False, conf=confidence_threshold, device=yolo_device)
            except torch.cuda.OutOfMemoryError:
                # Fall back to CPU
                print("Still OOM - falling back to CPU")
                self.has_gpu = False
                self.device = 'cpu'
                self.device_name = 'CPU (OOM fallback)'
                results = self.model(image, verbose=False, conf=confidence_threshold, device='cpu')
        
        detections: List[Detection] = []
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                cls_name = r.names.get(cls_idx, "unknown")
                
                print(f"  Detection {i+1}: class={cls_name} (idx={cls_idx}), conf={conf:.2f}")
                
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        label=cls_name,
                        confidence=conf,
                    )
                )
        
        print(f"Total detections: {len(detections)}")
        return detections


# Global detector instance
detector = YOLODetector()
