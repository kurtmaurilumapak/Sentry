"""YOLO model loading and detection logic."""

import os
from typing import List, Optional
from PIL import Image

from ultralytics import YOLO

from config import MODEL_PATHS
from schemas import Detection


class YOLODetector:
    """Handles YOLO model loading and inference."""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
    
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
        self.model = YOLO(self.model_path)
        
        print("=" * 50)
        print("MODEL LOADED SUCCESSFULLY")
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
    
    def detect(self, image: Image.Image, confidence_threshold: float = 0.25) -> List[Detection]:
        """Run detection on an image and return list of Detection objects."""
        if self.model is None:
            if not self.load():
                return []
        
        # Run inference
        results = self.model(image, verbose=False, conf=confidence_threshold)
        
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

