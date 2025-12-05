"""
Video processing with YOLO tracking - YOLOSpotlight style.
Auto-detects NVIDIA CUDA or falls back to AMD/CPU optimized mode.
Supports multiple model selection.
"""

import cv2
import os
import base64
import numpy as np
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
        # Tracking-only mode
        self.use_tracking = True
        # Allow tuning image size for low-VRAM GPUs/CPU via env
        self.gpu_imgsz = int(os.getenv("SENTRY_GPU_IMGSZ", "640"))
        self.cpu_imgsz = int(os.getenv("SENTRY_CPU_IMGSZ", "480"))
        
        # Load default model
        self.load_model(DEFAULT_MODEL)
        
        # Warm up GPU with a dummy inference if GPU is available
        if self.has_gpu and self.model is not None:
            try:
                import torch
                import numpy as np
                if torch.cuda.is_available():
                    print("Warming up GPU with dummy inference...")
                    # Use smaller warmup to avoid OOM on laptop GPUs
                    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                    yolo_device = 0 if self.has_gpu and 'cuda' in self.device else self.device
                    # Run warmup using the same method as actual processing
                    for _ in range(1):
                        try:
                            if self.use_tracking:
                                _ = self.model.track(dummy_img, verbose=False, device=yolo_device, imgsz=320, half=self.has_gpu)
                            else:
                                _ = self.model.predict(dummy_img, verbose=False, device=yolo_device, imgsz=320, half=self.has_gpu)
                            torch.cuda.synchronize()
                        except torch.cuda.OutOfMemoryError:
                            print("GPU warmup skipped (OOM); continuing without warmup")
                            torch.cuda.empty_cache()
                            break
                    print("GPU warmup complete")
            except Exception as e:
                print(f"GPU warmup warning: {e}")
        
        mode_str = "tracking" if self.use_tracking else "prediction (faster)"
        print(f"Ready on: {self.device_name} (mode: {mode_str})")
    
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
        print(f"Loading on device: {self.device} ({self.device_name})")
        
        # Determine device for YOLO (0 for CUDA, 'cpu' for CPU)
        yolo_device = 0 if self.has_gpu and 'cuda' in self.device else self.device
        
        # Load model with explicit device if possible
        # Some YOLO versions support device in constructor
        try:
            # Try loading with device parameter (if supported)
            self.model = YOLO(model_path)
        except Exception:
            self.model = YOLO(model_path)
        
        # Move the underlying PyTorch model to GPU immediately
        if self.has_gpu and torch.cuda.is_available():
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                    self.model.model.to(torch.device(self.device))
                    torch.cuda.synchronize()
                    
                    # CRITICAL: Also try to set device on YOLO's predictor if it exists
                    # This might be the key to forcing GPU usage
                    if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                        try:
                            # Try to set device on predictor's model
                            if hasattr(self.model.predictor, 'model'):
                                pred_model = self.model.predictor.model
                                if hasattr(pred_model, 'to'):
                                    pred_model.to(torch.device(self.device))
                        except Exception as e:
                            print(f"Warning: Could not set predictor device: {e}")
            except Exception as e:
                print(f"Warning: Could not move model to GPU: {e}")
        
        # Verify model is on GPU
        if self.has_gpu and torch.cuda.is_available():
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                    next_param = next(self.model.model.parameters(), None)
                    if next_param is not None:
                        actual_device = next_param.device
                        if actual_device.type == 'cuda':
                            print(f"Model verified on GPU: {actual_device}")
                        else:
                            print(f"WARNING: Model is on {actual_device}, expected GPU!")
            except Exception as e:
                print(f"Warning during device verification: {e}")
        
        self.names = self.model.names
        self.current_model_id = model_id
        
        # Verify device if GPU
        if self.has_gpu and torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"GPU memory allocated: {memory_allocated:.1f} MB")
            print(f"GPU memory reserved: {memory_reserved:.1f} MB")
        
        print(f"Model loaded. Classes: {self.names}")
        return True
    
    def reset_tracker(self):
        """Reset the tracker state for a new video."""
        if self.current_model_id:
            self.load_model(self.current_model_id)
            print(f"Tracker reset (model: {self.current_model_id})")
    
    def cleanup(self):
        """Release model and clear GPU memory."""
        print("[*] Cleaning up VideoProcessor...")
        self.model = None
        self.names = {}
        self.current_model_id = None
        
        if self.has_gpu:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[OK] VideoProcessor GPU memory released")
            except Exception as e:
                print(f"[WARNING] VideoProcessor cleanup error: {e}")
    
    def get_device_info(self):
        """Get current device information."""
        return {
            'device': self.device,
            'name': self.device_name,
            'has_gpu': self.has_gpu
        }
    
    def process_frame(self, im0: np.ndarray, conf: float = 0.25, imgsz_override: int = None) -> tuple:
        """
        Process a single frame with YOLO tracking.
        Auto-optimized for GPU or CPU.
        """
        if self.model is None:
            return im0, []
        
        # Use larger size for GPU, smaller for CPU
        imgsz = imgsz_override if imgsz_override else (self.gpu_imgsz if self.has_gpu else self.cpu_imgsz)

        # YOLO device format: 0 for CUDA GPU, 'cpu' for CPU/MPS
        yolo_device = 0 if (self.has_gpu and 'cuda' in str(self.device)) else self.device
        
        try:
            # Object tracking with device-optimized settings
            results = self.model.track(
                im0, 
                persist=True, 
                conf=conf, 
                verbose=False,
                imgsz=imgsz,
                device=yolo_device,
                half=self.has_gpu  # FP16 for GPU acceleration
            )
        except torch.cuda.OutOfMemoryError:
            # Graceful fallback for low-VRAM GPUs
            print("CUDA OOM during track(); falling back to CPU with smaller image size")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.has_gpu = False
            self.device = 'cpu'
            self.device_name = 'CPU (OOM fallback)'
            # Reload model on CPU to avoid device mismatch
            if self.current_model_id:
                self.load_model(self.current_model_id)
            # Retry on CPU
            return self.process_frame(im0, conf=conf, imgsz_override=self.cpu_imgsz)
        
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
