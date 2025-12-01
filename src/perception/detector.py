"""
Object Detection Module

Provides object detection capabilities for autonomous driving scenarios.
Supports both simulated detections and real YOLO-based detection.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2


@dataclass
class Detection:
    """Represents a single detected object."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[float, float] = None
    
    def __post_init__(self):
        if self.center is None:
            x1, y1, x2, y2 = self.bbox
            self.center = ((x1 + x2) / 2, (y1 + y2) / 2)


class ObjectDetector:
    """
    Object detector for autonomous driving scenarios.
    
    Supports:
    - Simulated detections for demo purposes
    - Integration with YOLO models (when available)
    """
    
    # Class definitions for autonomous driving
    CLASSES = {
        0: "car",
        1: "truck", 
        2: "pedestrian",
        3: "cyclist",
        4: "motorcycle",
        5: "bus",
        6: "traffic_light",
        7: "stop_sign"
    }
    
    # Colors for each class (BGR format)
    CLASS_COLORS = {
        0: (0, 255, 0),      # car - green
        1: (0, 165, 255),    # truck - orange
        2: (0, 0, 255),      # pedestrian - red
        3: (255, 255, 0),    # cyclist - cyan
        4: (255, 0, 255),    # motorcycle - magenta
        5: (0, 255, 255),    # bus - yellow
        6: (128, 0, 128),    # traffic_light - purple
        7: (0, 128, 255),    # stop_sign - orange-red
    }
    
    def __init__(self, mode: str = "simulated", model_path: Optional[str] = None):
        """
        Initialize the object detector.
        
        Args:
            mode: "simulated" for synthetic detections, "yolo" for real detection
            model_path: Path to YOLO model weights (only for yolo mode)
        """
        self.mode = mode
        self.model = None
        self.frame_count = 0
        
        if mode == "yolo" and model_path:
            self._load_yolo_model(model_path)
    
    def _load_yolo_model(self, model_path: str):
        """Load YOLO model for real detection."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            print("Ultralytics not installed. Falling back to simulated mode.")
            self.mode = "simulated"
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        self.frame_count += 1
        
        if self.mode == "yolo" and self.model is not None:
            return self._detect_yolo(frame)
        else:
            return self._detect_simulated(frame)
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection on frame."""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, "unknown")
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf
                ))
        
        return detections
    
    def _detect_simulated(self, frame: np.ndarray) -> List[Detection]:
        """
        Generate simulated detections for demonstration.
        Creates realistic-looking detections that move over time.
        """
        h, w = frame.shape[:2]
        detections = []
        
        # Seed based on frame for reproducibility
        np.random.seed(self.frame_count % 1000)
        
        # Generate 3-7 simulated vehicles
        num_vehicles = np.random.randint(3, 8)
        
        for i in range(num_vehicles):
            # Simulate vehicles at different distances (sizes)
            distance_factor = np.random.uniform(0.3, 1.0)
            
            # Base size varies with distance
            base_w = int(80 * distance_factor + 40)
            base_h = int(60 * distance_factor + 30)
            
            # Position with some temporal consistency
            t = self.frame_count * 0.02
            x_base = (i * 150 + int(50 * np.sin(t + i))) % (w - base_w)
            y_base = int(h * 0.4 + (h * 0.4 * distance_factor))
            
            # Add some noise
            x1 = max(0, x_base + np.random.randint(-10, 10))
            y1 = max(0, y_base + np.random.randint(-5, 5))
            x2 = min(w, x1 + base_w)
            y2 = min(h, y1 + base_h)
            
            # Assign class (mostly cars, some trucks, pedestrians)
            class_weights = [0.6, 0.15, 0.1, 0.05, 0.03, 0.05, 0.01, 0.01]
            class_id = np.random.choice(len(class_weights), p=class_weights)
            
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=self.CLASSES[class_id],
                confidence=np.random.uniform(0.75, 0.98)
            ))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                        show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: Input image
            detections: List of detections to draw
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.CLASS_COLORS.get(det.class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            if show_labels:
                label = det.class_name
                if show_confidence:
                    label += f" {det.confidence:.2f}"
                
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 5, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return annotated
    
    def reset(self):
        """Reset frame counter."""
        self.frame_count = 0

