"""
Scene Classifier

Classifies driving scenes based on visual and contextual features:
- Road type (intersection, highway, urban, residential, parking)
- Traffic elements (traffic lights, stop signs, crosswalks)
- Environmental conditions (congestion level, time of day)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RoadType(Enum):
    UNKNOWN = "unknown"
    INTERSECTION = "intersection"
    HIGHWAY = "highway"
    URBAN = "urban"
    RESIDENTIAL = "residential"
    PARKING = "parking"


class TrafficElement(Enum):
    TRAFFIC_LIGHT = "traffic_light"
    STOP_SIGN = "stop_sign"
    CROSSWALK = "crosswalk"
    YIELD_SIGN = "yield_sign"
    SPEED_LIMIT = "speed_limit"


class Condition(Enum):
    CLEAR = "clear"
    CONGESTED = "congested"
    NIGHT = "night"
    DAY = "day"
    RAIN = "rain"
    FOG = "fog"


@dataclass
class SceneTags:
    """Container for scene classification results."""
    road_type: RoadType = RoadType.UNKNOWN
    road_type_confidence: float = 0.0
    traffic_elements: List[Tuple[TrafficElement, float]] = field(default_factory=list)
    conditions: List[Tuple[Condition, float]] = field(default_factory=list)
    lane_count: int = 0
    has_pedestrian_area: bool = False
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'road_type': self.road_type.value,
            'road_type_confidence': self.road_type_confidence,
            'traffic_elements': [(e.value, c) for e, c in self.traffic_elements],
            'conditions': [(c.value, conf) for c, conf in self.conditions],
            'lane_count': self.lane_count,
            'has_pedestrian_area': self.has_pedestrian_area,
            'timestamp': self.timestamp
        }
    
    def get_tags_list(self) -> List[str]:
        """Get flat list of tag strings."""
        tags = [self.road_type.value]
        tags.extend([e.value for e, _ in self.traffic_elements])
        tags.extend([c.value for c, _ in self.conditions])
        if self.has_pedestrian_area:
            tags.append("pedestrian_area")
        return tags


class SceneClassifier:
    """
    Classifies driving scenes using visual analysis and detection results.
    
    Uses a combination of:
    - Visual features (color, edges, layout)
    - Object detection results (traffic signs, lights)
    - Lane detection results
    """
    
    def __init__(self):
        self.frame_count = 0
        self.history: List[SceneTags] = []
        self.smoothing_window = 5
        
    def classify(self, 
                 frame: np.ndarray,
                 detections: List = None,
                 lanes: Tuple = None,
                 vehicle_state = None) -> SceneTags:
        """
        Classify the current scene.
        
        Args:
            frame: Input image (BGR format)
            detections: List of Detection objects
            lanes: Tuple of (left_lane, right_lane) from lane detector
            vehicle_state: VehicleState object
            
        Returns:
            SceneTags with classification results
        """
        tags = SceneTags()
        tags.timestamp = self.frame_count / 30.0  # Assume 30fps
        
        # Analyze visual features
        road_type, road_conf = self._classify_road_type(frame, lanes, detections)
        tags.road_type = road_type
        tags.road_type_confidence = road_conf
        
        # Detect traffic elements from detections
        if detections:
            tags.traffic_elements = self._detect_traffic_elements(detections)
            tags.has_pedestrian_area = self._check_pedestrian_area(detections)
        
        # Analyze conditions
        tags.conditions = self._analyze_conditions(frame, vehicle_state)
        
        # Estimate lane count
        if lanes:
            tags.lane_count = self._estimate_lane_count(frame, lanes)
        
        # Apply temporal smoothing
        self.history.append(tags)
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)
        
        self.frame_count += 1
        return self._smooth_tags(tags)
    
    def _classify_road_type(self, 
                            frame: np.ndarray,
                            lanes: Tuple,
                            detections: List) -> Tuple[RoadType, float]:
        """Classify road type based on visual and contextual cues."""
        h, w = frame.shape[:2]
        scores = {rt: 0.0 for rt in RoadType}
        
        # Analyze frame layout
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for intersection patterns (high edge density in center)
        center_region = edges[h//3:2*h//3, w//3:2*w//3]
        center_density = np.sum(center_region > 0) / center_region.size
        
        if center_density > 0.15:
            scores[RoadType.INTERSECTION] += 0.4
        
        # Check for highway patterns (long straight lines, few edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None and len(lines) > 5:
            # Many long lines suggest highway
            avg_length = np.mean([np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) 
                                  for l in lines])
            if avg_length > 150:
                scores[RoadType.HIGHWAY] += 0.5
        
        # Check detections for traffic elements
        if detections:
            traffic_classes = ['traffic_light', 'stop_sign']
            traffic_count = sum(1 for d in detections 
                               if hasattr(d, 'class_name') and d.class_name in traffic_classes)
            if traffic_count > 0:
                scores[RoadType.INTERSECTION] += 0.3
                scores[RoadType.URBAN] += 0.2
            
            # Many vehicles suggest urban/highway
            vehicle_count = sum(1 for d in detections 
                               if hasattr(d, 'class_name') and d.class_name in ['car', 'truck', 'bus'])
            if vehicle_count > 3:
                scores[RoadType.URBAN] += 0.3
                scores[RoadType.HIGHWAY] += 0.2
            elif vehicle_count <= 1:
                scores[RoadType.RESIDENTIAL] += 0.3
        
        # Analyze color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # More green suggests residential
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        if green_ratio > 0.15:
            scores[RoadType.RESIDENTIAL] += 0.3
        
        # Check lanes for structure
        if lanes and lanes[0] is not None and lanes[1] is not None:
            scores[RoadType.HIGHWAY] += 0.2
            scores[RoadType.URBAN] += 0.1
        
        # Normalize and find best
        total = sum(scores.values()) + 0.001
        scores = {k: v/total for k, v in scores.items()}
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Default to urban if uncertain
        if confidence < 0.3:
            best_type = RoadType.URBAN
            confidence = 0.3
            
        return best_type, confidence
    
    def _detect_traffic_elements(self, detections: List) -> List[Tuple[TrafficElement, float]]:
        """Detect traffic elements from object detections."""
        elements = []
        
        element_mapping = {
            'traffic_light': TrafficElement.TRAFFIC_LIGHT,
            'stop_sign': TrafficElement.STOP_SIGN,
        }
        
        for det in detections:
            if hasattr(det, 'class_name') and det.class_name in element_mapping:
                elements.append((element_mapping[det.class_name], det.confidence))
        
        return elements
    
    def _check_pedestrian_area(self, detections: List) -> bool:
        """Check if pedestrians are present indicating pedestrian area."""
        pedestrian_count = sum(1 for d in detections 
                               if hasattr(d, 'class_name') and d.class_name == 'pedestrian')
        return pedestrian_count > 0
    
    def _analyze_conditions(self, 
                           frame: np.ndarray,
                           vehicle_state) -> List[Tuple[Condition, float]]:
        """Analyze environmental conditions."""
        conditions = []
        
        # Analyze brightness for day/night
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 60:
            conditions.append((Condition.NIGHT, 0.8))
        elif avg_brightness > 120:
            conditions.append((Condition.DAY, 0.8))
        else:
            conditions.append((Condition.DAY, 0.5))
        
        # Check for congestion based on vehicle state
        if vehicle_state and hasattr(vehicle_state, 'speed'):
            if vehicle_state.speed < 2.0:  # Nearly stopped
                conditions.append((Condition.CONGESTED, 0.7))
            elif vehicle_state.speed > 15.0:
                conditions.append((Condition.CLEAR, 0.7))
        
        # Simple rain detection (look for high contrast variations)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Low variance might indicate fog/rain
            conditions.append((Condition.FOG, 0.3))
        
        return conditions
    
    def _estimate_lane_count(self, frame: np.ndarray, lanes: Tuple) -> int:
        """Estimate number of lanes."""
        # Simple estimation based on lane width
        if lanes[0] is None or lanes[1] is None:
            return 2  # Default
        
        # Calculate lane width at bottom of frame
        h = frame.shape[0]
        left_x = lanes[0][1] * h + lanes[0][0] if len(lanes[0]) >= 2 else frame.shape[1] // 3
        right_x = lanes[1][1] * h + lanes[1][0] if len(lanes[1]) >= 2 else 2 * frame.shape[1] // 3
        
        lane_width = abs(right_x - left_x)
        
        # Estimate total road width and divide
        if lane_width > 200:
            return 3
        elif lane_width > 100:
            return 2
        else:
            return 1
    
    def _smooth_tags(self, current: SceneTags) -> SceneTags:
        """Apply temporal smoothing to reduce flickering."""
        if len(self.history) < 2:
            return current
        
        # Vote on road type
        road_type_votes = {}
        for tags in self.history:
            rt = tags.road_type
            road_type_votes[rt] = road_type_votes.get(rt, 0) + 1
        
        # Use majority vote
        best_road_type = max(road_type_votes, key=road_type_votes.get)
        if road_type_votes[best_road_type] > len(self.history) // 2:
            current.road_type = best_road_type
        
        return current
    
    def reset(self):
        """Reset classifier state."""
        self.frame_count = 0
        self.history = []


