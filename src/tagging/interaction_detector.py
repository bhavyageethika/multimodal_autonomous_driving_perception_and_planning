"""
Interaction Detector

Detects traffic interactions between ego vehicle and other agents:
- Following behavior
- Yielding situations
- Cut-ins and cut-outs
- Pedestrian interactions
- Near-miss events
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque


class InteractionType(Enum):
    NONE = "no_interaction"
    FOLLOWING = "following_vehicle"
    BEING_FOLLOWED = "being_followed"
    YIELDING = "yielding"
    VEHICLE_CUT_IN = "vehicle_cut_in"
    VEHICLE_CUT_OUT = "vehicle_cut_out"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    PEDESTRIAN_WAITING = "pedestrian_waiting"
    CYCLIST_NEARBY = "cyclist_nearby"
    NEAR_MISS = "near_miss"
    MERGING = "merging"
    PASSING = "passing"
    BEING_PASSED = "being_passed"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Interaction:
    """Single interaction event."""
    type: InteractionType
    confidence: float
    risk_level: RiskLevel
    agent_id: Optional[int] = None
    agent_class: Optional[str] = None
    distance: float = 0.0
    relative_speed: float = 0.0
    time_to_collision: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'confidence': self.confidence,
            'risk_level': self.risk_level.value,
            'agent_id': self.agent_id,
            'agent_class': self.agent_class,
            'distance': self.distance,
            'relative_speed': self.relative_speed,
            'time_to_collision': self.time_to_collision
        }


@dataclass 
class InteractionTags:
    """Container for interaction detection results."""
    interactions: List[Interaction] = field(default_factory=list)
    primary_interaction: Optional[InteractionType] = None
    overall_risk: RiskLevel = RiskLevel.LOW
    agent_count: int = 0
    pedestrian_count: int = 0
    cyclist_count: int = 0
    vehicle_count: int = 0
    closest_agent_distance: float = float('inf')
    min_ttc: Optional[float] = None  # Time to collision
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'interactions': [i.to_dict() for i in self.interactions],
            'primary_interaction': self.primary_interaction.value if self.primary_interaction else None,
            'overall_risk': self.overall_risk.value,
            'agent_count': self.agent_count,
            'pedestrian_count': self.pedestrian_count,
            'cyclist_count': self.cyclist_count,
            'vehicle_count': self.vehicle_count,
            'closest_agent_distance': self.closest_agent_distance,
            'min_ttc': self.min_ttc,
            'timestamp': self.timestamp
        }
    
    def get_tags_list(self) -> List[str]:
        """Get flat list of tag strings."""
        tags = []
        for interaction in self.interactions:
            if interaction.confidence > 0.5:
                tags.append(interaction.type.value)
        if self.overall_risk != RiskLevel.LOW:
            tags.append(f"risk_{self.overall_risk.value}")
        return list(set(tags))  # Remove duplicates


class InteractionDetector:
    """
    Detects interactions between ego vehicle and other traffic participants.
    
    Uses:
    - Object tracks with velocity estimates
    - Ego vehicle state
    - Spatial relationships
    """
    
    # Distance thresholds (meters)
    FOLLOWING_DISTANCE_MAX = 30.0
    FOLLOWING_DISTANCE_MIN = 5.0
    NEAR_MISS_DISTANCE = 3.0
    PEDESTRIAN_DANGER_DISTANCE = 10.0
    CUT_IN_DISTANCE = 15.0
    
    # Time thresholds (seconds)
    TTC_CRITICAL = 1.5
    TTC_WARNING = 3.0
    
    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        self.track_history: Dict[int, deque] = {}  # track_id -> position history
        self.frame_count = 0
        
    def detect(self,
               tracks: List,
               vehicle_state,
               frame_shape: Tuple[int, int] = (480, 640)) -> InteractionTags:
        """
        Detect interactions with tracked agents.
        
        Args:
            tracks: List of Track objects from tracker
            vehicle_state: VehicleState object
            frame_shape: (height, width) of frame for position estimation
            
        Returns:
            InteractionTags with detection results
        """
        tags = InteractionTags()
        tags.timestamp = self.frame_count / 30.0
        
        if not tracks:
            self.frame_count += 1
            return tags
        
        # Count agents by type
        for track in tracks:
            class_name = getattr(track, 'class_name', 'unknown')
            if class_name in ['pedestrian']:
                tags.pedestrian_count += 1
            elif class_name in ['cyclist', 'bicycle']:
                tags.cyclist_count += 1
            elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                tags.vehicle_count += 1
        
        tags.agent_count = len(tracks)
        
        # Get ego state
        ego_speed = getattr(vehicle_state, 'speed', 10.0) if vehicle_state else 10.0
        ego_x = getattr(vehicle_state, 'x', 0.0) if vehicle_state else 0.0
        ego_y = getattr(vehicle_state, 'y', 0.0) if vehicle_state else 0.0
        
        # Analyze each track
        interactions = []
        min_distance = float('inf')
        min_ttc = float('inf')
        
        for track in tracks:
            track_id = getattr(track, 'track_id', 0)
            class_name = getattr(track, 'class_name', 'unknown')
            bbox = getattr(track, 'bbox', (0, 0, 0, 0))
            velocity = getattr(track, 'velocity', (0, 0))
            
            # Estimate distance based on bbox position and size
            distance = self._estimate_distance(bbox, frame_shape)
            min_distance = min(min_distance, distance)
            
            # Estimate relative speed
            rel_speed = self._estimate_relative_speed(velocity, ego_speed)
            
            # Calculate time to collision
            ttc = self._calculate_ttc(distance, rel_speed)
            if ttc is not None and ttc > 0:
                min_ttc = min(min_ttc, ttc)
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.history_length)
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[track_id].append(center)
            
            # Detect specific interactions
            interaction = self._analyze_interaction(
                track, distance, rel_speed, ttc, 
                class_name, frame_shape
            )
            
            if interaction:
                interactions.append(interaction)
        
        # Store results
        tags.interactions = interactions
        tags.closest_agent_distance = min_distance if min_distance != float('inf') else 0
        tags.min_ttc = min_ttc if min_ttc != float('inf') else None
        
        # Determine primary interaction and overall risk
        if interactions:
            # Sort by confidence and risk
            interactions.sort(key=lambda x: (x.risk_level.value, -x.confidence), reverse=True)
            tags.primary_interaction = interactions[0].type
            tags.overall_risk = self._calculate_overall_risk(interactions, min_ttc)
        
        self.frame_count += 1
        return tags
    
    def _estimate_distance(self, bbox: Tuple, frame_shape: Tuple) -> float:
        """Estimate distance to object based on bounding box size."""
        h, w = frame_shape
        x1, y1, x2, y2 = bbox
        
        # Use vertical position and box height
        box_height = y2 - y1
        box_bottom = y2
        
        # Simple perspective model
        # Objects lower in frame and larger are closer
        if box_height <= 0:
            return 50.0
        
        # Normalize position (0 = top, 1 = bottom)
        y_normalized = box_bottom / h
        
        # Estimate distance (closer objects are lower and larger)
        # This is a simplified model
        base_distance = 50.0 * (1 - y_normalized) + 5.0
        size_factor = 100.0 / (box_height + 10)
        
        distance = (base_distance + size_factor) / 2
        return max(2.0, min(100.0, distance))
    
    def _estimate_relative_speed(self, velocity: Tuple, ego_speed: float) -> float:
        """Estimate relative speed between ego and tracked object."""
        if velocity is None:
            return 0.0
        
        vx, vy = velocity
        # Assume objects moving down in image are moving away
        # Objects moving up are approaching
        relative = ego_speed - vy  # Simplified longitudinal relative speed
        return relative
    
    def _calculate_ttc(self, distance: float, relative_speed: float) -> Optional[float]:
        """Calculate time to collision."""
        if relative_speed <= 0.1:  # Not approaching or moving away
            return None
        
        ttc = distance / relative_speed
        return ttc if ttc > 0 else None
    
    def _analyze_interaction(self,
                             track,
                             distance: float,
                             rel_speed: float,
                             ttc: Optional[float],
                             class_name: str,
                             frame_shape: Tuple) -> Optional[Interaction]:
        """Analyze interaction type for a specific track."""
        bbox = getattr(track, 'bbox', (0, 0, 0, 0))
        track_id = getattr(track, 'track_id', 0)
        confidence = getattr(track, 'confidence', 0.5)
        
        h, w = frame_shape
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Check for near-miss
        if distance < self.NEAR_MISS_DISTANCE:
            return Interaction(
                type=InteractionType.NEAR_MISS,
                confidence=0.9,
                risk_level=RiskLevel.CRITICAL,
                agent_id=track_id,
                agent_class=class_name,
                distance=distance,
                relative_speed=rel_speed,
                time_to_collision=ttc
            )
        
        # Pedestrian interactions
        if class_name == 'pedestrian':
            if distance < self.PEDESTRIAN_DANGER_DISTANCE:
                # Check if pedestrian is in path
                if abs(center_x - w/2) < w/4:  # Near center
                    return Interaction(
                        type=InteractionType.PEDESTRIAN_CROSSING,
                        confidence=0.8,
                        risk_level=RiskLevel.HIGH if distance < 8 else RiskLevel.MEDIUM,
                        agent_id=track_id,
                        agent_class=class_name,
                        distance=distance,
                        relative_speed=rel_speed,
                        time_to_collision=ttc
                    )
                else:
                    return Interaction(
                        type=InteractionType.PEDESTRIAN_WAITING,
                        confidence=0.6,
                        risk_level=RiskLevel.LOW,
                        agent_id=track_id,
                        agent_class=class_name,
                        distance=distance
                    )
        
        # Cyclist interactions
        if class_name in ['cyclist', 'bicycle']:
            if distance < 15:
                return Interaction(
                    type=InteractionType.CYCLIST_NEARBY,
                    confidence=0.7,
                    risk_level=RiskLevel.MEDIUM if distance < 8 else RiskLevel.LOW,
                    agent_id=track_id,
                    agent_class=class_name,
                    distance=distance,
                    relative_speed=rel_speed
                )
        
        # Vehicle interactions
        if class_name in ['car', 'truck', 'bus']:
            # Check for following
            if center_x > w/4 and center_x < 3*w/4:  # In front
                if self.FOLLOWING_DISTANCE_MIN < distance < self.FOLLOWING_DISTANCE_MAX:
                    risk = RiskLevel.LOW
                    if distance < 10:
                        risk = RiskLevel.MEDIUM
                    if ttc and ttc < self.TTC_WARNING:
                        risk = RiskLevel.HIGH
                    
                    return Interaction(
                        type=InteractionType.FOLLOWING,
                        confidence=0.75,
                        risk_level=risk,
                        agent_id=track_id,
                        agent_class=class_name,
                        distance=distance,
                        relative_speed=rel_speed,
                        time_to_collision=ttc
                    )
            
            # Check for cut-in (track moving into center)
            if track_id in self.track_history and len(self.track_history[track_id]) >= 10:
                history = list(self.track_history[track_id])
                start_x = history[0][0]
                end_x = history[-1][0]
                
                # Object moving toward center of frame
                if abs(end_x - w/2) < abs(start_x - w/2) and distance < self.CUT_IN_DISTANCE:
                    return Interaction(
                        type=InteractionType.VEHICLE_CUT_IN,
                        confidence=0.7,
                        risk_level=RiskLevel.MEDIUM,
                        agent_id=track_id,
                        agent_class=class_name,
                        distance=distance,
                        relative_speed=rel_speed
                    )
        
        return None
    
    def _calculate_overall_risk(self, 
                                interactions: List[Interaction],
                                min_ttc: float) -> RiskLevel:
        """Calculate overall risk level."""
        if not interactions:
            return RiskLevel.LOW
        
        # Check for critical TTC
        if min_ttc and min_ttc < self.TTC_CRITICAL:
            return RiskLevel.CRITICAL
        
        # Check for any high-risk interactions
        risk_levels = [i.risk_level for i in interactions]
        
        if RiskLevel.CRITICAL in risk_levels:
            return RiskLevel.CRITICAL
        if RiskLevel.HIGH in risk_levels:
            return RiskLevel.HIGH
        if RiskLevel.MEDIUM in risk_levels:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def get_interaction_summary(self) -> Dict:
        """Get summary of recent interactions."""
        return {
            'tracked_agents': len(self.track_history),
            'frame_count': self.frame_count
        }
    
    def reset(self):
        """Reset detector state."""
        self.track_history.clear()
        self.frame_count = 0


