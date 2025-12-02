"""
Maneuver Detector

Detects driving maneuvers based on vehicle state and trajectory:
- Lane keeping, lane changes
- Turning maneuvers
- Speed changes (accelerating, braking, stopped)
- Following behavior
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque


class LateralManeuver(Enum):
    LANE_KEEPING = "lane_keeping"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    SWERVING = "swerving"


class LongitudinalManeuver(Enum):
    CRUISING = "cruising"
    ACCELERATING = "accelerating"
    BRAKING = "braking"
    HARD_BRAKING = "hard_braking"
    STOPPED = "stopped"


class TurningManeuver(Enum):
    STRAIGHT = "straight"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    U_TURN = "u_turn"
    CURVING_LEFT = "curving_left"
    CURVING_RIGHT = "curving_right"


@dataclass
class ManeuverTags:
    """Container for maneuver detection results."""
    lateral: LateralManeuver = LateralManeuver.LANE_KEEPING
    lateral_confidence: float = 0.0
    longitudinal: LongitudinalManeuver = LongitudinalManeuver.CRUISING
    longitudinal_confidence: float = 0.0
    turning: TurningManeuver = TurningManeuver.STRAIGHT
    turning_confidence: float = 0.0
    speed_kmh: float = 0.0
    acceleration: float = 0.0
    yaw_rate_deg: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'lateral': self.lateral.value,
            'lateral_confidence': self.lateral_confidence,
            'longitudinal': self.longitudinal.value,
            'longitudinal_confidence': self.longitudinal_confidence,
            'turning': self.turning.value,
            'turning_confidence': self.turning_confidence,
            'speed_kmh': self.speed_kmh,
            'acceleration': self.acceleration,
            'yaw_rate_deg': self.yaw_rate_deg,
            'timestamp': self.timestamp
        }
    
    def get_tags_list(self) -> List[str]:
        """Get flat list of tag strings."""
        return [
            self.lateral.value,
            self.longitudinal.value,
            self.turning.value
        ]


class ManeuverDetector:
    """
    Detects driving maneuvers from vehicle state and trajectory history.
    
    Uses:
    - Vehicle state (position, velocity, heading, acceleration)
    - Historical trajectory
    - Lane offset information
    """
    
    # Thresholds for maneuver detection
    LANE_CHANGE_YAW_THRESHOLD = 5.0  # degrees
    LANE_CHANGE_LATERAL_THRESHOLD = 0.5  # meters
    TURN_YAW_RATE_THRESHOLD = 15.0  # degrees/sec
    HARD_BRAKE_THRESHOLD = -3.0  # m/s²
    BRAKE_THRESHOLD = -1.0  # m/s²
    ACCEL_THRESHOLD = 1.0  # m/s²
    STOPPED_SPEED_THRESHOLD = 0.5  # m/s
    
    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        self.state_history: deque = deque(maxlen=history_length)
        self.position_history: deque = deque(maxlen=history_length)
        self.frame_count = 0
        
    def detect(self, 
               vehicle_state,
               lane_offset: float = None) -> ManeuverTags:
        """
        Detect current driving maneuvers.
        
        Args:
            vehicle_state: VehicleState object with position, velocity, etc.
            lane_offset: Offset from lane center in meters
            
        Returns:
            ManeuverTags with detection results
        """
        tags = ManeuverTags()
        tags.timestamp = self.frame_count / 30.0
        
        if vehicle_state is None:
            return tags
        
        # Extract state values
        speed = getattr(vehicle_state, 'speed', 0.0)
        heading = getattr(vehicle_state, 'heading', 0.0)
        acceleration = getattr(vehicle_state, 'acceleration', 0.0)
        yaw_rate = getattr(vehicle_state, 'yaw_rate', 0.0)
        x = getattr(vehicle_state, 'x', 0.0)
        y = getattr(vehicle_state, 'y', 0.0)
        
        # Store in history
        self.state_history.append({
            'speed': speed,
            'heading': heading,
            'acceleration': acceleration,
            'yaw_rate': yaw_rate,
            'x': x,
            'y': y
        })
        self.position_history.append((x, y))
        
        # Set basic values
        tags.speed_kmh = speed * 3.6
        tags.acceleration = acceleration
        tags.yaw_rate_deg = np.degrees(yaw_rate)
        
        # Detect maneuvers
        tags.lateral, tags.lateral_confidence = self._detect_lateral_maneuver(
            yaw_rate, lane_offset
        )
        tags.longitudinal, tags.longitudinal_confidence = self._detect_longitudinal_maneuver(
            speed, acceleration
        )
        tags.turning, tags.turning_confidence = self._detect_turning_maneuver(
            yaw_rate, speed
        )
        
        self.frame_count += 1
        return tags
    
    def _detect_lateral_maneuver(self, 
                                  yaw_rate: float,
                                  lane_offset: float) -> Tuple[LateralManeuver, float]:
        """Detect lateral maneuvers (lane changes, swerving)."""
        yaw_rate_deg = np.degrees(yaw_rate)
        
        # Check for consistent lateral movement
        if len(self.state_history) >= 10:
            recent_yaw_rates = [s['yaw_rate'] for s in list(self.state_history)[-10:]]
            avg_yaw_rate = np.mean(recent_yaw_rates)
            yaw_rate_std = np.std(recent_yaw_rates)
            
            # Swerving detection (high variance)
            if yaw_rate_std > 0.1:  # High variation in yaw
                return LateralManeuver.SWERVING, min(0.9, yaw_rate_std * 5)
            
            # Lane change detection
            avg_yaw_deg = np.degrees(avg_yaw_rate)
            if avg_yaw_deg > self.LANE_CHANGE_YAW_THRESHOLD:
                confidence = min(0.9, abs(avg_yaw_deg) / 20.0)
                return LateralManeuver.LANE_CHANGE_LEFT, confidence
            elif avg_yaw_deg < -self.LANE_CHANGE_YAW_THRESHOLD:
                confidence = min(0.9, abs(avg_yaw_deg) / 20.0)
                return LateralManeuver.LANE_CHANGE_RIGHT, confidence
        
        # Check lane offset if available
        if lane_offset is not None:
            if abs(lane_offset) > self.LANE_CHANGE_LATERAL_THRESHOLD:
                if lane_offset > 0:
                    return LateralManeuver.LANE_CHANGE_LEFT, 0.6
                else:
                    return LateralManeuver.LANE_CHANGE_RIGHT, 0.6
        
        return LateralManeuver.LANE_KEEPING, 0.8
    
    def _detect_longitudinal_maneuver(self,
                                       speed: float,
                                       acceleration: float) -> Tuple[LongitudinalManeuver, float]:
        """Detect longitudinal maneuvers (accel, brake, stop)."""
        
        # Stopped
        if speed < self.STOPPED_SPEED_THRESHOLD:
            return LongitudinalManeuver.STOPPED, 0.95
        
        # Hard braking
        if acceleration < self.HARD_BRAKE_THRESHOLD:
            confidence = min(0.95, abs(acceleration) / 5.0)
            return LongitudinalManeuver.HARD_BRAKING, confidence
        
        # Normal braking
        if acceleration < self.BRAKE_THRESHOLD:
            confidence = min(0.9, abs(acceleration) / 3.0)
            return LongitudinalManeuver.BRAKING, confidence
        
        # Accelerating
        if acceleration > self.ACCEL_THRESHOLD:
            confidence = min(0.9, acceleration / 3.0)
            return LongitudinalManeuver.ACCELERATING, confidence
        
        # Cruising
        return LongitudinalManeuver.CRUISING, 0.8
    
    def _detect_turning_maneuver(self,
                                  yaw_rate: float,
                                  speed: float) -> Tuple[TurningManeuver, float]:
        """Detect turning maneuvers."""
        yaw_rate_deg = np.degrees(yaw_rate)
        
        # Need sufficient history for turn detection
        if len(self.position_history) < 15:
            return TurningManeuver.STRAIGHT, 0.5
        
        # Calculate cumulative heading change
        if len(self.state_history) >= 15:
            recent_headings = [s['heading'] for s in list(self.state_history)[-15:]]
            heading_change = np.degrees(recent_headings[-1] - recent_headings[0])
            
            # Normalize to -180, 180
            while heading_change > 180:
                heading_change -= 360
            while heading_change < -180:
                heading_change += 360
            
            # U-turn detection
            if abs(heading_change) > 120:
                return TurningManeuver.U_TURN, 0.8
            
            # Turn detection
            if heading_change > 60:
                return TurningManeuver.TURNING_LEFT, min(0.9, heading_change / 90)
            elif heading_change < -60:
                return TurningManeuver.TURNING_RIGHT, min(0.9, abs(heading_change) / 90)
            
            # Curve detection
            if heading_change > 15:
                return TurningManeuver.CURVING_LEFT, min(0.8, heading_change / 45)
            elif heading_change < -15:
                return TurningManeuver.CURVING_RIGHT, min(0.8, abs(heading_change) / 45)
        
        # Use instantaneous yaw rate as fallback
        if abs(yaw_rate_deg) > self.TURN_YAW_RATE_THRESHOLD:
            if yaw_rate_deg > 0:
                return TurningManeuver.CURVING_LEFT, 0.6
            else:
                return TurningManeuver.CURVING_RIGHT, 0.6
        
        return TurningManeuver.STRAIGHT, 0.8
    
    def get_maneuver_summary(self) -> Dict:
        """Get summary of recent maneuvers."""
        if len(self.state_history) < 5:
            return {}
        
        recent = list(self.state_history)[-30:]
        
        return {
            'avg_speed_kmh': np.mean([s['speed'] for s in recent]) * 3.6,
            'max_speed_kmh': np.max([s['speed'] for s in recent]) * 3.6,
            'min_speed_kmh': np.min([s['speed'] for s in recent]) * 3.6,
            'avg_acceleration': np.mean([s['acceleration'] for s in recent]),
            'max_acceleration': np.max([s['acceleration'] for s in recent]),
            'min_acceleration': np.min([s['acceleration'] for s in recent]),
            'total_distance': self._calculate_distance(),
        }
    
    def _calculate_distance(self) -> float:
        """Calculate total distance traveled."""
        if len(self.position_history) < 2:
            return 0.0
        
        positions = list(self.position_history)
        total_dist = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_dist += np.sqrt(dx*dx + dy*dy)
        
        return total_dist
    
    def reset(self):
        """Reset detector state."""
        self.state_history.clear()
        self.position_history.clear()
        self.frame_count = 0


