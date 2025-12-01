"""
Multi-Object Tracking Module

Implements IoU-based multi-object tracking with trajectory history.
Tracks agents across frames and maintains their state history.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque


@dataclass
class Track:
    """Represents a tracked object with history."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # Current bounding box
    class_id: int
    class_name: str
    confidence: float
    age: int = 0  # Frames since track started
    hits: int = 1  # Number of successful detections
    misses: int = 0  # Consecutive frames without detection
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of current bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """Get current estimated velocity."""
        if len(self.velocities) > 0:
            return self.velocities[-1]
        return None
    
    def predict_next_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity."""
        cx, cy = self.center
        if self.velocity:
            vx, vy = self.velocity
            return (cx + vx, cy + vy)
        return (cx, cy)


class MultiObjectTracker:
    """
    Multi-object tracker using IoU-based association.
    
    Features:
    - Track ID assignment and management
    - Trajectory history for each track
    - Velocity estimation
    - Track lifecycle management (birth, death)
    """
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_age: int = 30,
                 min_hits: int = 3,
                 trajectory_length: int = 50):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Maximum frames to keep unmatched track
            min_hits: Minimum hits before track is confirmed
            trajectory_length: Maximum trajectory history length
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trajectory_length = trajectory_length
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_distance(self, track: Track, detection_center: Tuple[float, float]) -> float:
        """Compute distance between track prediction and detection."""
        pred_x, pred_y = track.predict_next_position()
        det_x, det_y = detection_center
        return np.sqrt((pred_x - det_x)**2 + (pred_y - det_y)**2)
    
    def _associate(self, detections: List) -> Tuple[List, List, List]:
        """
        Associate detections with existing tracks using IoU.
        
        Returns:
            matched_pairs: List of (track_id, detection_idx) tuples
            unmatched_tracks: List of track_ids
            unmatched_detections: List of detection indices
        """
        if not detections or not self.tracks:
            unmatched_tracks = list(self.tracks.keys())
            unmatched_detections = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_detections
        
        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det.bbox)
        
        # Greedy matching
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        # Sort by IoU (descending) and match greedily
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            track_idx, det_idx = max_idx
            
            matched_pairs.append((track_ids[track_idx], det_idx))
            used_tracks.add(track_ids[track_idx])
            used_detections.add(det_idx)
            
            # Mark as used
            iou_matrix[track_idx, :] = -1
            iou_matrix[:, det_idx] = -1
        
        unmatched_tracks = [t for t in track_ids if t not in used_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of Detection objects from detector
            
        Returns:
            List of confirmed tracks
        """
        self.frame_count += 1
        
        # Associate detections with tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            # Compute velocity before updating position
            old_center = track.center
            new_center = det.center
            velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            
            # Update track
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.age += 1
            track.hits += 1
            track.misses = 0
            
            # Update trajectory
            track.trajectory.append(new_center)
            track.velocities.append(velocity)
            
            # Limit trajectory length
            if len(track.trajectory) > self.trajectory_length:
                track.trajectory = track.trajectory[-self.trajectory_length:]
                track.velocities = track.velocities[-self.trajectory_length:]
        
        # Handle unmatched tracks (increment miss count)
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.age += 1
            track.misses += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                trajectory=[det.center]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Remove dead tracks
        dead_tracks = [
            track_id for track_id, track in self.tracks.items()
            if track.misses > self.max_age
        ]
        for track_id in dead_tracks:
            del self.tracks[track_id]
        
        # Return confirmed tracks
        confirmed = [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
        
        return confirmed
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Get trajectories for all active tracks."""
        return {
            track_id: track.trajectory.copy()
            for track_id, track in self.tracks.items()
            if track.hits >= self.min_hits
        }
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track],
                    draw_trajectories: bool = True,
                    draw_ids: bool = True,
                    draw_velocities: bool = False) -> np.ndarray:
        """
        Draw tracks on frame.
        
        Args:
            frame: Input image
            tracks: List of tracks to draw
            draw_trajectories: Whether to draw trajectory trails
            draw_ids: Whether to show track IDs
            draw_velocities: Whether to draw velocity vectors
            
        Returns:
            Annotated frame
        """
        import cv2
        
        annotated = frame.copy()
        
        # Color palette for tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
        ]
        
        for track in tracks:
            color = colors[track.track_id % len(colors)]
            x1, y1, x2, y2 = track.bbox
            cx, cy = int(track.center[0]), int(track.center[1])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            if draw_ids:
                label = f"ID:{track.track_id} {track.class_name}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if draw_trajectories and len(track.trajectory) > 1:
                points = np.array(track.trajectory, dtype=np.int32)
                
                # Draw trail with fading effect
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    pt1 = tuple(points[i - 1])
                    pt2 = tuple(points[i])
                    cv2.line(annotated, pt1, pt2, color, thickness)
            
            # Draw velocity vector
            if draw_velocities and track.velocity:
                vx, vy = track.velocity
                scale = 5  # Scale for visibility
                end_x = int(cx + vx * scale)
                end_y = int(cy + vy * scale)
                cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), 
                               (0, 255, 255), 2, tipLength=0.3)
        
        return annotated
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0

