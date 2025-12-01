"""
Auto Tagger

Aggregates all tagging modules into a unified auto-tagging system.
Provides a single interface for tagging driving scenarios.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .scene_classifier import SceneClassifier, SceneTags
from .maneuver_detector import ManeuverDetector, ManeuverTags
from .interaction_detector import InteractionDetector, InteractionTags


@dataclass
class FrameTags:
    """Complete tags for a single frame."""
    frame_idx: int
    timestamp: float
    scene: SceneTags = None
    maneuver: ManeuverTags = None
    interaction: InteractionTags = None
    all_tags: List[str] = field(default_factory=list)
    tag_confidences: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/export."""
        return {
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'scene': self.scene.to_dict() if self.scene else {},
            'maneuver': self.maneuver.to_dict() if self.maneuver else {},
            'interaction': self.interaction.to_dict() if self.interaction else {},
            'all_tags': self.all_tags,
            'tag_confidences': self.tag_confidences
        }
    
    def get_summary_string(self) -> str:
        """Get human-readable summary."""
        parts = []
        if self.scene:
            parts.append(f"Scene: {self.scene.road_type.value}")
        if self.maneuver:
            parts.append(f"Maneuver: {self.maneuver.lateral.value}, {self.maneuver.longitudinal.value}")
        if self.interaction and self.interaction.primary_interaction:
            parts.append(f"Interaction: {self.interaction.primary_interaction.value}")
        return " | ".join(parts) if parts else "No tags"


@dataclass
class TaggingSession:
    """Metadata for a tagging session."""
    session_id: str
    video_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_frames: int = 0
    fps: float = 30.0
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'video_path': self.video_path,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_frames': self.total_frames,
            'fps': self.fps
        }


class AutoTagger:
    """
    Unified auto-tagging system for driving scenarios.
    
    Combines:
    - Scene classification
    - Maneuver detection  
    - Interaction detection
    
    Provides searchable tags and metrics for data curation.
    """
    
    def __init__(self, video_path: str = "unknown", fps: float = 30.0):
        """
        Initialize the auto-tagger.
        
        Args:
            video_path: Path to video being tagged
            fps: Frames per second for timestamp calculation
        """
        # Initialize sub-modules
        self.scene_classifier = SceneClassifier()
        self.maneuver_detector = ManeuverDetector()
        self.interaction_detector = InteractionDetector()
        
        # Session info
        self.session = TaggingSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            video_path=video_path,
            start_time=datetime.now(),
            fps=fps
        )
        
        # Storage
        self.frame_tags: List[FrameTags] = []
        self.tag_counts: Dict[str, int] = {}
        self.frame_count = 0
        
    def tag_frame(self,
                  frame: np.ndarray,
                  detections: List = None,
                  tracks: List = None,
                  lanes: Tuple = None,
                  vehicle_state = None) -> FrameTags:
        """
        Tag a single frame with all available information.
        
        Args:
            frame: Input image (BGR format)
            detections: List of Detection objects
            tracks: List of Track objects
            lanes: Tuple of (left_lane, right_lane)
            vehicle_state: VehicleState object
            
        Returns:
            FrameTags with all detected tags
        """
        timestamp = self.frame_count / self.session.fps
        
        # Get scene tags
        scene_tags = self.scene_classifier.classify(
            frame, detections, lanes, vehicle_state
        )
        
        # Get maneuver tags
        lane_offset = None
        if lanes and lanes[0] is not None and lanes[1] is not None:
            # Simple lane offset estimation
            h, w = frame.shape[:2]
            # This would need proper lane center calculation
            lane_offset = 0.0
        
        maneuver_tags = self.maneuver_detector.detect(vehicle_state, lane_offset)
        
        # Get interaction tags
        interaction_tags = self.interaction_detector.detect(
            tracks, vehicle_state, frame.shape[:2]
        )
        
        # Aggregate all tags
        all_tags = []
        tag_confidences = {}
        
        # Scene tags
        if scene_tags:
            scene_tag_list = scene_tags.get_tags_list()
            all_tags.extend(scene_tag_list)
            tag_confidences[scene_tags.road_type.value] = scene_tags.road_type_confidence
            for elem, conf in scene_tags.traffic_elements:
                tag_confidences[elem.value] = conf
        
        # Maneuver tags
        if maneuver_tags:
            maneuver_tag_list = maneuver_tags.get_tags_list()
            all_tags.extend(maneuver_tag_list)
            tag_confidences[maneuver_tags.lateral.value] = maneuver_tags.lateral_confidence
            tag_confidences[maneuver_tags.longitudinal.value] = maneuver_tags.longitudinal_confidence
            tag_confidences[maneuver_tags.turning.value] = maneuver_tags.turning_confidence
        
        # Interaction tags
        if interaction_tags:
            interaction_tag_list = interaction_tags.get_tags_list()
            all_tags.extend(interaction_tag_list)
            for interaction in interaction_tags.interactions:
                tag_confidences[interaction.type.value] = interaction.confidence
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in all_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        # Create frame tags
        frame_tags = FrameTags(
            frame_idx=self.frame_count,
            timestamp=timestamp,
            scene=scene_tags,
            maneuver=maneuver_tags,
            interaction=interaction_tags,
            all_tags=unique_tags,
            tag_confidences=tag_confidences
        )
        
        # Update counts
        for tag in unique_tags:
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1
        
        # Store
        self.frame_tags.append(frame_tags)
        self.frame_count += 1
        self.session.total_frames = self.frame_count
        
        return frame_tags
    
    def get_tag_statistics(self) -> Dict:
        """Get statistics about detected tags."""
        if not self.frame_tags:
            return {}
        
        # Tag frequency
        total_frames = len(self.frame_tags)
        tag_frequency = {
            tag: count / total_frames 
            for tag, count in self.tag_counts.items()
        }
        
        # Sort by frequency
        sorted_tags = sorted(
            tag_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Speed statistics
        speeds = [ft.maneuver.speed_kmh for ft in self.frame_tags if ft.maneuver]
        
        # Risk statistics
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for ft in self.frame_tags:
            if ft.interaction:
                risk_counts[ft.interaction.overall_risk.value] += 1
        
        return {
            'total_frames': total_frames,
            'unique_tags': len(self.tag_counts),
            'tag_frequency': dict(sorted_tags[:20]),  # Top 20
            'tag_counts': self.tag_counts,
            'speed_stats': {
                'min': min(speeds) if speeds else 0,
                'max': max(speeds) if speeds else 0,
                'avg': np.mean(speeds) if speeds else 0
            },
            'risk_distribution': risk_counts,
            'session_info': self.session.to_dict()
        }
    
    def search_by_tag(self, tag: str) -> List[FrameTags]:
        """Search for frames containing a specific tag."""
        return [ft for ft in self.frame_tags if tag in ft.all_tags]
    
    def search_by_tags(self, tags: List[str], match_all: bool = True) -> List[FrameTags]:
        """
        Search for frames containing specified tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, frame must contain all tags. If False, any tag matches.
        """
        results = []
        for ft in self.frame_tags:
            if match_all:
                if all(tag in ft.all_tags for tag in tags):
                    results.append(ft)
            else:
                if any(tag in ft.all_tags for tag in tags):
                    results.append(ft)
        return results
    
    def get_high_risk_frames(self) -> List[FrameTags]:
        """Get frames with high or critical risk interactions."""
        return [
            ft for ft in self.frame_tags 
            if ft.interaction and ft.interaction.overall_risk.value in ['high', 'critical']
        ]
    
    def get_event_segments(self, event_tag: str, min_duration: int = 5) -> List[Tuple[int, int]]:
        """
        Get continuous segments containing an event.
        
        Args:
            event_tag: Tag to search for
            min_duration: Minimum frames for a segment
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        segments = []
        current_start = None
        
        for i, ft in enumerate(self.frame_tags):
            has_tag = event_tag in ft.all_tags
            
            if has_tag and current_start is None:
                current_start = i
            elif not has_tag and current_start is not None:
                if i - current_start >= min_duration:
                    segments.append((current_start, i - 1))
                current_start = None
        
        # Handle segment ending at last frame
        if current_start is not None:
            if len(self.frame_tags) - current_start >= min_duration:
                segments.append((current_start, len(self.frame_tags) - 1))
        
        return segments
    
    def export_tags(self, format: str = 'dict') -> Any:
        """
        Export all tags.
        
        Args:
            format: 'dict', 'json', or 'csv'
            
        Returns:
            Exported data in requested format
        """
        if format == 'dict':
            return {
                'session': self.session.to_dict(),
                'statistics': self.get_tag_statistics(),
                'frames': [ft.to_dict() for ft in self.frame_tags]
            }
        elif format == 'json':
            import json
            return json.dumps(self.export_tags('dict'), indent=2)
        elif format == 'csv':
            # Create CSV-friendly format
            rows = []
            for ft in self.frame_tags:
                row = {
                    'frame_idx': ft.frame_idx,
                    'timestamp': ft.timestamp,
                    'road_type': ft.scene.road_type.value if ft.scene else '',
                    'lateral_maneuver': ft.maneuver.lateral.value if ft.maneuver else '',
                    'longitudinal_maneuver': ft.maneuver.longitudinal.value if ft.maneuver else '',
                    'turning_maneuver': ft.maneuver.turning.value if ft.maneuver else '',
                    'speed_kmh': ft.maneuver.speed_kmh if ft.maneuver else 0,
                    'risk_level': ft.interaction.overall_risk.value if ft.interaction else 'low',
                    'agent_count': ft.interaction.agent_count if ft.interaction else 0,
                    'all_tags': '|'.join(ft.all_tags)
                }
                rows.append(row)
            return rows
        
        return None
    
    def reset(self):
        """Reset the auto-tagger for a new session."""
        self.scene_classifier.reset()
        self.maneuver_detector.reset()
        self.interaction_detector.reset()
        
        self.frame_tags = []
        self.tag_counts = {}
        self.frame_count = 0
        
        self.session = TaggingSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            video_path=self.session.video_path,
            start_time=datetime.now(),
            fps=self.session.fps
        )
    
    def finalize(self):
        """Finalize the tagging session."""
        self.session.end_time = datetime.now()
        self.session.total_frames = self.frame_count

