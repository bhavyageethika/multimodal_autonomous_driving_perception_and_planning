"""
Auto-Tagging Module

Provides automated tagging of driving scenarios for:
- Scene classification (intersection, highway, urban, etc.)
- Driving maneuver detection (lane change, turning, braking, etc.)
- Interaction detection (pedestrian crossing, following, yielding, etc.)
"""

from .scene_classifier import SceneClassifier
from .maneuver_detector import ManeuverDetector
from .interaction_detector import InteractionDetector
from .auto_tagger import AutoTagger

__all__ = [
    'SceneClassifier',
    'ManeuverDetector', 
    'InteractionDetector',
    'AutoTagger'
]

