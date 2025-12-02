"""
Auto-Tagging Module

Provides automated tagging of driving scenarios using:
- Vision-Language Models (VLM) for dynamic tag generation
- Scene classification, maneuver detection, interaction detection

The VLMTagger uses BLIP-2 to generate natural language descriptions
and extract structured tags WITHOUT hardcoded tag definitions.
"""

from .scene_classifier import SceneClassifier
from .maneuver_detector import ManeuverDetector
from .interaction_detector import InteractionDetector
from .auto_tagger import AutoTagger
from .vlm_tagger import VLMTagger, VLMTags

__all__ = [
    'SceneClassifier',
    'ManeuverDetector', 
    'InteractionDetector',
    'AutoTagger',
    'VLMTagger',
    'VLMTags'
]


