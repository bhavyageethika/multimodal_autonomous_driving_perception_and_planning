"""
VLM Tagger

Uses Vision-Language Models (BLIP-2) to generate dynamic,
natural language descriptions and tags for driving scenes.

No hardcoded tags - the model generates descriptions that
are then parsed into structured, searchable tags.
"""

import numpy as np
import cv2
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import torch


@dataclass
class VLMTags:
    """Container for VLM-generated tags."""
    frame_idx: int = 0
    timestamp: float = 0.0
    
    # Raw VLM outputs
    scene_description: str = ""
    safety_assessment: str = ""
    
    # Parsed structured tags
    extracted_tags: List[str] = field(default_factory=list)
    
    # Specific queries
    road_type: str = "unknown"
    weather: str = "unknown"
    time_of_day: str = "unknown"
    vehicles_description: str = ""
    pedestrians_description: str = ""
    maneuver_description: str = ""
    risk_level: str = "low"
    risk_reason: str = ""
    
    # Confidence (based on model's response)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'scene_description': self.scene_description,
            'safety_assessment': self.safety_assessment,
            'extracted_tags': self.extracted_tags,
            'road_type': self.road_type,
            'weather': self.weather,
            'time_of_day': self.time_of_day,
            'vehicles_description': self.vehicles_description,
            'pedestrians_description': self.pedestrians_description,
            'maneuver_description': self.maneuver_description,
            'risk_level': self.risk_level,
            'risk_reason': self.risk_reason,
            'confidence': self.confidence
        }
    
    def get_tags_list(self) -> List[str]:
        """Get all tags as a flat list."""
        tags = list(self.extracted_tags)
        if self.road_type != "unknown":
            tags.append(self.road_type)
        if self.weather != "unknown":
            tags.append(self.weather)
        if self.time_of_day != "unknown":
            tags.append(self.time_of_day)
        if self.risk_level != "low":
            tags.append(f"risk_{self.risk_level}")
        return list(set(tags))


class VLMTagger:
    """
    Vision-Language Model based tagger using BLIP.
    
    Generates natural language descriptions of driving scenes
    and extracts structured tags without hardcoding.
    
    Uses lightweight BLIP model for fast inference.
    """
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip-image-captioning-base",
                 device: str = None,
                 use_fast_mode: bool = True):
        """
        Initialize the VLM tagger.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
            use_fast_mode: If True, use smaller model for speed
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fast_mode = use_fast_mode
        
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.load_error = None
        
        self.frame_count = 0
        self.tag_history: List[VLMTags] = []
        
        # Cache for performance
        self._cache_interval = 10  # Only run VLM every N frames
        self._last_tags = None
        
        print(f"VLMTagger initialized (device: {self.device})")
        print("Model will be loaded on first use...")
    
    def _load_model(self):
        """Load the BLIP model (lazy loading)."""
        if self.is_loaded:
            return True
        
        if self.load_error:
            return False
        
        print(f"Loading VLM model (lightweight)...")
        print("This may take a minute on first run...")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Use lightweight BLIP model (~1GB instead of 3GB)
            model_name = "Salesforce/blip-image-captioning-base"
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Use float32 for CPU compatibility
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"✓ VLM model loaded successfully on {self.device}")
            return True
            
        except ImportError as e:
            self.load_error = f"Missing library: {e}"
            print(f"Error: {self.load_error}")
            print("Run: pip install transformers")
            return False
        except Exception as e:
            self.load_error = str(e)
            print(f"Error loading model: {e}")
            return False
    
    def _generate_response(self, 
                           image: Image.Image, 
                           prompt: str = None,
                           max_tokens: int = 50) -> str:
        """Generate a response from the VLM."""
        if not self._load_model():
            return f"Model load failed: {self.load_error}"
        
        try:
            # BLIP captioning (simpler API)
            if prompt:
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=3
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def tag_frame(self, 
                  frame: np.ndarray,
                  vehicle_state=None,
                  tracks: List = None,
                  force_update: bool = False) -> VLMTags:
        """
        Generate tags for a frame using the VLM.
        
        Args:
            frame: BGR image (numpy array)
            vehicle_state: Optional vehicle state for context
            tracks: Optional list of tracked objects
            force_update: Force VLM inference even if cached
            
        Returns:
            VLMTags with generated descriptions and tags
        """
        timestamp = self.frame_count / 30.0
        
        # Use cache for performance (VLM is slow)
        if not force_update and self._last_tags is not None:
            if self.frame_count % self._cache_interval != 0:
                # Return cached tags with updated frame info
                cached = VLMTags(
                    frame_idx=self.frame_count,
                    timestamp=timestamp,
                    scene_description=self._last_tags.scene_description,
                    safety_assessment=self._last_tags.safety_assessment,
                    extracted_tags=self._last_tags.extracted_tags,
                    road_type=self._last_tags.road_type,
                    weather=self._last_tags.weather,
                    time_of_day=self._last_tags.time_of_day,
                    vehicles_description=self._last_tags.vehicles_description,
                    pedestrians_description=self._last_tags.pedestrians_description,
                    maneuver_description=self._last_tags.maneuver_description,
                    risk_level=self._last_tags.risk_level,
                    risk_reason=self._last_tags.risk_reason,
                    confidence=self._last_tags.confidence
                )
                self.frame_count += 1
                return cached
        
        # Convert BGR to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        tags = VLMTags(frame_idx=self.frame_count, timestamp=timestamp)
        
        try:
            # Generate scene caption (BLIP is optimized for this)
            tags.scene_description = self._generate_response(
                pil_image, 
                "a photo of a driving scene showing",
                max_tokens=75
            )
            
            # If description looks like an error, try without prompt
            if "error" in tags.scene_description.lower() or "failed" in tags.scene_description.lower():
                tags.scene_description = self._generate_response(
                    pil_image,
                    max_tokens=75
                )
            
            # Generate safety-focused caption
            tags.safety_assessment = self._generate_response(
                pil_image,
                "this driving situation is",
                max_tokens=50
            )
            
            # Extract structured tags from the caption
            tags.extracted_tags = self._extract_tags(
                tags.scene_description, 
                tags.safety_assessment
            )
            
            # Infer road type from description
            desc_lower = tags.scene_description.lower()
            if any(w in desc_lower for w in ['highway', 'freeway', 'motorway']):
                tags.road_type = 'highway'
            elif any(w in desc_lower for w in ['intersection', 'traffic light', 'crossroad']):
                tags.road_type = 'intersection'
            elif any(w in desc_lower for w in ['city', 'urban', 'street', 'building']):
                tags.road_type = 'urban'
            elif any(w in desc_lower for w in ['residential', 'neighborhood', 'house']):
                tags.road_type = 'residential'
            else:
                tags.road_type = 'road'
            
            # Infer weather
            if any(w in desc_lower for w in ['rain', 'wet', 'rainy']):
                tags.weather = 'rainy'
            elif any(w in desc_lower for w in ['snow', 'snowy', 'winter']):
                tags.weather = 'snowy'
            elif any(w in desc_lower for w in ['fog', 'foggy', 'mist']):
                tags.weather = 'foggy'
            else:
                tags.weather = 'clear'
            
            # Infer time of day
            if any(w in desc_lower for w in ['night', 'dark', 'evening']):
                tags.time_of_day = 'night'
            else:
                tags.time_of_day = 'day'
            
            # Parse risk level from safety assessment
            tags.risk_level, tags.risk_reason = self._parse_risk(
                tags.safety_assessment
            )
            
            # Add vehicle state context if available
            if vehicle_state:
                speed = getattr(vehicle_state, 'speed', 0) * 3.6
                if speed < 5:
                    tags.extracted_tags.append("stopped")
                elif speed > 100:
                    tags.extracted_tags.append("high_speed")
                
                accel = getattr(vehicle_state, 'acceleration', 0)
                if accel < -3:
                    tags.extracted_tags.append("hard_braking")
                elif accel < -1:
                    tags.extracted_tags.append("braking")
                elif accel > 1:
                    tags.extracted_tags.append("accelerating")
            
            # Add track info
            if tracks:
                if len(tracks) > 5:
                    tags.extracted_tags.append("heavy_traffic")
                pedestrians = sum(1 for t in tracks 
                                  if getattr(t, 'class_name', '') == 'pedestrian')
                if pedestrians > 0:
                    tags.extracted_tags.append("pedestrians_present")
            
            tags.confidence = 0.8
            
        except Exception as e:
            print(f"VLM inference error: {e}")
            tags.scene_description = "Error generating description"
            tags.confidence = 0.0
        
        # Update cache
        self._last_tags = tags
        self.tag_history.append(tags)
        self.frame_count += 1
        
        return tags
    
    def _extract_tags(self, 
                      scene_desc: str, 
                      safety_desc: str) -> List[str]:
        """Extract structured tags from natural language descriptions."""
        text = (scene_desc + " " + safety_desc).lower()
        tags = []
        
        # Road types
        road_keywords = {
            'highway': ['highway', 'freeway', 'motorway', 'expressway'],
            'intersection': ['intersection', 'crossroads', 'junction', 'traffic light'],
            'urban': ['urban', 'city', 'downtown', 'street'],
            'residential': ['residential', 'neighborhood', 'suburb'],
            'parking': ['parking', 'parked', 'parking lot']
        }
        
        for tag, keywords in road_keywords.items():
            if any(kw in text for kw in keywords):
                tags.append(tag)
        
        # Weather
        weather_keywords = {
            'rainy': ['rain', 'rainy', 'wet', 'raining'],
            'foggy': ['fog', 'foggy', 'mist', 'hazy'],
            'snowy': ['snow', 'snowy', 'winter'],
            'clear': ['clear', 'sunny', 'bright']
        }
        
        for tag, keywords in weather_keywords.items():
            if any(kw in text for kw in keywords):
                tags.append(tag)
        
        # Time of day
        if any(w in text for w in ['night', 'dark', 'nighttime']):
            tags.append('night')
        elif any(w in text for w in ['day', 'daytime', 'daylight', 'sunny']):
            tags.append('daytime')
        
        # Agents
        if any(w in text for w in ['pedestrian', 'people', 'person', 'walking']):
            tags.append('pedestrians')
        if any(w in text for w in ['cyclist', 'bicycle', 'bike']):
            tags.append('cyclists')
        if any(w in text for w in ['truck', 'lorry']):
            tags.append('trucks')
        if any(w in text for w in ['bus', 'buses']):
            tags.append('buses')
        
        # Safety/Risk
        if any(w in text for w in ['dangerous', 'hazard', 'risk', 'unsafe', 'caution']):
            tags.append('potential_hazard')
        if any(w in text for w in ['safe', 'clear road', 'no obstacles']):
            tags.append('safe_conditions')
        if any(w in text for w in ['close', 'near miss', 'almost', 'too close']):
            tags.append('close_call')
        
        # Traffic conditions
        if any(w in text for w in ['heavy traffic', 'congested', 'traffic jam', 'busy']):
            tags.append('heavy_traffic')
        if any(w in text for w in ['empty', 'no traffic', 'clear road']):
            tags.append('light_traffic')
        
        # Actions
        if any(w in text for w in ['turning', 'turn left', 'turn right']):
            tags.append('turning')
        if any(w in text for w in ['lane change', 'changing lanes', 'merging']):
            tags.append('lane_change')
        if any(w in text for w in ['stopping', 'stopped', 'brake', 'braking']):
            tags.append('stopping')
        if any(w in text for w in ['crossing', 'crosswalk', 'cross the']):
            tags.append('crossing')
        
        return list(set(tags))
    
    def _parse_risk(self, safety_text: str) -> Tuple[str, str]:
        """Parse risk level from safety assessment."""
        text = safety_text.lower()
        
        if any(w in text for w in ['very dangerous', 'extremely', 'critical', 'emergency', 'collision']):
            return 'critical', safety_text
        elif any(w in text for w in ['dangerous', 'hazard', 'risk', 'unsafe', 'caution needed']):
            return 'high', safety_text
        elif any(w in text for w in ['moderate', 'some risk', 'attention', 'careful']):
            return 'medium', safety_text
        else:
            return 'low', safety_text
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated tags."""
        if not self.tag_history:
            return {}
        
        all_tags = []
        for tags in self.tag_history:
            all_tags.extend(tags.extracted_tags)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_frames': len(self.tag_history),
            'unique_tags': len(tag_counts),
            'tag_frequency': dict(sorted_tags[:20]),
            'frames_with_risk': sum(1 for t in self.tag_history if t.risk_level != 'low')
        }
    
    def search_by_description(self, query: str) -> List[VLMTags]:
        """Search for frames matching a natural language query."""
        query_lower = query.lower()
        results = []
        
        for tags in self.tag_history:
            if query_lower in tags.scene_description.lower():
                results.append(tags)
            elif query_lower in tags.safety_assessment.lower():
                results.append(tags)
            elif any(query_lower in tag for tag in tags.extracted_tags):
                results.append(tags)
        
        return results
    
    def reset(self):
        """Reset the tagger state."""
        self.frame_count = 0
        self.tag_history = []
        self._last_tags = None

