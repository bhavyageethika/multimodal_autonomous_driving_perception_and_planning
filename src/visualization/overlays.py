"""
Overlay Renderer Module

Renders overlays on camera images including:
- Detection boxes
- Lane lines
- State information
- Planning visualizations
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class OverlayRenderer:
    """
    Renders various overlays on camera frames.
    """
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
    def draw_info_panel(self, frame: np.ndarray, 
                        vehicle_state=None,
                        fps: float = 0.0,
                        frame_num: int = 0) -> np.ndarray:
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw info text
        y_offset = 30
        line_height = 20
        
        info_lines = [
            f"Frame: {frame_num}",
            f"FPS: {fps:.1f}",
        ]
        
        if vehicle_state:
            info_lines.extend([
                f"Speed: {vehicle_state.speed * 3.6:.1f} km/h",
                f"Heading: {np.degrees(vehicle_state.heading):.1f} deg",
                f"Accel: {vehicle_state.acceleration:.2f} m/s2",
                f"Pos: ({vehicle_state.x:.1f}, {vehicle_state.y:.1f})",
            ])
        
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       self.font, self.font_scale, (255, 255, 255), 
                       self.font_thickness)
            y_offset += line_height
        
        return frame
    
    def draw_detection_summary(self, frame: np.ndarray,
                                detections: List,
                                position: str = "top_right") -> np.ndarray:
        """Draw detection count summary."""
        h, w = frame.shape[:2]
        
        # Count by class
        class_counts = {}
        for det in detections:
            class_name = det.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Position
        if position == "top_right":
            x_start = w - 150
            y_start = 10
        else:
            x_start = 10
            y_start = h - 100
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + 140, y_start + 20 + len(class_counts) * 18),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw counts
        cv2.putText(frame, "Detections:", (x_start + 5, y_start + 15),
                   self.font, 0.4, (255, 255, 255), 1)
        
        y_offset = y_start + 35
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"  {class_name}: {count}", 
                       (x_start + 5, y_offset),
                       self.font, 0.35, (200, 200, 200), 1)
            y_offset += 18
        
        return frame
    
    def draw_lane_offset_indicator(self, frame: np.ndarray,
                                    offset: Optional[float]) -> np.ndarray:
        """Draw lane center offset indicator."""
        h, w = frame.shape[:2]
        
        # Draw at bottom center
        indicator_w = 200
        indicator_h = 30
        x_start = (w - indicator_w) // 2
        y_start = h - 50
        
        # Background
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + indicator_w, y_start + indicator_h),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + indicator_w, y_start + indicator_h),
                     (100, 100, 100), 1)
        
        # Center line
        center_x = x_start + indicator_w // 2
        cv2.line(frame, (center_x, y_start), (center_x, y_start + indicator_h),
                (255, 255, 255), 1)
        
        if offset is not None:
            # Draw offset indicator
            max_offset = 100  # pixels
            offset_px = int(np.clip(offset, -max_offset, max_offset))
            indicator_x = center_x + offset_px
            
            # Color based on offset magnitude
            if abs(offset) < 20:
                color = (0, 255, 0)  # Green - centered
            elif abs(offset) < 50:
                color = (0, 255, 255)  # Yellow - slight offset
            else:
                color = (0, 0, 255)  # Red - large offset
            
            cv2.circle(frame, (indicator_x, y_start + indicator_h // 2),
                      8, color, -1)
            
            cv2.putText(frame, f"Offset: {offset:.0f}px", 
                       (x_start + 5, y_start - 5),
                       self.font, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_tracking_stats(self, frame: np.ndarray,
                            tracks: List,
                            position: str = "bottom_left") -> np.ndarray:
        """Draw tracking statistics."""
        h, w = frame.shape[:2]
        
        if position == "bottom_left":
            x_start = 10
            y_start = h - 80
        else:
            x_start = w - 150
            y_start = h - 80
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + 140, y_start + 70),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Stats
        active_tracks = len(tracks)
        avg_age = np.mean([t.age for t in tracks]) if tracks else 0
        
        cv2.putText(frame, "Tracking Stats:", (x_start + 5, y_start + 15),
                   self.font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"  Active: {active_tracks}", 
                   (x_start + 5, y_start + 35),
                   self.font, 0.35, (200, 200, 200), 1)
        cv2.putText(frame, f"  Avg Age: {avg_age:.0f} frames", 
                   (x_start + 5, y_start + 55),
                   self.font, 0.35, (200, 200, 200), 1)
        
        return frame
    
    def create_side_by_side(self, frame1: np.ndarray, 
                            frame2: np.ndarray,
                            labels: Tuple[str, str] = ("Camera", "BEV")) -> np.ndarray:
        """Create side-by-side view of two frames."""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize to same height
        target_h = max(h1, h2)
        if h1 != target_h:
            scale = target_h / h1
            frame1 = cv2.resize(frame1, (int(w1 * scale), target_h))
        if h2 != target_h:
            scale = target_h / h2
            frame2 = cv2.resize(frame2, (int(w2 * scale), target_h))
        
        # Concatenate
        combined = np.hstack([frame1, frame2])
        
        # Add labels
        cv2.putText(combined, labels[0], (10, 25),
                   self.font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, labels[1], (frame1.shape[1] + 10, 25),
                   self.font, 0.6, (255, 255, 255), 2)
        
        return combined

