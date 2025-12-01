"""
Lane Detection Module

Provides lane line detection using classical computer vision techniques.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LaneLine:
    """Represents a detected lane line."""
    points: np.ndarray  # Array of (x, y) points
    side: str  # "left", "right", or "center"
    confidence: float
    polynomial: Optional[np.ndarray] = None  # Polynomial coefficients


class LaneDetector:
    """
    Lane detector using edge detection and Hough transform.
    
    Pipeline:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Canny edge detection
    4. Region of interest masking
    5. Hough line transform
    6. Lane line fitting
    """
    
    def __init__(self, roi_vertices: Optional[np.ndarray] = None):
        """
        Initialize lane detector.
        
        Args:
            roi_vertices: Custom region of interest vertices
        """
        self.roi_vertices = roi_vertices
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.smoothing_factor = 0.7
    
    def _get_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create region of interest mask."""
        h, w = shape[:2]
        
        if self.roi_vertices is not None:
            vertices = self.roi_vertices
        else:
            # Default trapezoid ROI for front-facing camera
            vertices = np.array([[
                (int(w * 0.1), h),           # Bottom left
                (int(w * 0.4), int(h * 0.6)), # Top left
                (int(w * 0.6), int(h * 0.6)), # Top right
                (int(w * 0.9), h)             # Bottom right
            ]], dtype=np.int32)
        
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for edge detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def _detect_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        # Adaptive thresholds based on image statistics
        median = np.median(preprocessed)
        low_thresh = int(max(0, 0.7 * median))
        high_thresh = int(min(255, 1.3 * median))
        
        edges = cv2.Canny(preprocessed, low_thresh, high_thresh)
        return edges
    
    def _apply_roi(self, edges: np.ndarray) -> np.ndarray:
        """Apply region of interest mask."""
        mask = self._get_roi_mask(edges.shape)
        masked = cv2.bitwise_and(edges, mask)
        return masked
    
    def _detect_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """Detect lines using Hough transform."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=150
        )
        
        return lines if lines is not None else []
    
    def _separate_lines(self, lines: List[np.ndarray], 
                        frame_width: int) -> Tuple[List, List]:
        """Separate lines into left and right based on slope."""
        left_lines = []
        right_lines = []
        
        center_x = frame_width / 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 == x1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out nearly horizontal lines
            if abs(slope) < 0.3:
                continue
            
            # Negative slope = left lane, positive slope = right lane
            # (y increases downward in image coordinates)
            midpoint_x = (x1 + x2) / 2
            
            if slope < 0 and midpoint_x < center_x:
                left_lines.append(line[0])
            elif slope > 0 and midpoint_x > center_x:
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def _fit_lane_line(self, lines: List, frame_height: int, 
                       prev_fit: Optional[np.ndarray] = None) -> Optional[LaneLine]:
        """Fit a polynomial to lane line points."""
        if not lines:
            return None
        
        # Collect all points
        x_coords = []
        y_coords = []
        
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        # Fit polynomial (y as function of x for drawing)
        try:
            # Fit x as function of y for vertical lanes
            coeffs = np.polyfit(y_coords, x_coords, 2)
            
            # Apply smoothing with previous fit
            if prev_fit is not None:
                coeffs = self.smoothing_factor * prev_fit + \
                         (1 - self.smoothing_factor) * coeffs
            
            # Generate lane points
            y_points = np.linspace(frame_height * 0.6, frame_height, 50)
            x_points = np.polyval(coeffs, y_points)
            
            points = np.column_stack((x_points, y_points)).astype(np.int32)
            
            return LaneLine(
                points=points,
                side="unknown",
                confidence=min(1.0, len(lines) / 10),
                polynomial=coeffs
            )
        except np.linalg.LinAlgError:
            return None
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[LaneLine], Optional[LaneLine]]:
        """
        Detect lane lines in frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Tuple of (left_lane, right_lane), either can be None
        """
        h, w = frame.shape[:2]
        
        # Preprocessing
        preprocessed = self._preprocess(frame)
        
        # Edge detection
        edges = self._detect_edges(preprocessed)
        
        # Apply ROI
        masked_edges = self._apply_roi(edges)
        
        # Detect lines
        lines = self._detect_lines(masked_edges)
        
        # Separate left and right
        left_lines, right_lines = self._separate_lines(lines, w)
        
        # Fit lane lines
        left_lane = self._fit_lane_line(left_lines, h, self.prev_left_fit)
        right_lane = self._fit_lane_line(right_lines, h, self.prev_right_fit)
        
        # Update previous fits
        if left_lane is not None:
            left_lane.side = "left"
            self.prev_left_fit = left_lane.polynomial
        
        if right_lane is not None:
            right_lane.side = "right"
            self.prev_right_fit = right_lane.polynomial
        
        return left_lane, right_lane
    
    def draw_lanes(self, frame: np.ndarray, 
                   left_lane: Optional[LaneLine],
                   right_lane: Optional[LaneLine],
                   fill_lane: bool = True) -> np.ndarray:
        """
        Draw detected lanes on frame.
        
        Args:
            frame: Input image
            left_lane: Left lane line
            right_lane: Right lane line
            fill_lane: Whether to fill the lane area
            
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        
        # Fill lane area
        if fill_lane and left_lane is not None and right_lane is not None:
            pts = np.vstack([left_lane.points, right_lane.points[::-1]])
            cv2.fillPoly(overlay, [pts], (0, 255, 100))
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw lane lines
        if left_lane is not None:
            cv2.polylines(frame, [left_lane.points], False, (255, 0, 0), 3)
        
        if right_lane is not None:
            cv2.polylines(frame, [right_lane.points], False, (0, 0, 255), 3)
        
        return frame
    
    def get_lane_center_offset(self, frame_width: int,
                                left_lane: Optional[LaneLine],
                                right_lane: Optional[LaneLine]) -> Optional[float]:
        """
        Calculate offset of vehicle from lane center.
        
        Returns:
            Offset in pixels (negative = left of center, positive = right)
        """
        if left_lane is None or right_lane is None:
            return None
        
        # Get x positions at bottom of frame
        left_x = left_lane.points[-1, 0]
        right_x = right_lane.points[-1, 0]
        
        lane_center = (left_x + right_x) / 2
        vehicle_center = frame_width / 2
        
        return vehicle_center - lane_center
    
    def reset(self):
        """Reset lane tracking state."""
        self.prev_left_fit = None
        self.prev_right_fit = None

