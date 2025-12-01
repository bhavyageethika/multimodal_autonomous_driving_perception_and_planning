"""
Bird's Eye View (BEV) Renderer

Renders top-down view of the driving scene including:
- Ego vehicle
- Other agents
- Trajectories
- Lane markings
- Planned paths
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict


class BEVRenderer:
    """
    Renders bird's eye view visualization of driving scene.
    
    Features:
    - Configurable view range and resolution
    - Ego vehicle rendering
    - Agent rendering with trajectories
    - Road/lane visualization
    - Motion planning overlay
    """
    
    def __init__(self, 
                 width: int = 600,
                 height: int = 600,
                 pixels_per_meter: float = 10.0,
                 x_range: Tuple[float, float] = (-30, 30),
                 y_range: Tuple[float, float] = (-10, 50)):
        """
        Initialize BEV renderer.
        
        Args:
            width: Output image width in pixels
            height: Output image height in pixels
            pixels_per_meter: Scale factor
            x_range: (min, max) x range in meters (lateral)
            y_range: (min, max) y range in meters (longitudinal)
        """
        self.width = width
        self.height = height
        self.pixels_per_meter = pixels_per_meter
        self.x_range = x_range
        self.y_range = y_range
        
        # Compute transformation parameters
        self.x_scale = width / (x_range[1] - x_range[0])
        self.y_scale = height / (y_range[1] - y_range[0])
        
        # Colors
        self.bg_color = (40, 40, 40)  # Dark gray background
        self.road_color = (60, 60, 60)  # Road surface
        self.lane_color = (200, 200, 200)  # Lane markings
        self.ego_color = (0, 200, 255)  # Ego vehicle (orange)
        self.agent_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            x: World x coordinate (lateral, right is positive)
            y: World y coordinate (longitudinal, forward is positive)
            
        Returns:
            (px, py) pixel coordinates
        """
        px = int((x - self.x_range[0]) * self.x_scale)
        py = int(self.height - (y - self.y_range[0]) * self.y_scale)
        return px, py
    
    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        x = px / self.x_scale + self.x_range[0]
        y = (self.height - py) / self.y_scale + self.y_range[0]
        return x, y
    
    def create_base_image(self) -> np.ndarray:
        """Create base BEV image with road."""
        img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        # Draw road surface
        road_left = self.world_to_pixel(-7, self.y_range[0])[0]
        road_right = self.world_to_pixel(7, self.y_range[0])[0]
        cv2.rectangle(img, (road_left, 0), (road_right, self.height), self.road_color, -1)
        
        # Draw lane markings
        for lane_x in [-3.5, 0, 3.5]:
            px, _ = self.world_to_pixel(lane_x, 0)
            
            if lane_x == 0:
                # Center line (dashed yellow)
                for y in range(0, self.height, 30):
                    cv2.line(img, (px, y), (px, min(y + 15, self.height)), 
                            (0, 200, 200), 2)
            else:
                # Lane markers (dashed white)
                for y in range(0, self.height, 40):
                    cv2.line(img, (px, y), (px, min(y + 20, self.height)), 
                            self.lane_color, 2)
        
        # Draw road edges
        for edge_x in [-7, 7]:
            px, _ = self.world_to_pixel(edge_x, 0)
            cv2.line(img, (px, 0), (px, self.height), (255, 255, 255), 2)
        
        return img
    
    def draw_vehicle(self, img: np.ndarray, 
                     x: float, y: float, heading: float,
                     color: Tuple[int, int, int],
                     length: float = 4.5, width: float = 2.0,
                     label: str = None) -> np.ndarray:
        """
        Draw a vehicle on the BEV.
        
        Args:
            img: Image to draw on
            x, y: Vehicle center position in world coordinates
            heading: Vehicle heading in radians
            color: RGB color tuple
            length: Vehicle length in meters
            width: Vehicle width in meters
            label: Optional label to display
            
        Returns:
            Annotated image
        """
        # Compute corners in world coordinates
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        half_l = length / 2
        half_w = width / 2
        
        corners = [
            (x + half_l * cos_h - half_w * sin_h, y + half_l * sin_h + half_w * cos_h),
            (x + half_l * cos_h + half_w * sin_h, y + half_l * sin_h - half_w * cos_h),
            (x - half_l * cos_h + half_w * sin_h, y - half_l * sin_h - half_w * cos_h),
            (x - half_l * cos_h - half_w * sin_h, y - half_l * sin_h + half_w * cos_h),
        ]
        
        # Convert to pixels
        pixel_corners = [self.world_to_pixel(cx, cy) for cx, cy in corners]
        pts = np.array(pixel_corners, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 1)
        
        # Draw heading indicator (arrow at front)
        front_x = x + half_l * cos_h
        front_y = y + half_l * sin_h
        front_px, front_py = self.world_to_pixel(front_x, front_y)
        center_px, center_py = self.world_to_pixel(x, y)
        cv2.arrowedLine(img, (center_px, center_py), (front_px, front_py),
                       (255, 255, 255), 2, tipLength=0.5)
        
        # Draw label
        if label:
            cv2.putText(img, label, (center_px - 20, center_py - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def draw_ego_vehicle(self, img: np.ndarray, state) -> np.ndarray:
        """Draw ego vehicle at its current state."""
        return self.draw_vehicle(
            img, state.x, state.y, state.heading,
            self.ego_color, label="EGO"
        )
    
    def draw_agents(self, img: np.ndarray, 
                    tracks: List,
                    draw_trajectories: bool = True) -> np.ndarray:
        """
        Draw tracked agents on BEV.
        
        Args:
            img: Image to draw on
            tracks: List of Track objects
            draw_trajectories: Whether to draw trajectory trails
            
        Returns:
            Annotated image
        """
        for i, track in enumerate(tracks):
            color = self.agent_colors[track.track_id % len(self.agent_colors)]
            
            # Convert image bbox center to world coordinates (approximate)
            cx, cy = track.center
            
            # Map from image space to BEV world space (simplified mapping)
            # Assumes camera is forward-facing and bbox y correlates with distance
            world_y = 50 - cy * 0.1  # Closer objects are lower in image
            world_x = (cx - 320) * 0.03  # Lateral position
            
            # Draw vehicle
            self.draw_vehicle(img, world_x, world_y, 0, color, 
                            length=3.0, width=1.5, label=f"ID:{track.track_id}")
            
            # Draw trajectory
            if draw_trajectories and len(track.trajectory) > 1:
                for j in range(1, len(track.trajectory)):
                    # Convert trajectory points
                    prev_cx, prev_cy = track.trajectory[j-1]
                    curr_cx, curr_cy = track.trajectory[j]
                    
                    prev_world_y = 50 - prev_cy * 0.1
                    prev_world_x = (prev_cx - 320) * 0.03
                    curr_world_y = 50 - curr_cy * 0.1
                    curr_world_x = (curr_cx - 320) * 0.03
                    
                    prev_px, prev_py = self.world_to_pixel(prev_world_x, prev_world_y)
                    curr_px, curr_py = self.world_to_pixel(curr_world_x, curr_world_y)
                    
                    # Fade effect
                    alpha = j / len(track.trajectory)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(img, (prev_px, prev_py), (curr_px, curr_py), 
                            color, thickness)
        
        return img
    
    def draw_trajectory(self, img: np.ndarray,
                        trajectory,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2,
                        draw_waypoints: bool = True) -> np.ndarray:
        """
        Draw a planned trajectory on BEV.
        
        Args:
            img: Image to draw on
            trajectory: Trajectory object
            color: Line color
            thickness: Line thickness
            draw_waypoints: Whether to draw waypoint markers
            
        Returns:
            Annotated image
        """
        if not trajectory or not trajectory.waypoints:
            return img
        
        positions = trajectory.get_positions()
        if len(positions) < 2:
            return img
        
        # Draw path
        points = [self.world_to_pixel(p[0], p[1]) for p in positions]
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, color, thickness)
        
        # Draw waypoints
        if draw_waypoints:
            for i, wp in enumerate(trajectory.waypoints[::3]):  # Every 3rd point
                px, py = self.world_to_pixel(wp.x, wp.y)
                cv2.circle(img, (px, py), 3, color, -1)
        
        return img
    
    def draw_uncertainty_ellipse(self, img: np.ndarray,
                                  x: float, y: float,
                                  uncertainty: float,
                                  color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Draw uncertainty ellipse at position."""
        px, py = self.world_to_pixel(x, y)
        radius = int(uncertainty * self.pixels_per_meter)
        if radius > 0:
            cv2.ellipse(img, (px, py), (radius, radius), 0, 0, 360, color, 1)
        return img
    
    def render(self, 
               ego_state=None,
               tracks: List = None,
               planned_trajectory=None,
               candidate_trajectories: List = None,
               show_grid: bool = False) -> np.ndarray:
        """
        Render complete BEV scene.
        
        Args:
            ego_state: VehicleState for ego vehicle
            tracks: List of tracked agents
            planned_trajectory: Optimal planned trajectory
            candidate_trajectories: All candidate trajectories
            show_grid: Whether to show distance grid
            
        Returns:
            Rendered BEV image
        """
        img = self.create_base_image()
        
        # Draw grid
        if show_grid:
            for x in range(-30, 31, 10):
                px, _ = self.world_to_pixel(x, 0)
                cv2.line(img, (px, 0), (px, self.height), (50, 50, 50), 1)
                cv2.putText(img, f"{x}m", (px, self.height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            for y in range(-10, 51, 10):
                _, py = self.world_to_pixel(0, y)
                cv2.line(img, (0, py), (self.width, py), (50, 50, 50), 1)
                cv2.putText(img, f"{y}m", (5, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw candidate trajectories (faded)
        if candidate_trajectories:
            for traj in candidate_trajectories:
                if traj != planned_trajectory:
                    self.draw_trajectory(img, traj, (80, 80, 80), 1, False)
        
        # Draw planned trajectory
        if planned_trajectory:
            self.draw_trajectory(img, planned_trajectory, (0, 255, 0), 3, True)
        
        # Draw tracked agents
        if tracks:
            self.draw_agents(img, tracks)
        
        # Draw ego vehicle
        if ego_state:
            self.draw_ego_vehicle(img, ego_state)
            
            # Draw uncertainty
            if hasattr(ego_state, 'pos_uncertainty'):
                self.draw_uncertainty_ellipse(
                    img, ego_state.x, ego_state.y, 
                    ego_state.pos_uncertainty
                )
        
        # Add legend
        self._draw_legend(img)
        
        return img
    
    def _draw_legend(self, img: np.ndarray):
        """Draw legend on image."""
        legend_items = [
            ("EGO", self.ego_color),
            ("Planned", (0, 255, 0)),
            ("Agents", self.agent_colors[0]),
        ]
        
        y_offset = 20
        for label, color in legend_items:
            cv2.rectangle(img, (10, y_offset - 10), (25, y_offset + 5), color, -1)
            cv2.putText(img, label, (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20

