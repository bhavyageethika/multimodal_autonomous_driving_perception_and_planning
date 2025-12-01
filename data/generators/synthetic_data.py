"""
Synthetic Data Generator

Generates synthetic driving scenarios for demonstration:
- Synthetic video frames
- Agent trajectories
- Sensor measurements
"""

import numpy as np
import cv2
from typing import Tuple, List, Generator


class SyntheticDataGenerator:
    """
    Generates synthetic driving data for testing and demonstration.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps
        self.frame_count = 0
        
    def generate_road_frame(self) -> np.ndarray:
        """Generate a synthetic road scene frame."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(self.height // 2):
            ratio = y / (self.height // 2)
            color = (int(200 - 80 * ratio), int(180 - 60 * ratio), int(255 - 55 * ratio))
            cv2.line(frame, (0, y), (self.width, y), color, 1)
        
        # Ground/road
        road_color = (60, 60, 60)
        cv2.rectangle(frame, (0, self.height // 2), (self.width, self.height), road_color, -1)
        
        # Horizon line
        horizon_y = self.height // 2
        
        # Vanishing point
        vp_x = self.width // 2 + int(20 * np.sin(self.frame_count * 0.02))
        vp_y = horizon_y
        
        # Road edges (perspective)
        left_bottom = (50, self.height)
        right_bottom = (self.width - 50, self.height)
        
        # Draw road surface
        road_pts = np.array([
            [vp_x, vp_y],
            [left_bottom[0], left_bottom[1]],
            [right_bottom[0], right_bottom[1]]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [road_pts], (80, 80, 80))
        
        # Lane markings
        self._draw_lane_markings(frame, vp_x, vp_y)
        
        # Add some visual elements
        self._draw_environment(frame, horizon_y)
        
        return frame
    
    def _draw_lane_markings(self, frame: np.ndarray, vp_x: int, vp_y: int):
        """Draw perspective lane markings."""
        # Center dashed line
        num_dashes = 10
        for i in range(num_dashes):
            t1 = i / num_dashes
            t2 = (i + 0.5) / num_dashes
            
            # Interpolate from vanishing point
            y1 = int(vp_y + (self.height - vp_y) * t1)
            y2 = int(vp_y + (self.height - vp_y) * t2)
            
            # Apply animation offset
            offset = (self.frame_count * 5) % (self.height // num_dashes)
            y1 = min(y1 + offset, self.height)
            y2 = min(y2 + offset, self.height)
            
            if y1 >= vp_y and y2 >= vp_y:
                cv2.line(frame, (vp_x, y1), (vp_x, y2), (255, 255, 200), 2)
        
        # Side lane markings
        for side in [-1, 1]:
            offset_x = side * 150
            for i in range(num_dashes):
                t1 = i / num_dashes
                t2 = (i + 0.6) / num_dashes
                
                y1 = int(vp_y + (self.height - vp_y) * t1)
                y2 = int(vp_y + (self.height - vp_y) * t2)
                
                x1 = int(vp_x + offset_x * t1)
                x2 = int(vp_x + offset_x * t2)
                
                offset = (self.frame_count * 5) % (self.height // num_dashes)
                y1 = min(y1 + offset, self.height)
                y2 = min(y2 + offset, self.height)
                
                if y1 >= vp_y and y2 >= vp_y:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    def _draw_environment(self, frame: np.ndarray, horizon_y: int):
        """Draw environmental elements."""
        # Simple trees/poles on sides
        for i in range(5):
            t = (i + 0.5) / 5
            y_base = int(horizon_y + (self.height - horizon_y) * t * 0.8)
            
            # Left side
            x_left = int(30 + 50 * t)
            height = int(30 + 40 * t)
            cv2.line(frame, (x_left, y_base), (x_left, y_base - height), (80, 50, 30), 2)
            cv2.circle(frame, (x_left, y_base - height - 10), int(15 * t + 5), (50, 120, 50), -1)
            
            # Right side
            x_right = self.width - int(30 + 50 * t)
            cv2.line(frame, (x_right, y_base), (x_right, y_base - height), (80, 50, 30), 2)
            cv2.circle(frame, (x_right, y_base - height - 10), int(15 * t + 5), (50, 120, 50), -1)
    
    def generate_vehicle(self, frame: np.ndarray, 
                         x: int, y: int, 
                         scale: float = 1.0,
                         color: Tuple[int, int, int] = (0, 100, 200)) -> np.ndarray:
        """Draw a simple vehicle on the frame."""
        w = int(60 * scale)
        h = int(40 * scale)
        
        # Car body
        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, -1)
        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 0), 1)
        
        # Windshield
        cv2.rectangle(frame, (x - w//3, y - h//2), (x + w//3, y - h//4), (100, 100, 100), -1)
        
        # Wheels
        wheel_r = int(8 * scale)
        cv2.circle(frame, (x - w//3, y + h//2), wheel_r, (30, 30, 30), -1)
        cv2.circle(frame, (x + w//3, y + h//2), wheel_r, (30, 30, 30), -1)
        
        return frame
    
    def generate_frame_with_vehicles(self) -> np.ndarray:
        """Generate a complete frame with road and vehicles."""
        frame = self.generate_road_frame()
        
        # Add some vehicles
        np.random.seed(self.frame_count % 100)
        num_vehicles = np.random.randint(2, 5)
        
        for i in range(num_vehicles):
            # Random position with perspective
            t = np.random.uniform(0.2, 0.9)
            y = int(self.height // 2 + (self.height // 2) * t)
            
            # Random lateral position
            lane_offset = np.random.choice([-80, 0, 80])
            x = self.width // 2 + int(lane_offset * t) + np.random.randint(-20, 20)
            
            # Add temporal variation
            x += int(30 * np.sin(self.frame_count * 0.05 + i))
            
            # Scale based on distance
            scale = 0.3 + 0.7 * t
            
            # Random color
            colors = [(0, 100, 200), (200, 50, 50), (50, 200, 50), (200, 200, 50)]
            color = colors[i % len(colors)]
            
            self.generate_vehicle(frame, x, y, scale, color)
        
        self.frame_count += 1
        return frame
    
    def generate_video_stream(self, num_frames: int = 300) -> Generator[np.ndarray, None, None]:
        """Generate a stream of video frames."""
        self.frame_count = 0
        for _ in range(num_frames):
            yield self.generate_frame_with_vehicles()
    
    def generate_ego_motion(self, num_steps: int = 300) -> List[Tuple[float, float, float, float]]:
        """
        Generate simulated ego vehicle motion.
        
        Returns:
            List of (x, y, vx, vy) measurements
        """
        measurements = []
        x, y = 0.0, 0.0
        heading = 0.0
        speed = 10.0  # m/s
        
        for i in range(num_steps):
            t = i * self.dt
            
            # Simulate lane changes and curves
            heading = 0.1 * np.sin(t * 0.3) + 0.05 * np.sin(t * 0.7)
            speed = 10 + 3 * np.sin(t * 0.2)
            
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            
            x += vx * self.dt
            y += vy * self.dt
            
            # Add measurement noise
            measurements.append((
                x + np.random.normal(0, 0.3),
                y + np.random.normal(0, 0.3),
                vx + np.random.normal(0, 0.1),
                vy + np.random.normal(0, 0.1)
            ))
        
        return measurements
    
    def generate_agent_trajectories(self, num_agents: int = 5, 
                                     num_steps: int = 100) -> dict:
        """
        Generate simulated agent trajectories.
        
        Returns:
            Dictionary mapping agent_id to list of (x, y, vx, vy)
        """
        trajectories = {}
        
        for agent_id in range(num_agents):
            # Random starting position
            x = np.random.uniform(-20, 20)
            y = np.random.uniform(10, 40)
            heading = np.random.uniform(-0.3, 0.3)
            speed = np.random.uniform(5, 15)
            
            agent_traj = []
            for i in range(num_steps):
                # Simple motion model
                heading += np.random.normal(0, 0.02)
                speed += np.random.normal(0, 0.1)
                speed = np.clip(speed, 3, 20)
                
                vx = speed * np.cos(heading)
                vy = speed * np.sin(heading)
                
                x += vx * self.dt
                y += vy * self.dt
                
                agent_traj.append((x, y, vx, vy))
            
            trajectories[agent_id] = agent_traj
        
        return trajectories
    
    def reset(self):
        """Reset generator state."""
        self.frame_count = 0

