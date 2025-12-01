"""
Motion Planning Module

Implements trajectory generation and motion planning visualization.
Generates candidate trajectories and selects optimal paths.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline


@dataclass
class Waypoint:
    """Represents a single waypoint in a trajectory."""
    x: float
    y: float
    heading: float  # radians
    velocity: float  # m/s
    timestamp: float  # seconds
    curvature: float = 0.0


@dataclass
class Trajectory:
    """Represents a planned trajectory."""
    waypoints: List[Waypoint]
    cost: float = 0.0
    is_feasible: bool = True
    trajectory_type: str = "nominal"  # "nominal", "left_lane", "right_lane", etc.
    
    @property
    def length(self) -> float:
        """Total trajectory length in meters."""
        if len(self.waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.waypoints)):
            dx = self.waypoints[i].x - self.waypoints[i-1].x
            dy = self.waypoints[i].y - self.waypoints[i-1].y
            total += np.sqrt(dx**2 + dy**2)
        return total
    
    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        if not self.waypoints:
            return 0.0
        return self.waypoints[-1].timestamp - self.waypoints[0].timestamp
    
    def get_positions(self) -> np.ndarray:
        """Get trajectory positions as numpy array."""
        return np.array([[wp.x, wp.y] for wp in self.waypoints])


class MotionPlanner:
    """
    Motion planner for autonomous driving.
    
    Features:
    - Generate candidate trajectories
    - Trajectory cost evaluation
    - Collision checking (simplified)
    - Optimal trajectory selection
    """
    
    def __init__(self, 
                 planning_horizon: float = 5.0,
                 dt: float = 0.1,
                 num_samples: int = 7):
        """
        Initialize motion planner.
        
        Args:
            planning_horizon: Planning time horizon in seconds
            dt: Time step for trajectory discretization
            num_samples: Number of lateral samples for trajectory generation
        """
        self.planning_horizon = planning_horizon
        self.dt = dt
        self.num_samples = num_samples
        
        # Cost weights
        self.w_lateral = 1.0  # Lateral deviation cost
        self.w_velocity = 0.5  # Velocity deviation cost
        self.w_acceleration = 0.3  # Acceleration cost
        self.w_jerk = 0.2  # Jerk cost
        self.w_curvature = 0.4  # Curvature cost
        
        self.reference_trajectory: Optional[Trajectory] = None
        
    def set_reference_path(self, waypoints: List[Tuple[float, float]]):
        """
        Set reference path for planning.
        
        Args:
            waypoints: List of (x, y) waypoints for reference path
        """
        if len(waypoints) < 2:
            return
        
        # Create reference trajectory with default velocities
        ref_waypoints = []
        for i, (x, y) in enumerate(waypoints):
            heading = 0.0
            if i < len(waypoints) - 1:
                dx = waypoints[i+1][0] - x
                dy = waypoints[i+1][1] - y
                heading = np.arctan2(dy, dx)
            elif i > 0:
                dx = x - waypoints[i-1][0]
                dy = y - waypoints[i-1][1]
                heading = np.arctan2(dy, dx)
            
            ref_waypoints.append(Waypoint(
                x=x, y=y, heading=heading, 
                velocity=10.0, timestamp=i * 0.5
            ))
        
        self.reference_trajectory = Trajectory(
            waypoints=ref_waypoints,
            trajectory_type="reference"
        )
    
    def generate_polynomial_trajectory(self, 
                                        start_state: Tuple[float, float, float, float],
                                        end_lateral_offset: float,
                                        target_velocity: float) -> Trajectory:
        """
        Generate a polynomial trajectory from start state.
        
        Args:
            start_state: (x, y, heading, velocity)
            end_lateral_offset: Lateral offset from reference at end
            target_velocity: Target velocity
            
        Returns:
            Generated trajectory
        """
        x0, y0, heading0, v0 = start_state
        
        n_points = int(self.planning_horizon / self.dt) + 1
        timestamps = np.linspace(0, self.planning_horizon, n_points)
        
        # Generate smooth longitudinal motion
        s_values = np.zeros(n_points)
        velocities = np.zeros(n_points)
        
        # Smooth velocity transition
        for i, t in enumerate(timestamps):
            # Smooth velocity profile
            alpha = 1 - np.exp(-t)  # Exponential approach
            velocities[i] = v0 + (target_velocity - v0) * alpha
            
            if i > 0:
                s_values[i] = s_values[i-1] + velocities[i] * self.dt
        
        # Generate lateral offset profile (quintic polynomial)
        d0, d1, d2 = 0.0, 0.0, 0.0  # Initial lateral state
        df = end_lateral_offset  # Final lateral offset
        
        lateral_offsets = np.zeros(n_points)
        for i, t in enumerate(timestamps):
            # Quintic polynomial for smooth lateral motion
            tau = t / self.planning_horizon
            tau = np.clip(tau, 0, 1)
            # Smooth transition using quintic
            lateral_offsets[i] = df * (10*tau**3 - 15*tau**4 + 6*tau**5)
        
        # Convert to global coordinates
        waypoints = []
        for i, t in enumerate(timestamps):
            # Position along reference direction
            x = x0 + s_values[i] * np.cos(heading0)
            y = y0 + s_values[i] * np.sin(heading0)
            
            # Add lateral offset (perpendicular to heading)
            x += lateral_offsets[i] * np.cos(heading0 + np.pi/2)
            y += lateral_offsets[i] * np.sin(heading0 + np.pi/2)
            
            # Compute heading from trajectory tangent
            if i < n_points - 1:
                next_x = x0 + s_values[min(i+1, n_points-1)] * np.cos(heading0)
                next_y = y0 + s_values[min(i+1, n_points-1)] * np.sin(heading0)
                next_x += lateral_offsets[min(i+1, n_points-1)] * np.cos(heading0 + np.pi/2)
                next_y += lateral_offsets[min(i+1, n_points-1)] * np.sin(heading0 + np.pi/2)
                heading = np.arctan2(next_y - y, next_x - x)
            else:
                heading = waypoints[-1].heading if waypoints else heading0
            
            # Compute curvature
            curvature = 0.0
            if i > 0 and i < n_points - 1:
                prev_heading = waypoints[-1].heading
                curvature = (heading - prev_heading) / (velocities[i] * self.dt + 1e-6)
            
            waypoints.append(Waypoint(
                x=x, y=y, heading=heading,
                velocity=velocities[i], timestamp=t,
                curvature=curvature
            ))
        
        return Trajectory(waypoints=waypoints)
    
    def evaluate_trajectory_cost(self, trajectory: Trajectory,
                                  obstacles: Optional[List[Tuple[float, float, float]]] = None) -> float:
        """
        Evaluate trajectory cost.
        
        Args:
            trajectory: Trajectory to evaluate
            obstacles: List of (x, y, radius) obstacles
            
        Returns:
            Total cost (lower is better)
        """
        if not trajectory.waypoints:
            return float('inf')
        
        cost = 0.0
        
        # Lateral deviation from reference
        if self.reference_trajectory:
            ref_positions = self.reference_trajectory.get_positions()
            traj_positions = trajectory.get_positions()
            
            # Sample-wise distance to reference
            for pos in traj_positions:
                min_dist = np.min(np.linalg.norm(ref_positions - pos, axis=1))
                cost += self.w_lateral * min_dist**2
        
        # Velocity cost (deviation from target)
        target_velocity = 10.0  # m/s
        for wp in trajectory.waypoints:
            cost += self.w_velocity * (wp.velocity - target_velocity)**2
        
        # Acceleration/jerk cost
        for i in range(1, len(trajectory.waypoints)):
            dt = trajectory.waypoints[i].timestamp - trajectory.waypoints[i-1].timestamp
            if dt > 0:
                accel = (trajectory.waypoints[i].velocity - 
                        trajectory.waypoints[i-1].velocity) / dt
                cost += self.w_acceleration * accel**2
        
        # Curvature cost
        for wp in trajectory.waypoints:
            cost += self.w_curvature * wp.curvature**2
        
        # Obstacle cost
        if obstacles:
            traj_positions = trajectory.get_positions()
            for ox, oy, radius in obstacles:
                for pos in traj_positions:
                    dist = np.sqrt((pos[0] - ox)**2 + (pos[1] - oy)**2)
                    if dist < radius * 2:
                        cost += 1000 * (radius * 2 - dist)  # High penalty for collision
                    elif dist < radius * 4:
                        cost += 10 / (dist - radius + 0.1)  # Soft penalty for proximity
        
        trajectory.cost = cost
        return cost
    
    def plan(self, current_state: Tuple[float, float, float, float],
             obstacles: Optional[List[Tuple[float, float, float]]] = None) -> Tuple[Trajectory, List[Trajectory]]:
        """
        Generate optimal trajectory from current state.
        
        Args:
            current_state: (x, y, heading, velocity)
            obstacles: List of (x, y, radius) obstacles
            
        Returns:
            (optimal_trajectory, all_candidate_trajectories)
        """
        candidates = []
        
        # Generate candidate trajectories with different lateral offsets
        lateral_offsets = np.linspace(-3.5, 3.5, self.num_samples)  # Lane width ~3.5m
        target_velocities = [8.0, 10.0, 12.0]  # Different speed options
        
        for lat_offset in lateral_offsets:
            for target_vel in target_velocities:
                traj = self.generate_polynomial_trajectory(
                    current_state, lat_offset, target_vel
                )
                
                # Label trajectory type
                if abs(lat_offset) < 0.5:
                    traj.trajectory_type = "lane_keep"
                elif lat_offset < 0:
                    traj.trajectory_type = "lane_change_left"
                else:
                    traj.trajectory_type = "lane_change_right"
                
                self.evaluate_trajectory_cost(traj, obstacles)
                candidates.append(traj)
        
        # Select best trajectory
        candidates.sort(key=lambda t: t.cost)
        optimal = candidates[0] if candidates else None
        
        return optimal, candidates
    
    def draw_trajectories(self, frame: np.ndarray,
                          optimal: Optional[Trajectory],
                          candidates: List[Trajectory],
                          transform_func=None,
                          draw_all: bool = True) -> np.ndarray:
        """
        Draw trajectories on frame.
        
        Args:
            frame: Input image
            optimal: Optimal trajectory to highlight
            candidates: All candidate trajectories
            transform_func: Function to transform world coords to pixel coords
            draw_all: Whether to draw all candidates
            
        Returns:
            Annotated frame
        """
        import cv2
        
        annotated = frame.copy()
        
        # Default transform (identity with scaling)
        if transform_func is None:
            h, w = frame.shape[:2]
            def transform_func(x, y):
                px = int(w/2 + x * 10)
                py = int(h - y * 10 - 50)
                return px, py
        
        # Draw candidate trajectories (faded)
        if draw_all:
            for traj in candidates:
                if traj == optimal:
                    continue
                    
                positions = traj.get_positions()
                if len(positions) < 2:
                    continue
                
                # Color based on cost (green = good, red = bad)
                max_cost = max(t.cost for t in candidates) + 1
                cost_ratio = traj.cost / max_cost
                color = (0, int(255 * (1 - cost_ratio)), int(255 * cost_ratio))
                
                points = np.array([transform_func(p[0], p[1]) for p in positions])
                points = points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(annotated, [points], False, color, 1)
        
        # Draw optimal trajectory (highlighted)
        if optimal:
            positions = optimal.get_positions()
            if len(positions) >= 2:
                points = np.array([transform_func(p[0], p[1]) for p in positions])
                points = points.reshape((-1, 1, 2)).astype(np.int32)
                
                # Draw with glow effect
                cv2.polylines(annotated, [points], False, (0, 255, 0), 4)
                cv2.polylines(annotated, [points], False, (100, 255, 100), 2)
                
                # Draw waypoints
                for i, wp in enumerate(optimal.waypoints[::5]):  # Every 5th waypoint
                    px, py = transform_func(wp.x, wp.y)
                    cv2.circle(annotated, (px, py), 3, (255, 255, 0), -1)
        
        return annotated
    
    def reset(self):
        """Reset planner state."""
        self.reference_trajectory = None

