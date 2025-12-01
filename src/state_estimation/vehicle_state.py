"""
Vehicle State Estimation Module

Implements Kalman filter-based vehicle state estimation for ego-vehicle.
Estimates position, velocity, heading, and provides uncertainty bounds.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from filterpy.kalman import KalmanFilter


@dataclass
class VehicleState:
    """Represents estimated vehicle state."""
    x: float  # Position x (meters)
    y: float  # Position y (meters)
    vx: float  # Velocity x (m/s)
    vy: float  # Velocity y (m/s)
    heading: float  # Heading angle (radians)
    speed: float  # Total speed (m/s)
    acceleration: float  # Longitudinal acceleration (m/s^2)
    yaw_rate: float  # Yaw rate (rad/s)
    timestamp: float  # Time in seconds
    
    # Uncertainty bounds (1-sigma)
    pos_uncertainty: float = 0.0
    vel_uncertainty: float = 0.0
    heading_uncertainty: float = 0.0


class VehicleStateEstimator:
    """
    Kalman filter-based vehicle state estimator.
    
    State vector: [x, y, vx, vy, ax, ay]
    - x, y: Position
    - vx, vy: Velocity
    - ax, ay: Acceleration
    
    Features:
    - Prediction and update steps
    - State history with timestamps
    - Uncertainty estimation
    - Derived quantities (heading, speed, yaw rate)
    """
    
    def __init__(self, dt: float = 0.033, 
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0):
        """
        Initialize state estimator.
        
        Args:
            dt: Time step (seconds), default ~30 FPS
            process_noise: Process noise magnitude
            measurement_noise: Measurement noise magnitude
        """
        self.dt = dt
        self.kf = self._init_kalman_filter(dt, process_noise, measurement_noise)
        
        self.state_history: List[VehicleState] = []
        self.time = 0.0
        self.prev_heading = 0.0
        self.prev_speed = 0.0
        
    def _init_kalman_filter(self, dt: float, 
                            process_noise: float,
                            measurement_noise: float) -> KalmanFilter:
        """Initialize Kalman filter with constant acceleration model."""
        kf = KalmanFilter(dim_x=6, dim_z=4)
        
        # State transition matrix (constant acceleration model)
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we measure x, y, vx, vy)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(4) * measurement_noise
        
        # Process noise
        kf.Q = np.eye(6) * process_noise
        kf.Q[4, 4] *= 10  # Higher noise for acceleration
        kf.Q[5, 5] *= 10
        
        # Initial state covariance
        kf.P = np.eye(6) * 10
        
        # Initial state (starting at origin, stationary)
        kf.x = np.zeros(6)
        
        return kf
    
    def predict(self) -> VehicleState:
        """
        Perform prediction step.
        
        Returns:
            Predicted vehicle state
        """
        self.kf.predict()
        self.time += self.dt
        return self._extract_state()
    
    def update(self, measurement: np.ndarray) -> VehicleState:
        """
        Perform update step with new measurement.
        
        Args:
            measurement: [x, y, vx, vy] measurement vector
            
        Returns:
            Updated vehicle state
        """
        self.kf.update(measurement)
        state = self._extract_state()
        self.state_history.append(state)
        
        # Limit history size
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        return state
    
    def step(self, measurement: Optional[np.ndarray] = None) -> VehicleState:
        """
        Perform one estimation step (predict + optional update).
        
        Args:
            measurement: Optional measurement [x, y, vx, vy]
            
        Returns:
            Estimated vehicle state
        """
        self.predict()
        
        if measurement is not None:
            return self.update(measurement)
        else:
            state = self._extract_state()
            self.state_history.append(state)
            return state
    
    def _extract_state(self) -> VehicleState:
        """Extract VehicleState from Kalman filter state."""
        x, y, vx, vy, ax, ay = self.kf.x.flatten()
        
        # Compute derived quantities
        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx) if speed > 0.1 else self.prev_heading
        
        # Compute acceleration (longitudinal)
        acceleration = (speed - self.prev_speed) / self.dt if self.dt > 0 else 0
        
        # Compute yaw rate
        heading_diff = heading - self.prev_heading
        # Handle wrap-around
        if heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        elif heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        yaw_rate = heading_diff / self.dt if self.dt > 0 else 0
        
        # Extract uncertainties from covariance
        pos_uncertainty = np.sqrt(self.kf.P[0, 0] + self.kf.P[1, 1])
        vel_uncertainty = np.sqrt(self.kf.P[2, 2] + self.kf.P[3, 3])
        
        # Update previous values
        self.prev_heading = heading
        self.prev_speed = speed
        
        return VehicleState(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            heading=heading,
            speed=speed,
            acceleration=acceleration,
            yaw_rate=yaw_rate,
            timestamp=self.time,
            pos_uncertainty=pos_uncertainty,
            vel_uncertainty=vel_uncertainty
        )
    
    def get_state_history(self, n: Optional[int] = None) -> List[VehicleState]:
        """
        Get state history.
        
        Args:
            n: Number of recent states to return (None for all)
            
        Returns:
            List of VehicleState objects
        """
        if n is None:
            return self.state_history.copy()
        return self.state_history[-n:]
    
    def get_trajectory(self) -> np.ndarray:
        """Get position trajectory as numpy array."""
        if not self.state_history:
            return np.array([])
        return np.array([[s.x, s.y] for s in self.state_history])
    
    def get_velocity_history(self) -> np.ndarray:
        """Get velocity history as numpy array."""
        if not self.state_history:
            return np.array([])
        return np.array([[s.vx, s.vy] for s in self.state_history])
    
    def get_speed_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get speed history with timestamps."""
        if not self.state_history:
            return np.array([]), np.array([])
        times = np.array([s.timestamp for s in self.state_history])
        speeds = np.array([s.speed for s in self.state_history])
        return times, speeds
    
    def get_heading_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get heading history with timestamps."""
        if not self.state_history:
            return np.array([]), np.array([])
        times = np.array([s.timestamp for s in self.state_history])
        headings = np.array([s.heading for s in self.state_history])
        return times, headings
    
    def set_initial_state(self, x: float, y: float, 
                          vx: float = 0, vy: float = 0,
                          ax: float = 0, ay: float = 0):
        """Set initial state."""
        self.kf.x = np.array([x, y, vx, vy, ax, ay])
        self.prev_heading = np.arctan2(vy, vx)
        self.prev_speed = np.sqrt(vx**2 + vy**2)
    
    def reset(self):
        """Reset estimator to initial state."""
        self.kf.x = np.zeros(6)
        self.kf.P = np.eye(6) * 10
        self.state_history.clear()
        self.time = 0.0
        self.prev_heading = 0.0
        self.prev_speed = 0.0


class SimulatedVehicleMotion:
    """
    Generates simulated vehicle motion for testing.
    
    Simulates realistic driving scenarios including:
    - Straight driving
    - Lane changes
    - Curves
    - Acceleration/deceleration
    """
    
    def __init__(self, dt: float = 0.033):
        self.dt = dt
        self.time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 10.0  # m/s (~36 km/h)
        
    def step(self) -> Tuple[float, float, float, float]:
        """
        Generate next position and velocity.
        
        Returns:
            (x, y, vx, vy) with some noise added
        """
        self.time += self.dt
        
        # Simulate various maneuvers based on time
        t = self.time
        
        # Speed variations
        self.speed = 10 + 3 * np.sin(t * 0.2)  # Varying speed
        
        # Heading variations (simulate curves and lane changes)
        self.heading = 0.1 * np.sin(t * 0.3) + 0.05 * np.sin(t * 0.7)
        
        # Update position
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)
        
        self.x += vx * self.dt
        self.y += vy * self.dt
        
        # Add measurement noise
        noise_x = np.random.normal(0, 0.5)
        noise_y = np.random.normal(0, 0.5)
        noise_vx = np.random.normal(0, 0.2)
        noise_vy = np.random.normal(0, 0.2)
        
        return (
            self.x + noise_x,
            self.y + noise_y,
            vx + noise_vx,
            vy + noise_vy
        )
    
    def get_ground_truth(self) -> Tuple[float, float, float, float]:
        """Get ground truth state without noise."""
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)
        return self.x, self.y, vx, vy
    
    def reset(self):
        """Reset simulation."""
        self.time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 10.0

