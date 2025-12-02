"""
Video Data Loader

Loads video files for processing through the perception and planning pipeline.
Supports MP4, AVI, MOV, MKV, and other OpenCV-compatible formats.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Generator, Optional, Tuple


class VideoDataLoader:
    """
    Loads and iterates through video files.
    
    Provides the same interface as SyntheticDataGenerator for easy swapping.
    """
    
    def __init__(self, video_path: str, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the video loader.
        
        Args:
            video_path: Path to the video file
            target_size: Optional (width, height) to resize frames. 
                        If None, uses original video resolution.
        """
        self.video_path = Path(video_path)
        self.target_size = target_size
        self.cap = None
        self.frame_count = 0
        
        # Validate file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video and get properties
        self._open_video()
        
    def _open_video(self):
        """Open the video file and read properties."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._total_frames / self._fps if self._fps > 0 else 0
        
    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return self._total_frames
    
    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps
    
    @property
    def width(self) -> int:
        """Video width (or target width if resizing)."""
        return self.target_size[0] if self.target_size else self._width
    
    @property
    def height(self) -> int:
        """Video height (or target height if resizing)."""
        return self.target_size[1] if self.target_size else self._height
    
    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self._duration
    
    @property
    def dt(self) -> float:
        """Time step between frames."""
        return 1.0 / self._fps if self._fps > 0 else 0.033
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the video.
        
        Returns:
            Frame as numpy array (BGR format), or None if end of video.
        """
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Resize if target size specified
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size)
        
        self.frame_count += 1
        return frame
    
    def read_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by index.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Frame as numpy array (BGR format), or None if invalid index.
        """
        if self.cap is None or frame_idx < 0 or frame_idx >= self._total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size)
        
        self.frame_count = frame_idx + 1
        return frame
    
    def generate_frame_with_vehicles(self) -> Optional[np.ndarray]:
        """
        Read next frame. 
        
        This method provides compatibility with SyntheticDataGenerator interface.
        
        Returns:
            Frame as numpy array (BGR format), or None if end of video.
        """
        return self.read_frame()
    
    def generate_video_stream(self, num_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Generate a stream of video frames.
        
        Args:
            num_frames: Maximum number of frames to yield. 
                       If None, yields all frames in the video.
        
        Yields:
            Video frames as numpy arrays (BGR format).
        """
        self.reset()
        frames_yielded = 0
        max_frames = num_frames if num_frames else self._total_frames
        
        while frames_yielded < max_frames:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
            frames_yielded += 1
    
    def generate_ego_motion(self, num_steps: Optional[int] = None) -> list:
        """
        Generate placeholder ego motion data.
        
        Since we don't have actual sensor data, this generates 
        simulated motion that can be used with the state estimator.
        
        Args:
            num_steps: Number of motion steps. Defaults to total frames.
            
        Returns:
            List of (x, y, vx, vy) tuples.
        """
        if num_steps is None:
            num_steps = self._total_frames
            
        measurements = []
        x, y = 0.0, 0.0
        speed = 10.0  # Assume constant speed
        heading = 0.0
        dt = self.dt
        
        for i in range(num_steps):
            t = i * dt
            # Simple forward motion with slight variations
            heading = 0.05 * np.sin(t * 0.5)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            
            x += vx * dt
            y += vy * dt
            
            measurements.append((
                x + np.random.normal(0, 0.1),
                y + np.random.normal(0, 0.1),
                vx + np.random.normal(0, 0.05),
                vy + np.random.normal(0, 0.05)
            ))
        
        return measurements
    
    def reset(self):
        """Reset video to the beginning."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
    
    def release(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return self._total_frames
    
    def __iter__(self):
        """Iterate through all frames."""
        self.reset()
        return self
    
    def __next__(self) -> np.ndarray:
        """Get next frame."""
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame
    
    def get_info(self) -> dict:
        """
        Get video information.
        
        Returns:
            Dictionary with video properties.
        """
        return {
            'path': str(self.video_path),
            'total_frames': self._total_frames,
            'fps': self._fps,
            'width': self._width,
            'height': self._height,
            'duration': self._duration,
            'target_size': self.target_size
        }
    
    def __repr__(self) -> str:
        return (f"VideoDataLoader(path='{self.video_path.name}', "
                f"frames={self._total_frames}, fps={self._fps:.1f}, "
                f"size={self._width}x{self._height})")


