"""
Multimodal Autonomous Driving Perception & Planning Dashboard

An interactive Streamlit application that demonstrates:
- Visual perception (object detection, lane detection)
- Multi-object tracking with trajectory history
- Vehicle state estimation using Kalman filtering
- Motion planning visualization

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# Import project modules
from src.perception import ObjectDetector, LaneDetector
from src.tracking import MultiObjectTracker
from src.state_estimation import VehicleStateEstimator
from src.planning import MotionPlanner
from src.visualization import BEVRenderer, OverlayRenderer
from data.generators import SyntheticDataGenerator


def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.frame_idx = 0
        st.session_state.playing = False
        st.session_state.detector = ObjectDetector(mode="simulated")
        st.session_state.lane_detector = LaneDetector()
        st.session_state.tracker = MultiObjectTracker()
        st.session_state.state_estimator = VehicleStateEstimator()
        st.session_state.motion_planner = MotionPlanner()
        st.session_state.bev_renderer = BEVRenderer()
        st.session_state.overlay_renderer = OverlayRenderer()
        st.session_state.data_generator = SyntheticDataGenerator()
        
        # Pre-generate data
        st.session_state.ego_motion = st.session_state.data_generator.generate_ego_motion(500)


def process_frame(frame_idx: int):
    """Process a single frame through the perception pipeline."""
    # Get components
    detector = st.session_state.detector
    lane_detector = st.session_state.lane_detector
    tracker = st.session_state.tracker
    state_estimator = st.session_state.state_estimator
    motion_planner = st.session_state.motion_planner
    bev_renderer = st.session_state.bev_renderer
    overlay_renderer = st.session_state.overlay_renderer
    data_gen = st.session_state.data_generator
    
    # Generate synthetic frame
    data_gen.frame_count = frame_idx
    frame = data_gen.generate_frame_with_vehicles()
    
    # Run perception
    detections = detector.detect(frame)
    left_lane, right_lane = lane_detector.detect(frame)
    
    # Run tracking
    tracks = tracker.update(detections)
    
    # Update state estimation
    if frame_idx < len(st.session_state.ego_motion):
        measurement = np.array(st.session_state.ego_motion[frame_idx])
        vehicle_state = state_estimator.step(measurement)
    else:
        vehicle_state = state_estimator.step()
    
    # Run motion planning
    current_state = (vehicle_state.x, vehicle_state.y, 
                     vehicle_state.heading, vehicle_state.speed)
    optimal_traj, candidate_trajs = motion_planner.plan(current_state)
    
    # Create visualizations
    # Camera view with overlays
    camera_view = detector.draw_detections(frame, detections)
    camera_view = lane_detector.draw_lanes(camera_view, left_lane, right_lane)
    camera_view = tracker.draw_tracks(camera_view, tracks, draw_trajectories=True)
    
    # Add info panel
    lane_offset = lane_detector.get_lane_center_offset(frame.shape[1], left_lane, right_lane)
    camera_view = overlay_renderer.draw_info_panel(
        camera_view, vehicle_state, fps=30.0, frame_num=frame_idx
    )
    camera_view = overlay_renderer.draw_detection_summary(camera_view, detections)
    camera_view = overlay_renderer.draw_tracking_stats(camera_view, tracks)
    if lane_offset is not None:
        camera_view = overlay_renderer.draw_lane_offset_indicator(camera_view, lane_offset)
    
    # BEV view
    bev_view = bev_renderer.render(
        ego_state=vehicle_state,
        tracks=tracks,
        planned_trajectory=optimal_traj,
        candidate_trajectories=candidate_trajs[:10],
        show_grid=True
    )
    
    return camera_view, bev_view, vehicle_state, tracks, optimal_traj


def create_state_plots(state_history):
    """Create state estimation plots."""
    import matplotlib.pyplot as plt
    
    if len(state_history) < 2:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    
    times = [s.timestamp for s in state_history]
    speeds = [s.speed * 3.6 for s in state_history]  # km/h
    headings = [np.degrees(s.heading) for s in state_history]
    accels = [s.acceleration for s in state_history]
    positions = [(s.x, s.y) for s in state_history]
    
    # Speed plot
    ax = axes[0, 0]
    ax.set_facecolor('#2d2d2d')
    ax.plot(times, speeds, color='#00ff88', linewidth=2)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Speed (km/h)', color='white')
    ax.set_title('Vehicle Speed', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Heading plot
    ax = axes[0, 1]
    ax.set_facecolor('#2d2d2d')
    ax.plot(times, headings, color='#ff8800', linewidth=2)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Heading (deg)', color='white')
    ax.set_title('Vehicle Heading', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Acceleration plot
    ax = axes[1, 0]
    ax.set_facecolor('#2d2d2d')
    ax.plot(times, accels, color='#ff4488', linewidth=2)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Acceleration (m/s²)', color='white')
    ax.set_title('Longitudinal Acceleration', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Trajectory plot
    ax = axes[1, 1]
    ax.set_facecolor('#2d2d2d')
    x_pos = [p[0] for p in positions]
    y_pos = [p[1] for p in positions]
    ax.plot(x_pos, y_pos, color='#44aaff', linewidth=2)
    ax.scatter([x_pos[-1]], [y_pos[-1]], color='#ff4444', s=100, zorder=5)
    ax.set_xlabel('X Position (m)', color='white')
    ax.set_ylabel('Y Position (m)', color='white')
    ax.set_title('Vehicle Trajectory', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multimodal AV Perception & Planning",
        page_icon="🚗",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main { background-color: #1a1a2e; }
    .stApp { background-color: #1a1a2e; }
    h1, h2, h3 { color: #00d4ff; }
    .stMarkdown { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚗 Multimodal Autonomous Driving System")
    st.markdown("*Demonstrating perception, tracking, state estimation, and motion planning*")
    
    # Initialize
    init_session_state()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    frame_idx = st.sidebar.slider(
        "Frame", 0, 299, st.session_state.frame_idx, key="frame_slider"
    )
    st.session_state.frame_idx = frame_idx
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("⏮️ Reset"):
        st.session_state.frame_idx = 0
        st.session_state.detector.reset()
        st.session_state.tracker.reset()
        st.session_state.state_estimator.reset()
        st.session_state.lane_detector.reset()
        st.rerun()
    
    auto_play = st.sidebar.checkbox("▶️ Auto Play", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Visualization Options")
    
    show_detections = st.sidebar.checkbox("Show Detections", value=True)
    show_lanes = st.sidebar.checkbox("Show Lane Lines", value=True)
    show_trajectories = st.sidebar.checkbox("Show Trajectories", value=True)
    show_planning = st.sidebar.checkbox("Show Planning", value=True)
    
    # Process current frame
    camera_view, bev_view, vehicle_state, tracks, planned_traj = process_frame(frame_idx)
    
    # Main display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Camera View (Perception)")
        # Convert BGR to RGB for display
        camera_rgb = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
        st.image(camera_rgb, use_container_width=True)
        
        # Perception metrics
        st.markdown("**Perception Metrics:**")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Detections", len(st.session_state.detector._detect_simulated(camera_view)))
        metric_cols[1].metric("Active Tracks", len(tracks))
        metric_cols[2].metric("Track IDs", ", ".join([str(t.track_id) for t in tracks[:5]]) or "None")
    
    with col2:
        st.subheader("🗺️ Bird's Eye View (Planning)")
        bev_rgb = cv2.cvtColor(bev_view, cv2.COLOR_BGR2RGB)
        st.image(bev_rgb, use_container_width=True)
        
        # Planning metrics
        st.markdown("**Planning Metrics:**")
        metric_cols = st.columns(3)
        if planned_traj:
            metric_cols[0].metric("Trajectory Length", f"{planned_traj.length:.1f} m")
            metric_cols[1].metric("Duration", f"{planned_traj.duration:.1f} s")
            metric_cols[2].metric("Cost", f"{planned_traj.cost:.2f}")
        else:
            metric_cols[0].metric("Trajectory", "N/A")
    
    # State estimation plots
    st.markdown("---")
    st.subheader("📊 Vehicle State Estimation")
    
    state_history = st.session_state.state_estimator.get_state_history()
    if len(state_history) >= 2:
        fig = create_state_plots(state_history)
        if fig:
            st.pyplot(fig)
    
    # Current state display
    if vehicle_state:
        state_cols = st.columns(6)
        state_cols[0].metric("Position X", f"{vehicle_state.x:.1f} m")
        state_cols[1].metric("Position Y", f"{vehicle_state.y:.1f} m")
        state_cols[2].metric("Speed", f"{vehicle_state.speed * 3.6:.1f} km/h")
        state_cols[3].metric("Heading", f"{np.degrees(vehicle_state.heading):.1f}°")
        state_cols[4].metric("Acceleration", f"{vehicle_state.acceleration:.2f} m/s²")
        state_cols[5].metric("Yaw Rate", f"{np.degrees(vehicle_state.yaw_rate):.2f}°/s")
    
    # Auto-advance
    if auto_play and frame_idx < 299:
        time.sleep(0.05)
        st.session_state.frame_idx = frame_idx + 1
        st.rerun()
    
    # Info section
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### Multimodal Perception & Planning Pipeline
        
        This demonstration showcases key components of an autonomous driving system:
        
        **1. Visual Perception**
        - Object detection (vehicles, pedestrians, cyclists)
        - Lane line detection using edge detection and Hough transform
        
        **2. Multi-Object Tracking**
        - IoU-based track association
        - Trajectory history maintenance
        - Velocity estimation
        
        **3. State Estimation**
        - Kalman filter for ego-vehicle state
        - Fuses position and velocity measurements
        - Estimates heading, acceleration, yaw rate
        
        **4. Motion Planning**
        - Polynomial trajectory generation
        - Multiple candidate trajectory evaluation
        - Cost-based optimal path selection
        
        **Data Modalities:**
        - Visual data (camera frames)
        - Structured perception (bounding boxes, tracks)
        - Vehicle state (position, velocity, heading)
        - Planning outputs (trajectories, waypoints)
        """)


if __name__ == "__main__":
    main()

