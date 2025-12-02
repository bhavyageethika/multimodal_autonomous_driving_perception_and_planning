"""
Multimodal Autonomous Driving Perception & Planning Dashboard

An interactive Streamlit application that demonstrates:
- Visual perception (object detection, lane detection)
- Multi-object tracking with trajectory history
- Vehicle state estimation using Kalman filtering
- Motion planning visualization
- Auto-tagging for driving scenarios
- Searchable tag database

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import json
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

# Import project modules
from src.perception import ObjectDetector, LaneDetector
from src.tracking import MultiObjectTracker
from src.state_estimation import VehicleStateEstimator
from src.planning import MotionPlanner
from src.visualization import BEVRenderer, OverlayRenderer
from src.tagging import AutoTagger, VLMTagger
from src.database import TagDatabase
from data.loaders import VideoDataLoader
import tempfile
import os


def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.frame_idx = 0
        st.session_state.playing = False
        st.session_state.detector = ObjectDetector(mode="yolo", model_path="yolov8n.pt")
        st.session_state.lane_detector = LaneDetector()
        st.session_state.tracker = MultiObjectTracker()
        st.session_state.state_estimator = VehicleStateEstimator()
        st.session_state.motion_planner = MotionPlanner()
        st.session_state.bev_renderer = BEVRenderer()
        st.session_state.overlay_renderer = OverlayRenderer()
        st.session_state.auto_tagger = None
        st.session_state.vlm_tagger = VLMTagger(use_fast_mode=True)  # Initialize VLM early
        st.session_state.use_vlm = True  # Use VLM by default
        st.session_state.tag_database = TagDatabase("driving_tags.db")
        st.session_state.video_loader = None
        st.session_state.video_loaded = False
        st.session_state.max_frames = 0
        st.session_state.temp_video_path = None
        st.session_state.ego_motion = []
        st.session_state.current_tags = None
        st.session_state.current_vlm_tags = None


def load_video_file(uploaded_file):
    """Load an uploaded video file."""
    # Save uploaded file to temp location
    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        os.remove(st.session_state.temp_video_path)
    
    # Create temp file with proper extension
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        st.session_state.temp_video_path = tmp_file.name
    
    # Load video
    try:
        video_loader = VideoDataLoader(st.session_state.temp_video_path, target_size=(640, 480))
        st.session_state.video_loader = video_loader
        st.session_state.video_loaded = True
        st.session_state.max_frames = video_loader.total_frames
        st.session_state.ego_motion = video_loader.generate_ego_motion(video_loader.total_frames)
        st.session_state.frame_idx = 0
        
        # Initialize taggers
        st.session_state.auto_tagger = AutoTagger(
            video_path=uploaded_file.name,
            fps=video_loader.fps
        )
        
        # Reset VLM tagger for new video (keep model loaded)
        if st.session_state.vlm_tagger:
            st.session_state.vlm_tagger.reset()
        
        # Reset components
        st.session_state.tracker.reset()
        st.session_state.state_estimator.reset()
        st.session_state.lane_detector.reset()
        
        return True, video_loader.get_info()
    except Exception as e:
        return False, str(e)


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
    auto_tagger = st.session_state.auto_tagger
    
    # Get frame from video
    frame = st.session_state.video_loader.read_frame_at(frame_idx)
    if frame is None:
        return None, None, None, None, None, None, None
    
    # Run perception
    detections = detector.detect(frame)
    left_lane, right_lane = lane_detector.detect(frame)
    lanes = (left_lane, right_lane)
    
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
    
    # Run auto-tagging
    frame_tags = None
    vlm_tags = None
    
    if st.session_state.use_vlm and st.session_state.vlm_tagger:
        # Use VLM-based tagging (dynamic, no hardcoded tags)
        vlm_tags = st.session_state.vlm_tagger.tag_frame(
            frame=frame,
            vehicle_state=vehicle_state,
            tracks=tracks
        )
        st.session_state.current_vlm_tags = vlm_tags
    
    if auto_tagger:
        # Also run rule-based tagger for comparison
        frame_tags = auto_tagger.tag_frame(
            frame=frame,
            detections=detections,
            tracks=tracks,
            lanes=lanes,
            vehicle_state=vehicle_state
        )
        st.session_state.current_tags = frame_tags
    
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
    
    return camera_view, bev_view, vehicle_state, tracks, optimal_traj, detections, frame_tags, vlm_tags


def create_state_plots(state_history):
    """Create state estimation plots."""
    if len(state_history) < 2:
        return None
    
    try:
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
    except Exception as e:
        print(f"Error creating state plots: {e}")
        return None


def create_tag_metrics_plot(auto_tagger):
    """Create tag distribution plot."""
    if not auto_tagger or not auto_tagger.tag_counts:
        return None
    
    try:
        # Get top 15 tags
        sorted_tags = sorted(
            auto_tagger.tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        if not sorted_tags:
            return None
        
        tags, counts = zip(*sorted_tags)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2d2d2d')
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tags)))
        bars = ax.barh(range(len(tags)), counts, color=colors)
        
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, color='white')
        ax.set_xlabel('Count', color='white')
        ax.set_title('Tag Distribution', color='white')
        ax.tick_params(colors='white')
        ax.invert_yaxis()
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.5, i, str(count), va='center', color='white', fontsize=9)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating tag metrics plot: {e}")
        return None


def display_current_tags(frame_tags):
    """Display current frame tags in a nice format."""
    if not frame_tags:
        return
    
    # Scene tags
    if frame_tags.scene:
        scene = frame_tags.scene
        cols = st.columns(4)
        cols[0].markdown(f"**🛣️ Road Type:** `{scene.road_type.value}`")
        cols[1].markdown(f"**🚦 Lanes:** `{scene.lane_count}`")
        
        conditions = [c.value for c, _ in scene.conditions]
        cols[2].markdown(f"**🌤️ Conditions:** `{', '.join(conditions) if conditions else 'N/A'}`")
        
        elements = [e.value for e, _ in scene.traffic_elements]
        cols[3].markdown(f"**🚧 Elements:** `{', '.join(elements) if elements else 'None'}`")
    
    # Maneuver tags
    if frame_tags.maneuver:
        maneuver = frame_tags.maneuver
        cols = st.columns(4)
        cols[0].markdown(f"**↔️ Lateral:** `{maneuver.lateral.value}`")
        cols[1].markdown(f"**↕️ Longitudinal:** `{maneuver.longitudinal.value}`")
        cols[2].markdown(f"**🔄 Turning:** `{maneuver.turning.value}`")
        cols[3].markdown(f"**⚡ Speed:** `{maneuver.speed_kmh:.1f} km/h`")
    
    # Interaction tags
    if frame_tags.interaction:
        interaction = frame_tags.interaction
        risk_colors = {
            'low': '🟢',
            'medium': '🟡', 
            'high': '🟠',
            'critical': '🔴'
        }
        risk_icon = risk_colors.get(interaction.overall_risk.value, '⚪')
        
        cols = st.columns(4)
        cols[0].markdown(f"**{risk_icon} Risk:** `{interaction.overall_risk.value}`")
        cols[1].markdown(f"**👥 Agents:** `{interaction.agent_count}`")
        cols[2].markdown(f"**🚶 Pedestrians:** `{interaction.pedestrian_count}`")
        cols[3].markdown(f"**🚗 Vehicles:** `{interaction.vehicle_count}`")
        
        if interaction.interactions:
            primary = interaction.interactions[0]
            st.markdown(f"**Primary Interaction:** `{primary.type.value}` (confidence: {primary.confidence:.2f})")
    
    # All tags as badges
    if frame_tags.all_tags:
        st.markdown("**🏷️ All Tags:**")
        tag_html = " ".join([
            f'<span style="background-color: #3d5a80; color: white; padding: 2px 8px; '
            f'border-radius: 10px; margin: 2px; display: inline-block; font-size: 0.85em;">{tag}</span>'
            for tag in frame_tags.all_tags[:15]
        ])
        st.markdown(tag_html, unsafe_allow_html=True)


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
    
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #16213e;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 12px 30px;
        font-size: 1.2rem !important;
        font-weight: 600;
        background-color: #1a1a2e;
        border-radius: 10px;
        color: #888;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2d3748;
        color: #00d4ff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #1a1a2e !important;
        border: 2px solid #00d4ff;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    .upload-prompt {
        text-align: center;
        padding: 100px 50px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        border: 2px dashed #00d4ff;
        margin: 50px auto;
        max-width: 600px;
    }
    .upload-prompt h2 {
        color: #00d4ff;
        margin-bottom: 20px;
    }
    .upload-prompt p {
        color: #888;
        font-size: 1.1em;
    }
    .tag-box {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚗 Multimodal Autonomous Driving System")
    st.markdown("*Perception, Tracking, Planning & Auto-Tagging for Driving Scenarios*")
    
    # Initialize
    init_session_state()
    
    # Sidebar - Video Upload
    st.sidebar.header("📁 Video Input")
    
    uploaded_video = st.sidebar.file_uploader(
        "Upload Driving Video", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a driving video to analyze"
    )
    
    if uploaded_video is not None:
        # Check if this is a new upload
        if ('last_uploaded' not in st.session_state or 
            st.session_state.last_uploaded != uploaded_video.name):
            st.session_state.last_uploaded = uploaded_video.name
            with st.sidebar.status("Loading video...", expanded=True) as status:
                success, info = load_video_file(uploaded_video)
                if success:
                    status.update(label="Video loaded!", state="complete")
                    st.sidebar.success(f"✅ {uploaded_video.name}")
                    st.sidebar.info(f"📊 {info['total_frames']} frames | {info['fps']:.1f} FPS")
                else:
                    status.update(label="Error loading video", state="error")
                    st.sidebar.error(f"❌ {info}")
    
    # Check if video is loaded
    if not st.session_state.video_loaded:
        # Show upload prompt
        st.markdown("""
        <div class="upload-prompt">
            <h2>📹 Upload a Video to Get Started</h2>
            <p>Use the sidebar to upload a driving video (MP4, AVI, MOV, or MKV format).</p>
            <p>The system will analyze the video using:</p>
            <ul style="text-align: left; color: #aaa; margin-top: 20px;">
                <li>🎯 Object Detection (YOLOv8)</li>
                <li>🛣️ Lane Detection</li>
                <li>📍 Multi-Object Tracking</li>
                <li>📊 State Estimation</li>
                <li>🗺️ Motion Planning</li>
                <li>🏷️ <b>Auto-Tagging System</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Video is loaded - show controls and processing
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Controls")
    
    max_frame = st.session_state.max_frames - 1 if st.session_state.max_frames > 0 else 0
    frame_idx = st.sidebar.slider(
        "Frame", 0, max_frame, min(st.session_state.frame_idx, max_frame), key="frame_slider"
    )
    st.session_state.frame_idx = frame_idx
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("⏮️ Reset"):
        st.session_state.frame_idx = 0
        st.session_state.detector.reset()
        st.session_state.tracker.reset()
        st.session_state.state_estimator.reset()
        st.session_state.lane_detector.reset()
        if st.session_state.auto_tagger:
            st.session_state.auto_tagger.reset()
        st.rerun()
    
    # Auto-play is ON by default
    auto_play = st.sidebar.checkbox("▶️ Auto Play", value=True)
    
    # Playback speed control
    playback_speed = st.sidebar.slider("⚡ Speed", 0.5, 3.0, 1.0, 0.5, 
                                        help="Playback speed multiplier")
    
    # Database controls
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Database")
    
    if st.sidebar.button("💾 Save Tags to Database"):
        if st.session_state.auto_tagger:
            st.session_state.auto_tagger.finalize()
            count = st.session_state.tag_database.save_all_tags(st.session_state.auto_tagger)
            st.sidebar.success(f"✅ Saved {count} frames to database")
    
    if st.sidebar.button("📊 View Statistics"):
        stats = st.session_state.tag_database.get_tag_statistics()
        st.sidebar.json(stats)
    
    # Process current frame
    result = process_frame(frame_idx)
    if result[0] is None:
        st.error("Failed to read frame from video")
        return
    
    camera_view, bev_view, vehicle_state, tracks, planned_traj, detections, frame_tags, vlm_tags = result
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎥 Live View", "🏷️ Auto-Tags", "📊 Metrics"])
    
    with tab1:
        # Main display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Camera View (Perception)")
            camera_rgb = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
            st.image(camera_rgb, use_container_width=True)
            
            # Perception metrics
            st.markdown("**Perception Metrics:**")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Detections", len(detections))
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
        
        # Current state display
        if vehicle_state:
            st.markdown("---")
            st.subheader("📊 Vehicle State")
            state_cols = st.columns(6)
            state_cols[0].metric("Position X", f"{vehicle_state.x:.1f} m")
            state_cols[1].metric("Position Y", f"{vehicle_state.y:.1f} m")
            state_cols[2].metric("Speed", f"{vehicle_state.speed * 3.6:.1f} km/h")
            state_cols[3].metric("Heading", f"{np.degrees(vehicle_state.heading):.1f}°")
            state_cols[4].metric("Acceleration", f"{vehicle_state.acceleration:.2f} m/s²")
            state_cols[5].metric("Yaw Rate", f"{np.degrees(vehicle_state.yaw_rate):.2f}°/s")
    
    with tab2:
        st.subheader("🏷️ VLM-Generated Tags (Dynamic)")
        
        # Toggle between VLM and rule-based
        st.session_state.use_vlm = st.checkbox("Use VLM (Vision-Language Model)", value=st.session_state.use_vlm)
        
        if st.session_state.use_vlm:
            if vlm_tags:
                # Display VLM-generated content
                st.markdown("### 📝 Scene Description")
                st.info(vlm_tags.scene_description or "Generating description...")
                
                st.markdown("### ⚠️ Safety Assessment")
                risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
                risk_icon = risk_colors.get(vlm_tags.risk_level, '⚪')
                st.warning(f"{risk_icon} **Risk Level: {vlm_tags.risk_level.upper()}**")
                st.caption(vlm_tags.safety_assessment or "Analyzing...")
                
                st.markdown("### 🏷️ Extracted Tags")
                all_tags = vlm_tags.get_tags_list()
                if all_tags:
                    # Display tags as colored badges
                    tags_html = " ".join([
                        f'<span style="background-color: #4CAF50; color: white; padding: 4px 8px; border-radius: 12px; margin: 2px; display: inline-block;">{tag}</span>'
                        for tag in all_tags
                    ])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("Processing tags from description...")
                
                # Scene details
                st.markdown("### 📋 Scene Details")
                detail_cols = st.columns(3)
                detail_cols[0].metric("Road Type", vlm_tags.road_type or "Unknown")
                detail_cols[1].metric("Weather", vlm_tags.weather or "Unknown")
                detail_cols[2].metric("Time of Day", vlm_tags.time_of_day or "Unknown")
            else:
                st.warning("⏳ VLM is processing... Please wait (first frame may take 5-10 seconds)")
                st.caption("The Vision-Language Model generates descriptions for each frame.")
            
            st.markdown("---")
            st.subheader("📈 VLM Tagging Progress")
            
            if st.session_state.vlm_tagger:
                vlm = st.session_state.vlm_tagger
                progress = (frame_idx + 1) / st.session_state.max_frames
                st.progress(progress, text=f"Processed {frame_idx + 1} / {st.session_state.max_frames} frames")
                
                stats = vlm.get_statistics()
                col1, col2, col3 = st.columns(3)
                col1.metric("Unique Tags Found", stats.get('unique_tags', 0))
                col2.metric("Frames Analyzed", stats.get('total_frames', 0))
                col3.metric("Risk Frames", stats.get('frames_with_risk', 0))
        
        elif frame_tags:
            st.caption("📌 Using rule-based tagging (VLM disabled)")
            display_current_tags(frame_tags)
            
            st.markdown("---")
            st.subheader("📈 Tagging Progress")
            
            if st.session_state.auto_tagger:
                tagger = st.session_state.auto_tagger
                progress = (frame_idx + 1) / st.session_state.max_frames
                st.progress(progress, text=f"Processed {frame_idx + 1} / {st.session_state.max_frames} frames")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Tags Detected", len(tagger.tag_counts))
                col2.metric("Frames Tagged", len(tagger.frame_tags))
                
                # Show high risk frame count
                high_risk = len(tagger.get_high_risk_frames())
                col3.metric("High Risk Frames", high_risk)
        else:
            st.info("Process frames to generate tags")
    
    with tab3:
        st.subheader("📊 Tag Statistics & Metrics")
        
        # VLM-based metrics
        if st.session_state.use_vlm and st.session_state.vlm_tagger:
            vlm = st.session_state.vlm_tagger
            stats = vlm.get_statistics()
            
            if stats.get('tag_frequency'):
                # Create VLM tag distribution chart
                fig, ax = plt.subplots(figsize=(10, 4))
                tags = list(stats['tag_frequency'].keys())[:15]
                counts = [stats['tag_frequency'][t] for t in tags]
                
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tags)))
                bars = ax.barh(tags, counts, color=colors)
                ax.set_xlabel('Count')
                ax.set_title('VLM-Generated Tag Distribution')
                ax.invert_yaxis()
                
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                           str(count), va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 VLM Statistics:**")
                st.markdown(f"- Total Frames: `{stats.get('total_frames', 0)}`")
                st.markdown(f"- Unique Tags: `{stats.get('unique_tags', 0)}`")
                st.markdown(f"- Risk Frames: `{stats.get('frames_with_risk', 0)}`")
            
            with col2:
                st.markdown("**🔝 Top Tags:**")
                for tag, count in list(stats.get('tag_frequency', {}).items())[:5]:
                    st.markdown(f"- `{tag}`: {count} occurrences")
            
            # Natural language search
            st.markdown("---")
            st.subheader("🔍 Natural Language Search")
            search_query = st.text_input(
                "Search by description:", 
                placeholder="e.g., pedestrian crossing, dangerous situation, highway"
            )
            
            if search_query:
                results = vlm.search_by_description(search_query)
                st.markdown(f"Found **{len(results)}** frames matching `{search_query}`")
                
                if results:
                    st.markdown("Matching frames:")
                    for tags in results[:10]:
                        with st.expander(f"Frame {tags.frame_idx} (t={tags.timestamp:.2f}s)"):
                            st.markdown(f"**Description:** {tags.scene_description}")
                            st.markdown(f"**Risk:** {tags.risk_level}")
                            st.markdown(f"**Tags:** {', '.join(tags.extracted_tags)}")
        
        # Rule-based metrics (fallback)
        elif st.session_state.auto_tagger and st.session_state.auto_tagger.tag_counts:
            tagger = st.session_state.auto_tagger
            
            # Tag distribution chart
            fig = create_tag_metrics_plot(tagger)
            if fig:
                st.pyplot(fig)
            
            # Statistics
            stats = tagger.get_tag_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Speed Statistics:**")
                if 'speed_stats' in stats:
                    st.markdown(f"- Min: `{stats['speed_stats']['min']:.1f}` km/h")
                    st.markdown(f"- Max: `{stats['speed_stats']['max']:.1f}` km/h")
                    st.markdown(f"- Avg: `{stats['speed_stats']['avg']:.1f}` km/h")
            
            with col2:
                st.markdown("**⚠️ Risk Distribution:**")
                if 'risk_distribution' in stats:
                    for risk, count in stats['risk_distribution'].items():
                        icon = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(risk, '⚪')
                        st.markdown(f"- {icon} {risk}: `{count}` frames")
            
            # Search functionality
            st.markdown("---")
            st.subheader("🔍 Search Tags")
            search_tag = st.text_input("Enter tag to search:", placeholder="e.g., pedestrian_crossing")
            
            if search_tag:
                results = tagger.search_by_tag(search_tag)
                st.markdown(f"Found **{len(results)}** frames with tag `{search_tag}`")
                
                if results:
                    st.markdown("First 10 matching frames:")
                    for ft in results[:10]:
                        st.markdown(f"- Frame {ft.frame_idx} (t={ft.timestamp:.2f}s)")
        else:
            st.info("Process more frames to see statistics")
        
        # State estimation plots
        st.markdown("---")
        st.subheader("📉 Vehicle State History")
        state_history = st.session_state.state_estimator.get_state_history()
        if len(state_history) >= 2:
            fig = create_state_plots(state_history)
            if fig:
                st.pyplot(fig)
    
    # Auto-advance with speed control
    if auto_play and frame_idx < max_frame:
        # Base delay is ~30fps (0.033s), adjusted by playback speed
        delay = 0.033 / playback_speed
        time.sleep(delay)
        st.session_state.frame_idx = frame_idx + 1
        st.rerun()
    
    # Info section
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### Multimodal Perception, Planning & Auto-Tagging System
        
        This system analyzes driving videos and automatically generates searchable tags:
        
        **🎯 Perception Pipeline:**
        - YOLOv8 object detection (vehicles, pedestrians, cyclists)
        - Lane detection using edge detection and Hough transform
        - Multi-object tracking with trajectory history
        
        **🏷️ Auto-Tagging System:**
        - **Scene Tags:** Road type, traffic elements, conditions
        - **Maneuver Tags:** Lane changes, turning, acceleration, braking
        - **Interaction Tags:** Following, yielding, pedestrian crossings, near-misses
        
        **💾 Database Features:**
        - SQLite storage for all tags
        - Search by tag, risk level, or time range
        - Export to JSON/CSV for dataset curation
        
        **Use Cases:**
        - Automated data labeling for ML training
        - Finding specific driving scenarios
        - Building test sets for AV development
        - Triaging issues and edge cases
        """)


if __name__ == "__main__":
    main()
