"""
Standalone Demo Script

Runs the perception and planning pipeline on video input.
Useful for quick testing and video generation.

Run with: python demo.py --video path/to/video.mp4
"""

import numpy as np
import cv2
import time
import sys
from pathlib import Path

# Import project modules
from src.perception import ObjectDetector, LaneDetector
from src.tracking import MultiObjectTracker
from src.state_estimation import VehicleStateEstimator
from src.planning import MotionPlanner
from src.visualization import BEVRenderer, OverlayRenderer
from data.loaders import VideoDataLoader


def run_demo(video_path: str, num_frames: int = None, save_video: bool = False, display: bool = True):
    """
    Run the complete perception and planning pipeline demo.
    
    Args:
        video_path: Path to input video file (required)
        num_frames: Number of frames to process (None = all frames)
        save_video: Whether to save output video
        display: Whether to display in window
    """
    print("=" * 60)
    print("Multimodal Autonomous Driving Perception & Planning Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/6] Initializing perception modules...")
    detector = ObjectDetector(mode="yolo", model_path="yolov8n.pt")
    lane_detector = LaneDetector()
    
    print("[2/6] Initializing tracking module...")
    tracker = MultiObjectTracker()
    
    print("[3/6] Initializing state estimation...")
    state_estimator = VehicleStateEstimator()
    
    print("[4/6] Initializing motion planner...")
    motion_planner = MotionPlanner()
    
    print("[5/6] Initializing visualization...")
    bev_renderer = BEVRenderer()
    overlay_renderer = OverlayRenderer()
    
    # Load video
    print(f"[6/6] Loading video: {video_path}")
    try:
        data_gen = VideoDataLoader(video_path, target_size=(640, 480))
    except FileNotFoundError:
        print(f"\n❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: Could not open video: {e}")
        sys.exit(1)
    
    print(f"      Video info: {data_gen.total_frames} frames, {data_gen.fps:.1f} FPS, "
          f"{data_gen._width}x{data_gen._height}")
    
    # Set number of frames to process
    if num_frames is None:
        num_frames = data_gen.total_frames
    else:
        num_frames = min(num_frames, data_gen.total_frames)
    
    ego_motion = data_gen.generate_ego_motion(num_frames)
    
    print("\n" + "=" * 60)
    print("Starting processing pipeline...")
    print("=" * 60)
    
    # Video writer
    video_writer = None
    if save_video:
        output_path = Path("output_demo.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, 30.0, (1240, 480)
        )
        print(f"\nSaving video to: {output_path}")
    
    # Processing loop
    start_time = time.time()
    frame_times = []
    
    for frame_idx in range(num_frames):
        frame_start = time.time()
        
        # Get frame from video
        frame = data_gen.read_frame_at(frame_idx)
        if frame is None:
            print(f"\nEnd of video reached at frame {frame_idx}")
            break
        
        # Run perception
        detections = detector.detect(frame)
        left_lane, right_lane = lane_detector.detect(frame)
        
        # Run tracking
        tracks = tracker.update(detections)
        
        # Update state estimation
        measurement = np.array(ego_motion[frame_idx])
        vehicle_state = state_estimator.step(measurement)
        
        # Run motion planning
        current_state = (vehicle_state.x, vehicle_state.y,
                        vehicle_state.heading, vehicle_state.speed)
        optimal_traj, candidate_trajs = motion_planner.plan(current_state)
        
        # Create visualizations
        # Camera view
        camera_view = detector.draw_detections(frame, detections)
        camera_view = lane_detector.draw_lanes(camera_view, left_lane, right_lane)
        camera_view = tracker.draw_tracks(camera_view, tracks)
        
        lane_offset = lane_detector.get_lane_center_offset(
            frame.shape[1], left_lane, right_lane
        )
        
        fps = 1.0 / (frame_times[-1] if frame_times else 0.033)
        camera_view = overlay_renderer.draw_info_panel(
            camera_view, vehicle_state, fps=fps, frame_num=frame_idx
        )
        camera_view = overlay_renderer.draw_detection_summary(camera_view, detections)
        
        # BEV view
        bev_view = bev_renderer.render(
            ego_state=vehicle_state,
            tracks=tracks,
            planned_trajectory=optimal_traj,
            candidate_trajectories=candidate_trajs[:10],
            show_grid=True
        )
        
        # Combine views
        combined = overlay_renderer.create_side_by_side(
            camera_view, bev_view, ("Camera View", "Bird's Eye View")
        )
        
        # Display
        if display:
            cv2.imshow("Multimodal AV Demo", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser interrupted.")
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)
        
        # Save video
        if video_writer:
            video_writer.write(combined)
        
        # Track timing
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        # Progress update
        if (frame_idx + 1) % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:])
            print(f"Frame {frame_idx + 1}/{num_frames} | "
                  f"FPS: {avg_fps:.1f} | "
                  f"Tracks: {len(tracks)} | "
                  f"Speed: {vehicle_state.speed * 3.6:.1f} km/h")
    
    # Cleanup
    if video_writer:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    data_gen.release()
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = len(frame_times) / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Processed {len(frame_times)} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average frame time: {np.mean(frame_times) * 1000:.1f} ms")
    
    if save_video:
        print(f"\nVideo saved to: output_demo.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal AV Perception & Planning Demo"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file (required)"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Number of frames to process (default: all frames)"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save output to video file"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't display output window"
    )
    
    args = parser.parse_args()
    
    run_demo(
        video_path=args.video,
        num_frames=args.frames,
        save_video=args.save_video,
        display=not args.no_display
    )
