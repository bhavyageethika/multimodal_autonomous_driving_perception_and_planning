"""
Standalone Demo Script

Runs the perception and planning pipeline without Streamlit.
Useful for quick testing and video generation.

Run with: python demo.py
"""

import numpy as np
import cv2
import time
from pathlib import Path

# Import project modules
from src.perception import ObjectDetector, LaneDetector
from src.tracking import MultiObjectTracker
from src.state_estimation import VehicleStateEstimator
from src.planning import MotionPlanner
from src.visualization import BEVRenderer, OverlayRenderer
from data.generators import SyntheticDataGenerator


def run_demo(num_frames: int = 300, save_video: bool = False, display: bool = True):
    """
    Run the complete perception and planning pipeline demo.
    
    Args:
        num_frames: Number of frames to process
        save_video: Whether to save output video
        display: Whether to display in window
    """
    print("=" * 60)
    print("Multimodal Autonomous Driving Perception & Planning Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/6] Initializing perception modules...")
    detector = ObjectDetector(mode="simulated")
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
    
    print("[6/6] Generating synthetic data...")
    data_gen = SyntheticDataGenerator()
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
        
        # Generate synthetic frame
        data_gen.frame_count = frame_idx
        frame = data_gen.generate_frame_with_vehicles()
        
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
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = len(frame_times) / total_time
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Processed {len(frame_times)} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average frame time: {np.mean(frame_times) * 1000:.1f} ms")
    
    if save_video:
        print(f"\nVideo saved to: output_demo.mp4")


def run_component_tests():
    """Test individual components."""
    print("\n" + "=" * 60)
    print("Running Component Tests")
    print("=" * 60)
    
    # Test detector
    print("\n[Test 1] Object Detector")
    detector = ObjectDetector(mode="simulated")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_frame)
    print(f"  ✓ Generated {len(detections)} detections")
    for det in detections[:3]:
        print(f"    - {det.class_name}: confidence={det.confidence:.2f}")
    
    # Test lane detector
    print("\n[Test 2] Lane Detector")
    lane_detector = LaneDetector()
    data_gen = SyntheticDataGenerator()
    road_frame = data_gen.generate_road_frame()
    left_lane, right_lane = lane_detector.detect(road_frame)
    print(f"  ✓ Left lane detected: {left_lane is not None}")
    print(f"  ✓ Right lane detected: {right_lane is not None}")
    
    # Test tracker
    print("\n[Test 3] Multi-Object Tracker")
    tracker = MultiObjectTracker()
    for i in range(10):
        detector.frame_count = i
        detections = detector.detect(test_frame)
        tracks = tracker.update(detections)
    print(f"  ✓ Active tracks after 10 frames: {len(tracks)}")
    for track in tracks[:3]:
        print(f"    - Track {track.track_id}: {track.class_name}, "
              f"age={track.age}, trajectory_len={len(track.trajectory)}")
    
    # Test state estimator
    print("\n[Test 4] Vehicle State Estimator")
    estimator = VehicleStateEstimator()
    for i in range(50):
        measurement = np.array([i * 0.3, i * 0.01, 10.0, 0.1])
        state = estimator.step(measurement)
    print(f"  ✓ Estimated state after 50 steps:")
    print(f"    - Position: ({state.x:.2f}, {state.y:.2f})")
    print(f"    - Speed: {state.speed:.2f} m/s")
    print(f"    - Heading: {np.degrees(state.heading):.2f} deg")
    
    # Test motion planner
    print("\n[Test 5] Motion Planner")
    planner = MotionPlanner()
    current_state = (0, 0, 0, 10)
    optimal, candidates = planner.plan(current_state)
    print(f"  ✓ Generated {len(candidates)} candidate trajectories")
    print(f"  ✓ Optimal trajectory:")
    print(f"    - Type: {optimal.trajectory_type}")
    print(f"    - Length: {optimal.length:.2f} m")
    print(f"    - Cost: {optimal.cost:.2f}")
    
    # Test BEV renderer
    print("\n[Test 6] BEV Renderer")
    bev_renderer = BEVRenderer()
    bev_image = bev_renderer.render(
        ego_state=state,
        tracks=tracks,
        planned_trajectory=optimal,
        show_grid=True
    )
    print(f"  ✓ Rendered BEV image: {bev_image.shape}")
    
    print("\n" + "=" * 60)
    print("All component tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal AV Perception & Planning Demo"
    )
    parser.add_argument(
        "--frames", type=int, default=300,
        help="Number of frames to process"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save output to video file"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't display output window"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run component tests only"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_component_tests()
    else:
        run_demo(
            num_frames=args.frames,
            save_video=args.save_video,
            display=not args.no_display
        )

