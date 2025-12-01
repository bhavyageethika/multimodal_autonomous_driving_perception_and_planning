# 🚗 Multimodal Autonomous Driving Perception & Planning System

A comprehensive demonstration of autonomous driving perception and planning capabilities, showcasing **multimodal data** including visual data, structured perception outputs, and motion planning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)

---

## 🎯 Overview

| type | Implementation |
|------------|----------------|
| **Visual Data (Images/Video)** | Object detection, lane detection, frame processing |
| **Structured Perception Outputs** | Agent tracks with IDs, bounding boxes, trajectories |
| **Vehicle State Estimation** | Kalman filter-based position, velocity, heading estimation |
| **Motion Planner Outputs** | Trajectory generation, candidate evaluation, path visualization |

---

## 🏗️ Architecture

```
mm/
├── src/
│   ├── perception/          # Object & lane detection
│   │   ├── detector.py      # YOLO-compatible object detector
│   │   └── lane_detector.py # Edge-based lane detection
│   ├── tracking/            # Multi-object tracking
│   │   └── multi_object_tracker.py  # IoU-based MOT
│   ├── state_estimation/    # Vehicle state estimation
│   │   └── vehicle_state.py # Kalman filter implementation
│   ├── planning/            # Motion planning
│   │   └── motion_planner.py # Trajectory generation
│   └── visualization/       # Rendering utilities
│       ├── bev_renderer.py  # Bird's eye view
│       └── overlays.py      # Camera overlays
├── data/
│   └── generators/          # Synthetic data generation
├── app.py                   # Streamlit dashboard
├── demo.py                  # Standalone demo
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone/navigate to project
cd mm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Interactive Dashboard

```bash
streamlit run app.py
```

### Run Standalone Demo

```bash
# With display
python demo.py

# Save to video file
python demo.py --save-video

# Run component tests
python demo.py --test
```

---

## 📊 Features

### 1. Visual Perception

**Object Detection**
- Supports simulated detections (for demo) and YOLO integration
- Detects: cars, trucks, pedestrians, cyclists, motorcycles, buses
- Outputs: bounding boxes, class labels, confidence scores

**Lane Detection**
- Edge detection with Canny algorithm
- Hough transform for line detection
- Polynomial fitting for smooth lane curves
- Lane center offset calculation

### 2. Multi-Object Tracking

- **IoU-based association**: Matches detections across frames
- **Track lifecycle management**: Birth, update, death of tracks
- **Trajectory history**: Maintains position history for each agent
- **Velocity estimation**: Computes agent velocities from trajectory

### 3. Vehicle State Estimation

- **Kalman Filter**: 6-state model (x, y, vx, vy, ax, ay)
- **Constant acceleration model**: Predicts future states
- **Measurement fusion**: Combines noisy sensor readings
- **Uncertainty quantification**: Provides confidence bounds

### 4. Motion Planning

- **Polynomial trajectory generation**: Smooth, feasible paths
- **Multiple candidate evaluation**: Explores lane-keeping and lane changes
- **Cost-based selection**: Balances safety, comfort, efficiency
- **Real-time visualization**: Shows planned vs candidate trajectories

---

## 🖥️ Streamlit Dashboard

The interactive dashboard provides:

| View | Description |
|------|-------------|
| **Camera View** | Detection boxes, lane lines, track trajectories |
| **Bird's Eye View** | Top-down scene with ego, agents, planned path |
| **State Plots** | Speed, heading, acceleration over time |
| **Metrics Panel** | Real-time perception and planning statistics |

### Controls
- **Frame Slider**: Scrub through frames
- **Auto Play**: Animate the sequence
- **Reset**: Restart from beginning
- **Visualization Toggles**: Show/hide overlays

---

## 📈 Sample Outputs

### Perception Pipeline
```
Frame 150/300 | FPS: 45.2 | Tracks: 4 | Speed: 38.5 km/h
  Detections: car(3), truck(1), pedestrian(0)
  Active Tracks: ID:1, ID:3, ID:5, ID:7
  Lane Offset: -12.3 px (centered)
```

### State Estimation
```
Vehicle State at t=5.0s:
  Position: (48.2, 2.1) m
  Velocity: (10.5, 0.3) m/s
  Speed: 37.9 km/h
  Heading: 1.6 deg
  Acceleration: 0.12 m/s²
```

### Motion Planning
```
Optimal Trajectory:
  Type: lane_keep
  Length: 52.3 m
  Duration: 5.0 s
  Cost: 12.45
  Waypoints: 51
```

---

## 🧪 Testing

Run component tests:

```bash
python demo.py --test
```

Output:
```
[Test 1] Object Detector ✓
[Test 2] Lane Detector ✓
[Test 3] Multi-Object Tracker ✓
[Test 4] Vehicle State Estimator ✓
[Test 5] Motion Planner ✓
[Test 6] BEV Renderer ✓

All component tests passed! ✓
```

---

## 🔧 Configuration

### Detector Settings
```python
detector = ObjectDetector(
    mode="simulated",  # or "yolo" with model_path
)
```

### Tracker Settings
```python
tracker = MultiObjectTracker(
    iou_threshold=0.3,
    max_age=30,
    min_hits=3,
    trajectory_length=50
)
```

### State Estimator Settings
```python
estimator = VehicleStateEstimator(
    dt=0.033,              # ~30 FPS
    process_noise=0.1,
    measurement_noise=1.0
)
```

### Motion Planner Settings
```python
planner = MotionPlanner(
    planning_horizon=5.0,  # seconds
    dt=0.1,
    num_samples=7          # lateral samples
)
```

---

## 📚 Technical Details

### Kalman Filter Model

State vector: `[x, y, vx, vy, ax, ay]`

State transition (constant acceleration):
```
F = | 1  0  dt  0  0.5dt²  0      |
    | 0  1  0   dt 0       0.5dt² |
    | 0  0  1   0  dt      0      |
    | 0  0  0   1  0       dt     |
    | 0  0  0   0  1       0      |
    | 0  0  0   0  0       1      |
```

### Trajectory Generation

Uses quintic polynomial for smooth lateral motion:
```
d(τ) = df * (10τ³ - 15τ⁴ + 6τ⁵)
```

Where `τ = t/T` is normalized time, ensuring:
- Zero initial/final velocity
- Zero initial/final acceleration
- Smooth, jerk-limited motion

---

## 🎓 Skills Demonstrated

1. **Computer Vision**: Object detection, lane detection, image processing
2. **Sensor Fusion**: Kalman filtering, measurement integration
3. **Robotics**: State estimation, motion planning, trajectory optimization
4. **Software Engineering**: Modular architecture, clean code, documentation
5. **Data Visualization**: Real-time dashboards, multi-view rendering

---

