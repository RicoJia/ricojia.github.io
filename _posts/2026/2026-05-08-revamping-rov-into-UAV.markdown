---
layout: post
title: "[Robotics] From Market-Ready ROVs to Low-Cost AUVs: Building Practical Underwater Autonomy"
date: 2026-05-08 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
This is a summary of [From Market-Ready ROVs to Low-Cost AUVs](https://bluerobotics.com/from-market-ready-rovs-to-low-cost-auvs/)

BlueROV2 is a small 6-DoF ROV ROV platform that can operate at depths up to 100 meters, weighs under 12 kg, and includes a tether of up to 300 meters, onboard battery power, basic sensors, and a Pixhawk/Raspberry Pi frontseat architecture.

### System Architecture: Frontseat and Backseat

The autonomy stack can be divided into two major layers: the **frontseat** and the **backseat**. The frontseat is responsible for low-level vehicle interaction, including actuator control and sensor access. In many commercial ROVs, the frontseat accepts only simple commands such as velocity, force, or joystick-style inputs.

The backseat, or navigation stack, provides higher-level autonomy. It includes an **interpreter node**, **PID controller**, **pose estimation system**, **waypoint pilot**, and additional application modules. The interpreter node is especially important because it bridges the platform-specific frontseat interface with the platform-agnostic autonomy stack. In other words, the interpreter allows the same backseat autonomy software to be reused across different vehicles, while only the interpreter needs to change from platform to platform.

A typical structure is:

Frontseat:

- Actuators
- Sensors
- Basic vehicle control
Backseat / Navigation Stack:
- Interpreter node
- PID controller
- Pose estimation
- Waypoint pilot
- High-level applications

The **interpreter node** is platform-dependent. It translates commands from the backseat into the format expected by the frontseat. This keeps the rest of the autonomy stack mostly platform-agnostic.

### Waypointing and Mission Planning

The waypoint pilot manages which waypoints the vehicle should visit and in what order. It can also interpolate paths or call a planner for collision-free motion.

Key planning components:

- **I-RRT*** in **OMPL** for geometric path planning.
- **High-level task planner** to generate mission-level action sequences.
- **Temporal planner** for missions with time-dependent costs, constraints, or preferences.
- **SEA: Situational Evaluation and Awareness**
  - Handles recovery.
  - Supports tolerance analysis.
  - Detects anomalies and faults.
  - Generates alternative plans when failures occur.

For example, if SLAM tracking fails, SEA can trigger relocalization behaviors and request new viewpoints to help merge maps.

## 3. State Estimation and SLAM

Because GPS is unavailable underwater, the vehicle relies on dead reckoning and SLAM. Basic state estimation uses:

- Pressure sensor for depth.
- IMU for orientation.
- DVL for velocity estimation.
- ROS `robot_localization` package for EKF-based sensor fusion.

Dead reckoning drifts over time, so visual SLAM improves long-term pose estimation. One approach uses **ORB-SLAM** with online DVL extrinsic calibration, fusing DVL velocity, depth, and orientation into visual pose estimation.

Important SLAM behaviors:

- Map merging when tracking is lost.
- Loop closure for consistency.
- Viewpoint generation for relocalization.
- New map creation if old visual features are lost.

## 4. Manipulator Control

The vehicle can support intervention tasks using a **Reach Alpha 5** underwater manipulator. Challenges:

- Underwater manipulator dynamics are highly nonlinear.
- Payload hydrodynamics are often unknown.
- Different payloads can quickly degrade controller performance.

Proposed control method:

- **Adaptive Neural Network Model Predictive Control**, or **AdaNNMPC**.
- Uses a neural network to learn a more accurate dynamics model.
- Accounts for environmental disturbances.
- Adds online adaptive tuning for variable payloads.
- Uses the prediction window to take predictive tuning actions.

In tests, the manipulator reached the desired joint position quickly with minimal overshoot, even while carrying an unknown object.

## 5. Compute

A practical compute setup separates simple tasks from heavy perception tasks:

- **Raspberry Pi / companion computer**
  - Sensor handling
  - Frontseat communication
  - Basic navigation
- **NVIDIA Jetson**
  - Visual SLAM
  - Perception
  - Neural-network inference
  - Higher-level autonomy

FPGA image processing can be efficient, but it is less flexible and better suited to mature, stable pipelines. For development, Jetson-style embedded GPUs are often more practical.

## 6. Networking and Communication

The onboard network is described as **1 GB**, likely meaning **1 Gb/s bandwidth**, and was sufficient for communication and control tasks.

Communication options:

- **Tether**
  - Best for development, debugging, and safety.
  - Supports high-bandwidth monitoring.
- **Communication buoy**
  - Allows freer vehicle motion.
  - Can relay data wirelessly to the operator.
- **Acoustic communication**
  - Long range.
  - Low bandwidth.
- **Optical / laser communication**
  - Higher bandwidth.
  - Requires direct line of sight.
  - Limited range and affected by lighting conditions.

A removable underwater connector, such as one from Suburban Marine, allows the vehicle to switch between tethered and untethered operation.

## 7. Perception and Cabling

Stereo cameras can support 3D reconstruction and SLAM. There are two main design choices:

1. **Single integrated housing**
    - Easier synchronization.
    - Easier onboard processing integration.
    - More expensive and harder to manufacture.
    - Fixed camera geometry.
2. **Separate modular housings**
    - Easier to rearrange.
    - Simpler enclosure design.
    - More flexible for development.
    - Requires underwater cables and connectors.

The concern about cables is not that underwater cables do not exist. Rather, the required connector may need to support a specific data protocol, power transfer, size, and pressure rating, which may not always be commercially available or affordable.
