---
layout: post
title: Robotics - Robot State and Joint State Publisher
date: 2024-09-01 13:19
subtitle: robot_state_publisher, joint_state_publisher, and the TF tree
header-img: img/post-bg-os-metro.jpg
tags:
  - Robotics
  - ROS2
comments: true
---

## TF Pipeline Overview

```
URDF (/robot_description) ───────────────────────────────────┐
         │                                                   │
         ↓                                                   ↓
[joint_state_publisher / gui]   [hardware driver]   robot_state_publisher
                    ↓                   ↓                    ↑
                    └───── /joint_states ────────────────────┘
                                                             ↓
                                         TF: base_link → child links

[static_transform_publisher]
                                ↓
                   TF: world/map → base_link (fixed)

[odometry]
                                ↓
                   TF: odom → base_link

[SLAM / AMCL]
                                ↓
                   TF: map → odom
```

All transforms are broadcast into the shared **TF tree**. Standard ROS navigation frame chain:

`map → odom → base_link → <sensor/limb links>`

## `robot_state_publisher`

Publishes `base_link → <rest of robot>` by:

1. Reading the URDF/xacro (robot description).
2. Subscribing to `/joint_states`.
3. Computing and broadcasting TF transforms for every link — both **fixed** joints (directly from URDF) and **moving** joints (from `/joint_states`).

Example for an arm:

```
base_link → shoulder → elbow → wrist
```

## `joint_state_publisher`

Publishes `/joint_states` — one `JointState` message per joint. Sources:

| Source | Description |
|---|---|
| `joint_state_publisher` node | Reads URDF; publishes default (zero) values for all joints. Useful for visualization/testing. |
| `joint_state_publisher_gui` | Same, but with a GUI slider for manual control. |
| Your hardware driver node | Reads real encoder/sensor values and publishes to `/joint_states`. |

## `static_transform_publisher`

Broadcasts fixed transforms that never change, e.g. `world → base_link` or `base_link → lidar_link`. These are published once (latched) rather than at a recurring rate.

## TF at Different Publishing Frequencies

Different publishers emit transforms at different rates (e.g., odometry at 50 Hz, robot_state_publisher at 10 Hz, static transforms once). **`tf2` handles this transparently** via a time-stamped transform buffer:

- Each transform is stored with a timestamp in the `tf2` buffer (default 10-second history).
- When a node queries `lookupTransform`, `tf2` **interpolates** between the two nearest timestamped transforms in the buffer.
