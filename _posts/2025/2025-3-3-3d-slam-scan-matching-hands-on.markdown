---
layout: post
title: Robotics - [3D SLAM - 2] Scan Matching Hands-On Notes
date: '2025-3-3 13:19'
subtitle: Preact-Mojave
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

## Preact-Mojave Build & Workflow

The ROS2 driver for Preact-Mojave [can be found here](https://github.com/preact-tech/tofcore_ros). When testing it, I ran into below questions:

- Why `make provision` & `make ros2`?
    - `make provision` installs any missing OS-level deps via apt-get install ….
    - `make ros2` runs colcon build in the ros2/ subfolder with the right flags.
        - The top-level `Makefile` actually does `cd ros2 && colcon build --merge-install`.
        - If you run colcon build at the root without an install space or without ROS_PACKAGE_PATH pointing at ros2/, your packages won’t be found.

- The driver's visualization seems noisy. Is it possible to make it look better?
    - Search for the official promo video and download the highest-quality version before visualizing.
    - I bumped up the point size and tweaked the colors so the 3D view is crystal-clear.

- Lidar Odometry: Keyframes vs. full matching
    - Only add keyframes into your point-cloud map.
    - But don’t skip frame-to-frame matching—this detailed check often exposes alignment issues that keyframe-only tests miss.
    - Leaf size tuning
        - Beware: lowering the leaf size too much can break alignment.
        - add_scan_leaf_size: 0.1 (default)

- How bad are scan distortions? It could definitely create odometry drift. In the below snapshot, a wall is "curved" instead of being straight.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/xd3m5xLr/2025-05-20-15-48-41.png" height="300" alt=""/>
    </figure>
</p>
</div>