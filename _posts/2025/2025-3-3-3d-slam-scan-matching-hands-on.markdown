---
layout: post
title: Robotics - [3D SLAM - 2] Scan Matching Hands-On Notes
date: '2025-3-3 13:19'
subtitle: Preact-Mojave, Lidar-Only Front-End
header-img: "img/post-bg-o.jpg"
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

## SLAM Pipeline Issues And Improvements

- `.getFitnessScore()` actually returns the **mean squared distance from each point in the source cloud to its closest point** in the target cloud. By itself, this metric can be misleading—two scans that share only **a large, flat plane** may outperform a correct pair that has **lots of small features**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/Yj92ZtdG/2025-05-24-17-56-53.png" height="200" alt=""/>
        <img src="https://i.postimg.cc/nzHhBJSM/2025-05-24-15-13-24.png" height="200" alt=""/>
    </figure>
</p>
</div>

### Front End Improvements

- Switch to PCL’s NDT implementation.

Point filtering

- [Moving-Least-Squares Smoothing](https://www.sci.utah.edu/~shachar/Publications/crpss.pdf) takes up around 80ms to per ~20k points. The surface of objects are smoother, but in many cases, this smoothing does not change point distribution much and is optional
  - How it works:
        1. Fit a plane H across a set of points.
        2. Calculate height of each point w.r.t H, $p_i$, and their projections onto H, $(x_i, y_i)$
        3. Fit a low-degree bivariate polynomial g(x,y) across all $p_i$. Then, the output MLS points are $g(x_i, y_i)$

    <div style="text-align: center;">
    <p align="center">
        <figure>
            <img src="https://i.postimg.cc/MM7pJV7S/pcl-no-smooth.png" height="200" alt=""/>
            <img src="https://i.postimg.cc/zHqXzv7R/pcl-with-smooth.png" height="200" alt=""/>
            <figcaption>Left: no MLS, Right: with MLS</figcaption>
        </figure>
    </p>
    </div>

### Loop Detection Improvements

- Reject single-plane match by hessian conditioning:

$$
\begin{gather*}
\begin{aligned}
& cond = \frac{\lambda_{max}}{\lambda_{min}}
\end{aligned}
\end{gather*}
$$

The ndt hessian

- Symmetric Fitness Check: Run NDT A->B and B->A
