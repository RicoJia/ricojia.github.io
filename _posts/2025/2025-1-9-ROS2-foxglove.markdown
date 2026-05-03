---
layout: post
title: ROS2 - Foxglove
date: 2025-1-9 13:19
subtitle: ROS2, Waveshare
header-img: img/post-bg-os-metro.jpg
tags:
  - Robotics
  - ROS2
comments: true
---
## Basic Architecture

- `foxglove_bridge` is the ROS-side bridge/server. It runs in your ROS environment as a node and exposes live ROS data to Foxglove over a WebSocket connection.
- `Foxglove` is the client UI. It connects to the bridge, often at something like ws://localhost:8765/, to visualize and interact with the live ROS system.
  - **Exception: Foxglove can also open recorded files such as MCAP directly for offline analysis, without a running bridge.**

The two modes of operations are below:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/mgJ56ZBx/e423408c-bb90-490c-94cb-eeba2d252b0c.png" height="300" alt=""/>
    </figure>
</p>
</div>

## Basic Layout

For Foxglove, the simplest setup is one `3D` panel. That single panel can show the robot model, live TF motion, and both multibeam point clouds in the same view. Use this sequence:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://assets.foxglove.dev/website/blog/announcing-foxgloves-new-3d-panel/settings.webp" height="300" alt=""/>
    </figure>
</p>
</div>

1. Install and open Foxglove Desktop.
2. Open the MCAP file inside your bag directory, not just the folder. In your case that will be the `.mcap` file under run_001_1776257690_1776258078_bag.
3. Add one `3D` panel.
4. In that 3D panel, set the fixed frame to `map` if you want to see the robot move through space. Use `base_link` only if you want the robot visually locked in place.
5. In the 3D panel, enable transforms from `/tf` and `/tf_static`.
6. Add a robot model layer. Point it at `/robot_description` if Foxglove detects the URDF topic automatically. If it does not, load the URDF file directly from your workspace and keep TF enabled.
7. Add point cloud layers for each point cloud topic
8. Set point size and decay time so the clouds remain readable. Start with a small point size and short decay, then adjust.
9. Press play and orbit the 3D camera. With fixed frame, you should see the robot move while the model and clouds stay aligned through TF.

For adding a robot model,

1. If you need to add a mesh, note that you need the absolute path to the mesh.

```xml
//instead of 
<mesh filename="package://aqmk2_description/model/meshes/base_link.dae"/>
// use
<mesh filename="/root/my_ws/install/MY_DESCRIPTION/share/MY_DESCRIPTION/model/meshes/base_link.dae"/>
```
