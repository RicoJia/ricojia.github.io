---
layout: post
title: ROS2 - Time Synchronizer
date: 2025-1-16 13:19
subtitle:
header-img: img/post-bg-os-metro.jpg
tags:
  - Robotics
  - ROS2
comments: true
---
## Time Synchronizer

The `ApproximateTimeSynchronizer` is useful when multiple ROS topics need to be processed together, but their timestamps do not match exactly. In this example, color images, depth images, camera info, and detection messages are synchronized using a queue size of `8` and a `slop` value of `0.15`. The `slop` parameter defines the maximum allowed timestamp difference between messages in the same synchronized group. In other words, if messages from the four topics arrive within `0.15` seconds of each other, they can be treated as belonging to the same sensor frame and passed to the callback together. This is especially helpful in real robotic perception pipelines, where different sensors or processing nodes may introduce small timing delays.

```python
self.sync = ApproximateTimeSynchronizer(
	[color_sub, depth_sub, info_sub, detection_sub],
	queue_size=8,
	slop=0.15,
)
```
- slop is the max amount of time being permitted to mismatch between messages. 
- If messages' are within 0.15s from each other

