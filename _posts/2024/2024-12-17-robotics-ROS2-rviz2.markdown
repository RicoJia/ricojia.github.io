---
layout: post
title: Robotics - rviz2
date: 2024-12-17 13:19
subtitle:
header-img: img/post-bg-os-metro.jpg
tags:
  - Robotics
  - ROS2
comments: true
---
---

## Visualization Marker

If you’ve ever wanted to display text directly in RViz2—for debugging, labeling, or status messages—you can do this using visualization markers.

The key is to use a marker of type `TEXT_VIEW_FACING`, which ensures the text always faces the camera.

### Example

```python
from visualization_msgs.msg import Marker  
from rclpy.node import Node  
  
marker = Marker()  
marker.header.frame_id = "map"  
marker.type = Marker.TEXT_VIEW_FACING  
marker.action = Marker.ADD  
  
marker.pose.position.x = 0.0  
marker.pose.position.y = 0.0  
marker.pose.position.z = 1.0  
  
marker.scale.z = 0.5 # text size  
marker.color.a = 1.0  
marker.color.r = 1.0  
  
marker.text = "Hello RViz"  
  
publisher.publish(marker)
```

- would it hurt to launch the rviz twice in two different launch files - no, but you will get two instances of rviz.

---

## Using Rviz2

- Launching RViz multiple times: It won’t hurt, but you’ll get two separate RViz instances. Each instance runs independently, opens its own window, and subscribes to topics separately. This can be useful for debugging different views, but it also means extra CPU/GPU usage and duplicated subscriptions. In most cases, it’s cleaner to launch a single RViz instance and reuse it across your system.
