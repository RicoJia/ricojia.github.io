---
layout: post
title: Robotics - ROS2 TF2
date: '2024-12-05 13:19'
subtitle: Transform Listener
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## TF2 and Sim Time

When working with ROS2 and tf2, you may encounter the following warning when visualizing data in RViz or processing buffers:

```
Warning: TF_OLD_DATA ignoring data from the past for frame base_link at time 4.294967 according to authority Authority undetectable
```

This message indicates that tf2 is dropping out-of-order or stale transforms because your TF publisher is using a "simulated time" (or older timestamps), while the receiver uses the default system clock. In that case, we can either hit the reset button, or we need to run nodes using `use_sim_time`

- You only need to drive the /clock topic when you’ve told your nodes / rviz to run in simulation time (i.e. you’ve set the use_sim_time parameter to true, by default, use_sim_time=false).
  - `node->declare_parameter("use_sim_time", true);`
  - or `rviz use_sim_time:=true`

- One example is replaying a bag and displaying on rviz. `ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true`

In your tf publisher node:

1. Enable Sim Time Parameter. In your node’s constructor or setup, set:

```cpp
node_->set_parameter(rclcpp::Parameter("use_sim_time", true));
```

2. Create a Clock Publisher. Declare and initialize a publisher for `rosgraph_msgs::msg::Clock`:

```cpp
rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
clock_pub_ = node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
```

3. Publish clock and tf together:

```cpp
// Build and publish clock
rosgraph_msgs::msg::Clock clk;
clk.clock = imu_msg->header.stamp;  // reuse the IMU’s timestamp
clock_pub_->publish(clk);
imu_tf_publisher_->publish(*imu_msg);
```

## Transform Listener

### Life Cycle

- tf2_ros::TransformListener starts its own subscription `/tf`, `/tf_static`:
    1. `node_->create_subscription<tf2_msgs::msg::TFMessage>( … )`
    2. stores that subscription in the same callback group / executor your node is using, and

### Quirks

- rclcpp does not support callback in tf_listener.
