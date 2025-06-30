---
layout: post
title: Robotics - ROS2 Behavior Tree Cpp
date: '2024-12-05 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---


## Nodes

- A `CoroAction` is to pause, send an action goal, wait, get an action result, resume.
- destructor order:
    1. the dtor body
    2.members in a reverse order

## Quirks & Known Bugs

### Do NOT put a tf_listener in `init()` (Till ROS Humble)

This is not a BT bug, yet a bug in `TransformListener`. The call chain (on the same thread) is:

```
FixedStandoffOrbit::init() -> initialize a tf_listener
```

The issue is that [an initialized tf_listener may cause a hanging issue](https://code.hmech.us/nautilus/commander/commander/-/blob/master/toolkitt_behaviors/src/path_planners/fixed_standoff_orbit.cpp#L91) if destruction comes too soon. [Here is an issue report detailing this issue](https://github.com/ros2/geometry2/issues/517). Here's why in summary:

Currently, the `tf_listener` creates its own `SingleThreadedExecutor` and spin the executor on a separate thread.

```cpp
auto executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
auto run_func =
[executor](rclcpp::node_interfaces::NodeBaseInterface::SharedPtr node_base_interface) {
    executor->add_node(node_base_interface);
    executor->spin();
    executor->remove_node(node_base_interface);
};
dedicated_listener_thread_ = thread_ptr(
new std::thread(run_func, node_base_interface),
[executor](std::thread * t) {
    executor->cancel();
    t->join();
    delete t;
});
```

In the case where the `dedicated_listener_thread_` does not get to start before main calls the destructor (and tries to join the thread), the destructor hangs because `dedicated_listener_thread_` is still running.

1. main thread creates `dedicated_listener_thread_`
2. main thread perform all its jobs and `TransformListener` destructor
3. Main thread in TransformListener destructor calls `executor->cancel()` which sets `executor.spinning = false`
4. Main thread waits for `dedicated_listener_thread_` to finish
5. Dedicated thread starts running and calls `executor->spin()`
6. Dedicated thread at the beginning of `executor->spin()` sets `executor.spinning = true`
7. Dedicated thread is spinning waiting for messages or cancel which will not happen because main thread is blocked in `t->join();`
