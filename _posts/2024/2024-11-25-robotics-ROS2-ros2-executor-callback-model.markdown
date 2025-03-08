---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Executor Callback Model
date: '2024-11-22 13:19'
subtitle: Executor, Callbacks, Threading Model, Rate Object
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---
## [Executors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html)

TODO

## [Callback groups](https://docs.ros.org/en/humble/How-To-Guides/Using-callback-groups.html)

In ROS2, callback groups are a synchronization mechanism for managing concurrent execution in multi-threaded executors. They determine which callbacks can run simultaneously, similar to how locks work for critical-section access control. 

 There are two groups:

- Mutually Exclusive Callback (MEC) Group: 
    - Prevents its callbacks from being executed in parallel, so that callbacks in the group were executed by a SingleThreadedExecutor.
    - A node’s default callback group is MEC
- Reentrant callback group: 
    - Allows the executor to schedule and execute the group’s callbacks without restrictions.

Key rules:

- Callbacks created by a ROS 2 entity (e.g., action clients) inherit the assigned callback group. 
    - E.g.,, if one assigns a callback group to an action client, all callbacks created by the client will be assigned to that callback group.

- Callbacks in different groups can execute in parallel.

- Multiple MECs or Reentrant groups can be created as needed.

### Examples of Callbacks

- Subscription callbacks (receiving and handling data from a topic),
- Timer callbacks,
- Service callbacks (for executing service requests in a server),
- Action callbacks in both action servers and clients
- Done-callbacks of Futures. E.g., `Client.call(request) in rclpy`

## Controlling Callback Execution

- Within a Single Callback:

    - Allowing Parallel Self-Execution: use a Reentrant Callback Group if the same callback should execute concurrently (e.g., processing multiple service requests in parallel).

    - Preventing Overlap: Use a Mutually Exclusive Callback Group if a callback must not run concurrently with itself (e.g., a timer callback running a control loop).

- Between Different Callbacks:
    - Non-Parallel Execution: Assign callbacks that share non-thread-safe resources to the same MEC, ensuring they do not run concurrently.
    - Parallel Execution: Use separate MECs if individual callbacks must not overlap themselves.
    Alternatively, use a single Reentrant group to allow concurrent execution.

An example case of running different callbacks in parallel is a Node that has a synchronous service client and a timer calling this service.

### Threading Model (Subscriber, Service, etc.)

ROS 2 provides two main threading models for subscription callbacks:
    - By default, all callbacks run on a single thread.

```python
rclpy.init()
node = rclpy.create_node("single_threaded")
executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(node)
executor.spin()  # All callbacks run in ONE thread
```

- Multi-Threaded Executor (Parallel Processing of Callbacks) with Reentrant Callback Groups (More Control over Threads) When using MultiThreadedExecutor, you can explicitly declare that certain callbacks should run in parallel by using ReentrantCallbackGroup.

```python
from rclpy.callback_groups import ReentrantCallbackGroup
group = ReentrantCallbackGroup()
node.create_subscription(Imu, "imu_data", imu_callback, 10, callback_group=group)
node.create_subscription(LaserScan, "scan", scan_callback, 10, callback_group=group)
```

## `rclpy.Rate`

`rclpy.Rate` is a wrapper around a ROS Timer callback. Any callback in ROS 2 needs an executor to execute. `rclpy.spin()` is a wrapper around an executor, and it checks if any callback needs to be executed. So, `rclpy.Rate()` needs to communicate with a `rclpy.spin()` on a separate thread. This is a key difference in exeuction model from ROS1

[Reference](https://robotics.stackexchange.com/questions/96684/rate-and-sleep-function-in-rclpy-library-for-ros2)

```python
import rclpy

rclpy.init()
node = rclpy.create_node('simple_node')

# Spin in a separate thread
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
thread.start()

rate = node.create_rate(2)

try:
    while rclpy.ok():
        print('Help me body, you are my only hope')
        rate.sleep()
except KeyboardInterrupt:
    pass

rclpy.shutdown()
thread.join()
```


