---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Executor Callback Model
date: '2024-11-22 13:19'
subtitle: Executor, Callbacks, Rate Object
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---
## [Executors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html)

TODO

## [Callback groups](https://docs.ros.org/en/humble/How-To-Guides/Using-callback-groups.html)

In a ROS2 multi-threaded executor, being in a callback group is kind of like acquiring a multi-threaded lock to grant permission to fire. There are two groups:

- Mutually Exclusive Callback Group: it prevents its callbacks from being executed in parallel, so that callbacks in the group were executed by a SingleThreadedExecutor.
- Reentrant callback group: it allows the executor to schedule and execute the group’s callbacks without restrictions.

Some other rules:

- Different ROS 2 entities relay their callback group to all callbacks they spawn. E.g.,, if one assigns a callback group to an action client, all callbacks created by the client will be assigned to that callback group.

- Callbacks belonging to different callback groups (of any type) can always be executed parallel to each other.

- A node’s default callback group is a MEC.

- **You are able to create multiple MEC or Reentrant groups!**


In ROS 2 executors, a callback means a function whose scheduling and execution is handled by an executor. Examples of callbacks in this context are:

- subscription callbacks (receiving and handling data from a topic),
- timer callbacks,
- service callbacks (for executing service requests in a server),
- different callbacks in action servers and clients,
- done-callbacks of Futures.

**ROS 2 is event driven!!**

- Every function that is run by an executor is, by definition, a callback. The non-callback functions in a ROS 2 system are found mainly at the edge of the system (user and sensor inputs etc).

- Sometimes the callbacks are hidden in the user/developer API. This is the case especially with any kind of “synchronous” call to a service or an action (in rclpy). 
    - For example, the synchronous call `Client.call(request)` to a service adds a Future`’s done-callback` that needs to be executed during the execution of the function call, but this callback is not directly visible to the user.

## Controlling Execution

For the interaction of an individual callback with itself:

    - Register it to a Reentrant Callback Group if it should be executed in parallel to itself. 
        - An example case could be an action/service server that needs to be able to process several action calls in parallel to each other. 
    - Register it to a Mutually Exclusive Callback Group if it should never be executed in parallel to itself. An example case could be a timer callback that runs a control loop that publishes control commands.

For the interaction of different callbacks with each other:

    - Register them to the same Mutually Exclusive Callback Group if they should never be executed in parallel. An example case could be that the callbacks are accessing shared critical and non-thread-safe resources.

If they should be executed in parallel, you have two options, depending on whether the individual callbacks should be able to overlap themselves or not:

    - Register them to different Mutually Exclusive Callback Groups (no overlap of the individual callbacks)
    - Register them to a Reentrant Callback Group (overlap of the individual callbacks)

An example case of running different callbacks in parallel is a Node that has a synchronous service client and a timer calling this service.

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


