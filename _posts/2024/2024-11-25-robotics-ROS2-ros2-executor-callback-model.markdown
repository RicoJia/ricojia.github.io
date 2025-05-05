---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Executor Callback Model
date: '2024-11-25 13:19'
subtitle: Executor, Callbacks, Threading Model, Rate Object
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---
## [Executors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html)

In ROS2, we can have multiple logical nodes executing their callbacks of timer, subscription, etc. in executors. There are three types of executors: `SingleThreadedExecutor`, `StaticSingleThreadedExecutor` and `MultiThreadedExecutor`

Regardless of the type of executor, the basic message mechanism is: 

1. In the middleware (RMW/DDS), each topic, timer, service and client has its own incoming buffer tracked by a boolean `rcl_wait_set_t`.
2. When new data arrives on a subscription/time/etc., its `rcl_wait_set_t` flag is set to `true`.
3. The executor calls `rcl_wait(&wait_set, timeout)`, blocking until at least one flag is `true`.
4. At some point, the new data will be processed by executor's `rcl_take`. **the timing is different based on the type of the executor**

**Key difference** from ROS 1: `spin()` offloads incoming-message storage to the RMW/DDS layer instead of buffering in the client library.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/qgftJtq5/2025-05-02-10-08-54.png" height="300" alt=""/>
    </figure>
</p>
</div>

Also, executors support multiple local nodes to be executed: 

```cpp
// two entirely independent nodes …
auto node_a = std::make_shared<rclcpp::Node>("node_a");
auto node_b = std::make_shared<rclcpp::Node>("node_b");

rclcpp::executors::StaticSingleThreadedExecutor exec;
exec.add_node(node_a);
exec.add_node(node_b);
exec.spin();
```

### `SingleThreadedExecutor` 

**The workflow is**

For every cycle of the `spin loop`: 
1. Executor scans all of your Nodes → all their callback-groups → all their subscriptions, timers, services, clients, guard-conditions, etc.
2. Executor calls `rcl_wait` for next incoming callback
3. RMW builds up a fresh `rcl_wait_set` and unblocks `rcl_wait`
4. Exeuctor walks the callback handlers **in the exact order the entities were added to the executor** and calls `rcl_take` (or timer callbacks, service callbacks, …) one by one

- This per-cycle “scan + rebuild wait_set” model gives you flexibility (you can add or remove subscriptions/timers at runtime) at the cost of overhead if you have a very high callback rate or a lot of entities to scan.

In code, this is 

```cpp
rclcpp::spin(node);

// or 
rclcpp::executors::SingleThreadedExecutor exec;
exec.add_node(node);
exec.spin();
```

### `StaticSingleThreadedExecutor` 

During initialization `exec.add_node(node)`, `StaticSingleThreadedExecutor` initialize timer/subscriptions and wouldn't remove/add to them during runtime.

- That includes a single `rcl_wait_set_t` (with fixed arrays of handles for each entity type). on the `rcl-ROS layer`

Then At the beginning of each cycle of the `spin loop`:

1. Executor calls rcl_wait for next incoming callback
2. RMW updates the `rcl_wait_set` array from that scan and unblocks `rcl_wait`
3. Exeuctor walks the ready‐flags in order and calls rcl_take (or timer callbacks, service callbacks, …) one by one

        
### `MultiThreadedExecutor` 

`MultiThreadedExecutor` is similar to the single-threaded version, but it hands off ready handles into a thread pool.

For every cycle of the `spin loop`:

1. Executor scans all of your Nodes → all their callback-groups → all their subscriptions, timers, services, clients, guard-conditions, etc.
2. Executor calls `rcl_wait` for next incoming callback
3. RMW builds up a fresh `rcl_wait_set` and unblocks `rcl_wait`
4. Exeuctor walks the callback handlers **in the exact order the entities were added to the executor** and calls `rcl_take` (or timer callbacks, service callbacks, …) one by one. **Then these handlers are dispatched to a thread pool**

- Note, only **Reentrant** group can be executed in parallel


## [Callback groups](https://docs.ros.org/en/humble/How-To-Guides/Using-callback-groups.html)

In ROS2, callback groups are a synchronization mechanism for managing concurrent execution in multi-threaded executors. They determine which callbacks can run simultaneously, similar to how locks work for critical-section access control. 

 There are two groups:

- Mutually Exclusive Callback (MEC) Group: 
    - Prevents its callbacks from being executed in parallel, so that callbacks in the group were executed by a SingleThreadedExecutor.
        - Imagine in a MultiThreadedExecutor, we would need a lock for functions in this group to make them serialized.
    - A node’s default callback group is MEC
- Reentrant callback group: 
    - "Callbacks in this group promise to be thread-safe, so run them however you like." 
    - So this allows the executor to schedule and execute the group’s callbacks without restrictions.

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


