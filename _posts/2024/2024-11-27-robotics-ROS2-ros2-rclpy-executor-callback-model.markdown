---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Rclpy Executor Callback Model
date: '2024-11-27 13:19'
subtitle: Rclpy Executors, Rate Object
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Rclpy `MultiThreadedExecutor`

`rclpy.executors.MultiThreadedExecutor` is implemented **entirely in Python, not as a direct wrapper around the C++ `rclcpp::executors::MultiThreadedExecutor`**. Internally it builds a `concurrent.futures.ThreadPoolExecutor` to run callbacks, so the threads it launches are ordinary Python threads that share the Global Interpreter Lock (GIL)

Here is the [source code of Rclpy's executors](https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/executors.py). In [MultiThreadedExecutor](https://github.com/ros2/rclpy/blob/9c3e9665e9ed60551980ddeaf24dba15d4140b0e/rclpy/rclpy/executors.py#L761),

```python
from concurrent.futures import ThreadPoolExecutor
class MultiThreadedExecutor(Executor):
    ...
    def __init__(self, num_threads=None, *, context=None):
        ...
        self._executor = ThreadPoolExecutor(num_threads)   # <-- Python thread pool
    def _spin_once_impl(
        self,
        timeout_sec: Optional[Union[float, TimeoutObject]] = None,
        wait_condition: Callable[[], bool] = lambda: False
    ) -> None:
        handler, entity, node = self.wait_for_ready_callbacks(
                timeout_sec, None, wait_condition)
        self._executor.submit(handler)
        self._futures.append(handler)
        for future in self._futures:  # check for any exceptions
            if future.done():
                self._futures.remove(future)
                future.result()
```
    - Note that if there's no `timeout_sec` specified, the `executor` could block indefinitely until a future becomes ready.

- Callbacks that spend most time C extensions, such as those in `rclpy._rclpy` bindings, `NumPy`, `OpenCV`, can overlap and really run in parallel on different OS threads.
- Both `rclpy.executors.MultiThreadedExecutor` and `rclcpp::executors::MultiThreadedExecutor` sit on top of the common C layer (rcl) and wait on the same DDS wait-sets. 
    - The C++ version does its scheduling and callback invocation in C++ threads with no GIL
    - Python implementation will time-slice under the GIL and does not use multiple cores.


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

`rclpy.Rate` is a wrapper around a ROS Timer callback. Any callback in ROS 2 needs an executor to execute. `rclpy.spin()` is **a wrapper around an executor**, and it checks if any callback needs to be executed. So, `rclpy.Rate()` needs to communicate with a `rclpy.spin()` **on a separate thread**. This is a key difference in exeuction model from ROS1

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


