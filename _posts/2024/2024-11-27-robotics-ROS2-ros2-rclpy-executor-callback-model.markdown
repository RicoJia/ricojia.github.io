---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Rclpy Executor Callback Model
date: '2024-11-27 13:19'
subtitle: Rclpy Executors, Rate Object, Custom Node Destruction
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

## Destruction

Sometimes, we want to publish / send service requests upon destructing a node. In that case, we need to disable the default `SIGINT` handler because it will shutdown our executor before destructing objects that publish / send service requests / roslog's own publisher:

```python
# We want our objects destructed first, so we can use adios services
rclpy.init(signal_handler_options=rclpy.SignalHandlerOptions.NO)
server = WebServerNode()
try:
    server.run()
except KeyboardInterrupt:
    pass                        # we caught SIGINT ourselves
finally:
    # executor is still spinning here
    server.destroy_node()
    rclpy.shutdown()            # now shut rclpy down
```

## Issues

### `ValueError: generator already executing`

The direct cause comes from Python’s generator machinery: you’ve got two threads both trying to drive the same spin‐loop generator in rclpy.  For example:

- if you spin a node on one thread

    ```python
    t = Thread(target=lambda: rclpy.spin(self.cmc), daemon=True)
    t.start()
    ```

- And call a service on another thread:

    ```python
    fut = self.load_cli.call_async(req)
    rclpy.spin_until_future_complete(self.cmc, fut)
    ```

- Under the hood, `spin()` calls `spin_once()`, but `spin_until_future_complete` will also invoke `spin_once()` on that same executor generator.
    - `SingleThreadedExecutor` is implemented around a Python generator (_wait_for_ready_callbacks)
    - `self.executor.spin_until_future_complete()` tries to re‑enter the same generator. Python generators are not re‑entrant, so the interpreter raises `ValueError: generator already executing`.
    - `MultiThreadedExecutor` doesn’t use that generator; it waits in C and dispatches callbacks through a thread‑pool, so the re‑entrancy never occurs—hence “it works with MultiThreadedExecutor.”


### `resp = fut.result()` returns None

`Future.result()` returning `None` just means the Future was never filled. The rclpy Future returns None when no result has been set yet

- fut = client.call_async(req) merely enqueues the request and returns a Future. That Future is populated only when the executor takes the response from DDS and runs the client’s response-handler callback.

[The rclpy docs spell the solution out: “Execute work until the future is complete.” You must either call rclpy.spin_until_future_complete() or keep an executor spinning concurrently.](https://docs.ros2.org/foxy/api/rclpy/api/init_shutdown.html?utm_source=chatgpt.com)
    - [ GitHub issue #1141 shows the exact symptom—Future is always None until an executor spins.](https://github.com/ros2/rclpy/issues/1141?utm_source=chatgpt.com)

E.g., Use a MultiThreadedExecutor executor (recommended)

```python
from rclpy.executors import MultiThreadedExecutor

self.executor = MultiThreadedExecutor()
self.executor.add_node(self.cmc)
# Spin in background
Thread(target=self.executor.spin, daemon=True).start()
```
