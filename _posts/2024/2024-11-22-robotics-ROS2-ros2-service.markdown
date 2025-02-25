---
layout: post
title: Robotics - ROS2 Service
date: '2024-11-22 13:19'
subtitle: How to Bring Up a Service 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Set up a ROS2 Service File

ROS2's ament build system does not support generating `msg` and `srv` files in pure python packages. So it's a common practice to define an interface package that could be imported into pure Python Packages. [See ROS2 interfaces](https://docs.ros.org/en/foxy/Concepts/About-ROS-Interfaces.html) 

1. Create a `cmake` package as an interface. 
    
    ```cmake
    ros2 pkg create --build-type ament_cmake mumble_interfaces
    ```

1. Place your srv file and `srv` there. 

    ```yaml
    # Request
    float64 start_x
    float64 start_y
    float64 goal_x
    float64 goal_y
    ---
    # Response
    float64[] path_x
    ```

1. Then, make sure these are in files:

    ```xml
    // CMakeLists.txt
    find_package(ament_cmake REQUIRED)
    find_package(rosidl_default_generators REQUIRED)
    rosidl_generate_interfaces(${PROJECT_NAME}
    "srv/ComputePath.srv"
    )
    
    // package.xml
    <build_depend>rosidl_default_generators</build_depend>
    <exec_depend>rosidl_default_runtime</exec_depend>
    ```

1. Add srv as a dependency to CMake:

    ```cmake
    find_package(Mumble_interface REQUIRED)
    ament_target_dependencies(your_target Mumble_interface)
    ```

1. Add srv as a dependency to `package.xml`

    ```xml
    <depend>Mumble_interface</depend>
    <build_depend>Mumble_interface</build_depend>
    <exec_depend>Mumble_interface</exec_depend>
    ```

1. Build and source `setup.bash`: `colcon build && source install/setup.bash`

1. To example if a srv file has been defined correctly: `ros2 interface list | grep YOUR_SERVICE`

1. Finally, add this into your code:

    ```python
    #Python
    from mumble_interfaces.srv import ComputePath
    ```

## Create a Service Server in Python

By default, **a ros2 service call back is not running on a separate thread.** 

- If we are using a single function:

```python
def motor_command(request, response):
    print(request)
    return response

node = Node("imu_broadcaster")
node.create_service(MotorCommand, "motor_command", motor_command)

executor = MultiThreadedExecutor()
executor.add_node(node)
```

- To validate:

``` bash
ros2 service call /motor_command mumble_interfaces/srv/MotorCommand "{left_speed: 0.0, right_speed: 0.0, duration: 0.0}"
```

## Create a Service Client In Python

- We might need to use a separate callback group (Mutually Exclusive or Reentrant) for ros services in general;
- Note we need to spin a daemon thread for `rclpy.spin()`, which really is a thin wrapper of a default executor

```python
import rclpy
import time
from rclpy.node import Node
from mumble_interfaces.srv import MotorCommand
import threading

def call_motor_service_periodically(node, client, rate_hz=1.0):
    """Calls the motor_command service at a fixed rate."""
    rate = node.create_rate(rate_hz)

    i = 0.0
    while rclpy.ok():
        i+=1.0
        request = MotorCommand.Request()
        request.left_speed = i
        request.right_speed = 0.0
        request.duration = 0.0

        future = client.call_async(request)
        node.get_logger().info("Sent motor command request.")
        # If in a class, we can pass make the class a child class of node, and pass it in.
        print(future.result())
        rate.sleep()

def main():
    rclpy.init()
    node = Node('motor_command_caller')
    client = node.create_client(MotorCommand, 'motor_command')
    spin_thread = threading.Thread(target = rclpy.spin, args=(node, ), daemon=True)
    spin_thread.start()

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("Waiting for service...")

    try:
        call_motor_service_periodically(node, client, rate_hz=2.0)  # Call at 2 Hz
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```