---
layout: post
title: Robotics - ROS2 Service
date: '2024-11-22 13:19'
subtitle: How to Bring Up a Service 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
    - Docker
comments: true
---

## Create a ROS2 Service

ROS2's ament build system does not support generating `msg` and `srv` files in pure python packages. So it's a common practice to define an interface package that could be imported into pure Python Packages. 

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