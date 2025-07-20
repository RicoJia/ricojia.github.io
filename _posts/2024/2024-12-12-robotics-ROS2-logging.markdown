---
layout: post
title: Robotics - ROS2 Logging
date: '2024-12-12 13:19'
subtitle: Logging Level Tweaking
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Logging Level Tweaking

1. Programmatic API (in-process)
In C++, any `rclcpp::Logger` can have its level tweaked at runtime:

    ```cpp
    #include <rclcpp/logger.hpp>
    auto my_logger = rclcpp::get_logger("my_node");
    my_logger.set_level(rclcpp::Logger::Level::Debug);
    ```

2. With the default `rcl_logging_spdlog` backend, ROS 2 exposes two services per processes:

    - `/<node_name>/get_logger_levels (rcl_interfaces/srv/GetLoggerLevels)`
    - `/<node_name>/set_logger_levels (rcl_interfaces/srv/SetLoggerLevels)`

So you can call them during runtime: 

    ```
    ros2 service call /my_node/set_logger_levels \
    rcl_interfaces/srv/SetLoggerLevels "{ levels: [ { name: 'my_node', level: 10 } ] }"
    ```

3. Also, one can browse logs on rqt_logger: 

    ```
    ros2 run rqt_logger_level rqt_logger_level
    ```

4. Or, `ros2 run pkg node --ros-args --log-level my_node:=DEBUG`
    - I was having issues with doing this in gtest though.