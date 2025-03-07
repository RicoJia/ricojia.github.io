---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Parameters
date: '2024-11-22 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## ROS2 Parameters Intro

[What are ROS2 parameters?](https://docs.ros.org/en/humble/Concepts/Basic/About-Parameters.html)

ROS2 Parameters are one means to change certain values in a node during runtime. They are associated initialized, set, and retrieved at the level of individual nodes, and hence their lifetimes are the same as their nodes' lifetimes.

Each parameter has **a key, value, and a descriptor**. 

- The key is a string
- The value is one of the following types: `bool, int64, float64, string, byte[], bool[], int64[], float64[] string[]`
- The descriptor is a parameter's description

[Here is an example of how we can use them](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Parameters/Understanding-ROS2-Parameters.html):

1. Initialize / declare
    - Command Delare params? 
        ```python
        from rclpy.node import Node
        class ParameterNode(Node):
            def __init__(self):
                super().__init__('parameter_node')
                self.declare_parameter('shared_value', 42.0)
        ```
    - When a parameter needs to take in different types, use `ParameterDescriptor` with the dynamic_typing member variable set to true
        ```python
        from rcl_interfaces.msg import ParameterDescriptor
        descriptor = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('dynamic_param', 0, descriptor)
        ```
    - In CLI:
        - `ros2 param list` shows all parameters:
            ```
            /teleop_turtle:
                qos_overrides./parameter_events.publisher.depth
                use_sim_time
            ```
- Get
    - `param_value = self.get_parameter_or(param_name, None)`
    - For parameters that can't be known ahead of time, they can be instantiated as  `allow_undeclared_parameters = True`
        ```python
        class MyNode(Node):
            def __init__(self):
                super().__init__('my_node', allow_undeclared_parameters=True) 
            def get_dynamic_param(self, param_name):
                param_value = self.get_parameter_or(param_name, None)
                return param_value
        def main(args=None):
            ...
            node.get_dynamic_param('undeclared_param')  # Should print a warning and return None
        ```
    - In CLI:
        - `ros2 param get <node_name> <parameter_name>`
        - View all params of a node, and they can be saved into an yaml:
            ```
            ros2 param dump <node_name>
            ros2 param dump /turtlesim > turtlesim.yaml
            ```
- Set.
    - In code?
        ```python
        from rclpy.parameter import Parameter
        class MyNode(Node):
            def update_param(self, key, val):
                param = Parameter(key, Parameter.Type.INTEGER, val)
                self.set_parameters([param])
        ```
    - In CLI: 
        - `ros2 param set <node_name> <parameter_name> <value>`
        - Load from a yaml file:
            ```
            ros2 param load <node_name> <parameter_file>
            ```
        - Load parameters on start up:
            ```
            ros2 run <package_name> <executable_name> --ros-args --params-file <file_name>
            ```

They can be persisted as well: 