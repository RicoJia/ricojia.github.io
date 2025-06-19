---
layout: post
title: Robotics - ROS2 Launchfile
date: '2024-11-9 13:19'
subtitle:
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Components

### Launch Actions

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    # Grab the value of a declared argument at runtime:
    mode = LaunchConfiguration('mode').perform(context)
    if mode == 'foo':
        return [ Node(package='pkg', executable='foo_node', output='screen') ]
    else:
        return [ Node(package='pkg', executable='bar_node', output='screen') ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='foo'),
        OpaqueFunction(function=launch_setup),
    ])
```

- `DeclareLaunchArgument('mode', default_value='foo')` defines an argument that could be overriden on commandline. E.g., `ros2 launch your_pkg your_launch.py mode:=bar`
    - You can see launchfile args like: `ros2 launch your_pkg your_launch.py --show-args`
- `LaunchConfiguration('mode').perform(context)` is one true way to pull in a CLI arg at runtime.
- `OpaqueFunction` is basically a hook for calling your custom function. It's "opaque" because it's a blackbox to the launch system - whose behavior is unknown until runtime.
    - This concept came from compiler theory, which means "a call whose internals are unknown to the optimizer"
    - C/C++: Opaque pointers `(typedef struct Foo Foo;)` hide implementation details behind an incomplete type. It's the core of Pimpl idiom


