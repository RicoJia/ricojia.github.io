---
layout: post
title: Robotics - ROS2 Gazebo Differential Drive Onboard
date: '2024-11-7 13:19'
subtitle: ROS2, Gazebo
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---
## Install Gazebo

```
sudo apt install ros-iron-gazebo-ros-pkgs ros-iron-gazebo-ros2-control
```

### Add A Launchfile

File Directory:

```
├── launch
│   └── empty_world.launch.py
```

- `mkdir launch`
- `setup.py` Make sure to have:

    ```python
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all files in the launch directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    ```

- `empty_world.launch.py`

```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        )
    ])
```

-

### Add An Arg In Launch File

```python
def generate_launch_description():
    # Declare launch arguments
    no_rviz_arg = DeclareLaunchArgument(
        'no_rviz',
        default_value='false',
        description='Flag to disable RViz'
    )
    no_rviz = LaunchConfiguration('no_rviz')
    # register the arg and the node
    ld.add_action(no_rviz_arg)
    ld.add_action(rviz_node)
```
