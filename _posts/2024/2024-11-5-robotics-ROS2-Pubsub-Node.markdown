---
layout: post
title: Robotics - ROS2 Basic Pub Sub Node
date: '2024-11-5 13:19'
subtitle: ROS2 Basics
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Install ROS2

This is only to install `ROS2-iron-desktop` for quick desktop experiments. In general I'd like to use docker containers though.

```
# Remove any corrupted ROS2 keyrings.
sudo rm /etc/apt/keyrings/ros-archive-keyring.gpg
sudo mkdir -p /etc/apt/keyrings

# Download ROS2 GPG key
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg

# Verify the GPG keys are added correctly
# Should see "ros-archive-keyring.gpg"
ls /etc/apt/keyrings/

sudo rm /etc/apt/sources.list.d/ros2.list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# install comprehensive ROS2, like rviz, rqt, and demos
sudo apt install -y ros-iron-desktop
sudo apt install -y python3-colcon-common-extensions
sudo apt install ros-iron-joint-state-publisher
sudo apt install ros-iron-xacro

echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Run a talker and listener.
ros2 run demo_nodes_cpp talker
ros2 run demo_nodes_cpp listener
```

## Create a Python Package For Publisher & Subscriber

- `mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src`
- Create a package `ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs`
  - `--build-type` specifies that this is a python pkg.
  - `ros2 pkg create my_mixed_pkg --build-type ament_cmake --dependencies rclcpp rclpy std_msgs` is the command for creating cmake and python package.

- File Directory

```
├── dummy_test
│   ├── demo.py
│   ├── __init__.py
├── package.xml
├── resource
│   └── dummy_test
├── setup.cfg
├── setup.py
└── test
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

    - `resource`: holds files that do not need executable permissions like config, launch files, data files etc.
    - python executables **NEED executable permissions**, so they need to placed in `dummy_test`

- `setup.py`: so this node executable could be run by `ros2 run` command

```python
from setuptools import setup, find_packages

package_name = 'dummy_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools', 'rclpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='A minimal ROS 2 publisher node',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'demo = dummy_test.demo:main',
        ],
    },
)
```

- `dummy_test/demo.py`: this is the console script that could be found by the Python environment's `bin`
  - Note: `main()` function must be defined

```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```

- Build and run

```bash
chmod +x dummy_test/resource/dummy_test
colcon build --packages-select dummy_test --symlink-install
source install/setup.bash
ros2 run dummy_test demo
```

    - `--symlink-install` allows in-place modifications in `colcon build --packages-select dummy_test --symlink-install`


