---
layout: post
title: Robotics - Differences and Similarities Between ROS1 & ROS2
date: '2024-11-2 13:19'
subtitle: Similarities, Differences, Bag Conversions
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Similarities

### Similar Constructs

- Both are based on a graph of nodes, topics, and services. In ros2, the user graph looks like:

    ```
    ros2 node list
    ros2 topic list
    ros2 service list
    ros2 action list
    ```

- Both have the same namespacing rules `/turtlesim`
  - Both provide node remapping: `ros2 run turtlesim turtlesim_node --ros-args --remap_node:=my_turtle`. Then we can see our node under `/my_turtle`
  - Be careful with the syntax `--ros-args --remap_node:`
  - Topic remapping: `ros2 run turtlesim turtle_teleop_key --ros-args --remap turtle1/cmd_vel:=turtle2/cmd_vel`

### Similar Packages

- `joint_state_publisher`
- `rqt` is available

    ```
    sudo apt install ~nros-foxy-rqt*
    rqt
    ```

  - `ros2 run rqt_console rqt_console` can be used to view logs
  - `ros2 topic pub -r 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}"` topic publishing is similar, too.
- `rviz2`
  - Make sure the RobotModel display is added to RViz to visualize the robot.
    - In RViz, go to Displays > Add.
    - Select RobotModel.
    - This should visualize the robot if robot_description is set correctly.

## Basic Differences

- C++ Standard: ROS1 uses C++03, C++11 is prevalent. ROS 2 uses C++ 11 and C++ 14.
- Python Standard: ROS: Python2, ROS 2: Python 3.5+.
  - Actually, Python 3.10+ is a safe bet. Packages like `RoboStack` (an rviz-like browswer visualization) are built in 3.10+.
  - ROS Noetic was built for Python 3.8. With Python 3.10, one might run into errors like `/usr/lib/python3/dist-packages/Cryptodome/Util/../Cipher/_raw_ecb.so: cannot open shared object file: No such file or directory`
- Build systems: ROS 1 packages must be `cmake` packages. Now ROS2 package can be a plain python packge as well. They can use everything in `setup.py` files, e.g., entrypoints since they are being invoked with `python3 setup.py install`

### CLI Differences

- ROS2: `source install/setup.bash`, ROS1: `source devel/setup.bash`
- ROS2: `colcon_cd <package>`, ROS1: `roscd <package>`
  - If you can't find the `colcon_cd` command, do `source /usr/share/colcon_cd/function/colcon_cd.sh`

#### Ros2 Params

- `ros2 param list /robot_state_publisher`. This will give a list of ros2 params
- `ros2 param get /robot_state_publisher robot_description` to get a single ros parameter server.

### Key Package Differences

### DDS

- Pub-Sub Data Flow

```
+----------------------+
|      user land       |   1) create a ROS message
+----------------------+      v
|  ROS client library  |   2) publish the ROS message
+----------------------+      v
| middleware interface |      Translate to middleware data format
+----------------------+      v
|      mw impl N       |   3) convert the ROS message into a DDS sample and publish the DDS sample
+----------------------+
```

```
|      user land       |   3) use the ROS message
+----------------------+      ^
|  ROS client library  |   2) callback passing a ROS message
+----------------------+      ^
| middleware interface |      Translate to middleware data format
+----------------------+      ^
|      mw impl N       |   1) convert the DDS sample into a ROS message and invoke subscriber callback
+----------------------+
```

QOS: Some DDS QOS parameters are exposed to ROS2: (TODO)


## Bag Conversions

I found a ROS_independent implementation of ROS1 <-> ROS2 conversion. I have an example of ROS1 bag post processing script that saves to a new ROS 1 bag.

- [See here for the overall framework](https://github.com/RicoJia/SimpleRoboticsUtils/blob/8d0cd750e7c3d9bea1d58902088c8177952450d6/SimpleRoboticsPythonUtils/simple_robotics_python_utils/common/ros1bag_ros2bag_conversions.py).
- [See here for how to do image processing on bag data](https://github.com/RicoJia/dream_cartographer/blob/9f29311bc65e55e983baf195b5bd1d8deefdd7d5/scripts/remove_objects_processing.py)
