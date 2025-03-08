---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Bags
date: '2024-11-30 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Introduction

[`ros2 bag` is a command line tool for recording data published on topics in your system](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html). It accumulates the data passed on any number of topics and saves it in a database. You can then replay the data to reproduce the results of your tests and experiments. Here are some common CLI actions:

- To record:
    - `ros2 bag record -o MyOutputBag /turtle1/cmd_vel /turtle1/pose`
- To check bag information:
    - `ros2 bag info MyOutputBag`

## Ros Bag Recording From a Node

[Here is an example from the ROS2 website](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)

```python
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from std_msgs.msg import String
import rosbag2_py

class SimpleBagRecorder(Node):
    def __init__(self):
        super().__init__('simple_bag_recorder')
        self.writer = rosbag2_py.SequentialWriter()

        storage_options = rosbag2_py._storage.StorageOptions(
            uri='my_bag',
            storage_id='sqlite3')
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='chatter',
            type='std_msgs/msg/String',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.topic_callback,
            10)
        self.subscription

    def topic_callback(self, msg):
        self.writer.write(
            'chatter',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)


def main(args=None):
    rclpy.init(args=args)
    sbr = SimpleBagRecorder()
    rclpy.spin(sbr)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- `SequentialWriter` writes messages by the order they are received

By default, ROS 2 bag files use SQLite3 as the storage backend. However, ROS 2 supports different storage formats (e.g., MCAP) depending on the plugin being used. The default storage format is sqlite3: ros2 bag record -s sqlite3 /imu_data /scan. The recorded data is stored as an SQLite database file (*.db) inside the bag directory. 

[One can even synthesize bag data and does not create ros topics](https://docs.ros.org/en/galactic/Tutorials/Advanced/Recording-A-Bag-From-Your-Own-Node-Py.html)
