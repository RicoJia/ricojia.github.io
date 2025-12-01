---
layout: post
title: Robotics - [3D SLAM - 5] LIO Hands-On
date: '2025-4-8 13:19'
subtitle: 
header-img: "img/post-bg-o.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

## Issues That I Encountered

- Synchronization
    - THere are cases where you lose IMU for a bit. There could be a lag from the liadr to the imu as well. 
    - What's strange is ROS publisher / ROS bag writer/ lidar driver is bursty. lidar data could come >10s later. And sometimes, IMU data could be missing for that period
    - IMU could arrive later than lidar. Sometimes, Lidar Messages appear bursty over the wire; However, we can always expect IMU messages will arrive at some point, in a chronologically increasing fashion. 
    - Coping strategy:
        1. buffer imu and lidar messages in buffer queues
        2. Every time a new imu / lidar message comes in, trigger processing:
            while lidar_buffer not empty:
                lidar_msg = lidar_buffer.front()
                last_imu_pointer = find_imu_msgs_until_lidar_msg(imu_buffer_)
                if (last_imu_pointer == imu_buffer_.begin())
                    should_use_lidar_odometry
                    print_warning
                    break;
                if (last_imu_pointer == imu_buffer_.end())
                    // all imu messages are valid but more valid ones might be coming  - break and come back here later
                    break;

        3. append [IMU_buf_, last_imu_pointer] to measurement, 
        4. call callback_(measurement)



