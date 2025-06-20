---
layout: post
title: Robotics General Design Notes 
date: '2024-11-11 13:19'
subtitle: What's New In ROS2, Affordance
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Software Structuring

- Build a simulation / dataset for debugging software is key. Otherwise, you will have a lot of overhead of on-robot hardware testing.
  - Have three general feature flags / launch flags of your robot software:
    - production
    - simulation: no redundant visualization, so you can run simulation with small amount of compute
    - debug: bring up necessary visualization, log messages. the real-time performance could be sub-par, or certain parts could crash due to compute constrants.

- 2D and 3D visualization for results.

### General Guidelines

- Use consistent, simple notations
- Use modern C++ (>C++17)
  - ROS2 humble's default cpp standard is cpp 17. What if I want to have code in cpp 20?
    - Generally, it is okay to use newer C++ versions as long as their features and flags do not end in public APIs.
    - I.e. using modules would not be possible, but e.g. some stdlib goodies would be okay to use if they are only used in the .cpp files.
- Use unit test and module tests

## Safety Critical Nodes Should Be Long Running

- What about services that must be run outside the docker?
  - Systemd Services include micro-controller restart
  - Hardware drivers should be implemented as daemons
    - If I need to implement hardware drivers, say talking through I2C outside of container, how would I achieve inter-process communication. Maybe a simple TCP port is a good starting point.
  - They can have a TCP port and accept JSON / protobuf
    - Need to profile JSON / Protobuf for our use case

## Docker Container Strategy

- Start with one container, expand along the way. Currently, we have one container. As the business grows, we might want to separate the firmware one, the robot-software one.
  - Start with a minimal base image to reduce the container size and overhead.
  - Different containers might need networking properly, such as bridging and host-networking, extra DDS config
  - Deployment could be trickier with docker versioning

- What about disk space overhead for two images?

- How much overhead does Docker container create on Rpi 5? From personal experiences, almost 0
  - Use docker stats to keep track of memory usage.

### Versioning?

- Each ROS2 version is tied to a set of Ubuntu versions. There are usually breaking changes between each version of ROS2. A typical tech debt situation here is: **"Rpi 5 needs newest ROS2 version for some hardware compatibility. This breaks code written for rpi 4."**
  - This creates multiple dependencies that turn into hours to solve bugs, incompatibilities and create workarounds.
  - Pre-built binaries do not have the same ABI either.
  - Solution: extract out firmcode in its own environment? (TODO) The firmware should have **consistent**, and **very simple** input/output ROS2 messaging system. They also shouldn't have many ROS2 dependencies per-se either. If so, they should be forked, and locally built. [1]

### Hardware Access Complexity?

- Do we want to be careful at the very beginning? (TODO) This might slow down our initial sprint velocity

    ```bash
    docker run -it 
        --privileged \
        --device /dev/ttyACM0 \
    my_ros2_image
    ```

- **ARM64 docker container to access SPI, GPIO, serial, etc. is a PAIN**
- Nvidia driver may not be available. This could impact camera capabilities

### Design-Deploy Cycle

- When designing, we need to modify files. How do we make the distinction of testing and deployment? Also, what if a bug comes up in deployment, and we want to change things?
- Make ROS2 into an Image (Procedure TODO). The procedure is as simple as "pull-run-done"

### Design Choices (Which Could Change Over Time, Of Course)

- As of 2025, ROS2 Humble's Hawksbill is the first Ubuntu22.04 distro. It's supported until May 2027

## Reproducibility

- Traceable Logging System

  - Prometheus, collectd->influxdb->Grafana (trace), time series
    - Victoria Metrics
    - <https://discourse.ros.org/t/how-do-you-mitigate-robot-bugs-and-issues-related-to-system-resources/15682/4>
    - A trace tree, or distributed tracing in microservices
      - opentracing? + yegger?
    - Datadog / elk

- Warnings might be helpful for when nodes go down

## Logging

## ROS2 Specific Designs

- It's good to have a struct for ROS message equivalents. Say we want to calculate IMU information. We can have 1 `IMUData` to represent it, and at the ROS Node level, we publish it using a ROS message. Similarly, once we receive an IMU message, we immediately convert it to `IMUData`. This way, we can modify internal states in `IMUData` without touching the ROS message (if unecessary).

## The "Soft" Architectures

### Maintainability

- Have a **searchable** runbook. Coda is kind of bad with searching. Confluence was better (2 yrs ago).
  - Be concise and to the point. Lesser important details are left to the back.
- As the team grows, have an on-call schedule (unfortunately..). But this probably wouldn't be too bad unless we debug in very different timezones.

### Code Sharing

"Fork -> submodule add -> submodule init"

- Check for versioning. BSD-X and MIT Licenses are usually fine. GPL is generally not good for forking.

LLM should be encouraged, **especially for boring boiler plates** of ROS2 or many other systems. However, try not to put any business logic in it to [prevent potential code leak](https://www.forbes.com/sites/siladityaray/2023/05/02/samsung-bans-chatgpt-and-other-chatbots-for-employees-after-sensitive-code-leak/). Ask generic, modular questions instead.

### Good Culture I Noticed From Successful Startups

- Let's listen to each other. It's a safe place to voice ideas
  - "we don't yuck others' yum"
- Let's help each other to achieve our common success
  - Decently-prepared brainstorms should be encouraged

## Affordance

Affordance源自心理学和认知科学的术语，最早由心理学家James J. Gibson提出，用于描述环境所允许个人能实现的功能. affordance被迁移到机器人学科，强调的是智能体通过感知外界输入，以决定对物体有哪些可以被执行的操作, basically it means "can be used for".

- “A wad of paper affords tossing.”纸团可以被扔
- “A wastebasket affords receiving tossed objects.”废纸篓可以接受扔来的东西
- “Buttons drawn as 3D shapes appear to “stick out” and hence afford pushing.”按钮被画成3D突起以被按压
  - In the SayCan paper: affordances refer to possible actions that can be executed, which is conceptually similar to inferring preconditions in planning – what actions are feasible in a certain situation

## References

[1] <https://medium.com/@forrestallison/my-problems-with-ros2-and-why-im-going-my-own-way-and-salty-about-it-4802146eca89>
