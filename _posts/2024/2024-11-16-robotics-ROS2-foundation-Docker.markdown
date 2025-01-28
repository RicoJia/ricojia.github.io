---
layout: post
title: Robotics - [ROS2 Foundation 2] ROS2 in Docker
date: '2024-11-16 13:19'
subtitle: Various Things to Note For Building Dockerized ROS2 App
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
    - Docker
comments: true
---

## Networking

- Using `--net=host` implies both DDS participants believe they are in the same machine and they try to communicate using SharedMemory instead of UDP. So we need to enable SharedMemory between host and container. For this you should share /dev/shm:

```
docker run -ti --net host -v /dev/shm:/dev/shm <DOCKER_IMAGE>
```