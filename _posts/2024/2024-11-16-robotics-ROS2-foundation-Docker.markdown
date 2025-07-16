---
layout: post
title: Robotics - [ROS2 Foundation 2] ROS2 in Docker
date: '2024-11-16 13:19'
subtitle: Docker Networking, Multi-stage Build
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

## Multi-stage Build

What is the multi-level build:
    - Dockerfile.base
    - Dockerfile.devel
    - Dockerfile.base

Docker compose:

```yaml
services:
  base:
      dockerfile: Dockerfile.base
    image: myapp/base:dev

  devel:
    build:
      dockerfile: Dockerfile.devel
      args:
        BASE_IMAGE: myapp/base:dev
    image: myapp/devel:dev

  app:
    build:
      dockerfile: Dockerfile
      args:
        BASE_IMAGE: myapp/base:dev
        DEVEL_IMAGE: myapp/devel:dev
    image: myapp/app:dev
    depends_on:
      - devel

```

- `Dockerfile.base`, `Dockerfile.devel`, `Dockerfile.app`:

```
ARG BASE_IMAGE="code.hmech.us:5050/nautilus/common/dockers/toolkitt_developer/devel:latest"
```
