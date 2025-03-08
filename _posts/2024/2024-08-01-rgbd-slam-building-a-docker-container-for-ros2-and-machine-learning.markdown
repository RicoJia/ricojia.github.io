---
layout: post
title: RGBD SLAM - Building A ROS 2 Docker Container For Object Detection 
date: '2024-08-01 13:19'
subtitle: ROS 2 Docker Container For Object Detection Training And Inferencing
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - RGBD Slam
    - ROS2
    - Deep Learning
comments: true
---

## Dockerfile

- I installed jupyter notebook so I can experiment with it remotely.

```bash
# -U is to upgrade to the latest version
RUN python3 -m pip install -U jupyter notebook 
# jupyter notebook port
EXPOSE 8888
```

## Docker Runtime Args

- `--runtime=nvidia`: enable Nvidia Container Runtime, a "runtime" that connects Nvidia GPU with docker. If your laptop doesn't have an Nvidia GPU, simply remove this arg.
- `-p 8888` to expose the Jupyter Notebook Port
- `jupyter notebook --ip=0.0.0.0 --no-browser --allow-root`follows the image name. This is the command to run. without `-d`, docker will run this process in the foreground, which will terminates the container upon `SIGINT` (ctrl-C)
- Make sure `/dev/shm` is shared properly
    - On the same machine, ROS2 uses the directory for messaging
    - But this wouldn't work `-d /dev/shm:/dev/shm`
    - use `-v` instead

## Push To Docker Hub

```bash
docker login  #put in docker creds
docker tag <MY_IMAGE>:latest <MY_USERNAME>/<MY_IMAGE>:latest
docker push <MY_USERNAME>/<MY_IMAGE>:latest
```

- Docker Hub requires images to be pushed to the repo that corresponds to our Docker Hub username. So, we need to **tag** the image with the proper username.
