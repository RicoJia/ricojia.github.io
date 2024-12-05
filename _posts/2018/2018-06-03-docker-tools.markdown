---
layout: post
title: Docker Image Building
date: '2024-06-03 13:19'
subtitle: Dockerfile
comments: true
tags:
    - Docker
---

## Construct Of An Docker Image And Its Building Process

A Docker image is like a box of Oreo. Its final image is like multiple Oreos (layer) stacked on top of each other. A layer could be built from multiple building stages (like our favorite chocolate topping), and each stage consists of smaller layers.

Now Let's walk through a basic example

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/dd3fa345-6311-4975-b33f-349e122373fb" height="300" alt=""/>
        <figcaption><a href="https://betterprogramming.pub/container-images-are-like-cakes-ba9040cf18e9">Source: Sunny Beatteay</a></figcaption>
    </figure>
</p>
</div>

```dockerfile
# Stage 0: Build Stage
FROM node:18-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
RUN npm run build

# Stage 1: Final Stage
FROM node:18-alpine

WORKDIR /app
COPY --from=build /app/dist ./dist
CMD ["node", "dist/main.js"]
```

**Layers:** Docker images are built in layers.

```bash
WORKDIR /app
COPY package*.json ./
RUN npm install --production
```

- Each layer is an instruction such as `RUN`, `COPY`, `ADD`, etc. Each layer is stacked on top of each other. The above is a three-layer model.

**Stages:** The above is a two layer build. Each stage starts from a `FROM` command, which launches a base image.

- Stage 0 creates artifacts / build of the custom app (e.g., `RUN npx run build`). It consists of several layers
- Stage 1 is created by launching a new base image `node:18-alpine` and copying the build artifacts from Stage 0. The final image **only contains layers from the last stage and files copied from previous stages**. So Stage 0 is effectively discarded.

- To use a custom Dockerfile, use the `-f` arg: `docker build -f Dockerfile_test_container . -t ros-noetic-simple-robotics-tests`

### Create An Entrypoint

An entrypoint is a script that runs once a container starts. Here, we need to copy the entry point to a common location, then execute it

```python
COPY entrypoint_rgbd_slam_docker.sh /entrypoint_rgbd_slam_docker.sh
# RUN chmod +x /entrypoint_rgbd_slam_docker.sh
ENTRYPOINT ["/entrypoint_rgbd_slam_docker.sh"]
# must follow ENTRYPOINT, otherwise the container won't execute bash
CMD ["bash"]    
```

In entrypoint, make sure the last line has

```bash
...

# Execute the CMD command (bash) by replacing the shell with the CMD command in Dockerfile
exec "$@"
```

### Changing Python Version to Python 3.10 (Not Recommended For Potential Code Breaking)

```dockerfile
# Add Deadsnakes PPA and install Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Update alternatives to set python3 to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Set environment variables for Python
ENV PYTHON_VERSION=3.10
ENV PYTHON_BIN=/usr/bin/python${PYTHON_VERSION}
ENV PIP_BIN=/usr/local/bin/pip

# Upgrade pip and setuptools, then install a compatible PyYAML version
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --force-reinstall --ignore-installed PyYAML==6.0
```
